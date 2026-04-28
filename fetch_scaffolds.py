#!/usr/bin/env python3
import json
import os
import sys
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import requests


SODA_BASE_URL = "https://data.cityofnewyork.us/resource"
GEOCLIENT_BASE_URL = "https://maps.nyc.gov/geoclient/v1"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PAGE_SIZE = 50000
OUTPUT_PATH = "scaffolds.json"

BOROUGH_MAP = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
}

USER_AGENT = "SidewalkShed/0.1 (+https://data.cityofnewyork.us/)"


def log(message: str) -> None:
    print(f"[SidewalkShed] {message}", flush=True)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    if app_token:
        session.headers["X-App-Token"] = app_token
    return session


SESSION = make_session()


def fetch_json(url: str, params: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    response = SESSION.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_float(raw: Optional[str]) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def borough_label(raw: Optional[str]) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    return BOROUGH_MAP.get(raw, raw.title())


def build_address(row: Dict[str, str]) -> str:
    house = (row.get("house__") or "").strip()
    street = (row.get("street_name") or "").strip()
    borough = borough_label(row.get("borough"))
    zip_code = (row.get("zip_code") or "").strip()

    street_line = " ".join(part for part in (house, street) if part).strip()
    if not street_line:
        return ""
    city_line = ", ".join(part for part in (borough, "NY") if part).strip(", ")
    address_parts = [part for part in (street_line, city_line, zip_code) if part]
    return ", ".join(address_parts)


def iso_date(raw: Optional[str]) -> Optional[str]:
    parsed = parse_date(raw)
    return parsed.isoformat() if parsed else None


def candidate_score(candidate: Dict[str, object]) -> Tuple[int, int, int]:
    has_coords = int(candidate.get("lat") is not None and candidate.get("lng") is not None)
    permit_end = candidate.get("_permit_end_date")
    permit_start = candidate.get("_permit_start_date")
    end_ord = permit_end.toordinal() if isinstance(permit_end, date) else -1
    start_ord = permit_start.toordinal() if isinstance(permit_start, date) else -1
    return (end_ord, start_ord, has_coords)


def merge_candidate(
    existing: Optional[Dict[str, object]], incoming: Dict[str, object]
) -> Dict[str, object]:
    if existing is None:
        return incoming

    winner = existing
    loser = incoming
    if candidate_score(incoming) > candidate_score(existing):
        winner = incoming
        loser = existing

    if winner.get("lat") is None and loser.get("lat") is not None:
        winner["lat"] = loser["lat"]
    if winner.get("lng") is None and loser.get("lng") is not None:
        winner["lng"] = loser["lng"]

    return winner


def print_sample(label: str, rows: List[Dict[str, str]]) -> None:
    log(f"{label}: {len(rows)} sample rows")
    if not rows:
        return
    log(f"{label}: raw fields -> {', '.join(sorted(rows[0].keys()))}")
    print(json.dumps(rows[0], indent=2, sort_keys=True), flush=True)


def explore_live_datasets() -> None:
    log("Exploring NYC Open Data endpoints")

    rows_29du = fetch_json(
        f"{SODA_BASE_URL}/29du-2wzn.json",
        params={"$limit": "2"},
    )
    print_sample("29du-2wzn scaffold filter view", rows_29du)

    rows_sc = fetch_json(
        f"{SODA_BASE_URL}/ipu4-2q9a.json",
        params={"$where": "job_type='SC'", "$limit": "2"},
    )
    print_sample("ipu4-2q9a with job_type='SC'", rows_sc)

    rows_sh = fetch_json(
        f"{SODA_BASE_URL}/ipu4-2q9a.json",
        params={"$where": "permit_subtype='SH'", "$limit": "2"},
    )
    print_sample("ipu4-2q9a with permit_subtype='SH'", rows_sh)

    if not rows_sc:
        log("Live dataset returned zero rows for job_type='SC'; using permit_subtype='SH' for final extraction")


def fetch_active_scaffolds() -> Dict[str, Dict[str, object]]:
    log("Fetching scaffold permit rows from ipu4-2q9a")

    select_fields = ",".join(
        [
            "borough",
            "house__",
            "street_name",
            "zip_code",
            "job__",
            "job_doc___",
            "permit_sequence__",
            "permit_si_no",
            "permit_status",
            "filing_status",
            "filing_date",
            "issuance_date",
            "expiration_date",
            "job_start_date",
            "gis_latitude",
            "gis_longitude",
        ]
    )

    active_by_address: Dict[str, Dict[str, object]] = {}
    today = date.today()
    total_rows = 0
    active_rows = 0
    expired_rows = 0
    offset = 0

    while True:
        params = {
            "$select": select_fields,
            "$where": "permit_subtype='SH'",
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        }
        rows = fetch_json(f"{SODA_BASE_URL}/ipu4-2q9a.json", params=params)
        if not rows:
            break

        total_rows += len(rows)
        log(f"Fetched {len(rows)} rows at offset {offset} (running total: {total_rows})")

        for row in rows:
            permit_end = parse_date(row.get("expiration_date"))
            if permit_end and permit_end < today:
                expired_rows += 1
                continue

            active_rows += 1
            address = build_address(row)
            if not address:
                continue

            permit_start_raw = (
                row.get("issuance_date") or row.get("job_start_date") or row.get("filing_date")
            )
            candidate = {
                "address": address,
                "lat": parse_float(row.get("gis_latitude")),
                "lng": parse_float(row.get("gis_longitude")),
                "permit_start": iso_date(permit_start_raw),
                "permit_end": iso_date(row.get("expiration_date")),
                "_permit_start_date": parse_date(permit_start_raw),
                "_permit_end_date": permit_end,
                "_house": (row.get("house__") or "").strip(),
                "_street": (row.get("street_name") or "").strip(),
                "_borough": borough_label(row.get("borough")),
                "_zip_code": (row.get("zip_code") or "").strip(),
            }
            active_by_address[address] = merge_candidate(active_by_address.get(address), candidate)

        offset += len(rows)
        if len(rows) < PAGE_SIZE:
            break

    log(f"Total scaffold rows seen: {total_rows}")
    log(f"Expired rows skipped: {expired_rows}")
    log(f"Active rows retained before dedupe: {active_rows}")
    log(f"Unique active addresses after dedupe: {len(active_by_address)}")
    return active_by_address


def geocode_with_geoclient(row: Dict[str, object]) -> Optional[Tuple[float, float]]:
    app_id = os.environ.get("NYC_GEOCLIENT_APP_ID")
    app_key = os.environ.get("NYC_GEOCLIENT_APP_KEY")
    if not app_id or not app_key:
        return None

    params = {
        "houseNumber": str(row.get("_house") or ""),
        "street": str(row.get("_street") or ""),
        "app_id": app_id,
        "app_key": app_key,
    }
    borough = str(row.get("_borough") or "")
    zip_code = str(row.get("_zip_code") or "")
    if borough:
        params["borough"] = borough
    elif zip_code:
        params["zip"] = zip_code
    else:
        return None

    response = SESSION.get(f"{GEOCLIENT_BASE_URL}/address.json", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json().get("address", {})
    lat = parse_float(payload.get("latitude"))
    lng = parse_float(payload.get("longitude"))
    if lat is None or lng is None:
        return None
    return lat, lng


def geocode_with_nominatim(row: Dict[str, object]) -> Optional[Tuple[float, float]]:
    params = {
        "q": str(row["address"]),
        "format": "jsonv2",
        "limit": "1",
    }
    email = os.environ.get("NOMINATIM_EMAIL")
    if email:
        params["email"] = email

    try:
        response = SESSION.get(NOMINATIM_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        top = payload[0]
        lat = parse_float(top.get("lat"))
        lng = parse_float(top.get("lon"))
        if lat is None or lng is None:
            return None
        return lat, lng
    finally:
        time.sleep(1.0)


def fill_missing_coordinates(active_by_address: Dict[str, Dict[str, object]]) -> None:
    missing = [row for row in active_by_address.values() if row["lat"] is None or row["lng"] is None]
    log(f"Rows needing geocoding: {len(missing)}")
    if not missing:
        return

    geocode_cache: Dict[str, Optional[Tuple[float, float]]] = {}
    successes = 0

    for index, row in enumerate(missing, start=1):
        address = str(row["address"])
        if address not in geocode_cache:
            log(f"Geocoding {index}/{len(missing)}: {address}")
            coords = None
            try:
                coords = geocode_with_geoclient(row)
                if coords is None:
                    coords = geocode_with_nominatim(row)
            except requests.RequestException as exc:
                log(f"Geocoding failed for {address}: {exc}")
            geocode_cache[address] = coords

        coords = geocode_cache[address]
        if coords is None:
            continue

        row["lat"], row["lng"] = coords
        successes += 1

    log(f"Geocoding filled coordinates for {successes} rows")


def write_output(active_by_address: Dict[str, Dict[str, object]]) -> None:
    output_rows = []
    for row in sorted(active_by_address.values(), key=lambda item: str(item["address"])):
        output_rows.append(
            {
                "address": row["address"],
                "lat": row["lat"],
                "lng": row["lng"],
                "permit_start": row["permit_start"],
                "permit_end": row["permit_end"],
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, indent=2)

    unresolved = sum(1 for row in output_rows if row["lat"] is None or row["lng"] is None)
    log(f"Wrote {len(output_rows)} scaffold locations to {OUTPUT_PATH}")
    log(f"Rows still missing coordinates after geocoding: {unresolved}")


def main() -> int:
    try:
        explore_live_datasets()
        active_by_address = fetch_active_scaffolds()
        fill_missing_coordinates(active_by_address)
        write_output(active_by_address)
    except requests.RequestException as exc:
        log(f"HTTP error: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - last-resort CLI guard
        log(f"Unexpected error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
