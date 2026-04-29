#!/usr/bin/env python3
import json
import os
import pickle
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from flask import Flask, jsonify, render_template, request

from router import ScaffoldRouter


BASE_DIR = Path(__file__).resolve().parent
SCAFFOLDS_PATH = BASE_DIR / "scaffolds.json"
GRAPH_DATA_PATH = BASE_DIR / "graph_data.pkl"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_TIMEOUT_SECONDS = 10
ROUTE_HIT_THRESHOLD_DEGREES = 0.0004
WALKING_SPEED_MPS = 1.4

DETOUR_BIASES = {
    "minimal": 0.18,
    "moderate": 0.42,
    "max": 0.68,
}

app = Flask(__name__)
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "SidewalkShed/0.5"})


def log(message: str) -> None:
    print(f"[SidewalkShed] {message}", flush=True)


GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


def parse_point(payload: Dict[str, object], key: str) -> Tuple[float, float]:
    raw = payload.get(key) or {}
    return float(raw["lat"]), float(raw["lng"])


def parse_address(payload: Dict[str, object]) -> str:
    raw = str(payload.get("address", "")).strip()
    if not raw:
        raise ValueError("missing address")
    return raw


def load_scaffolds() -> List[Dict[str, object]]:
    if not SCAFFOLDS_PATH.exists():
        return []

    with SCAFFOLDS_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    cleaned: List[Dict[str, object]] = []
    for row in rows:
        if row.get("lat") is None or row.get("lng") is None:
            continue
        cleaned.append(
            {
                "address": row.get("address"),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "permit_start": row.get("permit_start"),
                "permit_end": row.get("permit_end"),
            }
        )
    return cleaned


def google_geocode(address: str):
    if not GOOGLE_MAPS_API_KEY:
        return None

    response = HTTP.get(
        GOOGLE_GEOCODE_URL,
        params={"address": address, "key": GOOGLE_MAPS_API_KEY, "region": "us"},
        timeout=GEOCODE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "OK":
        return None

    results = payload.get("results") or []
    if not results:
        return None

    first = results[0]
    location = ((first.get("geometry") or {}).get("location") or {})
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        return None

    return {
        "address": first.get("formatted_address") or address,
        "lat": float(lat),
        "lng": float(lng),
        "provider": "google",
    }


def nominatim_geocode(address: str):
    response = HTTP.get(
        NOMINATIM_SEARCH_URL,
        params={"q": address, "format": "jsonv2", "limit": "1", "addressdetails": "1"},
        timeout=GEOCODE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return None

    first = payload[0]
    lat = first.get("lat")
    lng = first.get("lon")
    if lat is None or lng is None:
        return None

    return {
        "address": first.get("display_name") or address,
        "lat": float(lat),
        "lng": float(lng),
        "provider": "nominatim",
    }


def geocode_address(address: str) -> Dict[str, object]:
    result = google_geocode(address)
    if result is not None:
        return result

    result = nominatim_geocode(address)
    if result is not None:
        return result

    raise LookupError("Address not found.")


def load_router() -> Tuple[dict, ScaffoldRouter]:
    if not GRAPH_DATA_PATH.exists():
        raise FileNotFoundError(
            "graph_data.pkl is missing. Run `python precompute.py` locally and commit the result."
        )

    tracemalloc.start()
    with GRAPH_DATA_PATH.open("rb") as handle:
        graph_data = pickle.load(handle)
    router = ScaffoldRouter(graph_data)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log(f"Router ready — {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    log(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    return graph_data, router


def build_route_payload(
    router: ScaffoldRouter,
    scaffolds: List[Dict[str, object]],
    node_path: List[int],
    distance_m: float,
) -> Dict[str, object]:
    coordinates = router.path_to_geojson(node_path)
    return {
        "geojson": {"type": "LineString", "coordinates": coordinates},
        "distance_m": float(distance_m),
        "duration_s": float(distance_m) / WALKING_SPEED_MPS,
        "scaffolds_covered": router.count_scaffolds_covered(
            coordinates,
            scaffolds,
            radius_deg=ROUTE_HIT_THRESHOLD_DEGREES,
        ),
        "waypoints": router.edge_waypoints(node_path),
    }


GRAPH_DATA, ROUTER = load_router()
SCAFFOLDS = load_scaffolds()
log(f"Scaffolds loaded: {len(SCAFFOLDS)}")
log("Ready.")


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/scaffolds")
def api_scaffolds():
    return jsonify(SCAFFOLDS)


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    try:
        payload = request.get_json(force=True)
        address = parse_address(payload)
        return jsonify(geocode_address(address))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid geocode payload."}), 400
    except LookupError:
        return jsonify({"error": "Address not found."}), 404
    except requests.RequestException as exc:
        return jsonify({"error": f"Geocoding request failed: {exc}"}), 502


@app.route("/api/route", methods=["POST"])
def api_route():
    try:
        payload = request.get_json(force=True)
        start = parse_point(payload, "start")
        end = parse_point(payload, "end")
        detour_mode = str(payload["detour_mode"])
        if detour_mode not in DETOUR_BIASES:
            raise ValueError("invalid detour mode")

        shortest_nodes, shortest_distance = ROUTER.route(start[0], start[1], end[0], end[1], 0.0)
        shortest = build_route_payload(ROUTER, SCAFFOLDS, shortest_nodes, shortest_distance)

        detour_bias = DETOUR_BIASES[detour_mode]
        scaffold_nodes, scaffold_distance = ROUTER.route(start[0], start[1], end[0], end[1], detour_bias)
        scaffold_route = build_route_payload(ROUTER, SCAFFOLDS, scaffold_nodes, scaffold_distance)

        return jsonify(
            {
                "shortest": shortest,
                "scaffold_route": scaffold_route,
            }
        )
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid route payload."}), 400
    except LookupError:
        return jsonify({"error": "No walking route found between A and B."}), 422
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
