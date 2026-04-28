#!/usr/bin/env python3
import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
SCAFFOLDS_PATH = BASE_DIR / "scaffolds.json"
SNAP_CACHE_PATH = BASE_DIR / ".scaffold_snap_cache.json"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/foot"
OSRM_NEAREST_URL = "https://router.project-osrm.org/nearest/v1/foot"
WTTR_URL = "https://wttr.in/New York"
EARTH_RADIUS_M = 6371000.0
ROUTE_HIT_THRESHOLD_M = 50.0
WAYPOINT_COVERAGE_RADIUS_M = 70.0
MAX_WAYPOINTS = 3
MAX_CANDIDATES = 8
BEAM_WIDTH = 3
OSRM_TIMEOUT_SECONDS = 8
WEATHER_TTL_SECONDS = 600

DETOUR_MODES = {
    "minimal": {
        "max_extra_pct": 10.0,
        "min_cos": 0.8,
        "label": "Stay close",
    },
    "moderate": {
        "max_extra_pct": 30.0,
        "min_cos": 0.5,
        "label": "A bit wetter is fine",
    },
    "max": {
        "max_extra_pct": 75.0,
        "min_cos": 0.2,
        "label": "Keep me dry",
    },
}

app = Flask(__name__)
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "SidewalkShed/0.3"})
SNAP_CACHE_LOCK = threading.Lock()
WEATHER_LOCK = threading.Lock()
WEATHER_CACHE: Dict[str, object] = {"fetched_at": 0.0, "payload": None}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def coord_key(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"


def load_snap_cache() -> Dict[str, Dict[str, float]]:
    if not SNAP_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(SNAP_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


SNAP_CACHE = load_snap_cache()


def save_snap_cache() -> None:
    try:
        SNAP_CACHE_PATH.write_text(json.dumps(SNAP_CACHE), encoding="utf-8")
    except OSError:
        pass


def load_scaffolds() -> List[Dict[str, object]]:
    with SCAFFOLDS_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    cleaned = []
    for index, row in enumerate(rows):
        if row.get("lat") is None or row.get("lng") is None:
            continue
        lat = float(row["lat"])
        lng = float(row["lng"])
        key = coord_key(lat, lng)
        snapped = SNAP_CACHE.get(key)
        cleaned.append(
            {
                **row,
                "_id": index,
                "_coord_key": key,
                "orig_lat": lat,
                "orig_lng": lng,
                "snap_lat": float(snapped["lat"]) if snapped else None,
                "snap_lng": float(snapped["lng"]) if snapped else None,
            }
        )
    return cleaned


SCAFFOLDS = load_scaffolds()


def get_scaffold_point(scaffold: Dict[str, object], snapped: bool = False) -> Tuple[float, float]:
    if snapped:
        if scaffold.get("snap_lat") is not None and scaffold.get("snap_lng") is not None:
            return float(scaffold["snap_lat"]), float(scaffold["snap_lng"])
    return float(scaffold["orig_lat"]), float(scaffold["orig_lng"])


def snap_coordinate(lat: float, lng: float) -> Tuple[float, float]:
    response = HTTP.get(
        f"{OSRM_NEAREST_URL}/{lng:.6f},{lat:.6f}",
        params={"number": "1"},
        timeout=OSRM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "Ok" or not payload.get("waypoints"):
        return lat, lng
    snapped_lng, snapped_lat = payload["waypoints"][0]["location"]
    return float(snapped_lat), float(snapped_lng)


def ensure_scaffold_snapped(scaffold: Dict[str, object]) -> Tuple[float, float]:
    if scaffold.get("snap_lat") is not None and scaffold.get("snap_lng") is not None:
        return float(scaffold["snap_lat"]), float(scaffold["snap_lng"])

    key = str(scaffold["_coord_key"])
    with SNAP_CACHE_LOCK:
        cached = SNAP_CACHE.get(key)
        if cached:
            scaffold["snap_lat"] = float(cached["lat"])
            scaffold["snap_lng"] = float(cached["lng"])
            return float(scaffold["snap_lat"]), float(scaffold["snap_lng"])

    lat, lng = get_scaffold_point(scaffold, snapped=False)
    try:
        snapped_lat, snapped_lng = snap_coordinate(lat, lng)
    except requests.RequestException:
        snapped_lat, snapped_lng = lat, lng

    scaffold["snap_lat"] = snapped_lat
    scaffold["snap_lng"] = snapped_lng
    with SNAP_CACHE_LOCK:
        SNAP_CACHE[key] = {"lat": snapped_lat, "lng": snapped_lng}
        save_snap_cache()
    return snapped_lat, snapped_lng


def latlng_to_xy(lat: float, lng: float, origin_lat_rad: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    return (
        lng_rad * math.cos(origin_lat_rad) * EARTH_RADIUS_M,
        lat_rad * EARTH_RADIUS_M,
    )


def meters_between(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    origin_lat_rad = math.radians((start[0] + end[0]) / 2.0)
    ax, ay = latlng_to_xy(start[0], start[1], origin_lat_rad)
    bx, by = latlng_to_xy(end[0], end[1], origin_lat_rad)
    return math.hypot(bx - ax, by - ay)


def cosine_forwardness(origin: Tuple[float, float], point: Tuple[float, float], end: Tuple[float, float]) -> float:
    origin_lat_rad = math.radians((origin[0] + end[0]) / 2.0)
    ox, oy = latlng_to_xy(origin[0], origin[1], origin_lat_rad)
    px, py = latlng_to_xy(point[0], point[1], origin_lat_rad)
    ex, ey = latlng_to_xy(end[0], end[1], origin_lat_rad)
    vx = px - ox
    vy = py - oy
    ux = ex - ox
    uy = ey - oy
    v_norm = math.hypot(vx, vy)
    u_norm = math.hypot(ux, uy)
    if v_norm == 0 or u_norm == 0:
        return 0.0
    return clamp(((vx * ux) + (vy * uy)) / (v_norm * u_norm), -1.0, 1.0)


def point_to_segment_distance_and_t(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[float, float]:
    origin_lat_rad = math.radians((start[0] + end[0]) / 2.0)
    px, py = latlng_to_xy(point[0], point[1], origin_lat_rad)
    ax, ay = latlng_to_xy(start[0], start[1], origin_lat_rad)
    bx, by = latlng_to_xy(end[0], end[1], origin_lat_rad)
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    length_sq = (abx * abx) + (aby * aby)
    if length_sq == 0:
        return math.hypot(px - ax, py - ay), 0.0
    t = clamp(((apx * abx) + (apy * aby)) / length_sq, 0.0, 1.0)
    closest_x = ax + (t * abx)
    closest_y = ay + (t * aby)
    return math.hypot(px - closest_x, py - closest_y), t


def point_to_segment_distance(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    distance, _ = point_to_segment_distance_and_t(point, start, end)
    return distance


def point_to_linestring_distance(point: Tuple[float, float], coordinates: Sequence[Sequence[float]]) -> float:
    if len(coordinates) < 2:
        return float("inf")
    minimum = float("inf")
    for first, second in zip(coordinates, coordinates[1:]):
        start = (first[1], first[0])
        end = (second[1], second[0])
        minimum = min(minimum, point_to_segment_distance(point, start, end))
        if minimum <= ROUTE_HIT_THRESHOLD_M:
            return minimum
    return minimum


def point_to_linestring_distance_and_progress(
    point: Tuple[float, float],
    coordinates: Sequence[Sequence[float]],
) -> Tuple[float, float]:
    if len(coordinates) < 2:
        return float("inf"), 0.0

    lengths = []
    total_length = 0.0
    for first, second in zip(coordinates, coordinates[1:]):
        start = (first[1], first[0])
        end = (second[1], second[0])
        segment_length = meters_between(start, end)
        lengths.append(segment_length)
        total_length += segment_length

    traversed = 0.0
    best_distance = float("inf")
    best_progress = 0.0
    for (first, second), segment_length in zip(zip(coordinates, coordinates[1:]), lengths):
        start = (first[1], first[0])
        end = (second[1], second[0])
        distance, t = point_to_segment_distance_and_t(point, start, end)
        if distance < best_distance:
            best_distance = distance
            best_progress = (traversed + (segment_length * t)) / total_length if total_length > 0 else 0.0
        traversed += segment_length

    return best_distance, best_progress


def scaffold_ids_near_route(route_geometry: Dict[str, object], scaffolds: Sequence[Dict[str, object]]) -> Set[int]:
    coordinates = route_geometry.get("coordinates", [])
    hits: Set[int] = set()
    for scaffold in scaffolds:
        point = get_scaffold_point(scaffold, snapped=False)
        if point_to_linestring_distance(point, coordinates) <= ROUTE_HIT_THRESHOLD_M:
            hits.add(int(scaffold["_id"]))
    return hits


def count_scaffolds_near_route(route_geometry: Dict[str, object], scaffolds: Sequence[Dict[str, object]]) -> int:
    return len(scaffold_ids_near_route(route_geometry, scaffolds))


def request_osrm_route(points: Sequence[Tuple[float, float]]) -> Optional[Dict[str, object]]:
    coordinates = ";".join(f"{lng:.6f},{lat:.6f}" for lat, lng in points)
    response = HTTP.get(
        f"{OSRM_ROUTE_URL}/{coordinates}",
        params={"overview": "full", "geometries": "geojson", "steps": "false"},
        timeout=OSRM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "Ok" or not payload.get("routes"):
        return None
    return payload["routes"][0]


def normalize_precip_mm(raw: object) -> float:
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def suggested_mode_for_precip(precip_mm_hr: float) -> str:
    if precip_mm_hr <= 0.0:
        return "none"
    if precip_mm_hr < 2.0:
        return "minimal"
    if precip_mm_hr <= 10.0:
        return "moderate"
    return "max"


def fetch_weather() -> Dict[str, object]:
    response = HTTP.get(WTTR_URL, params={"format": "j1"}, timeout=OSRM_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    current = (payload.get("current_condition") or [{}])[0]
    precip_mm_hr = normalize_precip_mm(current.get("precipMM"))
    condition = ((current.get("weatherDesc") or [{"value": "Unknown"}])[0]).get("value", "Unknown")
    return {
        "condition": condition,
        "precip_mm_hr": precip_mm_hr,
        "suggested_mode": suggested_mode_for_precip(precip_mm_hr),
    }


def get_weather(force: bool = False) -> Dict[str, object]:
    with WEATHER_LOCK:
        cached = WEATHER_CACHE.get("payload")
        fetched_at = float(WEATHER_CACHE.get("fetched_at", 0.0))
        if not force and cached and (time.time() - fetched_at) < WEATHER_TTL_SECONDS:
            return cached  # type: ignore[return-value]

    try:
        payload = fetch_weather()
    except requests.RequestException:
        payload = {
            "condition": "Unknown",
            "precip_mm_hr": 0.0,
            "suggested_mode": "none",
        }

    with WEATHER_LOCK:
        WEATHER_CACHE["payload"] = payload
        WEATHER_CACHE["fetched_at"] = time.time()
    return payload


def corridor_scaffolds_for_route(
    shortest_route_geometry: Dict[str, object],
    shortest_distance_m: float,
    detour_mode: str,
) -> List[Dict[str, object]]:
    mode = DETOUR_MODES[detour_mode]
    corridor_m = {
        "minimal": 180.0,
        "moderate": 320.0,
        "max": 520.0,
    }[detour_mode]
    corridor_m = max(corridor_m, min(900.0, shortest_distance_m * 0.05))
    coordinates = shortest_route_geometry.get("coordinates", [])
    corridor = []

    for scaffold in SCAFFOLDS:
        point = get_scaffold_point(scaffold, snapped=False)
        distance_m, progress = point_to_linestring_distance_and_progress(point, coordinates)
        if distance_m > corridor_m or progress <= 0.01 or progress >= 0.99:
            continue
        corridor.append(
            {
                **scaffold,
                "_progress": progress,
                "_corridor_distance_m": distance_m,
                "_mode": mode,
            }
        )

    return corridor


def build_seed_candidates(corridor_scaffolds: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    raw_candidates = []
    for scaffold in corridor_scaffolds:
        point = get_scaffold_point(scaffold, snapped=False)
        coverage_ids: Set[int] = set()
        for neighbor in corridor_scaffolds:
            neighbor_point = get_scaffold_point(neighbor, snapped=False)
            if meters_between(point, neighbor_point) <= WAYPOINT_COVERAGE_RADIUS_M:
                coverage_ids.add(int(neighbor["_id"]))
        raw_candidates.append(
            {
                "scaffold": scaffold,
                "seed_score": len(coverage_ids) - (float(scaffold["_corridor_distance_m"]) / 100.0),
                "coverage_ids": coverage_ids,
            }
        )

    raw_candidates.sort(
        key=lambda item: (-item["seed_score"], float(item["scaffold"]["_progress"]))
    )

    selected = []
    for item in raw_candidates[:MAX_CANDIDATES]:
        scaffold = item["scaffold"]
        snapped_lat, snapped_lng = ensure_scaffold_snapped(scaffold)
        waypoint_point = (snapped_lat, snapped_lng)
        coverage_ids: Set[int] = set()
        for neighbor in corridor_scaffolds:
            neighbor_point = get_scaffold_point(neighbor, snapped=False)
            if meters_between(waypoint_point, neighbor_point) <= WAYPOINT_COVERAGE_RADIUS_M:
                coverage_ids.add(int(neighbor["_id"]))
        selected.append(
            {
                "id": int(scaffold["_id"]),
                "point": waypoint_point,
                "progress": float(scaffold["_progress"]),
                "coverage_ids": coverage_ids,
            }
        )
    return selected


def candidate_state(
    candidate: Dict[str, object],
    vertices: Sequence[Tuple[float, float]],
    end: Tuple[float, float],
    claimed_ids: Set[int],
    min_cos: float,
) -> Optional[Dict[str, object]]:
    new_coverage_ids = set(candidate["coverage_ids"]) - claimed_ids
    if not new_coverage_ids:
        return None

    best_insert_index = None
    best_delta = None
    best_forwardness = 0.0

    for segment_index in range(len(vertices) - 1):
        segment_start = vertices[segment_index]
        segment_end = vertices[segment_index + 1]
        forwardness = max(0.0, cosine_forwardness(segment_start, candidate["point"], end))
        if forwardness <= min_cos:
            continue
        delta_d = (
            meters_between(segment_start, candidate["point"])
            + meters_between(candidate["point"], segment_end)
            - meters_between(segment_start, segment_end)
        )
        if best_delta is None or delta_d < best_delta:
            best_delta = delta_d
            best_insert_index = segment_index + 1
            best_forwardness = forwardness

    if best_insert_index is None or best_delta is None:
        return None

    adjusted_delta = max(best_delta, 1.0)
    score = (math.sqrt(len(new_coverage_ids)) * best_forwardness) / (adjusted_delta ** 1.3)
    if score <= 0:
        return None

    return {
        "candidate": candidate,
        "new_coverage_ids": new_coverage_ids,
        "insert_index": best_insert_index,
        "delta_d": best_delta,
        "forwardness": best_forwardness,
        "score": score,
    }


def insert_waypoint(waypoints: Sequence[Dict[str, object]], candidate: Dict[str, object], insert_index: int) -> List[Dict[str, object]]:
    updated = list(waypoints)
    updated.insert(insert_index - 1, candidate)
    return updated


def build_vertices(
    start: Tuple[float, float], waypoints: Sequence[Dict[str, object]], end: Tuple[float, float]
) -> List[Tuple[float, float]]:
    return [start, *[tuple(waypoint["point"]) for waypoint in waypoints], end]


def beam_ranked_states(
    candidates: Sequence[Dict[str, object]],
    waypoints: Sequence[Dict[str, object]],
    start: Tuple[float, float],
    end: Tuple[float, float],
    claimed_ids: Set[int],
    min_cos: float,
) -> List[Dict[str, object]]:
    vertices = build_vertices(start, waypoints, end)
    states = []
    for candidate in candidates:
        if any(int(existing["id"]) == int(candidate["id"]) for existing in waypoints):
            continue
        state = candidate_state(candidate, vertices, end, claimed_ids, min_cos)
        if state is not None:
            states.append(state)

    states.sort(key=lambda item: item["score"], reverse=True)
    top_states = states[:BEAM_WIDTH]
    ranked = []
    for state in top_states:
        simulated_waypoints = insert_waypoint(waypoints, state["candidate"], int(state["insert_index"]))
        simulated_claimed = claimed_ids | set(state["new_coverage_ids"])
        next_vertices = build_vertices(start, simulated_waypoints, end)
        best_next_score = 0.0
        for candidate in candidates:
            if any(int(existing["id"]) == int(candidate["id"]) for existing in simulated_waypoints):
                continue
            next_state = candidate_state(candidate, next_vertices, end, simulated_claimed, min_cos)
            if next_state is not None:
                best_next_score = max(best_next_score, float(next_state["score"]))
        ranked.append(
            {
                **state,
                "lookahead_score": float(state["score"]) + (0.5 * best_next_score),
            }
        )

    ranked.sort(key=lambda item: item["lookahead_score"], reverse=True)
    return ranked


def build_scaffold_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    shortest_route: Dict[str, object],
    detour_mode: str,
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]], int]:
    mode = DETOUR_MODES[detour_mode]
    shortest_distance_m = float(shortest_route["distance"])
    max_distance_m = shortest_distance_m * (1.0 + (mode["max_extra_pct"] / 100.0))

    corridor_scaffolds = corridor_scaffolds_for_route(
        shortest_route["geometry"],
        shortest_distance_m,
        detour_mode,
    )
    candidates = build_seed_candidates(corridor_scaffolds)
    selected_waypoints: List[Dict[str, object]] = []
    claimed_ids: Set[int] = set()
    current_route = shortest_route

    while len(selected_waypoints) < MAX_WAYPOINTS:
        ranked_states = beam_ranked_states(
            candidates,
            selected_waypoints,
            start,
            end,
            claimed_ids,
            float(mode["min_cos"]),
        )
        if not ranked_states:
            break

        accepted = False
        for state in ranked_states:
            tentative_waypoints = insert_waypoint(
                selected_waypoints,
                state["candidate"],
                int(state["insert_index"]),
            )
            route_points = [start, *[tuple(waypoint["point"]) for waypoint in tentative_waypoints], end]
            try:
                route = request_osrm_route(route_points)
            except requests.RequestException:
                continue
            if route is None:
                continue

            route_distance = float(route["distance"])
            if route_distance > max_distance_m:
                continue
            if route_distance < (shortest_distance_m * 0.98):
                continue

            selected_waypoints = tentative_waypoints
            claimed_ids |= set(state["new_coverage_ids"])
            current_route = route
            accepted = True
            break

        if not accepted:
            break

    if not selected_waypoints:
        return None, [], 0

    scaffolds_covered = count_scaffolds_near_route(current_route["geometry"], SCAFFOLDS)
    return current_route, selected_waypoints, scaffolds_covered


def parse_point(payload: Dict[str, object], key: str) -> Tuple[float, float]:
    raw = payload.get(key) or {}
    return float(raw["lat"]), float(raw["lng"])


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/scaffolds")
def api_scaffolds():
    public_rows = [
        {
            "address": row["address"],
            "lat": row["orig_lat"],
            "lng": row["orig_lng"],
            "permit_start": row.get("permit_start"),
            "permit_end": row.get("permit_end"),
        }
        for row in SCAFFOLDS
    ]
    return jsonify(public_rows)


@app.route("/api/weather")
def api_weather():
    return jsonify(get_weather())


@app.route("/api/route", methods=["POST"])
def api_route():
    try:
        payload = request.get_json(force=True)
        start = parse_point(payload, "start")
        end = parse_point(payload, "end")
        detour_mode = str(payload["detour_mode"])
        if detour_mode not in DETOUR_MODES:
            raise ValueError("invalid detour mode")

        shortest_route = request_osrm_route([start, end])
        if shortest_route is None:
            return jsonify({"error": "No walking route found between A and B."}), 422

        scaffold_route, selected_waypoints, scaffold_covered = build_scaffold_route(
            start,
            end,
            shortest_route,
            detour_mode,
        )

        weather = get_weather()
        shortest_geojson = {
            "type": "Feature",
            "geometry": shortest_route["geometry"],
            "properties": {"label": "Shortest route"},
        }
        response = {
            "shortest": {
                "geojson": shortest_geojson,
                "distance_m": float(shortest_route["distance"]),
                "duration_s": float(shortest_route["duration"]),
                "scaffolds_covered": count_scaffolds_near_route(shortest_route["geometry"], SCAFFOLDS),
            },
            "scaffold_route": {
                "geojson": (
                    {
                        "type": "Feature",
                        "geometry": scaffold_route["geometry"],
                        "properties": {"label": "Scaffold-maximizing route"},
                    }
                    if scaffold_route is not None
                    else None
                ),
                "distance_m": float(scaffold_route["distance"]) if scaffold_route is not None else None,
                "duration_s": float(scaffold_route["duration"]) if scaffold_route is not None else None,
                "scaffolds_covered": scaffold_covered if scaffold_route is not None else None,
                "waypoints": [
                    {
                        "lat": waypoint["point"][0],
                        "lng": waypoint["point"][1],
                        "covered": len(waypoint["coverage_ids"]),
                    }
                    for waypoint in selected_waypoints
                ],
            },
            "weather": weather,
        }
        return jsonify(response)
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid route payload."}), 400
    except requests.RequestException as exc:
        return jsonify({"error": f"Routing request failed: {exc}"}), 502


if __name__ == "__main__":
    app.run(debug=True)
