#!/usr/bin/env python3
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import requests
from flask import Flask, jsonify, render_template, request
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point

BASE_DIR = Path(__file__).resolve().parent
SCAFFOLDS_PATH = BASE_DIR / "scaffolds.json"
GRAPHML_PATH = BASE_DIR / "manhattan_walk.graphml"
WTTR_URL = "https://wttr.in/New York"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
WEATHER_TIMEOUT_SECONDS = 8
GEOCODE_TIMEOUT_SECONDS = 10
WEATHER_TTL_SECONDS = 600
ROUTE_HIT_THRESHOLD_DEGREES = 0.0004
WALKING_SPEED_MPS = 1.4

DETOUR_LAMBDAS = {
    "minimal": 0.3,
    "moderate": 0.7,
    "max": 1.2,
}

EdgeKey = Tuple[Hashable, Hashable, Hashable]

app = Flask(__name__)
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "SidewalkShed/0.4"})
WEATHER_LOCK = threading.Lock()
WEATHER_CACHE: Dict[str, object] = {"fetched_at": 0.0, "payload": None}


def log(message: str) -> None:
    print(f"[SidewalkShed] {message}", flush=True)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEV_BAD_WEATHER_MODE = env_flag("SIDEWALKSHED_DEV_BAD_WEATHER")
DEV_WEATHER_OVERRIDE = {
    "condition": "Heavy rain",
    "precip_mm_hr": 12.0,
    "suggested_mode": "max",
}
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


def normalize_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


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


def coerce_graph_types(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    for _, data in graph.nodes(data=True):
        data["x"] = normalize_float(data.get("x"))
        data["y"] = normalize_float(data.get("y"))

    for _, _, _, data in graph.edges(keys=True, data=True):
        data["length"] = normalize_float(data.get("length"), 1.0)
        geometry = data.get("geometry")
        if isinstance(geometry, str) and geometry:
            try:
                data["geometry"] = wkt.loads(geometry)
            except Exception:
                data.pop("geometry", None)

    return graph


def load_graph() -> nx.MultiDiGraph:
    if GRAPHML_PATH.exists():
        graph = ox.load_graphml(filepath=str(GRAPHML_PATH))
    else:
        graph = ox.graph_from_place(
            "Manhattan, New York, USA",
            network_type="walk",
            retain_all=False,
        )
        ox.save_graphml(graph, filepath=str(GRAPHML_PATH))

    graph = coerce_graph_types(graph)
    log(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def edge_length(data: Dict[str, Any]) -> float:
    return normalize_float(data.get("length"), 1.0)


def route_edge_weight(
    scaffold_counts: Dict[EdgeKey, int],
    lambda_val: float,
    u: Hashable,
    v: Hashable,
    key: Hashable,
    data: Dict[str, Any],
) -> float:
    base = edge_length(data)
    bonus = scaffold_counts.get((u, v, key), 0) * lambda_val
    return max(base * 0.1, base - bonus)


def make_weight_fn(scaffold_counts: Dict[EdgeKey, int], lambda_val: float):
    def weight_fn(u: Hashable, v: Hashable, data: Dict[str, Any]) -> float:
        if "length" in data:
            return route_edge_weight(scaffold_counts, lambda_val, u, v, 0, data)

        return min(
            route_edge_weight(scaffold_counts, lambda_val, u, v, key, attrs)
            for key, attrs in data.items()
        )

    return weight_fn


def precompute_scaffold_coverage(
    graph: nx.MultiDiGraph,
    transformer: Transformer,
    scaffolds: Sequence[Dict[str, object]],
) -> Dict[EdgeKey, int]:
    scaffold_counts: Dict[EdgeKey, int] = {}
    if not scaffolds:
        return scaffold_counts

    lngs = [float(scaffold["lng"]) for scaffold in scaffolds]
    lats = [float(scaffold["lat"]) for scaffold in scaffolds]
    xs, ys = transformer.transform(lngs, lats)
    nearest_edges = ox.nearest_edges(graph, xs, ys)

    if (
        isinstance(nearest_edges, tuple)
        and len(nearest_edges) == 3
        and all(hasattr(part, "__len__") for part in nearest_edges)
    ):
        edge_iter = zip(nearest_edges[0], nearest_edges[1], nearest_edges[2])
    else:
        edge_iter = nearest_edges

    for edge in edge_iter:
        scaffold_counts[edge] = scaffold_counts.get(edge, 0) + 1
    return scaffold_counts


def node_xy(graph: nx.MultiDiGraph, node: Hashable) -> Tuple[float, float]:
    data = graph.nodes[node]
    return float(data["x"]), float(data["y"])


def sq_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx) + (dy * dy)


def edge_coordinates(
    graph: nx.MultiDiGraph,
    u: Hashable,
    v: Hashable,
    data: Dict[str, Any],
) -> List[Tuple[float, float]]:
    start = node_xy(graph, u)
    end = node_xy(graph, v)
    geometry = data.get("geometry")

    if geometry is not None and hasattr(geometry, "coords"):
        coords = [(float(x), float(y)) for x, y in geometry.coords]
    else:
        coords = [start, end]

    direct = sq_dist(coords[0], start) + sq_dist(coords[-1], end)
    reverse = sq_dist(coords[0], end) + sq_dist(coords[-1], start)
    if reverse < direct:
        coords.reverse()
    return coords


def select_edge_variant(
    graph: nx.MultiDiGraph,
    scaffold_counts: Dict[EdgeKey, int],
    lambda_val: float,
    u: Hashable,
    v: Hashable,
    weighted: bool,
) -> Tuple[Hashable, Dict[str, Any]]:
    edge_bundle = graph.get_edge_data(u, v)
    if edge_bundle is None:
        raise KeyError(f"Missing edge for {u}->{v}")

    if "length" in edge_bundle:
        return 0, edge_bundle

    def sort_key(item: Tuple[Hashable, Dict[str, Any]]) -> Tuple[float, float]:
        key, data = item
        if weighted:
            return (
                route_edge_weight(scaffold_counts, lambda_val, u, v, key, data),
                edge_length(data),
            )
        return (edge_length(data), edge_length(data))

    return min(edge_bundle.items(), key=sort_key)


def edge_midpoint(coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    line = LineString(coords)
    midpoint = line.interpolate(0.5, normalized=True)
    return float(midpoint.x), float(midpoint.y)


def route_scaffold_count(
    route_geometry: Dict[str, object],
    scaffolds: Sequence[Dict[str, object]],
) -> int:
    coordinates = route_geometry.get("coordinates", [])
    if len(coordinates) < 2:
        return 0

    line = LineString(coordinates)
    return sum(
        1
        for scaffold in scaffolds
        if line.distance(Point(float(scaffold["lng"]), float(scaffold["lat"])))
        < ROUTE_HIT_THRESHOLD_DEGREES
    )


def build_route_payload(
    graph: nx.MultiDiGraph,
    scaffold_counts: Dict[EdgeKey, int],
    scaffolds: Sequence[Dict[str, object]],
    nodes: Sequence[Hashable],
    lambda_val: float,
    weighted: bool,
) -> Dict[str, object]:
    coordinates: List[Tuple[float, float]] = []
    scaffold_waypoints: List[Dict[str, object]] = []
    distance_m = 0.0

    for u, v in zip(nodes, nodes[1:]):
        key, data = select_edge_variant(graph, scaffold_counts, lambda_val, u, v, weighted)
        edge_coords = edge_coordinates(graph, u, v, data)
        distance_m += edge_length(data)

        if coordinates and coordinates[-1] == edge_coords[0]:
            coordinates.extend(edge_coords[1:])
        else:
            coordinates.extend(edge_coords)

        nearby_scaffolds = scaffold_counts.get((u, v, key), 0)
        if nearby_scaffolds > 0:
            midpoint_lng, midpoint_lat = edge_midpoint(edge_coords)
            scaffold_waypoints.append(
                {
                    "lat": midpoint_lat,
                    "lng": midpoint_lng,
                    "covered": nearby_scaffolds,
                }
            )

    geometry = {
        "type": "LineString",
        "coordinates": coordinates,
    }
    return {
        "geojson": geometry,
        "distance_m": distance_m,
        "duration_s": distance_m / WALKING_SPEED_MPS,
        "scaffolds_covered": route_scaffold_count(geometry, scaffolds),
        "waypoints": scaffold_waypoints,
    }


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
    response = HTTP.get(WTTR_URL, params={"format": "j1"}, timeout=WEATHER_TIMEOUT_SECONDS)
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


def google_geocode(address: str) -> Optional[Dict[str, object]]:
    if not GOOGLE_MAPS_API_KEY:
        return None

    response = HTTP.get(
        GOOGLE_GEOCODE_URL,
        params={
            "address": address,
            "key": GOOGLE_MAPS_API_KEY,
            "region": "us",
        },
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


def nominatim_geocode(address: str) -> Optional[Dict[str, object]]:
    response = HTTP.get(
        NOMINATIM_SEARCH_URL,
        params={
            "q": address,
            "format": "jsonv2",
            "limit": "1",
            "addressdetails": "1",
        },
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


def get_weather(force: bool = False) -> Dict[str, object]:
    if DEV_BAD_WEATHER_MODE:
        return DEV_WEATHER_OVERRIDE.copy()

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


def build_node_index(graph: nx.MultiDiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_ids = np.array(list(graph.nodes()), dtype=object)
    node_xs = np.array([float(graph.nodes[node]["x"]) for node in node_ids], dtype=float)
    node_ys = np.array([float(graph.nodes[node]["y"]) for node in node_ids], dtype=float)
    return node_ids, node_xs, node_ys


def nearest_node(
    transformer: Transformer,
    node_ids: np.ndarray,
    node_xs: np.ndarray,
    node_ys: np.ndarray,
    lat: float,
    lng: float,
) -> Hashable:
    x, y = transformer.transform(lng, lat)
    distances = ((node_xs - x) ** 2) + ((node_ys - y) ** 2)
    return node_ids[int(np.argmin(distances))]


def initialize_state() -> Tuple[
    nx.MultiDiGraph,
    Transformer,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, object]],
    Dict[EdgeKey, int],
]:
    graph = load_graph()
    projected_graph = ox.project_graph(graph)
    transformer = Transformer.from_crs(graph.graph["crs"], projected_graph.graph["crs"], always_xy=True)
    node_ids, node_xs, node_ys = build_node_index(projected_graph)
    scaffolds = load_scaffolds()
    log(f"Scaffolds loaded: {len(scaffolds)}")
    scaffold_counts = precompute_scaffold_coverage(projected_graph, transformer, scaffolds)
    log(f"Edges with scaffold coverage: {len(scaffold_counts)}")
    if DEV_BAD_WEATHER_MODE:
        log(
            "Dev bad weather mode enabled: "
            f"{DEV_WEATHER_OVERRIDE['condition']} at {DEV_WEATHER_OVERRIDE['precip_mm_hr']} mm/hr"
        )
    log("Ready.")
    return graph, transformer, node_ids, node_xs, node_ys, scaffolds, scaffold_counts


GRAPH, GRAPH_TRANSFORMER, GRAPH_NODE_IDS, GRAPH_NODE_XS, GRAPH_NODE_YS, SCAFFOLDS, SCAFFOLD_COUNTS = initialize_state()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/scaffolds")
def api_scaffolds():
    return jsonify(SCAFFOLDS)


@app.route("/api/weather")
def api_weather():
    return jsonify(get_weather())


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
        if detour_mode not in DETOUR_LAMBDAS:
            raise ValueError("invalid detour mode")

        orig_node = nearest_node(
            GRAPH_TRANSFORMER,
            GRAPH_NODE_IDS,
            GRAPH_NODE_XS,
            GRAPH_NODE_YS,
            start[0],
            start[1],
        )
        dest_node = nearest_node(
            GRAPH_TRANSFORMER,
            GRAPH_NODE_IDS,
            GRAPH_NODE_XS,
            GRAPH_NODE_YS,
            end[0],
            end[1],
        )

        shortest_nodes = nx.shortest_path(GRAPH, orig_node, dest_node, weight="length")
        shortest = build_route_payload(
            GRAPH,
            SCAFFOLD_COUNTS,
            SCAFFOLDS,
            shortest_nodes,
            0.0,
            weighted=False,
        )

        lambda_val = DETOUR_LAMBDAS[detour_mode]
        scaffold_nodes = nx.shortest_path(
            GRAPH,
            orig_node,
            dest_node,
            weight=make_weight_fn(SCAFFOLD_COUNTS, lambda_val),
        )
        scaffold_route = build_route_payload(
            GRAPH,
            SCAFFOLD_COUNTS,
            SCAFFOLDS,
            scaffold_nodes,
            lambda_val,
            weighted=True,
        )

        return jsonify(
            {
                "shortest": shortest,
                "scaffold_route": scaffold_route,
                "weather": get_weather(),
            }
        )
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid route payload."}), 400
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return jsonify({"error": "No walking route found between A and B."}), 422


if __name__ == "__main__":
    app.run(debug=True)
