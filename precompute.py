#!/usr/bin/env python3
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import osmnx as ox
from shapely import wkt


BASE_DIR = Path(__file__).resolve().parent
SCAFFOLDS_PATH = BASE_DIR / "scaffolds.json"
GRAPHML_PATH = BASE_DIR / "manhattan_walk.graphml"
GRAPH_DATA_PATH = BASE_DIR / "graph_data.pkl"
SCAFFOLD_EDGES_PATH = BASE_DIR / "scaffold_edges.json"


def log(message: str) -> None:
    print(f"[SidewalkShed] {message}", flush=True)


def normalize_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def load_scaffolds() -> List[Dict[str, object]]:
    with SCAFFOLDS_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    cleaned: List[Dict[str, object]] = []
    for row in rows:
        if row.get("lat") is None or row.get("lng") is None:
            continue
        cleaned.append(
            {
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
            }
        )
    return cleaned


def load_graph():
    if GRAPHML_PATH.exists():
        graph = ox.load_graphml(filepath=str(GRAPHML_PATH))
    else:
        graph = ox.graph_from_place(
            "Manhattan, New York, USA",
            network_type="walk",
            retain_all=False,
        )
        ox.save_graphml(graph, filepath=str(GRAPHML_PATH))

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


def sq_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx) + (dy * dy)


def orient_geometry(
    graph,
    u: int,
    v: int,
    data: Dict[str, object],
) -> Optional[List[Tuple[float, float]]]:
    geometry = data.get("geometry")
    if geometry is None or not hasattr(geometry, "coords"):
        return None

    start = (float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"]))
    end = (float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"]))
    coords = [(float(lat), float(lng)) for lng, lat in geometry.coords]
    direct = sq_dist(coords[0], start) + sq_dist(coords[-1], end)
    reverse = sq_dist(coords[0], end) + sq_dist(coords[-1], start)
    if reverse < direct:
        coords.reverse()

    if len(coords) <= 2:
        return None
    return coords


def build_lightweight_graph(graph):
    nodes: Dict[int, Tuple[float, float]] = {}
    for node_id, data in graph.nodes(data=True):
        nodes[int(node_id)] = (float(data["y"]), float(data["x"]))

    best_edges: Dict[Tuple[int, int], Tuple[float, Optional[List[Tuple[float, float]]]]] = {}
    for u, v, _, data in graph.edges(keys=True, data=True):
        edge_key = (int(u), int(v))
        length = float(data.get("length", 1.0))
        geometry = orient_geometry(graph, int(u), int(v), data)
        existing = best_edges.get(edge_key)
        if existing is None or length < existing[0]:
            best_edges[edge_key] = (length, geometry)

    edges = [(u, v, length) for (u, v), (length, _) in best_edges.items()]
    edge_geometry = {
        edge_key: geometry
        for edge_key, (_, geometry) in best_edges.items()
        if geometry is not None
    }
    return nodes, edges, edge_geometry


def map_scaffolds_to_edges(graph, scaffolds: List[Dict[str, object]]) -> Dict[Tuple[int, int], int]:
    scaffold_counts: Dict[Tuple[int, int], int] = {}
    if not scaffolds:
        return scaffold_counts

    lngs = [float(scaffold["lng"]) for scaffold in scaffolds]
    lats = [float(scaffold["lat"]) for scaffold in scaffolds]
    nearest_edges = ox.nearest_edges(graph, lngs, lats)
    for u, v, _ in nearest_edges:
        edge_key = (int(u), int(v))
        scaffold_counts[edge_key] = scaffold_counts.get(edge_key, 0) + 1
    return scaffold_counts


def write_scaffold_edges_json(scaffold_counts: Dict[Tuple[int, int], int]) -> None:
    payload = {f"{u}|{v}": count for (u, v), count in scaffold_counts.items()}
    SCAFFOLD_EDGES_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    scaffolds = load_scaffolds()
    graph = load_graph()
    nodes, edges, edge_geometry = build_lightweight_graph(graph)
    scaffold_counts = map_scaffolds_to_edges(graph, scaffolds)

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "scaffold_counts": scaffold_counts,
        "edge_geometry": edge_geometry,
    }
    with GRAPH_DATA_PATH.open("wb") as handle:
        pickle.dump(graph_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    write_scaffold_edges_json(scaffold_counts)

    serialized_size_mb = os.path.getsize(GRAPH_DATA_PATH) / 1024 / 1024
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Scaffolds mapped to edges: {sum(scaffold_counts.values())}")
    print(f"Serialized size: {serialized_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
