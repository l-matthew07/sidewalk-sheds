#!/usr/bin/env python3
import heapq
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


NodeId = int
LatLng = Tuple[float, float]
EdgeKey = Tuple[NodeId, NodeId]


class ScaffoldRouter:
    def __init__(self, graph_data: Dict[str, object]) -> None:
        raw_nodes = graph_data["nodes"]
        raw_edges = graph_data["edges"]
        raw_scaffold_counts = graph_data["scaffold_counts"]
        raw_edge_geometry = graph_data["edge_geometry"]

        self.nodes: Dict[NodeId, LatLng] = {
            int(node_id): (float(lat), float(lng))
            for node_id, (lat, lng) in raw_nodes.items()
        }
        self.edges: List[Tuple[NodeId, NodeId, float]] = [
            (int(u), int(v), float(length))
            for u, v, length in raw_edges
        ]
        self.scaffold_counts: Dict[EdgeKey, int] = {
            (int(u), int(v)): int(count)
            for (u, v), count in raw_scaffold_counts.items()
        }
        self.edge_geometry: Dict[EdgeKey, List[LatLng]] = {
            (int(u), int(v)): [(float(lat), float(lng)) for lat, lng in coords]
            for (u, v), coords in raw_edge_geometry.items()
        }

        self.adjacency: Dict[NodeId, List[Tuple[NodeId, float, int]]] = {}
        self.edge_lengths: Dict[EdgeKey, float] = {}
        for u, v, length in self.edges:
            count = self.scaffold_counts.get((u, v), 0)
            self.adjacency.setdefault(u, []).append((v, length, count))
            self.edge_lengths[(u, v)] = length

        self.node_ids = np.array(list(self.nodes.keys()), dtype=np.int64)
        self.node_lats = np.array([self.nodes[node_id][0] for node_id in self.node_ids], dtype=np.float32)
        self.node_lngs = np.array([self.nodes[node_id][1] for node_id in self.node_ids], dtype=np.float32)

    def nearest_node(self, lat: float, lng: float) -> NodeId:
        lat = float(lat)
        lng = float(lng)
        scale = math.cos(math.radians(lat))
        dx = (self.node_lngs - lng) * scale
        dy = self.node_lats - lat
        distances = (dx * dx) + (dy * dy)
        return int(self.node_ids[int(np.argmin(distances))])

    def route(
        self,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        detour_bias: float,
    ) -> Tuple[List[NodeId], float]:
        start_node = self.nearest_node(start_lat, start_lng)
        end_node = self.nearest_node(end_lat, end_lng)

        frontier: List[Tuple[float, NodeId]] = [(0.0, start_node)]
        best_weight: Dict[NodeId, float] = {start_node: 0.0}
        best_distance: Dict[NodeId, float] = {start_node: 0.0}
        previous: Dict[NodeId, NodeId] = {}

        while frontier:
            current_weight, node = heapq.heappop(frontier)
            if current_weight > best_weight.get(node, float("inf")):
                continue
            if node == end_node:
                break

            for neighbor, length, scaffold_count in self.adjacency.get(node, []):
                # Treat scaffold-covered edges as meaningfully "cheaper" so preference modes
                # produce visibly different paths even with a relatively sparse active dataset.
                if scaffold_count > 0:
                    discount = detour_bias * min(scaffold_count, 3)
                    edge_weight = max(length * 0.08, length * (1.0 - discount))
                else:
                    edge_weight = length
                next_weight = current_weight + edge_weight
                if next_weight >= best_weight.get(neighbor, float("inf")):
                    continue

                best_weight[neighbor] = next_weight
                best_distance[neighbor] = best_distance[node] + length
                previous[neighbor] = node
                heapq.heappush(frontier, (next_weight, neighbor))

        if end_node not in best_weight:
            raise LookupError("No path found.")

        path: List[NodeId] = [end_node]
        cursor = end_node
        while cursor != start_node:
            cursor = previous[cursor]
            path.append(cursor)
        path.reverse()
        return path, best_distance[end_node]

    def path_to_geojson(self, node_path: Sequence[NodeId]) -> List[List[float]]:
        coordinates: List[List[float]] = []
        for u, v in zip(node_path, node_path[1:]):
            latlngs = self.edge_geometry.get((u, v))
            if latlngs is None:
                start = self.nodes[u]
                end = self.nodes[v]
                latlngs = [start, end]

            segment = [[lng, lat] for lat, lng in latlngs]
            if coordinates and coordinates[-1] == segment[0]:
                coordinates.extend(segment[1:])
            else:
                coordinates.extend(segment)
        return coordinates

    def edge_waypoints(self, node_path: Sequence[NodeId]) -> List[Dict[str, float]]:
        waypoints: List[Dict[str, float]] = []
        for u, v in zip(node_path, node_path[1:]):
            count = self.scaffold_counts.get((u, v), 0)
            if count <= 0:
                continue

            latlngs = self.edge_geometry.get((u, v))
            if latlngs:
                midpoint = latlngs[len(latlngs) // 2]
                lat, lng = midpoint
            else:
                start_lat, start_lng = self.nodes[u]
                end_lat, end_lng = self.nodes[v]
                lat = (start_lat + end_lat) / 2.0
                lng = (start_lng + end_lng) / 2.0

            waypoints.append({"lat": float(lat), "lng": float(lng), "covered": int(count)})
        return waypoints

    def count_scaffolds_covered(
        self,
        geojson_coords: Sequence[Sequence[float]],
        scaffolds: Sequence[Dict[str, object]],
        radius_deg: float = 0.0004,
    ) -> int:
        if len(geojson_coords) < 2 or not scaffolds:
            return 0

        scaffold_lats = np.array([float(scaffold["lat"]) for scaffold in scaffolds], dtype=np.float32)
        scaffold_lngs = np.array([float(scaffold["lng"]) for scaffold in scaffolds], dtype=np.float32)
        route = np.asarray(geojson_coords, dtype=np.float32)

        ax = route[:-1, 0][None, :]
        ay = route[:-1, 1][None, :]
        bx = route[1:, 0][None, :]
        by = route[1:, 1][None, :]

        px = scaffold_lngs[:, None]
        py = scaffold_lats[:, None]

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        length_sq = (abx * abx) + (aby * aby)
        safe_length_sq = np.where(length_sq == 0, 1.0, length_sq)
        t = np.clip(((apx * abx) + (apy * aby)) / safe_length_sq, 0.0, 1.0)

        closest_x = ax + (t * abx)
        closest_y = ay + (t * aby)
        dist_sq = ((px - closest_x) ** 2) + ((py - closest_y) ** 2)
        min_dist_sq = np.min(dist_sq, axis=1)
        return int(np.count_nonzero(min_dist_sq < (radius_deg * radius_deg)))
