"""Candidate position generation for RL-based AP placement.

The RL agent should choose from a compact, valid action space instead of free
continuous coordinates. This module generates room-aware candidate AP locations
while avoiding walls and near-duplicate points.
"""

from __future__ import annotations

from math import hypot
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .room_profiles import (
    room_candidate_score_multiplier,
    room_effective_priority,
    room_is_high_density,
    room_requires_service,
)


Point = Tuple[float, float]


def point_in_polygon(x: float, y: float, polygon: Iterable[Dict[str, float]]) -> bool:
    """Return True when point is inside a polygon using ray casting."""
    points = list(polygon or [])
    if len(points) < 3:
        return False

    inside = False
    j = len(points) - 1
    for i, point in enumerate(points):
        xi, yi = float(point.get("x", 0)), float(point.get("y", 0))
        xj, yj = float(points[j].get("x", 0)), float(points[j].get("y", 0))
        intersects = ((yi > y) != (yj > y)) and (
            x < ((xj - xi) * (y - yi)) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from point p to segment ab."""
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return hypot(px - proj_x, py - proj_y)


def nearest_wall_distance(x: float, y: float, walls: List[Dict[str, Any]]) -> Tuple[float, Optional[str]]:
    """Return distance to nearest wall plus its id."""
    best_dist = float("inf")
    best_id = None

    for wall in walls:
        points = wall.get("points") or []
        if len(points) < 2:
            continue
        (x1, y1), (x2, y2) = points[0], points[1]
        dist = distance_point_to_segment(x, y, float(x1), float(y1), float(x2), float(y2))
        if dist < best_dist:
            best_dist = dist
            best_id = wall.get("id")

    return (best_dist if best_dist != float("inf") else 999.0, best_id)


def candidate_key(x: float, y: float, precision: float) -> Tuple[int, int]:
    """Quantized key used for deduplication."""
    return (round(x / precision), round(y / precision))


def room_bbox(room: Dict[str, Any]) -> Optional[Dict[str, float]]:
    polygon = room.get("polygon") or []
    if len(polygon) < 3:
        return None
    xs = [float(point.get("x", 0)) for point in polygon]
    ys = [float(point.get("y", 0)) for point in polygon]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}


def add_candidate(
    candidates: List[Dict[str, Any]],
    used: set,
    x: float,
    y: float,
    z: float,
    source: str,
    room: Optional[Dict[str, Any]],
    walls: List[Dict[str, Any]],
    min_wall_distance_m: float,
    dedupe_radius_m: float,
) -> None:
    key = candidate_key(x, y, dedupe_radius_m)
    if key in used:
        return

    wall_dist, wall_id = nearest_wall_distance(x, y, walls)
    if wall_dist < min_wall_distance_m:
        return

    area = float(room.get("area_m2", 0)) if room else 0.0
    clients = float(room.get("clients", 0)) if room else 0.0
    priority = room_effective_priority(room) if room else 1.0
    type_multiplier = room_candidate_score_multiplier(room) if room else 1.0
    demand_multiplier = 1.0 + min(clients, 300.0) / 100.0
    if room and room_is_high_density(room):
        demand_multiplier += 1.5
    score = priority * (1.0 + min(area, 80.0) / 80.0) * demand_multiplier
    score *= type_multiplier
    score += min(wall_dist, 3.0) * 0.15

    used.add(key)
    candidates.append({
        "id": f"cand_{len(candidates) + 1}",
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "z": round(float(z), 3),
        "source": source,
        "room_id": room.get("id") if room else None,
        "room_name": room.get("name") if room else None,
        "room_type": room.get("type") if room else None,
        "min_wall_distance_m": round(float(wall_dist), 3),
        "nearest_wall_id": wall_id,
        "score": round(float(score), 4),
    })


def generate_room_candidates(
    room: Dict[str, Any],
    walls: List[Dict[str, Any]],
    z: float,
    grid_step_m: float,
    min_wall_distance_m: float,
    dedupe_radius_m: float,
    max_candidates_per_room: int,
) -> List[Dict[str, Any]]:
    """Generate candidates inside one room polygon."""
    polygon = room.get("polygon") or []
    bbox = room_bbox(room)
    if not bbox:
        return []

    candidates: List[Dict[str, Any]] = []
    used = set()

    centroid = room.get("centroid") or {}
    cx = float(centroid.get("x", (bbox["min_x"] + bbox["max_x"]) / 2))
    cy = float(centroid.get("y", (bbox["min_y"] + bbox["max_y"]) / 2))
    if point_in_polygon(cx, cy, polygon):
        add_candidate(
            candidates, used, cx, cy, z, "room_centroid", room, walls,
            min_wall_distance_m, dedupe_radius_m
        )

    x = bbox["min_x"] + grid_step_m / 2
    while x <= bbox["max_x"]:
        y = bbox["min_y"] + grid_step_m / 2
        while y <= bbox["max_y"]:
            if point_in_polygon(x, y, polygon):
                add_candidate(
                    candidates, used, x, y, z, "room_grid", room, walls,
                    min_wall_distance_m, dedupe_radius_m
                )
            y += grid_step_m
        x += grid_step_m

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates[:max_candidates_per_room]


def generate_ap_candidates(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate AP candidate positions for a full RL scenario."""
    constraints = scenario.get("constraints") or {}
    floor_plan = scenario.get("floor_plan") or {}
    floor_height = float(floor_plan.get("floor_height_m") or 3.0)
    z = max(1.8, floor_height - 0.25)
    grid_step_m = float(constraints.get("candidate_grid_step_m") or 0.75)
    min_wall_distance_m = float(constraints.get("min_wall_distance_m") or 0.35)
    dedupe_radius_m = float(constraints.get("candidate_dedupe_radius_m") or 0.35)
    max_candidates = int(constraints.get("max_candidates") or 240)
    max_candidates_per_room = int(constraints.get("max_candidates_per_room") or 10)

    walls = scenario.get("walls") or []
    all_rooms = scenario.get("rooms", [])
    rooms = [room for room in all_rooms if room_requires_service(room)]
    excluded_rooms = [room for room in all_rooms if not room_requires_service(room)]
    candidates: List[Dict[str, Any]] = []
    used = set()
    room_best_candidates: Dict[str, Dict[str, Any]] = {}
    service_candidates: List[Dict[str, Any]] = []

    def in_excluded_area(x: float, y: float) -> bool:
        return any(
            point_in_polygon(x, y, room.get("polygon") or [])
            for room in excluded_rooms
        )

    for room in rooms:
        for candidate in generate_room_candidates(
            room, walls, z, grid_step_m, min_wall_distance_m,
            dedupe_radius_m, max_candidates_per_room
        ):
            if in_excluded_area(float(candidate["x"]), float(candidate["y"])):
                continue
            key = candidate_key(candidate["x"], candidate["y"], dedupe_radius_m)
            if key in used:
                continue
            used.add(key)
            candidate["id"] = f"cand_{len(candidates) + 1}"
            candidates.append(candidate)
            room_id = str(candidate.get("room_id"))
            if room_id and (
                room_id not in room_best_candidates
                or float(candidate.get("score", 0.0)) > float(room_best_candidates[room_id].get("score", 0.0))
            ):
                room_best_candidates[room_id] = candidate

    # Keep candidate options in corridors and unlabeled service areas too.
    service_boundary = floor_plan.get("service_boundary") or []
    if len(service_boundary) < 3:
        bounds = floor_plan.get("bounds") or {}
        min_x = float(bounds.get("min_x", 0))
        max_x = float(bounds.get("max_x", 0))
        min_y = float(bounds.get("min_y", 0))
        max_y = float(bounds.get("max_y", 0))
        if max_x > min_x and max_y > min_y:
            service_boundary = [
                {"x": min_x, "y": min_y},
                {"x": max_x, "y": min_y},
                {"x": max_x, "y": max_y},
                {"x": min_x, "y": max_y},
            ]
    if len(service_boundary) >= 3:
        xs = [float(point.get("x", 0)) for point in service_boundary]
        ys = [float(point.get("y", 0)) for point in service_boundary]
        x = min(xs) + grid_step_m
        while x <= max(xs) - grid_step_m:
            y = min(ys) + grid_step_m
            while y <= max(ys) - grid_step_m:
                inside_room = any(point_in_polygon(x, y, room.get("polygon") or []) for room in all_rooms)
                if point_in_polygon(x, y, service_boundary) and not inside_room:
                    add_candidate(
                        service_candidates, used, x, y, z, "service_grid", None, walls,
                        min_wall_distance_m, dedupe_radius_m
                    )
                y += grid_step_m
            x += grid_step_m
        service_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        candidates.extend(service_candidates[:max(12, max_candidates // 5)])

    # If there are no rooms yet, fall back to a coarse floor-bounds grid.
    if not candidates:
        bounds = floor_plan.get("bounds") or {}
        min_x = float(bounds.get("min_x", 0))
        max_x = float(bounds.get("max_x", 0))
        min_y = float(bounds.get("min_y", 0))
        max_y = float(bounds.get("max_y", 0))
        x = min_x + grid_step_m
        while x <= max_x - grid_step_m:
            y = min_y + grid_step_m
            while y <= max_y - grid_step_m:
                if not in_excluded_area(x, y):
                    add_candidate(
                        candidates, used, x, y, z, "floor_grid", None, walls,
                        min_wall_distance_m, dedupe_radius_m
                    )
                y += grid_step_m
            x += grid_step_m

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    if not room_best_candidates or len(candidates) <= max_candidates:
        return candidates[:max_candidates]

    # Preserve coverage of rooms first, then fill remaining slots by score.
    selected = sorted(room_best_candidates.values(), key=lambda candidate: candidate["score"], reverse=True)
    selected.extend(service_candidates[:min(24, max(0, max_candidates - len(selected)))])
    selected_ids = {id(candidate) for candidate in selected}
    remaining = [candidate for candidate in candidates if id(candidate) not in selected_ids]
    selected.extend(remaining)
    selected = selected[:max_candidates]
    for index, candidate in enumerate(selected, start=1):
        candidate["id"] = f"cand_{index}"
    return selected
