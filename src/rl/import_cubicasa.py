"""Import a representative CubiCasa5K SVG subset into RL scenario JSON files."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .candidate_generator import generate_ap_candidates
from .rf_calibration import attach_calibration, calibrated_wall_profiles
from .room_profiles import estimated_room_clients, normalize_room_type, room_type_profile
from .train_ppo import calculate_polygon_area, calculate_polygon_centroid


ROOM_ALIASES = {
    "livingroom": "lobby",
    "bedroom": "office",
    "kitchen": "other",
    "bath": "restroom",
    "bathroom": "restroom",
    "storage": "storage",
    "closet": "storage",
    "corridor": "corridor",
    "hall": "corridor",
}


def parse_points(value: str) -> List[Dict[str, float]]:
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value or "")]
    return [{"x": numbers[index], "y": numbers[index + 1]} for index in range(0, len(numbers) - 1, 2)]


def element_classes(element: ET.Element) -> str:
    return " ".join([
        element.attrib.get("class", ""),
        element.attrib.get("id", ""),
        element.attrib.get("label", ""),
    ]).lower()


def find_polygons(root: ET.Element) -> Iterable[Tuple[str, List[Dict[str, float]]]]:
    for element in root.iter():
        classes = element_classes(element)
        if "space" in classes:
            if "outdoor" in classes:
                continue
            for child in element:
                if child.tag.lower().endswith("polygon"):
                    points = parse_points(child.attrib.get("points", ""))
                    if len(points) >= 3:
                        yield classes, points
                        break
            continue
        if not element.tag.lower().endswith("polygon"):
            continue
        points = parse_points(element.attrib.get("points", ""))
        if len(points) >= 3:
            yield classes, points


def classify_room(classes: str) -> str:
    for key, room_type in ROOM_ALIASES.items():
        if key in classes:
            return room_type
    return "other"


def import_svg(svg_path: Path, scale_m_per_unit: float = 0.02) -> Dict:
    root = ET.parse(svg_path).getroot()
    rooms = []
    for classes, raw_polygon in find_polygons(root):
        if "space" not in classes and not any(alias in classes for alias in ROOM_ALIASES):
            continue
        polygon = [
            {"x": round(point["x"] * scale_m_per_unit, 3), "y": round(point["y"] * scale_m_per_unit, 3)}
            for point in raw_polygon
        ]
        area = calculate_polygon_area(polygon)
        if area < 1.2 or area > 500:
            continue
        room_type = normalize_room_type(classify_room(classes))
        profile = room_type_profile(room_type)
        rooms.append({
            "id": f"room_{len(rooms) + 1}",
            "name": f"{room_type.title()} {len(rooms) + 1}",
            "type": room_type,
            "polygon": polygon,
            "area_m2": round(area, 3),
            "centroid": calculate_polygon_centroid(polygon),
            "clients": estimated_room_clients(room_type, area),
            "priority": profile["default_priority"],
            "coverage_target_dbm": profile["default_coverage_dbm"],
            "excluded": room_type == "stairs",
        })
    if not rooms:
        raise ValueError(f"No room polygons recognized in {svg_path}")

    all_points = [point for room in rooms for point in room["polygon"]]
    min_x, max_x = min(p["x"] for p in all_points), max(p["x"] for p in all_points)
    min_y, max_y = min(p["y"] for p in all_points), max(p["y"] for p in all_points)
    wall_profile = calibrated_wall_profiles()[0]
    walls = []
    for room in rooms:
        polygon = room["polygon"]
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            walls.append({
                "id": f"wall_{len(walls) + 1}",
                "points": [[point["x"], point["y"]], [next_point["x"], next_point["y"]]],
                "material_id": wall_profile["material_id"],
                "attenuation_db": wall_profile["attenuation_db"],
            })

    scenario = {
        "name": f"CubiCasa5K / {svg_path.parent.name}",
        "dataset_source": {
            "name": "CubiCasa5K",
            "url": "https://zenodo.org/records/2613548",
            "license": "CC BY-NC 4.0",
            "source_file": str(svg_path),
        },
        "floor_plan": {
            "selected_frequency_ghz": 5.0,
            "floor_height_m": 3.0,
            "bounds": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        },
        "walls": walls,
        "rooms": rooms,
        "constraints": {
            "observation_version": 3, "max_candidates": 160, "max_candidates_per_room": 8,
            "max_ap_count": 16, "min_ap_count": max(1, (len(rooms) + 5) // 6),
            "min_ap_separation_m": 3.5,
        },
        "targets": {"default_coverage_dbm": -67, "target_sample_coverage_ratio": 0.95},
    }
    attach_calibration(scenario)
    scenario["candidate_positions"] = generate_ap_candidates(scenario)
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Extracted CubiCasa5K directory")
    parser.add_argument("--output-dir", default="floor_plans/public/cubicasa")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scale-m-per-unit", type=float, default=0.02)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    imported = 0
    for svg_path in sorted(Path(args.dataset_dir).rglob("model.svg")):
        try:
            scenario = import_svg(svg_path, args.scale_m_per_unit)
        except (ValueError, ET.ParseError):
            continue
        with (output / f"cubicasa_{imported + 1:04d}.json").open("w", encoding="utf-8") as target:
            json.dump(scenario, target, indent=2)
        imported += 1
        if imported >= args.limit:
            break
    print(f"Imported {imported} CubiCasa5K scenarios into {output}")


if __name__ == "__main__":
    main()
