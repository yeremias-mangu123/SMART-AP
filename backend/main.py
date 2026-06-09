from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import sys
import os
import json
import io
import base64
import time
from datetime import datetime, timezone
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'src'))
sys.path.append(os.path.join(parent_dir, 'gui'))

try:
    from main_four_ap import _place_aps_intelligent_grid, estimate_initial_ap_count
except ImportError as e:
    print(f"Failed to import core modules: {e}")

try:
    from rl.candidate_generator import generate_ap_candidates, point_in_polygon
    from rl.rf_calibration import attach_calibration
    from rl.room_profiles import estimated_room_clients, room_is_high_density, room_requires_service
except ImportError as e:
    generate_ap_candidates = None
    point_in_polygon = None
    attach_calibration = None
    estimated_room_clients = None
    room_is_high_density = None
    room_requires_service = None
    print(f"Failed to import RL candidate generator: {e}")

try:
    from rl.baseline_trainer import run_baselines
except ImportError as e:
    run_baselines = None
    print(f"Failed to import RL baseline trainer: {e}")

try:
    from rl.self_training_runner import run_self_training_batch
except ImportError as e:
    run_self_training_batch = None
    print(f"Failed to import RL self-training runner: {e}")

from backend.database import engine, get_db
from backend import models

try:
    from stable_baselines3 import PPO
    from rl.gym_wrapper import GymWiFiAPPlacementEnv
    from rl.environment import PlacedAP
    from rl.train_ppo import adapt_scenario_for_rl, candidate_diagnostics
except ImportError as e:
    PPO = None
    GymWiFiAPPlacementEnv = None
    PlacedAP = None
    adapt_scenario_for_rl = None
    candidate_diagnostics = None
    print(f"Failed to import PPO dependencies: {e}")

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    MaskablePPO = None

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="WiFi Planner API")

# Configure CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Point(BaseModel):
    x: float
    y: float

class FloorPlanElement(BaseModel):
    type: str
    points: List[List[float]] # Nested array like [[x, y], [x, y]]
    material: str = "concrete"
    height: float = 3.0
    color: str = "#000000"

class AccessPoint(BaseModel):
    x: float
    y: float
    z: float = 2.5
    name: str = "AP"
    power: float = 20.0
    frequency: float = 2.4
    color: str = "#FF0000"
    model_id: Optional[str] = None
    model: Optional[str] = None
    vendor: Optional[str] = None
    gain24: Optional[float] = None
    gain5: Optional[float] = None
    power24: Optional[float] = None
    power5: Optional[float] = None
    channel24: Optional[int] = None
    channel5: Optional[int] = None

class DetectWallsRequest(BaseModel):
    image: str  # base64 encoded image
    pixels_per_meter: float = 45.0
    min_wall_length_m: float = 0.5  # minimum wall length in meters
    wall_type: str = "medium"  # default wall type to assign
    sensitivity: int = 50  # 0-100, higher = more walls detected

class DetectRoomsRequest(BaseModel):
    image: str
    pixels_per_meter: float = 45.0
    min_room_area_m2: float = 1.2
    max_room_area_m2: float = 180.0

class AutoPlaceRequest(BaseModel):
    width: float
    length: float
    height: float = 3.0
    elements: List[FloorPlanElement] = []
    resolution: float = 0.2

class ProjectData(BaseModel):
    elements: List[FloorPlanElement]
    access_points: List[AccessPoint]
    parameters: Dict[str, Any]
    version: str = "2.0"

class ProjectCreate(BaseModel):
    name: str

DEFAULT_WALL_TYPES = [
    {"id": "common_brick", "name": "Common brick wall", "db24": 10, "db5": 15, "db6": 17, "thickness": 120, "category": "Common"},
    {"id": "thick_brick", "name": "Thick brick wall", "db24": 15, "db5": 25, "db6": 28, "thickness": 240, "category": "Common"},
    {"id": "gypsum", "name": "Gypsum board", "db24": 3, "db5": 4, "db6": 5, "thickness": 8, "category": "Materials"},
    {"id": "foam", "name": "Foam materials", "db24": 3, "db5": 4, "db6": 5, "thickness": 8, "category": "Materials"},
    {"id": "hollow_wood", "name": "Hollow wood", "db24": 2, "db5": 3, "db6": 4, "thickness": 20, "category": "Materials"},
    {"id": "common_wood_door", "name": "Common wooden door", "db24": 3, "db5": 4, "db6": 5, "thickness": 40, "category": "Materials"},
    {"id": "solid_wood_door", "name": "Solid wood door", "db24": 10, "db5": 15, "db6": 17, "thickness": 40, "category": "Materials"},
    {"id": "common_glass", "name": "Common glass", "db24": 4, "db5": 7, "db6": 8, "thickness": 8, "category": "Materials"},
    {"id": "thick_glass", "name": "Thick glass", "db24": 8, "db5": 10, "db6": 12, "thickness": 12, "category": "Materials"},
    {"id": "armored_glass", "name": "Armored glass", "db24": 25, "db5": 35, "db6": 40, "thickness": 30, "category": "Materials"},
    {"id": "load_bearing_column", "name": "Load-bearing column", "db24": 25, "db5": 30, "db6": 33, "thickness": 500, "category": "Materials"},
    {"id": "shutter_door", "name": "Shutter door", "db24": 15, "db5": 20, "db6": 23, "thickness": 10, "category": "Materials"},
    {"id": "color_steel_panel", "name": "Color steel sandwich panel", "db24": 30, "db5": 35, "db6": 40, "thickness": 80, "category": "Materials"},
    {"id": "thin", "name": "Thin Wall", "db24": 3, "db5": 9, "db6": 12, "thickness": 100, "category": "Common"},
    {"id": "medium", "name": "Medium Wall", "db24": 10, "db5": 20, "db6": 24, "thickness": 200, "category": "Common"},
    {"id": "thick", "name": "Thick Wall", "db24": 20, "db5": 32, "db6": 38, "thickness": 300, "category": "Common"},
    {"id": "concrete", "name": "Concrete Wall", "db24": 25, "db5": 30, "db6": 33, "thickness": 240, "category": "Materials"},
    {"id": "elevator", "name": "Elevator", "db24": 30, "db5": 35, "db6": 40, "thickness": 80, "category": "Materials"},
    {"id": "brick", "name": "Brick Wall", "db24": 10, "db5": 12, "db6": 15, "thickness": 150, "category": "Materials"},
    {"id": "metal_wall", "name": "Metal Wall", "db24": 20, "db5": 25, "db6": 30, "thickness": 100, "category": "Materials"},
    {"id": "solid_wood", "name": "Solid Wooden Door", "db24": 4, "db5": 6, "db6": 8, "thickness": 40, "category": "Materials"},
    {"id": "glass_thick", "name": "Thick Glass", "db24": 3, "db5": 5, "db6": 6, "thickness": 12, "category": "Materials"},
    {"id": "partition", "name": "Workstation Partition", "db24": 1, "db5": 2, "db6": 3, "thickness": 50, "category": "Materials"},
    {"id": "shelf", "name": "Shelf", "db24": 15, "db5": 20, "db6": 23, "thickness": 80, "category": "Materials"},
    {"id": "window", "name": "Window", "db24": 1, "db5": 2, "db6": 2, "thickness": 5, "category": "Materials"},
]

@app.get("/")
def read_root():
    return {"message": "WiFi Planner API is running"}

def load_ap_catalog_data():
    catalog_path = os.path.join(parent_dir, "src", "rl", "ap_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/ap-catalog")
def get_ap_catalog():
    """Return the AP model catalog used by UI and RL placement training."""
    try:
        return load_ap_catalog_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="AP catalog not found")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid AP catalog JSON: {e}")

def calculate_polygon_area(points):
    if not isinstance(points, list) or len(points) < 3:
        return 0.0
    area = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        area += float(point.get("x", 0)) * float(nxt.get("y", 0))
        area -= float(nxt.get("x", 0)) * float(point.get("y", 0))
    return abs(area) / 2.0

def calculate_polygon_centroid(points):
    if not isinstance(points, list) or len(points) < 3:
        return {"x": 0.0, "y": 0.0}

    area_factor = 0.0
    cx = 0.0
    cy = 0.0
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        x0, y0 = float(point.get("x", 0)), float(point.get("y", 0))
        x1, y1 = float(nxt.get("x", 0)), float(nxt.get("y", 0))
        cross = x0 * y1 - x1 * y0
        area_factor += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(area_factor) < 1e-8:
        return {
            "x": sum(float(point.get("x", 0)) for point in points) / len(points),
            "y": sum(float(point.get("y", 0)) for point in points) / len(points),
        }

    return {"x": cx / (3 * area_factor), "y": cy / (3 * area_factor)}

def build_rl_training_scenario(name: str, elements: List[Dict[str, Any]],
                               access_points: List[Dict[str, Any]],
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
    wall_types = parameters.get("wallTypes") or DEFAULT_WALL_TYPES
    wall_type_map = {wall_type.get("id"): wall_type for wall_type in wall_types}
    pixels_per_meter = float(parameters.get("pixelsPerMeter") or 30.7)
    floor_height = float(parameters.get("floorHeight") or 3.0)
    selected_frequency = float(parameters.get("selectedFrequency") or 2.4)

    walls = []
    xs = []
    ys = []
    for index, element in enumerate(elements or []):
        if element.get("type") != "wall" or len(element.get("points", [])) < 2:
            continue
        p1, p2 = element["points"][0], element["points"][1]
        material_id = element.get("material") or parameters.get("selectedWallType") or "common_brick"
        material = (
            wall_type_map.get(material_id)
            or element.get("properties")
            or wall_type_map.get("common_brick")
            or wall_type_map.get("thin", {})
        )
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        walls.append({
            "id": f"wall_{index + 1}",
            "points": [[x1, y1], [x2, y2]],
            "material_id": material_id,
            "material_name": material.get("name", material_id),
            "attenuation_db": {
                "2.4": float(material.get("db24", 3)),
                "5": float(material.get("db5", material.get("db24", 3))),
                "6": float(material.get("db6", material.get("db5", material.get("db24", 3)))),
            },
            "thickness_mm": float(material.get("thickness", 100)),
            "length_m": float(np.hypot(x2 - x1, y2 - y1)),
        })

    rooms = []
    for index, room in enumerate(parameters.get("rooms") or []):
        polygon = room.get("polygon") or []
        normalized_polygon = [
            {"x": float(point.get("x", 0)), "y": float(point.get("y", 0))}
            for point in polygon
            if isinstance(point, dict)
        ]
        area_m2 = float(room.get("areaM2") or calculate_polygon_area(normalized_polygon))
        centroid = room.get("centroid") or calculate_polygon_centroid(normalized_polygon)
        room_type = str(room.get("roomType") or room.get("type") or "room").lower()
        is_stairs = room_type == "stairs"
        # Authoritative area-based demand: ~1 active user per m² so AP placement
        # tracks room size (large rooms carry more users, tiny rooms few/none).
        # Computed here (not taken from the UI's flat default) so the behaviour is
        # consistent regardless of frontend state. Stairs/excluded stay at 0.
        clients_per_sqm = float(parameters.get("clientsPerSqm") or 1.0)
        clients = 0 if is_stairs else max(0, int(round(area_m2 * clients_per_sqm)))
        service_excluded = is_stairs or bool(room.get("serviceExcluded", room.get("service_excluded", False))) or clients <= 0
        xs.extend(point["x"] for point in normalized_polygon)
        ys.extend(point["y"] for point in normalized_polygon)
        rooms.append({
            "id": room.get("id") or f"room_{index + 1}",
            "name": room.get("name") or f"Room {index + 1}",
            "type": room_type,
            "polygon": normalized_polygon,
            "area_m2": area_m2,
            "centroid": {
                "x": float(centroid.get("x", 0)),
                "y": float(centroid.get("y", 0)),
            },
            "clients": clients,
            # Area-based priority: large rooms High(3), medium Medium(2), small Low(1).
            "priority": 0.0 if is_stairs else (3.0 if area_m2 >= 30 else 2.0 if area_m2 >= 15 else 1.0),
            "coverage_target_dbm": float(room.get("coverageTarget", -80 if is_stairs else -67)),
            "excluded": is_stairs or bool(room.get("excluded", False)),
            "service_excluded": service_excluded,
        })

    service_boundary = [
        {"x": float(point.get("x", 0)), "y": float(point.get("y", 0))}
        for point in (parameters.get("serviceBoundary") or [])
        if isinstance(point, dict)
    ]
    xs.extend(point["x"] for point in service_boundary)
    ys.extend(point["y"] for point in service_boundary)

    if xs and ys:
        bounds = {
            "min_x": min(xs),
            "min_y": min(ys),
            "max_x": max(xs),
            "max_y": max(ys),
            "width_m": max(xs) - min(xs),
            "height_m": max(ys) - min(ys),
        }
    else:
        bounds = {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0, "width_m": 0, "height_m": 0}

    catalog_data = load_ap_catalog_data()
    catalog_models = catalog_data.get("models", [])
    catalog_by_id = {str(model.get("id")): model for model in catalog_models}
    parameter_models = parameters.get("apModels") or []
    merged_models = [
        {**catalog_by_id.get(str(model.get("id")), {}), **model}
        for model in parameter_models
    ] if parameter_models else catalog_models
    ap_catalog = {
        **catalog_data,
        "models": merged_models,
    }

    scenario = {
        "schema_version": "1.0",
        "scenario_type": "wifi_ap_placement_rl",
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": {
            "unit": "meter",
            "origin": "top_left",
            "pixels_per_meter": pixels_per_meter,
        },
        "floor_plan": {
            "bounds": bounds,
            "floor_height_m": floor_height,
            "selected_frequency_ghz": selected_frequency,
            "service_boundary": service_boundary,
        },
        "walls": walls,
        "rooms": rooms,
        "ap_catalog": ap_catalog,
        "initial_access_points": access_points or [],
        "constraints": {
            "observation_version": 3,
            "candidate_grid_step_m": 0.75,
            "min_ap_separation_m": 4.0,
            "min_wall_distance_m": 0.35,
            "candidate_dedupe_radius_m": 0.35,
            "max_candidates_per_room": 10,
            "max_candidates": 240,
            "room_sample_step_m": 1.2,
            "max_samples_per_room": 20,
            "floor_sample_step_m": 1.2,
            "max_floor_samples": 320,
            "max_ap_count": max(1, min(12, len([room for room in rooms if room_requires_service(room)]) or 12)),
            "allowed_mounting": ["ceiling"],
            "allowed_bands": ["2.4", "5", "6"],
        },
        "targets": {
            "default_coverage_dbm": -67,
            "min_coverage_ratio": 0.85,
            "min_room_sample_coverage_ratio": 0.85,
            "target_sample_coverage_ratio": 0.90,
            "target_acceptable_coverage_ratio": 0.95,
            "min_capacity_ratio": 0.90,
            "max_weak_room_count": 0,
            "target_sinr_coverage_ratio": 0.85,
            "target_floor_coverage_ratio": 0.80,
            "target_floor_sinr_coverage_ratio": 0.80,
            "max_overlap_ratio": 0.25,
            "min_sinr_db": 15.0,
            "interference_sinr_db": 20.0,
            "capacity_headroom_ratio": 1.2,
            "optimize_for": ["coverage", "capacity", "cost", "interference"],
        },
        "reward_weights": {
            "coverage": 1.0,
            "sample_coverage": 2.0,
            "acceptable_coverage": 0.8,
            "room_priority": 0.9,
            "capacity": 0.5,
            "cost": -0.18,
            "ap_count": -0.22,
            "overlap": -1.0,
            "sinr_coverage": 1.4,
            "interference": -1.4,
            "floor_coverage": 1.8,
            "floor_sinr_coverage": 1.0,
            "invalid_position": -2.0,
            "terminal_sample_coverage": 4.5,
            "terminal_acceptable_coverage": 1.2,
            "terminal_blankspot": 4.0,
            "terminal_overlap": 3.0,
            "terminal_sinr_coverage": 2.5,
            "terminal_interference": 2.5,
            "terminal_floor_coverage": 3.5,
            "terminal_floor_blankspot": 4.0,
            "terminal_ap_count": 0.55,
            "stop_target_met": 2.0,
            "stop_efficiency": 3.0,
            "max_ap_without_stop": -2.0,
        },
    }

    if generate_ap_candidates:
        scenario["candidate_positions"] = generate_ap_candidates(scenario)
        scenario["candidate_count"] = len(scenario["candidate_positions"])
    else:
        scenario["candidate_positions"] = []
        scenario["candidate_count"] = 0

    if attach_calibration is not None:
        attach_calibration(scenario, str(parameters.get("rfProfile") or "office"))
    return scenario

def wall_material_summary(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Summarize the wall attenuation values actually used by the RF engine."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for wall in scenario.get("walls") or []:
        material_id = str(wall.get("material_id") or "unknown")
        entry = grouped.setdefault(material_id, {
            "material_id": material_id,
            "material_name": wall.get("material_name") or material_id,
            "count": 0,
            "attenuation_db": wall.get("attenuation_db") or {},
        })
        entry["count"] += 1
    return sorted(grouped.values(), key=lambda item: (-item["count"], item["material_id"]))

@app.post("/api/rl/scenario")
def create_rl_scenario(project_data: ProjectData):
    """Build an RL training scenario from project data posted by the UI."""
    return build_rl_training_scenario(
        name=project_data.parameters.get("name", "WiFi RL Scenario"),
        elements=[e.model_dump() for e in project_data.elements],
        access_points=[a.model_dump() for a in project_data.access_points],
        parameters=project_data.parameters,
    )

@app.post("/api/rl/candidates")
def create_rl_candidates(project_data: ProjectData):
    """Return only RL candidate AP positions for quick debugging."""
    scenario = build_rl_training_scenario(
        name=project_data.parameters.get("name", "WiFi RL Scenario"),
        elements=[e.model_dump() for e in project_data.elements],
        access_points=[a.model_dump() for a in project_data.access_points],
        parameters=project_data.parameters,
    )
    return {
        "success": True,
        "candidate_count": scenario.get("candidate_count", 0),
        "candidate_positions": scenario.get("candidate_positions", []),
        "constraints": scenario.get("constraints", {}),
    }

def baseline_result_to_access_points(result: Dict[str, Any], scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = (scenario.get("ap_catalog") or {}).get("models", [])
    frequency = float((scenario.get("floor_plan") or {}).get("selected_frequency_ghz") or 5.0)
    access_points = []

    for index, placed in enumerate(result.get("best", {}).get("placed_aps", [])):
        model_index = int(placed.get("model_index", 0))
        model = models[model_index] if 0 <= model_index < len(models) else {}
        tx_power = float(placed.get("tx_power_dbm", model.get("maxPower", 20)))
        power24 = float(model.get("defaultPower24", tx_power))
        power5 = float(model.get("defaultPower5", tx_power))
        if frequency >= 5:
            power5 = tx_power
        else:
            power24 = tx_power

        access_points.append({
            "name": f"RL AP {index + 1}",
            "x": float(placed.get("x", 0)),
            "y": float(placed.get("y", 0)),
            "z": float(placed.get("z", 2.7)),
            "power": tx_power,
            "frequency": frequency,
            "color": model.get("color", "#334155"),
            "model_id": model.get("id", placed.get("model_id")),
            "model": model.get("model", placed.get("model_id", "AP")),
            "vendor": model.get("vendor", "RL"),
            "gain24": float(model.get("gain24", 2)),
            "gain5": float(model.get("gain5", model.get("gain24", 2))),
            "power24": power24,
            "power5": power5,
        })

    return access_points

@app.post("/api/rl/baseline-placement")
def create_rl_baseline_placement(project_data: ProjectData):
    """Run random/greedy baseline placement and return APs usable by the UI."""
    if run_baselines is None:
        raise HTTPException(status_code=500, detail="RL baseline trainer is not available")

    scenario = build_rl_training_scenario(
        name=project_data.parameters.get("name", "WiFi RL Scenario"),
        elements=[e.model_dump() for e in project_data.elements],
        access_points=[a.model_dump() for a in project_data.access_points],
        parameters=project_data.parameters,
    )
    episodes = int(project_data.parameters.get("rlRandomEpisodes", 50))
    seed = int(project_data.parameters.get("rlSeed", 42))
    candidate_limit = int(project_data.parameters.get("rlGreedyCandidateLimit", 80))
    result = run_baselines(
        scenario,
        random_episodes=max(0, min(episodes, 500)),
        seed=seed,
        greedy_candidate_limit=max(1, min(candidate_limit, 240)),
    )

    return {
        "success": True,
        "access_points": baseline_result_to_access_points(result, scenario),
        "baseline": result,
        "candidate_count": result.get("candidate_count", 0),
    }

def load_latest_ppo_run():
    """Locate the latest PPO run that already has a usable model checkpoint."""
    base_dir = os.path.join(parent_dir, "training_runs")
    if not os.path.exists(base_dir): return None
    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("ppo_")
        and os.path.exists(os.path.join(base_dir, d, "best_model.zip"))
    ]
    if not runs: return None
    return max(runs, key=os.path.getmtime)

def load_best_ppo_model():
    """Helper to locate the latest trained PPO best_model.zip in training_runs/."""
    latest_run = load_latest_ppo_run()
    if not latest_run: return None
    model_path = os.path.join(latest_run, "best_model.zip")
    if not os.path.exists(model_path): return None
    return model_path

def load_ppo_training_summary(run_dir: str) -> Dict[str, Any]:
    """Load training metadata so inference uses the same action-space settings."""
    summary_path = os.path.join(run_dir, "training_summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_ppo_summary_to_scenario(scenario: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Apply training constraints/targets before regenerating PPO candidates."""
    constraints = scenario.setdefault("constraints", {})
    if "observation_version" not in (summary.get("constraints") or {}):
        constraints["observation_version"] = 1
    constraints.update(summary.get("constraints") or {})
    targets = scenario.setdefault("targets", {})
    targets.update(summary.get("targets") or {})
    scenario.pop("candidate_positions", None)
    return adapt_scenario_for_rl(scenario) if adapt_scenario_for_rl is not None else scenario

def recommended_min_ap_count(rooms: List[Dict[str, Any]], scenario: Dict[str, Any]) -> int:
    """Estimate a practical minimum AP count for office-style room layouts."""
    active_rooms = [room for room in rooms if room_requires_service(room)]
    if not active_rooms:
        return 0

    total_area = sum(float(room.get("area_m2") or room.get("areaM2") or 0.0) for room in active_rooms)
    by_rooms = (len(active_rooms) + 4) // 5
    by_area = int((total_area + 74.999) // 75.0) if total_area > 0 else 1
    planning_clients = float((scenario.get("constraints") or {}).get("planning_clients_per_ap") or 40.0)
    total_demand = sum(
        float(room.get("clients") or 0.0) * float(room.get("capacity_headroom_ratio") or 1.2)
        for room in active_rooms
    )
    by_capacity = int(np.ceil(total_demand / max(1.0, planning_clients)))
    max_ap_count = int((scenario.get("constraints") or {}).get("max_ap_count") or 6)
    return max(1, min(max_ap_count, max(by_rooms, by_area, by_capacity)))

def apply_inference_ap_floor(scenario: Dict[str, Any],
                             deployment_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Set layout-specific AP and RF floors for deployment inference."""
    constraints = scenario.setdefault("constraints", {})
    rooms = [room for room in scenario.get("rooms", []) if room_requires_service(room)]
    planning_clients = float(
        (deployment_model or {}).get("planningClients")
        or (deployment_model or {}).get("recommendedClients")
        or 40.0
    )
    constraints["planning_clients_per_ap"] = planning_clients
    capacity_minimum = int(np.ceil(sum(
        float(room.get("clients") or 0.0) * float(room.get("capacity_headroom_ratio") or 1.2)
        for room in rooms
    ) / max(1.0, planning_clients)))
    constraints["max_ap_count"] = min(
        32,
        max(int(constraints.get("max_ap_count") or 0), capacity_minimum + 2),
    )
    suggested_min = recommended_min_ap_count(rooms, scenario)
    trained_min = int(constraints.get("min_ap_count") or 0)
    constraints["min_ap_count"] = max(trained_min, suggested_min)

    # Demand-based acceptance criteria: cover rooms that have real users, let empty
    # rooms/corridors stay uncovered, and let small adjacent rooms share signal from
    # a neighbour. Whole-building (floor) coverage is intentionally NOT required so
    # the supplemental filler does not blanket the building with redundant APs.
    targets = scenario.setdefault("targets", {})
    minimum_targets = {
        "min_coverage_ratio": 0.85,
        "min_room_sample_coverage_ratio": 0.85,
        "target_sample_coverage_ratio": 0.90,
        "target_acceptable_coverage_ratio": 0.95,
        "min_capacity_ratio": 0.90,
        "target_sinr_coverage_ratio": 0.85,
        "target_floor_coverage_ratio": 0.80,
        "target_floor_sinr_coverage_ratio": 0.80,
    }
    for key, minimum in minimum_targets.items():
        targets[key] = max(float(targets.get(key) or 0.0), minimum)
    # One marginal room is acceptable on large, highly segmented layouts when
    # whole-floor and sample coverage are already strong. Requiring zero can
    # otherwise double the AP count to fix a tiny residual corner.
    targets["max_weak_room_count"] = 1 if len(rooms) >= 10 else 0
    targets["max_capacity_shortfall_room_count"] = 1 if len(rooms) >= 10 else 0
    return scenario

def load_ppo_model_for_inference(model_path: str):
    """Load the PPO model without an env so we can inspect its action space first."""
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(model_path), True
        except Exception:
            pass
    return PPO.load(model_path), False

def get_ppo_action_space_nvec(model) -> List[int]:
    """Return trained MultiDiscrete action dimensions as regular ints."""
    nvec = getattr(getattr(model, "action_space", None), "nvec", None)
    if nvec is None:
        raise HTTPException(status_code=500, detail="Trained PPO model does not use MultiDiscrete action space")
    return [int(value) for value in nvec]

def select_candidates_for_action_space(scenario: Dict[str, Any],
                                       candidates: List[Dict[str, Any]],
                                       limit: int) -> List[Dict[str, Any]]:
    """Trim candidates to the model action space while preserving room coverage.

    Every serviceable room keeps its single best candidate before remaining
    slots are filled by score, so no room loses its only reachable position.
    """
    rooms = scenario.get("rooms", [])
    serviceable_ids = {
        str(room.get("id")) for room in rooms if room_requires_service(room)
    }
    best_by_room: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        room_id = str(candidate.get("room_id"))
        if room_id not in serviceable_ids:
            continue
        if (
            room_id not in best_by_room
            or float(candidate.get("score", 0.0)) > float(best_by_room[room_id].get("score", 0.0))
        ):
            best_by_room[room_id] = candidate

    reserved = sorted(
        best_by_room.values(),
        key=lambda candidate: float(candidate.get("score", 0.0)),
        reverse=True,
    )
    if len(reserved) >= limit:
        # More serviceable rooms than action slots: keep the highest-scored bests.
        return reserved[:limit]

    reserved_ids = {id(candidate) for candidate in reserved}
    remaining = sorted(
        (candidate for candidate in candidates if id(candidate) not in reserved_ids),
        key=lambda candidate: float(candidate.get("score", 0.0)),
        reverse=True,
    )
    selected = reserved + remaining[: limit - len(reserved)]
    return selected[:limit]


def point_in_excluded_room(x: float, y: float, scenario: Dict[str, Any]) -> bool:
    """Return whether a point falls inside a room excluded from service (e.g. stairs)."""
    if point_in_polygon is None:
        return False
    for room in scenario.get("rooms", []):
        if room_requires_service(room):
            continue
        if point_in_polygon(x, y, room.get("polygon") or []):
            return True
    return False


def strip_aps_in_excluded_rooms(scenario: Dict[str, Any],
                                placed_aps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop any AP that ended up inside an excluded area such as a staircase."""
    if point_in_polygon is None:
        return placed_aps
    return [
        ap for ap in placed_aps
        if not point_in_excluded_room(float(ap.get("x", 0.0)), float(ap.get("y", 0.0)), scenario)
    ]


def ensure_serviceable_rooms_have_ap(scenario: Dict[str, Any],
                                     placed_aps: List[Dict[str, Any]],
                                     deployment_model: Dict[str, Any],
                                     tx_power_dbm: float) -> List[Dict[str, Any]]:
    """Guarantee that every HIGH-DEMAND room gets at least one AP inside it.

    Only large / busy rooms qualify (area or client count above a threshold), so
    big rooms with real demand always get their own AP for coverage + capacity,
    while small rooms are left to share a neighbour's signal and empty/excluded
    rooms are untouched. Runs in pure-agent mode too: a busy room missing an AP is
    a genuine gap, not redundant blanketing. Runs after pruning so it sticks.
    """
    if point_in_polygon is None:
        return placed_aps
    constraints = scenario.get("constraints") or {}
    min_area = float(constraints.get("dedicated_ap_min_area_m2") or 25.0)
    min_clients = float(constraints.get("dedicated_ap_min_clients") or 20.0)

    def is_high_demand(room: Dict[str, Any]) -> bool:
        area = float(room.get("area_m2") or room.get("areaM2") or 0.0)
        clients = float(room.get("clients") or 0.0)
        return area >= min_area or clients >= min_clients

    rooms = [
        room for room in scenario.get("rooms", [])
        if room_requires_service(room) and is_high_demand(room)
    ]
    candidates = scenario.get("candidate_positions") or []
    result = list(placed_aps)

    def has_interior_ap(room: Dict[str, Any]) -> bool:
        polygon = room.get("polygon") or []
        return any(
            point_in_polygon(float(ap.get("x", 0.0)), float(ap.get("y", 0.0)), polygon)
            for ap in result
        )

    for room in rooms:
        if has_interior_ap(room):
            continue
        room_id = str(room.get("id"))
        room_candidates = sorted(
            (c for c in candidates if str(c.get("room_id")) == room_id),
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )
        if room_candidates:
            spot = room_candidates[0]
        else:
            centroid = room.get("centroid") or {}
            spot = {
                "x": float(centroid.get("x", 0.0)),
                "y": float(centroid.get("y", 0.0)),
                "z": 2.7,
            }
        result.append({
            "candidate_index": -1,
            "model_index": 0,
            "power_index": 0,
            "x": float(spot.get("x", 0.0)),
            "y": float(spot.get("y", 0.0)),
            "z": float(spot.get("z", 2.7)),
            "model_id": str(deployment_model.get("id", "ap")),
            "tx_power_dbm": float(tx_power_dbm),
            "supplemental": True,
            "room_guarantee": True,
        })
    return result


def align_scenario_to_ppo_action_space(scenario: Dict[str, Any], expected_nvec: List[int]) -> Dict[str, Any]:
    """Keep PPO inference scenario compatible with the trained model dimensions."""
    if len(expected_nvec) != 4:
        raise HTTPException(status_code=500, detail=f"Unexpected PPO action space: {expected_nvec}")

    expected_candidates, expected_models, expected_powers, expected_stop = expected_nvec
    if expected_powers != 5 or expected_stop != 2:
        raise HTTPException(status_code=500, detail=f"Unsupported PPO action space: {expected_nvec}")

    candidates = scenario.get("candidate_positions") or []
    if len(candidates) < expected_candidates:
        fallback = candidates[-1] if candidates else {"x": 0.0, "y": 0.0, "z": 2.7}
        padding_count = expected_candidates - len(candidates)
        scenario["candidate_positions"] = candidates + [
            {
                **fallback,
                "id": f"action_padding_{index + 1}",
                "source": "action_padding",
                "score": 0.0,
                "_action_padding": True,
            }
            for index in range(padding_count)
        ]
    if len(candidates) > expected_candidates:
        # Naive head-truncation drops the lowest-scored candidates, which often
        # belong to small / low-priority rooms - leaving those rooms with no
        # reachable candidate at all. Instead, reserve a slot for each
        # serviceable room's best candidate first, then fill the rest by score.
        scenario["candidate_positions"] = select_candidates_for_action_space(
            scenario, candidates, expected_candidates,
        )

    ap_catalog = scenario.setdefault("ap_catalog", {})
    ap_models = ap_catalog.get("models") or []
    if len(ap_models) < expected_models:
        raise HTTPException(
            status_code=500,
            detail=f"PPO model expects {expected_models} AP models, but only {len(ap_models)} are available",
        )
    if len(ap_models) > expected_models:
        ap_catalog["models"] = ap_models[:expected_models]

    return scenario

def choose_deployment_ap_model(ap_models: List[Dict[str, Any]], parameters: Dict[str, Any],
                               selected_frequency: float) -> Dict[str, Any]:
    """Choose the AP profile used for UI heatmap after PPO picks locations."""
    selected_id = parameters.get("selectedApModel")
    if selected_id:
        selected = next((model for model in ap_models if str(model.get("id")) == str(selected_id)), None)
        if selected:
            return selected

    band_key = "gain5" if selected_frequency >= 5 else "gain24"
    return max(
        ap_models or [{}],
        key=lambda model: (
            float(model.get("defaultPower5" if selected_frequency >= 5 else "defaultPower24") or model.get("maxPower") or 0)
            + float(model.get(band_key) or 0),
            float(model.get("recommendedClients") or model.get("clients") or 0),
        ),
    )

def ap_default_power_for_band(ap_model: Dict[str, Any], frequency: float, boost_db: float = 2.0) -> float:
    """Return a practical deployment TX power, capped by model max power."""
    if frequency >= 6:
        default_power = ap_model.get("defaultPower6") or ap_model.get("defaultPower5")
    elif frequency >= 5:
        default_power = ap_model.get("defaultPower5")
    else:
        default_power = ap_model.get("defaultPower24")
    max_power = float(ap_model.get("maxPower") or default_power or 20)
    return min(max_power, float(default_power or max_power) + boost_db)

def target_smart_ap_count(rooms: List[Dict[str, Any]], selected_frequency: float,
                          current_count: int) -> int:
    """Choose a practical AP count for dense room layouts."""
    active_rooms = [room for room in rooms if room_requires_service(room)]
    total_area = sum(float(room.get("area_m2") or room.get("areaM2") or 0.0) for room in active_rooms)
    rooms_based = (len(active_rooms) + 2) // 3
    area_per_ap = 42.0 if selected_frequency >= 5 else 65.0
    area_based = int((total_area + area_per_ap - 0.001) // area_per_ap) if total_area > 0 else 1
    return min(16, max(1, rooms_based, area_based))

def room_containing_point(x: float, y: float,
                          rooms: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the room containing a deployment point, if any."""
    if point_in_polygon is None:
        return None
    return next(
        (
            room for room in rooms
            if point_in_polygon(x, y, room.get("polygon") or [])
        ),
        None,
    )

def room_ap_limit(room: Dict[str, Any]) -> int:
    """Allow a second AP only for genuinely large or high-density rooms."""
    area_m2 = float(room.get("area_m2") or room.get("areaM2") or 0.0)
    clients = int(room.get("clients") or 0)
    room_type = str(room.get("type") or room.get("roomType") or "").lower()
    high_density_types = {"auditorium", "high_density", "event", "hall"}
    return 2 if area_m2 >= 70.0 or clients >= 100 or room_type in high_density_types else 1

def required_local_ap_count(room: Dict[str, Any], deployment_model: Dict[str, Any]) -> int:
    """Return local AP capacity required by a dense room."""
    if not room_is_high_density(room):
        return 0
    clients = float(room.get("clients") or 0.0)
    headroom = float(room.get("capacity_headroom_ratio") or 1.2)
    recommended_clients = max(1.0, float(
        deployment_model.get("planningClients")
        or deployment_model.get("recommendedClients")
        or 40.0
    ))
    return min(room_ap_limit(room), max(1, int(np.ceil(clients * headroom / recommended_clients))))

def deployment_meets_targets(scenario: Dict[str, Any], metrics: Dict[str, float]) -> bool:
    """Return whether a deployment satisfies the planner's hard RF targets."""
    targets = scenario.get("targets") or {}
    return bool(metrics) and (
        metrics.get("sample_coverage_ratio", 0.0)
        >= float(targets.get("target_sample_coverage_ratio") or 0.98)
        and metrics.get("acceptable_sample_ratio", 0.0)
        >= float(targets.get("target_acceptable_coverage_ratio") or 0.98)
        and metrics.get("capacity_ratio", 0.0)
        >= float(targets.get("min_capacity_ratio") or 0.95)
        and metrics.get("capacity_shortfall_room_count", 999.0)
        <= float(targets.get("max_capacity_shortfall_room_count", 0))
        and metrics.get("high_density_room_without_local_ap_count", 999.0) <= 0
        and metrics.get("weak_room_count", 999.0)
        <= float(targets.get("max_weak_room_count", 0))
        and metrics.get("floor_coverage_ratio", 0.0)
        >= float(targets.get("target_floor_coverage_ratio") or 0.97)
        and metrics.get("floor_sinr_coverage_ratio", 0.0)
        >= float(targets.get("target_floor_sinr_coverage_ratio") or 0.90)
    )

def enforce_room_ap_limits(scenario: Dict[str, Any], placed_aps: List[Dict[str, Any]],
                           deployment_model: Dict[str, Any],
                           tx_power_dbm: float) -> List[Dict[str, Any]]:
    """Remove excess APs from rooms while retaining the strongest deployment."""
    rooms = [room for room in scenario.get("rooms", []) if room_requires_service(room)]
    result = list(placed_aps)
    if point_in_polygon is None or len(result) <= 1:
        return result

    def deployment_score(items: List[Dict[str, Any]]) -> float:
        metrics = evaluate_deployment_metrics(scenario, items, deployment_model, tx_power_dbm)
        return (
            0.40 * metrics.get("sample_coverage_ratio", 0.0)
            + 0.25 * metrics.get("floor_coverage_ratio", 0.0)
            + 0.15 * metrics.get("acceptable_sample_ratio", 0.0)
            + 0.15 * metrics.get("floor_sinr_coverage_ratio", 0.0)
            + 0.05 * metrics.get("capacity_ratio", 0.0)
            - 0.10 * metrics.get("cochannel_interference_ratio", 0.0)
            - 0.05 * metrics.get("overlap_ratio", 0.0)
        )

    for room in rooms:
        while True:
            indices = [
                index for index, ap in enumerate(result)
                if point_in_polygon(
                    float(ap.get("x", 0.0)),
                    float(ap.get("y", 0.0)),
                    room.get("polygon") or [],
                )
            ]
            if len(indices) <= room_ap_limit(room):
                break
            current_metrics = evaluate_deployment_metrics(
                scenario, result, deployment_model, tx_power_dbm,
            )
            current_score = deployment_score(result)
            current_meets_targets = deployment_meets_targets(scenario, current_metrics)
            alternatives = []
            for index in indices:
                candidate = result[:index] + result[index + 1:]
                metrics = evaluate_deployment_metrics(
                    scenario, candidate, deployment_model, tx_power_dbm,
                )
                score = deployment_score(candidate)
                if current_meets_targets and not deployment_meets_targets(scenario, metrics):
                    continue
                if not current_meets_targets and (
                    score < current_score
                    or metrics.get("sample_coverage_ratio", 0.0)
                    < current_metrics.get("sample_coverage_ratio", 0.0) - 0.005
                    or metrics.get("floor_coverage_ratio", 0.0)
                    < current_metrics.get("floor_coverage_ratio", 0.0) - 0.005
                ):
                    continue
                alternatives.append((score, candidate))
            if not alternatives:
                break
            result = max(alternatives, key=lambda item: item[0])[1]

    return result

def complete_weak_room_coverage(scenario: Dict[str, Any], placed_aps: List[Dict[str, Any]],
                                deployment_model: Dict[str, Any],
                                selected_frequency: float,
                                tx_power_dbm: float) -> List[Dict[str, Any]]:
    """Add supplemental APs in rooms that remain weak after PPO placement."""
    if PlacedAP is None:
        return placed_aps
    if (scenario.get("constraints") or {}).get("rl_agent_placement"):
        return placed_aps

    rooms = [room for room in scenario.get("rooms", []) if room_requires_service(room)]
    max_ap_count = int((scenario.get("constraints") or {}).get("max_ap_count") or 12)
    practical_count = target_smart_ap_count(rooms, selected_frequency, len(placed_aps))
    required_min_count = int((scenario.get("constraints") or {}).get("min_ap_count") or 0)
    max_completion_count = min(
        max_ap_count,
        max(len(placed_aps), practical_count, required_min_count) + 2,
    )

    eval_scenario = dict(scenario)
    eval_scenario["ap_catalog"] = {"models": [deployment_model]}
    eval_env = GymWiFiAPPlacementEnv(eval_scenario).env

    def as_env_ap(ap: Dict[str, Any], candidate_index: int) -> PlacedAP:
        return PlacedAP(
            candidate_index=candidate_index,
            model_index=0,
            power_index=0,
            x=float(ap.get("x", 0.0)),
            y=float(ap.get("y", 0.0)),
            z=float(ap.get("z", 2.7)),
            model_id=str(deployment_model.get("id", "ap")),
            tx_power_dbm=float(tx_power_dbm),
        )

    env_aps = [as_env_ap(ap, index) for index, ap in enumerate(placed_aps)]
    # Honour the same spacing the RL policy was trained with so supplemental APs
    # spread out instead of crowding next to existing ones.
    min_separation = float((scenario.get("constraints") or {}).get("min_ap_separation_m") or 5.0)

    def room_score(room: Dict[str, Any]) -> Dict[str, Any]:
        eval_env.placed_aps = env_aps
        samples = eval_env.room_samples.get(str(room.get("id"))) or [
            (
                float((room.get("centroid") or {}).get("x", 0)),
                float((room.get("centroid") or {}).get("y", 0)),
            )
        ]
        target = float(room.get("coverage_target_dbm") or eval_env.good_signal_dbm)
        best_signals = []
        for sample_x, sample_y in samples:
            signals = [eval_env.rssi_at_room(ap, sample_x, sample_y) for ap in env_aps]
            best_signals.append(max(signals) if signals else -100.0)
        coverage = sum(1 for signal in best_signals if signal >= target) / max(1, len(best_signals))
        local_ap_count = sum(
            1 for ap in env_aps
            if point_in_polygon(ap.x, ap.y, room.get("polygon") or [])
        )
        required_local_count = required_local_ap_count(room, deployment_model)
        return {
            "room": room,
            "coverage": coverage,
            "min_signal": min(best_signals) if best_signals else -100.0,
            "avg_signal": sum(best_signals) / max(1, len(best_signals)),
            "needs_local_ap": local_ap_count < required_local_count,
        }

    def far_enough(x: float, y: float) -> bool:
        return all(float(np.hypot(x - ap.x, y - ap.y)) >= min_separation for ap in env_aps)

    def room_has_capacity(room: Dict[str, Any]) -> bool:
        polygon = room.get("polygon") or []
        count = sum(1 for ap in env_aps if point_in_polygon(ap.x, ap.y, polygon))
        return count < room_ap_limit(room)

    def candidate_room_has_capacity(candidate: Dict[str, Any]) -> bool:
        room = room_containing_point(
            float(candidate.get("x", 0.0)),
            float(candidate.get("y", 0.0)),
            rooms,
        )
        return room is None or room_has_capacity(room)

    candidates = scenario.get("candidate_positions") or []
    allowed_weak_rooms = int((scenario.get("targets") or {}).get("max_weak_room_count", 1))
    target_sample_coverage = float((scenario.get("targets") or {}).get("target_sample_coverage_ratio") or 0.98)
    target_acceptable_coverage = float((scenario.get("targets") or {}).get("target_acceptable_coverage_ratio") or 0.98)
    target_floor_coverage = float((scenario.get("targets") or {}).get("target_floor_coverage_ratio") or 0.97)
    target_floor_sinr = float((scenario.get("targets") or {}).get("target_floor_sinr_coverage_ratio") or 0.90)
    while len(placed_aps) < max_completion_count:
        ranked_rooms = sorted(
            [score for score in (room_score(room) for room in rooms)],
            key=lambda score: (
                not score["needs_local_ap"],
                score["coverage"],
                score["min_signal"],
                score["avg_signal"],
            ),
        )
        if not ranked_rooms:
            break
        weak_scores = [
            score for score in ranked_rooms
            if score["coverage"] < eval_env.min_room_sample_coverage or score["needs_local_ap"]
        ]
        eval_env.placed_aps = env_aps
        current_metrics = eval_env.evaluate()
        floor_target_met = current_metrics.get("floor_coverage_ratio", 0.0) >= target_floor_coverage
        all_targets_met = (
            len(weak_scores) <= allowed_weak_rooms
            and current_metrics.get("sample_coverage_ratio", 0.0) >= target_sample_coverage
            and current_metrics.get("acceptable_sample_ratio", 0.0) >= target_acceptable_coverage
            and current_metrics.get("capacity_shortfall_room_count", 999.0)
            <= float((scenario.get("targets") or {}).get("max_capacity_shortfall_room_count", 0))
            and current_metrics.get("high_density_room_without_local_ap_count", 999.0) <= 0
            and floor_target_met
            and current_metrics.get("floor_sinr_coverage_ratio", 0.0) >= target_floor_sinr
        )
        if all_targets_met:
            break

        selected = None
        for score in weak_scores:
            room = score["room"]
            if not room_has_capacity(room):
                continue
            area_m2 = float(room.get("area_m2") or room.get("areaM2") or 0.0)
            if (
                area_m2 < 6.0
                and not score["needs_local_ap"]
                and score["avg_signal"] >= eval_env.acceptable_signal_dbm
            ):
                continue
            room_id = str(room.get("id"))
            room_candidates = [
                candidate for candidate in candidates
                if str(candidate.get("room_id")) == room_id
                and far_enough(float(candidate.get("x", 0)), float(candidate.get("y", 0)))
            ]
            room_candidates.sort(key=lambda candidate: float(candidate.get("score", 0.0)), reverse=True)
            if room_candidates:
                selected = room_candidates[0]
                break
            centroid = room.get("centroid") or {}
            cx = float(centroid.get("x", 0))
            cy = float(centroid.get("y", 0))
            if far_enough(cx, cy):
                selected = {"x": cx, "y": cy, "z": 2.7}
                break
        capacity_target_unmet = (
            current_metrics.get("capacity_shortfall_room_count", 999.0) > 0
            or current_metrics.get("capacity_ratio", 0.0)
            < float((scenario.get("targets") or {}).get("min_capacity_ratio") or 0.95)
            or len(placed_aps) < required_min_count
        )
        if selected is None and capacity_target_unmet:
            capacity_candidates = []
            for room in sorted(
                rooms,
                key=lambda item: (
                    float(item.get("clients") or 0.0)
                    * float(item.get("capacity_headroom_ratio") or 1.2)
                ),
                reverse=True,
            ):
                if not room_has_capacity(room):
                    continue
                room_id = str(room.get("id"))
                candidate = next(
                    (
                        item for item in candidates
                        if str(item.get("room_id")) == room_id
                        and far_enough(float(item.get("x", 0)), float(item.get("y", 0)))
                    ),
                    None,
                )
                if candidate:
                    capacity_candidates.append(candidate)
            if capacity_candidates:
                selected = capacity_candidates[0]
            else:
                fallback_capacity_candidates = [
                    candidate for candidate in candidates
                    if far_enough(float(candidate.get("x", 0)), float(candidate.get("y", 0)))
                    and candidate_room_has_capacity(candidate)
                ]
                fallback_capacity_candidates.sort(
                    key=lambda candidate: float(candidate.get("score", 0.0)),
                    reverse=True,
                )
                if fallback_capacity_candidates:
                    selected = fallback_capacity_candidates[0]
        if selected is None and not floor_target_met:
            weak_floor_samples = []
            for sample_x, sample_y in eval_env.floor_samples:
                signals = [eval_env.rssi_at_room(ap, sample_x, sample_y) for ap in env_aps]
                if not signals or max(signals) < eval_env.good_signal_dbm:
                    weak_floor_samples.append((sample_x, sample_y))
            floor_candidates = [
                candidate for candidate in candidates
                if far_enough(float(candidate.get("x", 0)), float(candidate.get("y", 0)))
                and candidate_room_has_capacity(candidate)
            ]
            floor_candidates.sort(key=lambda candidate: (
                sum(
                    float(np.hypot(
                        float(candidate.get("x", 0)) - sample_x,
                        float(candidate.get("y", 0)) - sample_y,
                    ))
                    for sample_x, sample_y in weak_floor_samples
                ) / max(1, len(weak_floor_samples)),
                -float(candidate.get("score", 0.0)),
            ))
            if floor_candidates:
                selected = floor_candidates[0]
        if selected is None:
            break

        supplemental = {
            "candidate_index": -1,
            "model_index": 0,
            "power_index": 0,
            "x": float(selected.get("x", 0)),
            "y": float(selected.get("y", 0)),
            "z": float(selected.get("z", 2.7)),
            "model_id": str(deployment_model.get("id", "ap")),
            "tx_power_dbm": float(tx_power_dbm),
            "supplemental": True,
        }
        placed_aps.append(supplemental)
        env_aps.append(as_env_ap(supplemental, len(env_aps)))

    return placed_aps

def prune_redundant_aps(scenario: Dict[str, Any], placed_aps: List[Dict[str, Any]],
                        deployment_model: Dict[str, Any],
                        tx_power_dbm: float) -> List[Dict[str, Any]]:
    """Remove APs whose loss does not materially reduce coverage or capacity."""
    if PlacedAP is None or len(placed_aps) <= 1:
        return placed_aps

    eval_scenario = dict(scenario)
    eval_scenario["ap_catalog"] = {"models": [deployment_model]}
    eval_env = GymWiFiAPPlacementEnv(eval_scenario).env
    constraints = scenario.get("constraints") or {}
    targets = scenario.get("targets") or {}
    min_count = recommended_min_ap_count(
        [room for room in scenario.get("rooms", []) if room_requires_service(room)],
        scenario,
    )
    target_coverage = float(targets.get("target_sample_coverage_ratio") or targets.get("min_coverage_ratio") or 0.95)
    min_acceptable = max(0.93, target_coverage - 0.02)
    max_weak_rooms = float(targets.get("max_weak_room_count", 1.0))
    target_floor_coverage = float(targets.get("target_floor_coverage_ratio") or 0.97)
    target_floor_sinr = float(targets.get("target_floor_sinr_coverage_ratio") or 0.90)

    def evaluate(items: List[Dict[str, Any]]) -> Dict[str, float]:
        eval_env.placed_aps = [
            PlacedAP(
                candidate_index=int(ap.get("candidate_index", index)),
                model_index=0,
                power_index=int(ap.get("power_index", 0)),
                x=float(ap.get("x", 0.0)),
                y=float(ap.get("y", 0.0)),
                z=float(ap.get("z", 2.7)),
                model_id=str(deployment_model.get("id", "ap")),
                tx_power_dbm=float(tx_power_dbm),
            )
            for index, ap in enumerate(items)
        ]
        return eval_env.evaluate()

    result = list(placed_aps)
    while len(result) > min_count:
        removable = []
        for index in range(len(result)):
            candidate = result[:index] + result[index + 1:]
            metrics = evaluate(candidate)
            if (
                metrics.get("sample_coverage_ratio", 0.0) >= target_coverage
                and metrics.get("acceptable_sample_ratio", 0.0) >= min_acceptable
                and metrics.get("capacity_ratio", 0.0) >= 0.95
                and metrics.get("weak_room_count", 999.0) <= max_weak_rooms
                and metrics.get("floor_coverage_ratio", 0.0) >= target_floor_coverage
                and metrics.get("floor_sinr_coverage_ratio", 0.0) >= target_floor_sinr
            ):
                score = (
                    metrics.get("sample_coverage_ratio", 0.0)
                    + 0.25 * metrics.get("acceptable_sample_ratio", 0.0)
                    + 0.20 * metrics.get("sinr_coverage_ratio", 0.0)
                    - 0.25 * metrics.get("cochannel_interference_ratio", 0.0)
                    - 0.10 * metrics.get("overlap_ratio", 0.0)
                )
                removable.append((score, candidate))
        if not removable:
            break
        result = max(removable, key=lambda item: item[0])[1]

    return result

def prune_close_aps(scenario: Dict[str, Any], placed_aps: List[Dict[str, Any]],
                    deployment_model: Dict[str, Any],
                    tx_power_dbm: float) -> List[Dict[str, Any]]:
    """Remove APs from overly close pairs when RF quality remains acceptable."""
    if len(placed_aps) <= 1:
        return placed_aps

    constraints = scenario.get("constraints") or {}
    minimum_count = recommended_min_ap_count(
        [room for room in scenario.get("rooms", []) if room_requires_service(room)],
        scenario,
    )
    separation = max(4.0, float(constraints.get("min_ap_separation_m") or 4.0))
    result = list(placed_aps)

    while len(result) > minimum_count:
        current = evaluate_deployment_metrics(scenario, result, deployment_model, tx_power_dbm)
        close_indices = set()
        for first_index, first in enumerate(result):
            for second_index in range(first_index + 1, len(result)):
                second = result[second_index]
                distance = float(np.hypot(
                    float(first.get("x", 0.0)) - float(second.get("x", 0.0)),
                    float(first.get("y", 0.0)) - float(second.get("y", 0.0)),
                ))
                if distance < separation:
                    close_indices.update((first_index, second_index))

        if not close_indices:
            break

        removable = []
        for index in close_indices:
            candidate = result[:index] + result[index + 1:]
            metrics = evaluate_deployment_metrics(scenario, candidate, deployment_model, tx_power_dbm)
            if (
                deployment_meets_targets(scenario, metrics)
                and metrics.get("sample_coverage_ratio", 0.0)
                >= current.get("sample_coverage_ratio", 0.0) - 0.015
                and metrics.get("floor_coverage_ratio", 0.0)
                >= current.get("floor_coverage_ratio", 0.0) - 0.02
            ):
                removable.append((deployment_alternative_score(metrics), candidate))

        if not removable:
            break
        result = max(removable, key=lambda item: item[0])[1]

    return result

def prune_cost_inefficient_aps(scenario: Dict[str, Any], placed_aps: List[Dict[str, Any]],
                               deployment_model: Dict[str, Any],
                               selected_frequency: float,
                               tx_power_dbm: float) -> List[Dict[str, Any]]:
    """Prefer a smaller deployment when extra APs add little usable coverage."""
    rooms = [room for room in scenario.get("rooms", []) if room_requires_service(room)]
    practical_count = target_smart_ap_count(rooms, selected_frequency, len(placed_aps))
    result = list(placed_aps)
    if len(result) <= practical_count:
        return result

    while len(result) > practical_count:
        current_metrics = evaluate_deployment_metrics(
            scenario, result, deployment_model, tx_power_dbm,
        )
        removable = []
        for index in range(len(result)):
            candidate = result[:index] + result[index + 1:]
            metrics = evaluate_deployment_metrics(scenario, candidate, deployment_model, tx_power_dbm)
            if deployment_preserves_quality(scenario, current_metrics, metrics):
                removable.append((deployment_alternative_score(metrics), candidate))
        if not removable:
            break
        result = max(removable, key=lambda item: item[0])[1]

    return result

def evaluate_deployment_metrics(scenario: Dict[str, Any], placed_aps: List[Dict[str, Any]],
                                deployment_model: Dict[str, Any],
                                tx_power_dbm: float) -> Dict[str, float]:
    """Evaluate the final post-processed deployment using the selected AP profile."""
    if PlacedAP is None:
        return {}
    eval_scenario = dict(scenario)
    eval_scenario["ap_catalog"] = {"models": [deployment_model]}
    eval_env = GymWiFiAPPlacementEnv(eval_scenario).env
    eval_env.placed_aps = [
        PlacedAP(
            candidate_index=int(ap.get("candidate_index", index)),
            model_index=0,
            power_index=int(ap.get("power_index", 0)),
            x=float(ap.get("x", 0.0)),
            y=float(ap.get("y", 0.0)),
            z=float(ap.get("z", 2.7)),
            model_id=str(deployment_model.get("id", "ap")),
            tx_power_dbm=float(ap.get("tx_power_dbm", tx_power_dbm)),
        )
        for index, ap in enumerate(placed_aps)
    ]
    return eval_env.evaluate()

def deployment_alternative_score(metrics: Dict[str, float]) -> float:
    """Rank complete alternatives by usable coverage, interference, and cost."""
    return (
        45.0 * metrics.get("sample_coverage_ratio", 0.0)
        + 15.0 * metrics.get("acceptable_sample_ratio", 0.0)
        + 25.0 * metrics.get("floor_coverage_ratio", 0.0)
        + 8.0 * metrics.get("floor_sinr_coverage_ratio", 0.0)
        + 8.0 * metrics.get("capacity_ratio", 0.0)
        - 1.8 * metrics.get("weak_room_count", 0.0)
        - 4.0 * metrics.get("capacity_shortfall_room_count", 0.0)
        - 6.0 * metrics.get("high_density_room_without_local_ap_count", 0.0)
        - 15.0 * metrics.get("cochannel_interference_ratio", 0.0)
        - 3.0 * metrics.get("overlap_ratio", 0.0)
        - 1.0 * metrics.get("ap_count", 0.0)
    )

def deployment_preserves_quality(scenario: Dict[str, Any], reference: Dict[str, float],
                                 candidate: Dict[str, float]) -> bool:
    """Allow a cheaper plan when removing an AP has no meaningful RF impact."""
    if deployment_meets_targets(scenario, reference):
        return deployment_meets_targets(scenario, candidate)

    return bool(candidate) and (
        candidate.get("sample_coverage_ratio", 0.0)
        >= reference.get("sample_coverage_ratio", 0.0) - 0.012
        and candidate.get("acceptable_sample_ratio", 0.0)
        >= reference.get("acceptable_sample_ratio", 0.0) - 0.01
        and candidate.get("floor_coverage_ratio", 0.0)
        >= reference.get("floor_coverage_ratio", 0.0) - 0.015
        and candidate.get("floor_sinr_coverage_ratio", 0.0)
        >= reference.get("floor_sinr_coverage_ratio", 0.0) - 0.02
        and candidate.get("capacity_ratio", 0.0)
        >= reference.get("capacity_ratio", 0.0) - 0.02
        and candidate.get("capacity_shortfall_room_count", 999.0)
        <= reference.get("capacity_shortfall_room_count", 999.0)
        and candidate.get("high_density_room_without_local_ap_count", 999.0)
        <= reference.get("high_density_room_without_local_ap_count", 999.0)
        and candidate.get("weak_room_count", 999.0)
        <= reference.get("weak_room_count", 999.0)
        and candidate.get("cochannel_interference_ratio", 1.0)
        <= reference.get("cochannel_interference_ratio", 1.0) + 0.02
    )

def select_cost_efficient_plan(scenario: Dict[str, Any],
                               plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the smallest plan whose RF quality is effectively tied for best."""
    best_quality = max(
        plans,
        key=lambda item: deployment_selection_key(scenario, item["metrics"]),
    )
    near_best = [
        item for item in plans
        if deployment_preserves_quality(scenario, best_quality["metrics"], item["metrics"])
    ]
    return max(
        near_best or [best_quality],
        key=lambda item: (
            -float(item["metrics"].get("ap_count", 999.0)),
            deployment_alternative_score(item["metrics"]),
        ),
    )

def deployment_selection_key(scenario: Dict[str, Any], metrics: Dict[str, float]) -> tuple:
    """Prefer visibly complete RF plans before considering AP efficiency."""
    targets = scenario.get("targets") or {}
    sample = float(metrics.get("sample_coverage_ratio", 0.0))
    floor = float(metrics.get("floor_coverage_ratio", 0.0))
    acceptable = float(metrics.get("acceptable_sample_ratio", 0.0))
    capacity = float(metrics.get("capacity_ratio", 0.0))
    floor_sinr = float(metrics.get("floor_sinr_coverage_ratio", 0.0))
    weak_rooms = float(metrics.get("weak_room_count", 999.0))
    capacity_shortfall_rooms = float(metrics.get("capacity_shortfall_room_count", 999.0))
    interference = float(metrics.get("cochannel_interference_ratio", 1.0))
    sample_target = float(targets.get("target_sample_coverage_ratio") or 0.98)
    floor_target = float(targets.get("target_floor_coverage_ratio") or 0.97)
    acceptable_target = float(targets.get("target_acceptable_coverage_ratio") or 0.99)
    capacity_target = float(targets.get("min_capacity_ratio") or 0.95)
    robust = bool(
        sample >= max(sample_target, 0.99)
        and floor >= max(floor_target, 0.98)
        and acceptable >= acceptable_target
        and capacity >= capacity_target
        and capacity_shortfall_rooms <= float(targets.get("max_capacity_shortfall_room_count", 0))
        and floor_sinr >= float(targets.get("target_floor_sinr_coverage_ratio") or 0.95)
        and weak_rooms <= float(targets.get("max_weak_room_count", 0))
    )

    return (
        deployment_meets_targets(scenario, metrics),
        robust,
        -float(metrics.get("ap_count", 999.0)) if robust else -999.0,
        min(
            sample / sample_target,
            floor / floor_target,
            acceptable / acceptable_target,
            capacity / capacity_target,
        ),
        sample + floor + 0.35 * acceptable + 0.20 * floor_sinr,
        -capacity_shortfall_rooms,
        -weak_rooms,
        -interference,
        deployment_alternative_score(metrics),
    )

def format_deployment_access_points(placed_aps: List[Dict[str, Any]],
                                    deployment_model: Dict[str, Any],
                                    selected_frequency: float,
                                    deployment_power24: float,
                                    deployment_power5: float) -> List[Dict[str, Any]]:
    """Format one evaluated alternative for the planner UI."""
    active_power = deployment_power5 if selected_frequency >= 5 else deployment_power24
    access_points = []
    for ap in placed_aps:
        ap_active_power = float(ap.get("tx_power_dbm", active_power))
        access_points.append({
            "id": f"ppo_ap_{len(access_points)}",
            "name": f"AP {len(access_points) + 1}",
            "x": float(ap.get("x", 0)),
            "y": float(ap.get("y", 0)),
            "z": float(ap.get("z", 2.7)),
            "model_id": str(deployment_model.get("id", ap.get("model_id", "ap"))),
            "vendor": deployment_model.get("vendor", "Custom"),
            "model": deployment_model.get("model", "Custom AP"),
            "color": deployment_model.get("color", "#2563eb"),
            "frequency": selected_frequency,
            "power": ap_active_power,
            "power24": min(deployment_power24, ap_active_power),
            "power5": min(deployment_power5, ap_active_power),
            "gain24": float(deployment_model.get("gain24", 2)),
            "gain5": float(deployment_model.get("gain5", deployment_model.get("gain24", 2))),
        })
    assign_ap_channels(access_points, selected_frequency)
    return access_points

def assign_ap_channels(access_points: List[Dict[str, Any]], selected_frequency: float) -> Dict[str, Any]:
    """Assign non-overlapping WiFi channels, prioritizing nearby AP separation."""
    if not access_points:
        return {"active_band": "5 GHz" if selected_frequency >= 5 else "2.4 GHz", "cochannel_conflicts": 0}

    channels_24 = [1, 6, 11]
    channels_5 = [36, 40, 44, 48, 149, 153, 157, 161]

    def assign_for_band(channels: List[int], avoid_radius_m: float) -> List[int]:
        coords = [(float(ap.get("x", 0.0)), float(ap.get("y", 0.0))) for ap in access_points]
        density = []
        for i, (x1, y1) in enumerate(coords):
            score = 0.0
            for j, (x2, y2) in enumerate(coords):
                if i == j:
                    continue
                distance = float(np.hypot(x1 - x2, y1 - y2))
                if distance < avoid_radius_m:
                    score += (avoid_radius_m - distance) / avoid_radius_m
            density.append(score)

        order = sorted(range(len(access_points)), key=lambda idx: density[idx], reverse=True)
        assigned: Dict[int, int] = {}
        usage = {channel: 0 for channel in channels}

        for idx in order:
            x1, y1 = coords[idx]
            best_channel = channels[0]
            best_score = None
            for channel in channels:
                score = usage[channel] * 0.08
                for other_idx, other_channel in assigned.items():
                    if other_channel != channel:
                        continue
                    x2, y2 = coords[other_idx]
                    distance = float(np.hypot(x1 - x2, y1 - y2))
                    if distance < avoid_radius_m:
                        proximity = (avoid_radius_m - distance) / avoid_radius_m
                        score += 10.0 * proximity * proximity
                    elif distance < avoid_radius_m * 1.7:
                        score += 0.6 * ((avoid_radius_m * 1.7 - distance) / (avoid_radius_m * 0.7))
                if best_score is None or score < best_score:
                    best_score = score
                    best_channel = channel
            assigned[idx] = best_channel
            usage[best_channel] += 1

        return [assigned[idx] for idx in range(len(access_points))]

    assigned_24 = assign_for_band(channels_24, 14.0)
    assigned_5 = assign_for_band(channels_5, 8.0)

    for index, ap in enumerate(access_points):
        ap["channel24"] = assigned_24[index]
        ap["channel5"] = assigned_5[index]
        ap["channel"] = assigned_5[index] if selected_frequency >= 5 else assigned_24[index]

    active_key = "channel5" if selected_frequency >= 5 else "channel24"
    active_radius = 8.0 if selected_frequency >= 5 else 14.0
    conflicts = 0
    for i, ap_a in enumerate(access_points):
        for ap_b in access_points[i + 1:]:
            same_channel = ap_a.get(active_key) == ap_b.get(active_key)
            distance = float(np.hypot(float(ap_a.get("x", 0.0)) - float(ap_b.get("x", 0.0)),
                                      float(ap_a.get("y", 0.0)) - float(ap_b.get("y", 0.0))))
            if same_channel and distance < active_radius:
                conflicts += 1

    return {
        "active_band": "5 GHz" if selected_frequency >= 5 else "2.4 GHz",
        "active_channels": channels_5 if selected_frequency >= 5 else channels_24,
        "cochannel_conflicts": conflicts,
    }

@app.post("/api/rl/ppo-placement")
def create_rl_ppo_placement(project_data: ProjectData):
    """Run inference using the latest trained PPO model."""
    if PPO is None or GymWiFiAPPlacementEnv is None or adapt_scenario_for_rl is None:
        raise HTTPException(status_code=500, detail="PPO or gym environment is not available")
        
    run_dir = load_latest_ppo_run()
    model_path = load_best_ppo_model()
    if not model_path:
        raise HTTPException(status_code=404, detail="No trained PPO model found in training_runs/")
    summary = load_ppo_training_summary(run_dir) if run_dir else {}

    scenario = build_rl_training_scenario(
        name=project_data.parameters.get("name", "WiFi RL Scenario"),
        elements=[e.model_dump() for e in project_data.elements],
        access_points=[a.model_dump() for a in project_data.access_points],
        parameters=project_data.parameters,
    )
    scenario = apply_ppo_summary_to_scenario(scenario, summary)
    selected_frequency = float((scenario.get("floor_plan") or {}).get("selected_frequency_ghz") or 2.4)
    deployment_model = choose_deployment_ap_model(
        (scenario.get("ap_catalog") or {}).get("models", []),
        project_data.parameters,
        selected_frequency,
    )
    scenario = apply_inference_ap_floor(scenario, deployment_model)
    # "RL agent" mode (default): deploy exactly the RL agent's placement, skipping
    # the AP-adding completion. Set rlAgentPlacement=false to re-enable the
    # cost-aware completion safety net.
    rl_agent = bool(project_data.parameters.get("rlAgentPlacement", True))
    scenario.setdefault("constraints", {})["rl_agent_placement"] = rl_agent
    active_rooms = [room for room in scenario.get("rooms", []) if room_requires_service(room)]
    if not active_rooms:
        raise HTTPException(status_code=400, detail="No rooms found for PPO placement")
    if not scenario.get("candidate_positions"):
        raise HTTPException(status_code=400, detail="No AP candidate positions generated for PPO placement")
    
    # Run inference in the gym wrapper
    model, use_action_mask = load_ppo_model_for_inference(model_path)
    expected_nvec = get_ppo_action_space_nvec(model)
    scenario = align_scenario_to_ppo_action_space(scenario, expected_nvec)
    ap_models = (scenario.get("ap_catalog") or {}).get("models", [])
    deployment_model = choose_deployment_ap_model(ap_models, project_data.parameters, selected_frequency)
    deployment_power24 = ap_default_power_for_band(deployment_model, 2.4, boost_db=2.0)
    deployment_power5 = ap_default_power_for_band(deployment_model, 5.0, boost_db=2.0)
    active_power = deployment_power5 if selected_frequency >= 5 else deployment_power24
    requested_alternatives = int(project_data.parameters.get("smartApAlternativeCount", 2))
    alternative_count = max(2, min(requested_alternatives, 6))
    if len(active_rooms) >= 15:
        alternative_count = min(alternative_count, 2)
    base_seed = int(project_data.parameters.get("smartApExplorationSeed") or time.time_ns() % 2_147_483_647)
    alternatives = []

    for alternative_index in range(alternative_count):
        env = GymWiFiAPPlacementEnv(scenario)
        model.set_env(env)
        seed = base_seed + alternative_index * 7919
        model.set_random_seed(seed)
        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        deterministic = alternative_index == 0
        while not done:
            if use_action_mask:
                action, _states = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=env.action_masks(),
                )
            else:
                action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        placed_aps = list(info.get("placed_aps", []))
        ppo_ap_count = len(placed_aps)
        stage_plans = [("ppo", list(placed_aps))]
        rl_agent = bool((scenario.get("constraints") or {}).get("rl_agent_placement"))
        capacity_driven_completion = (
            ppo_ap_count < int((scenario.get("constraints") or {}).get("min_ap_count") or 0)
        )
        # Removal-only pruning runs in BOTH modes: it trims redundant / close /
        # cost-inefficient APs (e.g. a small room already covered by a neighbour
        # loses its AP). It only REMOVES APs, so it is safe in pure-agent mode.
        if not capacity_driven_completion:
            placed_aps = enforce_room_ap_limits(
                scenario, placed_aps, deployment_model, active_power,
            )
            stage_plans.append(("room_limits", list(placed_aps)))
            placed_aps = prune_close_aps(
                scenario, placed_aps, deployment_model, active_power,
            )
            stage_plans.append(("close_prune", list(placed_aps)))
            placed_aps = prune_redundant_aps(
                scenario, placed_aps, deployment_model, active_power,
            )
            stage_plans.append(("redundancy_prune", list(placed_aps)))
            placed_aps = prune_cost_inefficient_aps(
                scenario, placed_aps, deployment_model, selected_frequency, active_power,
            )
            stage_plans.append(("cost_prune", list(placed_aps)))
        # AP-ADDING completion only runs when RL-agent mode is OFF.
        if not rl_agent:
            placed_aps = complete_weak_room_coverage(
                scenario, placed_aps, deployment_model, selected_frequency, active_power,
            )
            stage_plans.append(("coverage_completion", list(placed_aps)))
            if not capacity_driven_completion:
                placed_aps = enforce_room_ap_limits(
                    scenario, placed_aps, deployment_model, active_power,
                )
                stage_plans.append(("completed_room_limits", list(placed_aps)))
                placed_aps = prune_close_aps(
                    scenario, placed_aps, deployment_model, active_power,
                )
                stage_plans.append(("completed_close_prune", list(placed_aps)))
                placed_aps = prune_redundant_aps(
                    scenario, placed_aps, deployment_model, active_power,
                )
                stage_plans.append(("completed_redundancy_prune", list(placed_aps)))
                placed_aps = prune_cost_inefficient_aps(
                    scenario, placed_aps, deployment_model, selected_frequency, active_power,
                )
                stage_plans.append(("cost_prune", list(placed_aps)))

        evaluated_stages = []
        seen_stage_plans = set()
        for stage, plan in stage_plans:
            plan_key = tuple(sorted(
                (round(float(ap.get("x", 0.0)), 3), round(float(ap.get("y", 0.0)), 3))
                for ap in plan
            ))
            if plan_key in seen_stage_plans:
                continue
            seen_stage_plans.add(plan_key)
            evaluated_stages.append({
                "stage": stage,
                "placed_aps": plan,
                "metrics": evaluate_deployment_metrics(
                    scenario, plan, deployment_model, active_power,
                ),
            })
        selected_stage = select_cost_efficient_plan(scenario, evaluated_stages)
        placed_aps = selected_stage["placed_aps"]
        metrics = selected_stage["metrics"]
        alternatives.append({
            "index": alternative_index + 1,
            "mode": "stable" if deterministic else "exploration",
            "seed": seed,
            "selected_stage": selected_stage["stage"],
            "score": deployment_alternative_score(metrics),
            "total_reward": total_reward,
            "placed_aps": placed_aps,
            "metrics": metrics,
            "ppo_ap_count": ppo_ap_count,
            "pruned_ap_count": max(0, ppo_ap_count - min(ppo_ap_count, len(placed_aps))),
            "supplemental_ap_count": sum(
                1 for ap in placed_aps if ap.get("supplemental")
            ),
        })

    best_alternative = select_cost_efficient_plan(scenario, alternatives)
    placed_aps = best_alternative["placed_aps"]
    # Final guarantees applied to the winning plan (after pruning so they stick):
    #  1. No AP may sit inside an excluded area such as a staircase.
    #  2. Every room with users gets at least one interior AP; empty rooms skip.
    placed_aps = strip_aps_in_excluded_rooms(scenario, placed_aps)
    placed_aps = ensure_serviceable_rooms_have_ap(
        scenario, placed_aps, deployment_model, active_power,
    )
    deployment_metrics = evaluate_deployment_metrics(
        scenario, placed_aps, deployment_model, active_power,
    )
    ppo_ap_count = best_alternative["ppo_ap_count"]
    pruned_ap_count = best_alternative["pruned_ap_count"]
    supplemental_ap_count = sum(1 for ap in placed_aps if ap.get("supplemental"))
    ui_access_points = format_deployment_access_points(
        placed_aps,
        deployment_model,
        selected_frequency,
        deployment_power24,
        deployment_power5,
    )
    requirements_met = deployment_meets_targets(scenario, deployment_metrics)
    channel_summary = assign_ap_channels(ui_access_points, selected_frequency)
    unit_cost = float(deployment_model.get("cost") or 0.0)
    economics = {
        "model_id": deployment_model.get("id"),
        "vendor": deployment_model.get("vendor", "Custom"),
        "model": deployment_model.get("model", "Custom AP"),
        "unit_cost": unit_cost,
        "ap_count": len(ui_access_points),
        "estimated_hardware_cost": unit_cost * len(ui_access_points),
    }

    return {
        "success": True,
        "access_points": ui_access_points,
        "metrics": deployment_metrics or info.get("metrics", {}),
        "channel_summary": channel_summary,
        "economics": economics,
        "rf_summary": {
            "frequency_ghz": selected_frequency,
            "tx_power_dbm": active_power,
            "antenna_gain_dbi": float(
                deployment_model.get("gain5" if selected_frequency >= 5 else "gain24") or 0.0
            ),
            "good_signal_dbm": float((scenario.get("targets") or {}).get("good_signal_dbm") or -67.0),
            "planning_clients_per_ap": float(deployment_model.get("planningClients") or 40.0),
        },
        "requirements_met": requirements_met,
        "candidate_count": scenario.get("candidate_positions", []).__len__(),
        "service_room_count": len(active_rooms),
        "no_service_room_count": len(scenario.get("rooms", [])) - len(active_rooms),
        "ppo_ap_count": ppo_ap_count,
        "pruned_ap_count": pruned_ap_count,
        "supplemental_ap_count": supplemental_ap_count,
        "alternatives_evaluated": len(alternatives),
        "selected_alternative": best_alternative["index"],
        "selected_alternative_mode": best_alternative["mode"],
        "selected_optimization_stage": best_alternative["selected_stage"],
        "alternative_summaries": [
            {
                "index": alternative["index"],
                "mode": alternative["mode"],
                "selected_stage": alternative["selected_stage"],
                "score": round(float(alternative["score"]), 4),
                "ap_count": len(alternative["placed_aps"]),
                "sample_coverage_ratio": alternative["metrics"].get("sample_coverage_ratio", 0.0),
                "floor_coverage_ratio": alternative["metrics"].get("floor_coverage_ratio", 0.0),
                "capacity_ratio": alternative["metrics"].get("capacity_ratio", 0.0),
                "cochannel_interference_ratio": alternative["metrics"].get("cochannel_interference_ratio", 0.0),
                "weak_room_count": alternative["metrics"].get("weak_room_count", 0.0),
            }
            for alternative in alternatives
        ],
        "alternative_plans": [
            {
                "index": alternative["index"],
                "mode": alternative["mode"],
                "selected_stage": alternative["selected_stage"],
                "score": round(float(alternative["score"]), 4),
                "metrics": alternative["metrics"],
                "access_points": format_deployment_access_points(
                    alternative["placed_aps"],
                    deployment_model,
                    selected_frequency,
                    deployment_power24,
                    deployment_power5,
                ),
            }
            for alternative in alternatives
        ],
        "model_path": model_path,
        "use_action_mask": use_action_mask,
        "wall_materials": wall_material_summary(scenario),
        "candidate_diagnostics": candidate_diagnostics(scenario) if candidate_diagnostics is not None else {},
    }

@app.post("/api/rl/self-training")
def create_rl_self_training(project_data: ProjectData):
    """Generate wall/demand variants from one project and run baseline batches."""
    if run_self_training_batch is None:
        raise HTTPException(status_code=500, detail="RL self-training runner is not available")

    scenario = build_rl_training_scenario(
        name=project_data.parameters.get("name", "WiFi RL Scenario"),
        elements=[e.model_dump() for e in project_data.elements],
        access_points=[a.model_dump() for a in project_data.access_points],
        parameters=project_data.parameters,
    )
    params = project_data.parameters
    variants = max(1, min(int(params.get("rlTrainingVariants", 20)), 200))
    episodes = max(0, min(int(params.get("rlTrainingEpisodes", 20)), 500))
    seed = int(params.get("rlTrainingSeed", 42))
    candidate_limit = max(1, min(int(params.get("rlGreedyCandidateLimit", 80)), 240))
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in scenario.get("name", "wifi_scenario")).strip("_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(parent_dir, "training_runs", f"{safe_name or 'wifi_scenario'}_{timestamp}")

    summary = run_self_training_batch(
        scenario,
        variants=variants,
        random_episodes=episodes,
        seed=seed,
        greedy_candidate_limit=candidate_limit,
        output_dir=output_dir,
        save_scenarios=bool(params.get("rlSaveTrainingScenarios", False)),
    )
    return {"success": True, "summary": summary}

@app.get("/api/projects/{project_id}/rl-scenario")
def get_project_rl_scenario(project_id: int, db: Session = Depends(get_db)):
    """Build an RL training scenario from a saved project."""
    project = db.query(models.ProjectModel).filter(models.ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dict = project.to_dict()
    return build_rl_training_scenario(
        name=project_dict.get("name") or f"Project {project_id}",
        elements=project_dict.get("elements", []),
        access_points=project_dict.get("access_points", []),
        parameters=project_dict.get("parameters", {}),
    )

@app.post("/api/projects")
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = models.ProjectModel(name=project.name)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project.to_dict()

@app.get("/api/projects")
def get_projects(db: Session = Depends(get_db)):
    projects = db.query(models.ProjectModel).order_by(models.ProjectModel.created_at.desc()).all()
    return [p.to_dict() for p in projects]

@app.get("/api/projects/{project_id}")
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(models.ProjectModel).filter(models.ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.to_dict()

@app.put("/api/projects/{project_id}")
def update_project(project_id: int, project_data: ProjectData, db: Session = Depends(get_db)):
    db_project = db.query(models.ProjectModel).filter(models.ProjectModel.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Use model_dump instead of dict for Pydantic V2
    db_project.elements_json = json.dumps([e.model_dump() for e in project_data.elements])
    db_project.access_points_json = json.dumps([a.model_dump() for a in project_data.access_points])
    db_project.parameters_json = json.dumps(project_data.parameters)
    
    db.commit()
    return {"success": True}

@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(models.ProjectModel).filter(models.ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    return {"success": True}

def create_materials_grid(width: float, length: float, elements: List[FloorPlanElement], resolution: float = 0.2):
    """Create a simplified materials grid from floor plan elements"""
    grid_width = int(width / resolution)
    grid_length = int(length / resolution)
    
    # Initialize with air
    materials_grid = [['air' for _ in range(grid_width)] for _ in range(grid_length)]
    
    # Add walls and obstacles (simplified)
    for element in elements:
        if element.type in ['wall', 'obstacle']:
            for point in element.points:
                grid_x = int(point[0] / resolution)
                grid_y = int(point[1] / resolution)
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_length:
                    materials_grid[grid_y][grid_x] = element.material
                    
    return materials_grid

@app.post("/api/auto-place")
def auto_place_aps(req: AutoPlaceRequest):
    try:
        # Estimate number of APs needed
        num_aps, _ = estimate_initial_ap_count(req.width, req.length, req.height)
        
        # Create materials grid
        materials_grid = create_materials_grid(req.width, req.length, req.elements, req.resolution)
        
        # Place APs using the intelligent grid algorithm
        ap_locations = _place_aps_intelligent_grid(
            num_aps=num_aps,
            building_width=req.width,
            building_length=req.length,
            building_height=req.height,
            materials_grid=materials_grid,
            min_ap_sep=7.0,
            min_wall_gap=1.0
        )
        
        # Format the output
        result = []
        for name, (x, y, z) in ap_locations.items():
            result.append({
                "name": name,
                "x": x,
                "y": y,
                "z": z
            })
            
        return {"success": True, "access_points": result}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/heatmap")
def generate_heatmap(project: ProjectData):
    try:
        if not project.access_points:
            return {"success": False, "error": "No access points available."}
            
        # Get building bounds
        # Simple bounds from elements
        min_x, min_y = 0.0, 0.0
        max_x, max_y = 10.0, 10.0 # Default if no elements
        
        if project.elements:
            all_x = []
            all_y = []
            for elem in project.elements:
                for pt in elem.points:
                    all_x.append(pt[0])
                    all_y.append(pt[1])
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
        # Add a small padding
        width = max_x - min_x + 2
        height = max_y - min_y + 2
        min_x -= 1
        min_y -= 1
        max_x += 1
        max_y += 1
        
        resolution = 100
        x_points = np.linspace(min_x, max_x, resolution)
        y_points = np.linspace(min_y, max_y, resolution)
        
        signal_strength = np.zeros((len(y_points), len(x_points)))
        
        # Helper: line intersects wall
        def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10: return False
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            return 0 <= t <= 1 and 0 <= u <= 1
            
        # Calculate for each point
        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                signals = []
                for ap in project.access_points:
                    distance = np.sqrt((x - ap.x)**2 + (y - ap.y)**2 + (1.5 - ap.z)**2)
                    if distance < 0.1: distance = 0.1
                    
                    # FSPL
                    if distance < 1.0:
                        fspl = 20 * np.log10(ap.frequency * 1000) + 20 * np.log10(distance) + 32.44
                    else:
                        fspl = 40 * np.log10(distance) + 20 * np.log10(ap.frequency * 1000) - 10
                        
                    # Wall loss
                    wall_loss = 0
                    for elem in project.elements:
                        if elem.type == 'wall' and len(elem.points) >= 2:
                            for k in range(len(elem.points) - 1):
                                wx1, wy1 = elem.points[k]
                                wx2, wy2 = elem.points[k + 1]
                                if lines_intersect(x, y, ap.x, ap.y, wx1, wy1, wx2, wy2):
                                    if elem.material == 'concrete': wall_loss += 15
                                    elif elem.material == 'drywall': wall_loss += 8
                                    elif elem.material == 'glass': wall_loss += 3
                                    else: wall_loss += 10
                    wall_loss = min(wall_loss, 40)
                    
                    signal = ap.power - fspl - wall_loss
                    signals.append(signal)
                    
                if signals:
                    signal_strength[i, j] = max(signals)
                else:
                    signal_strength[i, j] = -100
                    
        # Generate image
        plt.figure(figsize=(10, 8), dpi=100)
        ax = plt.gca()
        # Create heatmap
        im = ax.imshow(signal_strength, extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]], 
                      cmap='RdYlGn', origin='lower', alpha=0.8, vmin=-100, vmax=-30)
                      
        plt.axis('off')
        # Remove margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            "success": True, 
            "image": f"data:image/png;base64,{img_str}",
            "bounds": {
                "min_x": min_x, "min_y": min_y,
                "max_x": max_x, "max_y": max_y
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-walls")
def detect_walls(req: DetectWallsRequest):
    """Detect walls from a floor plan image using trained UNet deep learning model."""
    try:
        from wall_detector.inference import get_detector
        
        # Get or create the wall detector (singleton, loads model once)
        detector = get_detector()
        
        # Run AI inference
        # Higher UI sensitivity should accept weaker AI wall probabilities.
        sensitivity = max(0, min(100, req.sensitivity)) / 100.0
        threshold = max(0.25, min(0.75, 0.8 - sensitivity * 0.55))
        result = detector.detect_from_base64(
            req.image,
            pixels_per_meter=req.pixels_per_meter,
            threshold=threshold,
            min_wall_length_m=req.min_wall_length_m
        )
        
        # A monochrome plan cannot reliably reveal every construction material.
        # Keep high-confidence thickness-based classifications and use the user's
        # selected material as the fallback for visually ambiguous segments.
        if result["success"]:
            for wall in result["walls"]:
                if not wall.get("wall_type") or wall.get("material_confidence", 0) < 0.6:
                    wall["wall_type"] = req.wall_type
                    wall["material_source"] = "selected_fallback"
                else:
                    wall["material_source"] = "visual_thickness"
        
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-rooms")
def detect_rooms(req: DetectRoomsRequest):
    """Detect full enclosed room polygons from filled/thick-wall floor plans."""
    try:
        image_data = req.image.split(",", 1)[-1]
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(image_data), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return {"success": False, "rooms": [], "message": "Failed to decode image"}

        ppm = max(5.0, float(req.pixels_per_meter))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Thin furniture outlines disappear from the distance-transform core,
        # while genuinely filled wall bands remain.
        dark = (gray < 125).astype(np.uint8)
        distance = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
        wall_core = (distance >= 1.8).astype(np.uint8) * 255
        wall_core = cv2.dilate(
            wall_core,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        # Temporarily bridge door/window gaps so rooms become closed regions.
        gap_px = max(5, min(120, int(round(ppm * 2.3))))
        horizontal = cv2.morphologyEx(
            wall_core, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (gap_px, 1)),
        )
        vertical = cv2.morphologyEx(
            wall_core, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_px)),
        )
        barriers = cv2.bitwise_or(horizontal, vertical)
        barriers = cv2.dilate(
            barriers,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        # Limit flood fill to the actual colored/drawn building footprint.
        # This keeps rooms with exterior windows or doors from leaking into the
        # white canvas around the floor plan.
        footprint_ink = (gray < 252).astype(np.uint8) * 255
        footprint_ink = cv2.morphologyEx(
            footprint_ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        )
        footprint_count, footprint_labels, footprint_stats, _ = cv2.connectedComponentsWithStats(
            (footprint_ink > 0).astype(np.uint8), connectivity=8
        )
        footprint = np.zeros_like(gray, dtype=np.uint8)
        if footprint_count > 1:
            largest_id = 1 + int(np.argmax(footprint_stats[1:, cv2.CC_STAT_AREA]))
            largest_mask = (footprint_labels == largest_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cv2.drawContours(
                    footprint, [max(contours, key=cv2.contourArea)], -1, 255, -1
                )

        free = ((barriers == 0) & (footprint > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        height, width = gray.shape[:2]
        rooms = []

        for label_id in range(1, count):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            component_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area_px = int(stats[label_id, cv2.CC_STAT_AREA])
            area_m2 = area_px / (ppm * ppm)
            if area_m2 < req.min_room_area_m2 or area_m2 > req.max_room_area_m2:
                continue

            component_mask = (labels == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = max(1.5, ppm * 0.06)
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) < 4 or len(polygon) > 36:
                continue

            moments = cv2.moments(contour)
            centroid_x = moments["m10"] / moments["m00"] if moments["m00"] else x + component_width / 2
            centroid_y = moments["m01"] / moments["m00"] if moments["m00"] else y + component_height / 2
            rooms.append({
                "polygon": [
                    {"x": float(px / ppm), "y": float(py / ppm)}
                    for px, py in polygon
                ],
                "centroid": {
                    "x": float(centroid_x / ppm),
                    "y": float(centroid_y / ppm),
                },
                "areaM2": float(round(area_m2, 2)),
            })

        rooms.sort(key=lambda room: (room["centroid"]["y"], room["centroid"]["x"]))
        return {
            "success": True,
            "rooms": rooms,
            "count": len(rooms),
            "message": f"Detected {len(rooms)} enclosed room areas",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _merge_similar_lines(segments, angle_threshold=5, distance_threshold=15):
    """Merge line segments that are nearly collinear and close together."""
    if not segments:
        return []

    def line_angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def line_midpoint(x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def perpendicular_distance(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx*dx + dy*dy
        if length_sq == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    merged = list(segments)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            best_j = -1
            best_score = float('inf')

            x1, y1, x2, y2 = merged[i]
            angle_i = line_angle(x1, y1, x2, y2)

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                x3, y3, x4, y4 = merged[j]
                angle_j = line_angle(x3, y3, x4, y4)

                # Check angle similarity
                angle_diff = abs(angle_i - angle_j)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                if angle_diff > angle_threshold:
                    continue

                # Check perpendicular distance between midpoints
                mx, my = line_midpoint(x3, y3, x4, y4)
                perp_dist = perpendicular_distance(mx, my, x1, y1, x2, y2)
                if perp_dist > distance_threshold:
                    continue

                # Check if endpoints are close enough
                min_endpoint_dist = min(
                    np.sqrt((x1-x3)**2 + (y1-y3)**2),
                    np.sqrt((x1-x4)**2 + (y1-y4)**2),
                    np.sqrt((x2-x3)**2 + (y2-y3)**2),
                    np.sqrt((x2-x4)**2 + (y2-y4)**2)
                )

                score = perp_dist + min_endpoint_dist * 0.5
                if score < best_score and min_endpoint_dist < distance_threshold * 5:
                    best_score = score
                    best_j = j

            if best_j >= 0:
                # Merge: create a line from the two most distant endpoints
                x3, y3, x4, y4 = merged[best_j]
                all_pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                max_dist = 0
                best_pair = (0, 1)
                for a in range(4):
                    for b in range(a + 1, 4):
                        d = np.sqrt((all_pts[a][0]-all_pts[b][0])**2 + (all_pts[a][1]-all_pts[b][1])**2)
                        if d > max_dist:
                            max_dist = d
                            best_pair = (a, b)

                new_merged.append((
                    all_pts[best_pair[0]][0], all_pts[best_pair[0]][1],
                    all_pts[best_pair[1]][0], all_pts[best_pair[1]][1]
                ))
                used[i] = True
                used[best_j] = True
                changed = True
            else:
                new_merged.append(merged[i])
                used[i] = True

        # Add any remaining unused
        for j in range(len(merged)):
            if not used[j]:
                new_merged.append(merged[j])

        merged = new_merged

    return merged
