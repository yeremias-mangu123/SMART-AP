"""Main script for training a PPO model on the AP placement environment."""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from math import hypot
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
except ImportError:  # pragma: no cover - exercised only when optional dep is absent
    MaskablePPO = None
    MaskableEvalCallback = None

from src.rl.gym_wrapper import GymWiFiAPPlacementEnv, MultiScenarioGymWiFiAPPlacementEnv
from src.rl.candidate_generator import generate_ap_candidates
from src.rl.room_profiles import (
    estimated_room_clients,
    normalize_room_type,
    room_effective_priority,
    room_requires_service,
    room_type_profile,
)


def apply_cli_overrides(scenario: dict, args: argparse.Namespace) -> None:
    """Apply training-time overrides before candidate generation."""
    constraints = scenario.setdefault("constraints", {})
    targets = scenario.setdefault("targets", {})

    if args.max_ap_count is not None:
        constraints["max_ap_count"] = args.max_ap_count
    if args.min_ap_count is not None:
        constraints["min_ap_count"] = args.min_ap_count
    if args.max_episode_steps is not None:
        constraints["max_episode_steps"] = args.max_episode_steps
    if args.max_candidates is not None:
        constraints["max_candidates"] = args.max_candidates
    if args.max_candidates_per_room is not None:
        constraints["max_candidates_per_room"] = args.max_candidates_per_room
    if args.candidate_grid_step is not None:
        constraints["candidate_grid_step_m"] = args.candidate_grid_step
    if args.min_ap_separation is not None:
        constraints["min_ap_separation_m"] = args.min_ap_separation
    if args.coverage_target_dbm is not None:
        targets["default_coverage_dbm"] = args.coverage_target_dbm
        targets["good_signal_dbm"] = args.coverage_target_dbm
    if args.acceptable_signal_dbm is not None:
        targets["acceptable_signal_dbm"] = args.acceptable_signal_dbm


def align_training_scenario(scenario: dict, reference: dict) -> dict:
    """Align variable-size layouts to one fixed PPO action space."""
    reference_constraints = reference.get("constraints") or {}
    constraints = scenario.setdefault("constraints", {})
    constraints["observation_version"] = int(reference_constraints.get("observation_version") or 3)
    constraints["max_ap_count"] = int(reference_constraints.get("max_ap_count") or 16)
    constraints["max_episode_steps"] = int(reference_constraints.get("max_episode_steps") or 32)
    expected_candidates = len(reference.get("candidate_positions") or [])
    candidates = list(scenario.get("candidate_positions") or [])
    if len(candidates) < expected_candidates:
        fallback = candidates[-1] if candidates else {"x": 0.0, "y": 0.0, "z": 2.7}
        candidates.extend({
            **fallback,
            "id": f"training_padding_{index + 1}",
            "_action_padding": True,
            "score": 0.0,
        } for index in range(expected_candidates - len(candidates)))
    scenario["candidate_positions"] = candidates[:expected_candidates]

    expected_models = len((reference.get("ap_catalog") or {}).get("models") or [])
    models = list((scenario.setdefault("ap_catalog", {})).get("models") or [])
    reference_models = list((reference.get("ap_catalog") or {}).get("models") or [])
    if len(models) < expected_models:
        models.extend(reference_models[len(models):expected_models])
    scenario["ap_catalog"]["models"] = models[:expected_models]
    return scenario


def calculate_polygon_area(points: list[dict]) -> float:
    """Calculate polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += float(point.get("x", 0)) * float(next_point.get("y", 0))
        area -= float(next_point.get("x", 0)) * float(point.get("y", 0))
    return abs(area) / 2.0


def calculate_polygon_centroid(points: list[dict]) -> dict:
    """Calculate polygon centroid with a mean-point fallback."""
    if not points:
        return {"x": 0.0, "y": 0.0}

    area_factor = 0.0
    cx = 0.0
    cy = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        x1 = float(point.get("x", 0))
        y1 = float(point.get("y", 0))
        x2 = float(next_point.get("x", 0))
        y2 = float(next_point.get("y", 0))
        cross = x1 * y2 - x2 * y1
        area_factor += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    if abs(area_factor) < 1e-9:
        return {
            "x": sum(float(point.get("x", 0)) for point in points) / len(points),
            "y": sum(float(point.get("y", 0)) for point in points) / len(points),
        }

    return {"x": cx / (3 * area_factor), "y": cy / (3 * area_factor)}


def normalize_ap_model(model: dict, index: int) -> dict:
    """Normalize UI AP model fields into the RL catalog shape."""
    normalized = dict(model)
    normalized.setdefault("id", model.get("model") or f"ap_model_{index + 1}")
    normalized.setdefault("bands", ["2.4", "5"])
    if "gain24" not in normalized and "gain2_4" in normalized:
        normalized["gain24"] = normalized["gain2_4"]
    normalized.setdefault("gain24", 3.0)
    normalized.setdefault("gain5", normalized.get("gain24", 3.0))
    normalized.setdefault("gain6", normalized.get("gain5", normalized.get("gain24", 3.0)))
    normalized.setdefault("maxPower", 26.0)
    normalized.setdefault("recommendedClients", normalized.get("clients", 80))
    normalized.setdefault("planningClients", min(normalized["recommendedClients"], 40))
    normalized.setdefault("cost", 100.0)
    return normalized


def normalize_project_rooms(project_rooms: list[dict]) -> list[dict]:
    """Convert web project rooms from parameters.rooms to RL rooms."""
    rooms = []
    for index, room in enumerate(project_rooms or []):
        polygon = [
            {"x": float(point.get("x", 0)), "y": float(point.get("y", 0))}
            for point in room.get("polygon", [])
            if isinstance(point, dict)
        ]
        if len(polygon) < 3:
            continue

        area_m2 = float(room.get("areaM2") or room.get("area_m2") or calculate_polygon_area(polygon))
        centroid = room.get("centroid") or calculate_polygon_centroid(polygon)
        room_type = normalize_room_type(room.get("roomType") or room.get("type"))
        is_stairs = room_type == "stairs"
        profile = room_type_profile(room_type)
        coverage_target = room.get("coverageTarget", room.get("coverage_target_dbm"))
        if coverage_target is None:
            coverage_target = profile["default_coverage_dbm"]
        priority = room.get("priority")
        if priority is None:
            priority = profile["default_priority"]

        normalized_room = {
            "id": room.get("id") or f"room_{index + 1}",
            "name": room.get("name") or f"Room {index + 1}",
            "type": room_type,
            "polygon": polygon,
            "area_m2": area_m2,
            "centroid": {
                "x": float(centroid.get("x", 0)),
                "y": float(centroid.get("y", 0)),
            },
            "clients": 0 if is_stairs else int(
                room.get("clients")
                if room.get("clients") is not None
                else estimated_room_clients(room_type, area_m2)
            ),
            "priority": float(priority),
            "priority_multiplier": float(profile["priority_multiplier"]),
            "effective_priority": 0.0,
            "coverage_target_dbm": float(coverage_target),
            "capacity_headroom_ratio": float(profile["capacity_headroom_ratio"]),
            "candidate_score_multiplier": float(profile["candidate_score_multiplier"]),
            "excluded": is_stairs or bool(room.get("excluded", False)),
        }
        normalized_room["service_excluded"] = bool(
            normalized_room["excluded"]
            or room.get("serviceExcluded", room.get("service_excluded", False))
            or normalized_room["clients"] <= 0
        )
        normalized_room["effective_priority"] = room_effective_priority(normalized_room)
        rooms.append(normalized_room)
    return rooms


def update_bounds_from_geometry(scenario: dict) -> None:
    """Ensure floor_plan.bounds covers walls and room polygons."""
    xs = []
    ys = []
    for wall in scenario.get("walls", []):
        for point in wall.get("points", []):
            if len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    for room in scenario.get("rooms", []):
        for point in room.get("polygon", []):
            xs.append(float(point.get("x", 0)))
            ys.append(float(point.get("y", 0)))

    if not xs or not ys:
        return

    scenario.setdefault("floor_plan", {})
    scenario["floor_plan"]["bounds"] = {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
        "width_m": max(xs) - min(xs),
        "height_m": max(ys) - min(ys),
    }


def adapt_scenario_for_rl(scenario: dict) -> dict:
    """Ensure the scenario has the exact fields required by WiFiAPPlacementEnv."""
    parameters = scenario.get("parameters") or {}

    # Add ap_catalog if missing
    if "ap_catalog" not in scenario:
        if parameters.get("apModels"):
            scenario["ap_catalog"] = {
                "models": [
                    normalize_ap_model(model, index)
                    for index, model in enumerate(parameters.get("apModels", []))
                ]
            }
        else:
            ap_catalog_path = os.path.join(os.path.dirname(__file__), "ap_catalog.json")
            if os.path.exists(ap_catalog_path):
                with open(ap_catalog_path, "r") as f:
                    scenario["ap_catalog"] = json.load(f)
            else:
                scenario["ap_catalog"] = {
                    "models": [{"id": "generic", "maxPower": 20.0, "gain24": 3.0, "gain5": 4.0}]
                }
    else:
        scenario["ap_catalog"]["models"] = [
            normalize_ap_model(model, index)
            for index, model in enumerate((scenario.get("ap_catalog") or {}).get("models", []))
        ]

    # Convert web project 'parameters.rooms' to RL rooms if needed.
    if "rooms" not in scenario and parameters.get("rooms"):
        scenario["rooms"] = normalize_project_rooms(parameters.get("rooms", []))

    # Convert planner 'regions' to 'rooms' if needed
    if "rooms" not in scenario and "regions" in scenario:
        scenario["rooms"] = []
        for i, reg in enumerate(scenario.get("regions", [])):
            if reg.get("type") == "office" and "coords" in reg:
                coords = reg["coords"]
                if len(coords) >= 4:
                    min_x, min_y, max_x, max_y = coords[0], coords[1], coords[2], coords[3]
                    scenario["rooms"].append({
                        "id": f"room_{i}",
                        "name": reg.get("name", f"Room {i}"),
                        "priority": 1.0,
                        "clients": 10.0,
                        "polygon": [
                            {"x": min_x, "y": min_y},
                            {"x": max_x, "y": min_y},
                            {"x": max_x, "y": max_y},
                            {"x": min_x, "y": max_y}
                        ],
                        "centroid": {"x": (min_x + max_x) / 2, "y": (min_y + max_y) / 2}
                    })

    # Convert DB 'elements' to 'walls' if needed
    if "walls" not in scenario and "elements" in scenario:
        scenario["walls"] = []
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for i, el in enumerate(scenario.get("elements", [])):
            if el.get("type") == "wall" and "points" in el:
                pts = el["points"]
                if len(pts) >= 2:
                    x1, y1 = float(pts[0][0]), float(pts[0][1])
                    x2, y2 = float(pts[1][0]), float(pts[1][1])
                    scenario["walls"].append({
                        "id": f"wall_{i}",
                        "points": [[x1, y1], [x2, y2]],
                        "material_id": el.get("material", "thin"),
                        "attenuation_db": {"2.4": 3.0, "5": 5.0, "6": 6.0},
                        "length_m": hypot(x2 - x1, y2 - y1),
                    })
                    for p in ([x1, y1], [x2, y2]):
                        min_x, min_y = min(min_x, p[0]), min(min_y, p[1])
                        max_x, max_y = max(max_x, p[0]), max(max_y, p[1])
        
        # Ensure floor_plan bounds exist for candidate generation
        if "floor_plan" not in scenario:
            scenario["floor_plan"] = {}
        if "bounds" not in scenario["floor_plan"] and min_x != float('inf'):
            scenario["floor_plan"]["bounds"] = {
                "min_x": min_x, "min_y": min_y,
                "max_x": max_x, "max_y": max_y
            }

    if "floor_plan" not in scenario:
        scenario["floor_plan"] = {}
    if parameters.get("floorHeight") and "floor_height_m" not in scenario["floor_plan"]:
        scenario["floor_plan"]["floor_height_m"] = float(parameters.get("floorHeight"))
    if parameters.get("selectedFrequency") and "selected_frequency_ghz" not in scenario["floor_plan"]:
        scenario["floor_plan"]["selected_frequency_ghz"] = float(parameters.get("selectedFrequency"))
    if "bounds" not in scenario.get("floor_plan", {}) and (
        scenario.get("walls") or scenario.get("rooms")
    ):
        update_bounds_from_geometry(scenario)

    # Ensure constraints exist
    if "constraints" not in scenario:
        scenario["constraints"] = {
            "observation_version": 3,
            "max_ap_count": 6,
            "min_ap_separation_m": 4.0,
            "room_sample_step_m": 1.0,
            "max_samples_per_room": 20,
            "candidate_grid_step_m": 1.0,
            "min_wall_distance_m": 0.3,
            "max_candidates": 100
        }
    scenario["constraints"].setdefault("observation_version", 3)
    scenario["constraints"].setdefault("floor_sample_step_m", 1.2)
    scenario["constraints"].setdefault("max_floor_samples", 320)
    if "max_ap_count" not in scenario["constraints"]:
        active_rooms = len([room for room in scenario.get("rooms", []) if room_requires_service(room)])
        scenario["constraints"]["max_ap_count"] = max(1, min(12, active_rooms or 6))

    targets = scenario.setdefault("targets", {})
    targets.setdefault("min_sinr_db", 15.0)
    targets.setdefault("interference_sinr_db", 20.0)
    targets.setdefault("target_sample_coverage_ratio", 0.98)
    targets.setdefault("target_acceptable_coverage_ratio", 0.99)
    targets.setdefault("min_coverage_ratio", 0.95)
    targets.setdefault("min_capacity_ratio", 0.95)
    targets.setdefault("max_weak_room_count", 0)
    targets.setdefault("target_sinr_coverage_ratio", 0.90)
    targets.setdefault("target_floor_coverage_ratio", 0.97)
    targets.setdefault("target_floor_sinr_coverage_ratio", 0.90)

    reward_weights = scenario.setdefault("reward_weights", {})
    reward_defaults = {
        "step": -0.02,
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
    }
    for key, value in reward_defaults.items():
        reward_weights.setdefault(key, value)

    # Generate candidate positions if missing
    if not scenario.get("candidate_positions"):
        scenario["candidate_positions"] = generate_ap_candidates(scenario)
        print(f"Generated {len(scenario['candidate_positions'])} AP candidate positions.")

    return scenario


def candidate_diagnostics(scenario: dict) -> dict:
    """Summarize generated AP candidates for quick sanity checks."""
    candidates = scenario.get("candidate_positions") or []
    rooms = [room for room in scenario.get("rooms", []) if room_requires_service(room)]
    room_ids = {str(room.get("id")) for room in rooms}
    candidate_room_ids = [str(candidate.get("room_id")) for candidate in candidates if candidate.get("room_id")]
    represented_rooms = set(candidate_room_ids) & room_ids
    wall_distances = [float(candidate.get("min_wall_distance_m", 0.0)) for candidate in candidates]
    room_type_counts = Counter(str(room.get("type", "unknown")) for room in rooms)

    return {
        "candidate_count": len(candidates),
        "room_count": len(rooms),
        "rooms_with_candidates": len(represented_rooms),
        "rooms_without_candidates": max(0, len(room_ids - represented_rooms)),
        "source_counts": dict(Counter(str(candidate.get("source", "unknown")) for candidate in candidates)),
        "room_type_counts": dict(room_type_counts),
        "min_wall_distance_m": min(wall_distances) if wall_distances else None,
        "avg_wall_distance_m": sum(wall_distances) / len(wall_distances) if wall_distances else None,
    }


def print_candidate_diagnostics(diagnostics: dict) -> None:
    """Print candidate diagnostics in a compact, presentation-friendly shape."""
    print("Candidate diagnostics:")
    print(f"  candidates            : {diagnostics['candidate_count']}")
    print(f"  rooms                 : {diagnostics['room_count']}")
    print(f"  rooms with candidates : {diagnostics['rooms_with_candidates']}")
    print(f"  rooms without cand.   : {diagnostics['rooms_without_candidates']}")
    print(f"  room types            : {diagnostics.get('room_type_counts', {})}")
    print(f"  source counts         : {diagnostics['source_counts']}")
    if diagnostics["min_wall_distance_m"] is not None:
        print(f"  min wall distance     : {diagnostics['min_wall_distance_m']:.2f} m")
        print(f"  avg wall distance     : {diagnostics['avg_wall_distance_m']:.2f} m")


def rollout_policy(model, scenario: dict, use_action_mask: bool, max_steps: int = 200) -> dict:
    """Run one deterministic policy rollout and return domain metrics."""
    eval_env = GymWiFiAPPlacementEnv(scenario)
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    last_info = info

    while not done and steps < max_steps:
        if use_action_mask:
            action, _ = model.predict(obs, deterministic=True, action_masks=eval_env.action_masks())
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, last_info = eval_env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    return {
        "steps": steps,
        "total_reward": round(total_reward, 6),
        "metrics": last_info.get("metrics", {}),
        "placed_aps": last_info.get("placed_aps", []),
    }


def print_domain_metrics(label: str, result: dict) -> None:
    """Print the metrics that matter for SMART AP, not only PPO reward."""
    metrics = result.get("metrics", {})
    print(f"{label}:")
    print(f"  reward                : {result.get('total_reward', 0.0):.3f}")
    print(f"  steps                 : {result.get('steps', 0)}")
    print(f"  coverage >= target    : {metrics.get('coverage_percent', 0.0):.1f}%")
    print(f"  coverage >= acceptable: {metrics.get('acceptable_coverage_percent', 0.0):.1f}%")
    print(f"  blank spot            : {metrics.get('blankspot_percent', 0.0):.1f}%")
    print(f"  average signal        : {metrics.get('average_best_rssi_dbm', -100.0):.1f} dBm")
    print(f"  minimum signal        : {metrics.get('minimum_best_rssi_dbm', -100.0):.1f} dBm")
    print(f"  AP used               : {metrics.get('ap_count', 0.0):.0f}")
    print(f"  overlap               : {metrics.get('overlap_ratio', 0.0) * 100.0:.1f}%")
    print(f"  SINR coverage         : {metrics.get('sinr_coverage_ratio', 0.0) * 100.0:.1f}%")
    print(f"  co-channel interference: {metrics.get('cochannel_interference_ratio', 0.0) * 100.0:.1f}%")
    print(f"  weak rooms            : {metrics.get('weak_room_count', 0.0):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO on a WiFi AP Placement scenario")
    parser.add_argument("scenario", help="Path to the JSON scenario (e.g. finalmap.json)")
    parser.add_argument("--scenario-dir", help="Optional directory of compatible JSON scenarios sampled during training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=2000, help="Evaluation frequency in steps")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--max-ap-count", type=int, help="Override max AP count for each episode")
    parser.add_argument("--min-ap-count", type=int, help="Override minimum AP count before the policy may stop")
    parser.add_argument("--max-episode-steps", type=int, help="Override maximum environment steps")
    parser.add_argument("--max-candidates", type=int, help="Override maximum generated AP candidates")
    parser.add_argument("--max-candidates-per-room", type=int, help="Override maximum generated candidates per room")
    parser.add_argument("--candidate-grid-step", type=float, help="Override candidate grid spacing in meters")
    parser.add_argument("--min-ap-separation", type=float, help="Override minimum AP separation in meters")
    parser.add_argument("--coverage-target-dbm", type=float, help="Override good coverage threshold, default -67 dBm")
    parser.add_argument("--acceptable-signal-dbm", type=float, help="Override acceptable signal threshold, default -70 dBm")
    parser.add_argument("--output-root", default="training_runs", help="Directory used for this training run")
    parser.add_argument("--device", default="auto", help="Stable-Baselines device, for example cpu or cuda")
    parser.add_argument(
        "--no-action-mask",
        action="store_true",
        help="Use vanilla PPO instead of MaskablePPO even when sb3-contrib is installed",
    )
    args = parser.parse_args()

    print(f"Loading scenario from {args.scenario}...")
    with open(args.scenario, 'r', encoding='utf-8') as f:
        scenario = json.load(f)
    apply_cli_overrides(scenario, args)

    # Adapt and fix missing fields
    scenario = adapt_scenario_for_rl(scenario)
    training_scenarios = [scenario]
    if args.scenario_dir:
        for scenario_path in sorted(Path(args.scenario_dir).glob("*.json")):
            with scenario_path.open("r", encoding="utf-8") as source:
                candidate_scenario = json.load(source)
            apply_cli_overrides(candidate_scenario, args)
            candidate_scenario = adapt_scenario_for_rl(candidate_scenario)
            training_scenarios.append(align_training_scenario(candidate_scenario, scenario))

    if len(scenario.get("candidate_positions", [])) == 0:
        raise SystemExit("No candidate positions generated. Check scenario bounds, rooms, and constraints.")
    if len([room for room in scenario.get("rooms", []) if room_requires_service(room)]) == 0:
        raise SystemExit("No RL rooms found. Add rooms/regions with polygons before PPO training.")

    diagnostics = candidate_diagnostics(scenario)
    print_candidate_diagnostics(diagnostics)

    # Initialize environment
    env = (
        MultiScenarioGymWiFiAPPlacementEnv(training_scenarios)
        if len(training_scenarios) > 1
        else GymWiFiAPPlacementEnv(scenario)
    )
    env = Monitor(env)

    # Eval environment
    eval_env = GymWiFiAPPlacementEnv(scenario)
    eval_env = Monitor(eval_env)

    run_name = datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    use_action_mask = MaskablePPO is not None and not args.no_action_mask
    callback_cls = MaskableEvalCallback if use_action_mask else EvalCallback
    eval_callback = callback_cls(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=log_dir,
        name_prefix="checkpoint",
    )

    print(f"Starting PPO training for {args.timesteps} timesteps...")
    print(f"TensorBoard logs will be saved to ./runs/")
    if use_action_mask:
        print("Using MaskablePPO with candidate/model/stop action masks.")
    else:
        print("Using vanilla PPO. Install sb3-contrib to enable action masking.")
    
    model_cls = MaskablePPO if use_action_mask else PPO
    batch_size = min(args.batch_size, args.n_steps)
    model = model_cls(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./runs/",
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=batch_size,
        gamma=0.99,
        device=args.device,
    )
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        tb_log_name=run_name,
    )

    final_result = rollout_policy(model, scenario, use_action_mask)
    print_domain_metrics("Final policy domain metrics", final_result)

    summary_path = os.path.join(log_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenario": args.scenario,
            "training_scenarios": [item.get("name", "WiFi Scenario") for item in training_scenarios],
            "timesteps": args.timesteps,
            "use_action_mask": use_action_mask,
            "candidate_diagnostics": diagnostics,
            "final_policy": final_result,
            "constraints": scenario.get("constraints", {}),
            "targets": scenario.get("targets", {}),
            "reward_weights": scenario.get("reward_weights", {}),
        }, f, indent=2)

    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}.zip")
    print(f"Best evaluation model saved to {log_dir}/best_model.zip")
    print(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
