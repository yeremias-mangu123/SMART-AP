"""Domain randomization for one-floor-plan WiFi AP placement training.

When the floor plan stays fixed, we can still create useful training variation
by changing wall attenuation, client demand, and coverage targets. This helps
the placement agent learn robust decisions instead of memorizing one exact
scenario.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from .candidate_generator import generate_ap_candidates
from .rf_calibration import calibrated_wall_profiles, propagation_profile


WALL_PROFILES: List[Dict[str, Any]] = calibrated_wall_profiles()


def jitter_value(value: float, rng: random.Random, jitter_ratio: float, minimum: float = 0.0) -> float:
    factor = rng.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio)
    return round(max(minimum, float(value) * factor), 3)


def randomize_wall(wall: Dict[str, Any], rng: random.Random, attenuation_jitter: float,
                   profiles: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return a wall with randomized material attenuation."""
    randomized = copy.deepcopy(wall)
    profile = rng.choice(profiles or WALL_PROFILES)

    attenuation = {}
    for band, value in profile["attenuation_db"].items():
        profile_jitter = float(profile.get("jitter_ratio", attenuation_jitter))
        attenuation[band] = jitter_value(float(value), rng, max(attenuation_jitter, profile_jitter))

    randomized["material_id"] = profile["material_id"]
    randomized["material_name"] = profile["material_name"]
    randomized["attenuation_db"] = attenuation
    randomized["thickness_mm"] = jitter_value(float(profile.get("thickness_mm", 100.0)), rng, 0.15, minimum=1.0)
    randomized["randomized_profile"] = profile["material_id"]
    return randomized


def randomize_room(room: Dict[str, Any], rng: random.Random,
                   client_jitter: float, target_jitter_db: float) -> Dict[str, Any]:
    """Return a room with varied demand and coverage target."""
    randomized = copy.deepcopy(room)
    if (
        randomized.get("excluded")
        or randomized.get("service_excluded")
        or float(randomized.get("clients", 1)) <= 0
        or str(randomized.get("type", "")).lower() == "stairs"
    ):
        randomized["clients"] = 0
        randomized["priority"] = 0.0
        randomized["service_excluded"] = True
        return randomized
    base_clients = float(randomized.get("clients", 1))
    randomized["clients"] = max(1, int(round(jitter_value(base_clients, rng, client_jitter, minimum=1.0))))

    base_target = float(randomized.get("coverage_target_dbm", -67))
    randomized["coverage_target_dbm"] = round(base_target + rng.uniform(-target_jitter_db, target_jitter_db), 1)

    base_priority = float(randomized.get("priority", 1.0))
    randomized["priority"] = round(max(0.5, min(3.0, base_priority * rng.uniform(0.85, 1.25))), 3)
    return randomized


def randomize_scenario(
    scenario: Dict[str, Any],
    variant_index: int,
    seed: int,
    attenuation_jitter: float = 0.18,
    client_jitter: float = 0.35,
    target_jitter_db: float = 2.5,
    min_ap_separation_choices: Optional[List[float]] = None,
    wall_strategy: str = "mixed",
    uniform_material_id: Optional[str] = None,
    frequency_choices: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Create one randomized scenario variant from a fixed layout."""
    rng = random.Random(seed + variant_index * 9973)
    variant = copy.deepcopy(scenario)
    variant["name"] = f"{scenario.get('name', 'WiFi Scenario')} / wall variant {variant_index + 1}"
    variant["variant"] = {
        "index": variant_index,
        "seed": seed + variant_index * 9973,
        "type": "wall_and_demand_randomization",
    }

    if uniform_material_id:
        palette = [
            profile for profile in WALL_PROFILES
            if profile.get("material_id") == uniform_material_id
        ]
        if not palette:
            raise ValueError(f"Unknown wall material: {uniform_material_id}")
    elif wall_strategy == "mixed":
        # Real buildings usually use a small material palette repeatedly,
        # rather than assigning an unrelated material to every wall segment.
        palette_size = min(len(WALL_PROFILES), rng.randint(2, 5))
        palette = rng.sample(WALL_PROFILES, palette_size)
        primary = rng.choice(palette)
        palette = [primary, primary, primary, *palette]
    else:
        palette = WALL_PROFILES
    variant["walls"] = [
        randomize_wall(wall, rng, attenuation_jitter, palette)
        for wall in variant.get("walls", [])
    ]
    variant["rooms"] = [
        randomize_room(room, rng, client_jitter, target_jitter_db)
        for room in variant.get("rooms", [])
    ]
    base_propagation = propagation_profile(rng.choice(["office", "dense_office"]))
    for parameters in base_propagation.values():
        parameters["distance_loss_coefficient"] = jitter_value(
            parameters["distance_loss_coefficient"], rng, 0.08, minimum=20.0,
        )
        parameters["shadow_margin_db"] = jitter_value(
            parameters["shadow_margin_db"], rng, 0.18, minimum=2.0,
        )
    variant["propagation_model"] = {
        "profile": "randomized_calibrated",
        "bands": base_propagation,
    }
    if frequency_choices:
        variant.setdefault("floor_plan", {})["selected_frequency_ghz"] = rng.choice(frequency_choices)

    constraints = variant.setdefault("constraints", {})
    choices = min_ap_separation_choices or [3.0, 3.5, 4.0]
    constraints["min_ap_separation_m"] = rng.choice(choices)
    constraints["room_sample_step_m"] = float(constraints.get("room_sample_step_m") or 1.2)
    constraints["max_samples_per_room"] = int(constraints.get("max_samples_per_room") or 20)

    # Candidate score depends on room demand, so regenerate candidates after room randomization.
    variant["candidate_positions"] = generate_ap_candidates(variant)
    variant["candidate_count"] = len(variant["candidate_positions"])
    return variant


def wall_profile_counts(scenario: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for wall in scenario.get("walls", []):
        profile = str(wall.get("randomized_profile") or wall.get("material_id") or "unknown")
        counts[profile] = counts.get(profile, 0) + 1
    return counts
