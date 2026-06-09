"""Traceable RF calibration defaults used by training and inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List


CALIBRATION_PATH = Path(__file__).with_name("rf_calibration.json")


def load_rf_calibration() -> Dict[str, Any]:
    with CALIBRATION_PATH.open("r", encoding="utf-8") as source:
        return json.load(source)


def propagation_profile(name: str = "office") -> Dict[str, Dict[str, float]]:
    calibration = load_rf_calibration()
    profiles = calibration.get("propagation_profiles") or {}
    return copy.deepcopy(profiles.get(name) or profiles["office"])


def calibrated_wall_profiles() -> List[Dict[str, Any]]:
    return copy.deepcopy(load_rf_calibration().get("wall_profiles") or [])


def attach_calibration(scenario: Dict[str, Any], profile_name: str = "office") -> Dict[str, Any]:
    """Attach calibrated propagation parameters while preserving explicit overrides."""
    propagation = scenario.get("propagation_model")
    if not isinstance(propagation, dict):
        propagation = {"legacy_model": propagation} if propagation else {}
        scenario["propagation_model"] = propagation
    propagation.setdefault("profile", profile_name)
    propagation.setdefault("bands", propagation_profile(propagation["profile"]))
    propagation.setdefault("calibration_sources", load_rf_calibration().get("sources") or [])
    return scenario
