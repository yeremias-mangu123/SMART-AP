"""Evaluate a candidate PPO model across unseen layouts before activation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Dict, List

from .domain_randomization import randomize_scenario
from .evaluate_ppo import load_model
from .train_ppo import adapt_scenario_for_rl, rollout_policy


def deployment_score(metrics: Dict[str, float]) -> float:
    return (
        45.0 * metrics.get("sample_coverage_ratio", 0.0)
        + 18.0 * metrics.get("acceptable_sample_ratio", 0.0)
        + 25.0 * metrics.get("floor_coverage_ratio", 0.0)
        + 8.0 * metrics.get("capacity_ratio", 0.0)
        - 2.0 * metrics.get("weak_room_count", 0.0)
        - 15.0 * metrics.get("cochannel_interference_ratio", 0.0)
        - 3.0 * metrics.get("overlap_ratio", 0.0)
        - 1.0 * metrics.get("ap_count", 0.0)
    )


def evaluate_model(model_path: str, scenarios: List[Dict]) -> List[Dict]:
    model, use_mask = load_model(model_path)
    records = []
    expected = tuple(int(value) for value in model.action_space.nvec)
    observation_keys = set(getattr(model.observation_space, "spaces", {}).keys())
    observation_version = (
        4 if "wall_mean_attenuation_db" in observation_keys
        else 3 if "floor_coverage_ratio" in observation_keys
        else 2 if "sinr_coverage_ratio" in observation_keys
        else 1
    )
    for scenario in scenarios:
        scenario = adapt_scenario_for_rl(scenario)
        scenario.setdefault("constraints", {})["observation_version"] = observation_version
        actual = (
            len(scenario.get("candidate_positions") or []),
            len((scenario.get("ap_catalog") or {}).get("models") or []),
            5,
            2,
        )
        if actual != expected:
            continue
        result = rollout_policy(model, scenario, use_mask)
        metrics = result["metrics"]
        records.append({
            "scenario": scenario.get("name", "WiFi Scenario"),
            "score": deployment_score(metrics),
            "metrics": metrics,
        })
    return records


def summarize(records: List[Dict]) -> Dict:
    if not records:
        return {"scenario_count": 0, "mean_score": -1e9}
    metric_mean = lambda key: mean(float(record["metrics"].get(key, 0.0)) for record in records)
    return {
        "scenario_count": len(records),
        "mean_score": mean(record["score"] for record in records),
        "mean_sample_coverage": metric_mean("sample_coverage_ratio"),
        "mean_floor_coverage": metric_mean("floor_coverage_ratio"),
        "mean_weak_rooms": metric_mean("weak_room_count"),
        "mean_interference": metric_mean("cochannel_interference_ratio"),
        "mean_ap_count": metric_mean("ap_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_model")
    parser.add_argument("active_model")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--variants-per-scenario", type=int, default=2)
    parser.add_argument("--output", default="scratch/model_promotion_report.json")
    parser.add_argument("--promote-dir", help="Copy candidate here only when the gate passes")
    args = parser.parse_args()

    scenarios = []
    for path in sorted(Path(args.scenario_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as source:
            base = json.load(source)
        scenarios.append(base)
        for variant_index in range(args.variants_per_scenario):
            scenarios.append(randomize_scenario(base, variant_index, seed=20260607))

    active_records = evaluate_model(args.active_model, scenarios)
    candidate_records = evaluate_model(args.candidate_model, scenarios)
    active = summarize(active_records)
    candidate = summarize(candidate_records)
    passed = bool(
        candidate["scenario_count"] >= max(3, active["scenario_count"])
        and candidate["mean_score"] >= active["mean_score"] + 1.0
        and candidate["mean_sample_coverage"] >= active["mean_sample_coverage"] - 0.01
        and candidate["mean_floor_coverage"] >= active["mean_floor_coverage"] - 0.015
        and candidate["mean_interference"] <= active["mean_interference"] + 0.01
        and candidate["mean_ap_count"] <= active["mean_ap_count"] + 0.25
    )
    report = {"passed": passed, "active": active, "candidate": candidate}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if passed and args.promote_dir:
        destination = Path(args.promote_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate_model, destination / "best_model.zip")
        shutil.copy2(output, destination / "promotion_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
