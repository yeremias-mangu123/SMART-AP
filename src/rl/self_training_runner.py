"""Batch runner for one-layout AP placement self-training data.

This runner creates many randomized wall/demand variants from one exported RL
scenario, runs the current baseline planner on each variant, and writes a JSONL
dataset. The output is useful as a benchmark now and as imitation/evaluation
data once PPO/DQN training is added.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .baseline_trainer import run_baselines
from .domain_randomization import randomize_scenario, wall_profile_counts


def compact_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    best = result.get("best") or {}
    metrics = best.get("metrics") or {}
    return {
        "strategy": best.get("strategy"),
        "objective": best.get("objective"),
        "ap_count": len(best.get("placed_aps") or []),
        "coverage_ratio": metrics.get("coverage_ratio", 0.0),
        "sample_coverage_ratio": metrics.get("sample_coverage_ratio", 0.0),
        "capacity_ratio": metrics.get("capacity_ratio", 0.0),
        "weak_room_count": metrics.get("weak_room_count", 0.0),
        "overlap_ratio": metrics.get("overlap_ratio", 0.0),
        "cost": metrics.get("cost", 0.0),
    }


def run_self_training_batch(
    scenario: Dict[str, Any],
    variants: int = 30,
    random_episodes: int = 30,
    seed: int = 42,
    greedy_candidate_limit: int = 80,
    output_dir: str = "training_runs/latest",
    save_scenarios: bool = False,
) -> Dict[str, Any]:
    """Generate variants, run baseline placement, and persist the results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scenarios_dir = output_path / "scenarios"
    if save_scenarios:
        scenarios_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "results.jsonl"
    records: List[Dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as results_file:
        for variant_index in range(max(1, variants)):
            variant = randomize_scenario(scenario, variant_index=variant_index, seed=seed)
            result = run_baselines(
                variant,
                random_episodes=random_episodes,
                seed=seed + variant_index,
                greedy_candidate_limit=greedy_candidate_limit,
            )
            record = {
                "variant_index": variant_index,
                "variant_seed": variant.get("variant", {}).get("seed"),
                "scenario_name": variant.get("name"),
                "candidate_count": result.get("candidate_count", 0),
                "wall_profiles": wall_profile_counts(variant),
                "metrics": compact_metrics(result),
                "placed_aps": (result.get("best") or {}).get("placed_aps", []),
            }
            records.append(record)
            results_file.write(json.dumps(record, separators=(",", ":")) + "\n")

            if save_scenarios:
                scenario_path = scenarios_dir / f"variant_{variant_index + 1:04d}.json"
                with scenario_path.open("w", encoding="utf-8") as scenario_file:
                    json.dump(variant, scenario_file, indent=2)

    if records:
        avg = lambda key: sum(float(item["metrics"].get(key, 0.0)) for item in records) / len(records)
        best_record = max(records, key=lambda item: float(item["metrics"].get("objective") or -1e9))
    else:
        avg = lambda key: 0.0
        best_record = None

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_scenario": scenario.get("name", "WiFi Scenario"),
        "variant_count": len(records),
        "random_episodes": random_episodes,
        "seed": seed,
        "greedy_candidate_limit": greedy_candidate_limit,
        "outputs": {
            "directory": os.fspath(output_path),
            "results_jsonl": os.fspath(results_path),
            "scenarios_directory": os.fspath(scenarios_dir) if save_scenarios else None,
        },
        "averages": {
            "ap_count": avg("ap_count"),
            "coverage_ratio": avg("coverage_ratio"),
            "sample_coverage_ratio": avg("sample_coverage_ratio"),
            "capacity_ratio": avg("capacity_ratio"),
            "weak_room_count": avg("weak_room_count"),
            "cost": avg("cost"),
        },
        "best_variant": best_record,
    }

    with (output_path / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wall variants and run AP placement baseline batch.")
    parser.add_argument("scenario", help="Path to an RL scenario JSON exported from the planner")
    parser.add_argument("--variants", type=int, default=30, help="Number of randomized wall/demand variants")
    parser.add_argument("--episodes", type=int, default=30, help="Random rollout episodes per variant")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--candidate-limit", type=int, default=80, help="Top candidates considered by greedy search")
    parser.add_argument("--output-dir", default="training_runs/latest", help="Directory for JSONL and summary outputs")
    parser.add_argument("--save-scenarios", action="store_true", help="Write every randomized scenario JSON")
    args = parser.parse_args()

    with open(args.scenario, "r", encoding="utf-8") as scenario_file:
        scenario = json.load(scenario_file)

    summary = run_self_training_batch(
        scenario,
        variants=args.variants,
        random_episodes=args.episodes,
        seed=args.seed,
        greedy_candidate_limit=args.candidate_limit,
        output_dir=args.output_dir,
        save_scenarios=args.save_scenarios,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
