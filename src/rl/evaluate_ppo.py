"""Evaluate a trained PPO AP placement model against simple baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # pragma: no cover - optional dependency
    MaskablePPO = None

from src.rl.baseline_trainer import run_baselines
from src.rl.train_ppo import (
    adapt_scenario_for_rl,
    apply_cli_overrides,
    candidate_diagnostics,
    print_candidate_diagnostics,
    print_domain_metrics,
    rollout_policy,
)


def load_model(model_path: str):
    """Load MaskablePPO when available, otherwise vanilla PPO."""
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(model_path), True
        except Exception:
            pass
    return PPO.load(model_path), False


def baseline_to_domain_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert baseline output to the same shape used by PPO rollout results."""
    best = result.get("best") or {}
    return {
        "steps": best.get("steps", 0),
        "total_reward": best.get("total_reward", 0.0),
        "metrics": best.get("metrics", {}),
        "placed_aps": best.get("placed_aps", []),
        "strategy": best.get("strategy"),
        "objective": best.get("objective"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PPO AP placement against random/greedy baselines.")
    parser.add_argument("scenario", help="Path to floor plan or RL scenario JSON")
    parser.add_argument("model", help="Path to best_model.zip or final_model.zip")
    parser.add_argument("--random-episodes", type=int, default=50, help="Random baseline rollout count")
    parser.add_argument("--candidate-limit", type=int, default=80, help="Top candidates considered by greedy baseline")
    parser.add_argument("--seed", type=int, default=42, help="Baseline random seed")
    parser.add_argument("--max-ap-count", type=int, help="Override max AP count")
    parser.add_argument("--max-episode-steps", type=int, help="Override maximum environment steps")
    parser.add_argument("--max-candidates", type=int, help="Override maximum AP candidates")
    parser.add_argument("--max-candidates-per-room", type=int, help="Override maximum generated candidates per room")
    parser.add_argument("--candidate-grid-step", type=float, help="Override candidate grid spacing in meters")
    parser.add_argument("--min-ap-separation", type=float, help="Override minimum AP separation in meters")
    parser.add_argument("--coverage-target-dbm", type=float, help="Override good coverage threshold")
    parser.add_argument("--acceptable-signal-dbm", type=float, help="Override acceptable signal threshold")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    with open(args.scenario, "r", encoding="utf-8") as f:
        scenario = json.load(f)
    apply_cli_overrides(scenario, args)
    scenario = adapt_scenario_for_rl(scenario)

    diagnostics = candidate_diagnostics(scenario)
    print_candidate_diagnostics(diagnostics)

    model, use_action_mask = load_model(args.model)
    ppo_result = rollout_policy(model, scenario, use_action_mask)
    baseline_result = run_baselines(
        scenario,
        random_episodes=args.random_episodes,
        seed=args.seed,
        greedy_candidate_limit=args.candidate_limit,
    )
    baseline_domain = baseline_to_domain_result(baseline_result)

    print_domain_metrics("PPO policy", ppo_result)
    print_domain_metrics(f"Best baseline ({baseline_domain.get('strategy')})", baseline_domain)

    output = {
        "scenario": args.scenario,
        "model": args.model,
        "use_action_mask": use_action_mask,
        "candidate_diagnostics": diagnostics,
        "ppo": ppo_result,
        "baseline_best": baseline_domain,
        "baseline_raw": baseline_result,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
