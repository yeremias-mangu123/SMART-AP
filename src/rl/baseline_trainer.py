"""Baseline search loops for the WiFi AP placement RL environment.

This is not PPO yet. It gives us a dependable benchmark and validates that the
scenario, candidate action space, and reward are coherent before adding a real
RL algorithm.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from typing import Any, Dict, List, Optional

from .candidate_generator import generate_ap_candidates
from .environment import WiFiAPPlacementEnv


def ensure_candidates(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Return a scenario that has candidate positions."""
    scenario = copy.deepcopy(scenario)
    if not scenario.get("candidate_positions"):
        scenario["candidate_positions"] = generate_ap_candidates(scenario)
        scenario["candidate_count"] = len(scenario["candidate_positions"])
    return scenario


def final_objective(metrics: Dict[str, float]) -> float:
    """Score a finished placement for baseline comparison."""
    sample_coverage = metrics.get("sample_coverage_ratio", metrics.get("coverage_ratio", 0.0))
    coverage_shortfall = max(0.0, 0.95 - sample_coverage)
    return (
        sample_coverage * 130.0
        + metrics.get("coverage_ratio", 0.0) * 85.0
        + metrics.get("priority_coverage_ratio", 0.0) * 70.0
        + metrics.get("capacity_ratio", 0.0) * 30.0
        - coverage_shortfall * 120.0
        - metrics.get("weak_room_count", 0.0) * 2.2
        - metrics.get("overlap_ratio", 0.0) * 25.0
        - metrics.get("ap_count", 0.0) * 1.8
        - metrics.get("cost", 0.0) / 450.0
    )


def valid_actions(env: WiFiAPPlacementEnv, candidate_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Enumerate valid placement actions for the current environment state."""
    candidate_indices = [
        index for index, is_valid in enumerate(env.candidate_action_mask()) if is_valid
    ]
    candidate_indices.sort(
        key=lambda index: float(env.candidates[index].get("score", 0.0)),
        reverse=True,
    )
    if candidate_limit is not None:
        candidate_indices = candidate_indices[:candidate_limit]

    actions = []
    for candidate_index in candidate_indices:
        for model_index in range(len(env.ap_models)):
            for power_index in range(len(env.power_levels)):
                action = {
                    "candidate_index": candidate_index,
                    "ap_model_index": model_index,
                    "power_index": power_index,
                    "stop": False,
                }
                if env.validate_action(action) is None:
                    actions.append(action)
    return actions


def rollout_random(
    scenario: Dict[str, Any],
    rng: random.Random,
    stop_probability: float = 0.12,
) -> Dict[str, Any]:
    """Run one random rollout."""
    env = WiFiAPPlacementEnv(scenario)
    _, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        actions = valid_actions(env)
        if not actions or (steps > 0 and rng.random() < stop_probability):
            action = {"stop": True}
        else:
            action = rng.choice(actions)

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    metrics = env.evaluate()
    return {
        "strategy": "random",
        "steps": steps,
        "total_reward": round(total_reward, 6),
        "objective": round(final_objective(metrics), 6),
        "metrics": metrics,
        "placed_aps": info.get("placed_aps", []),
    }


def rollout_greedy(
    scenario: Dict[str, Any],
    candidate_limit: int = 80,
    min_improvement: float = 0.001,
) -> Dict[str, Any]:
    """Greedily choose the action with the best immediate objective gain."""
    env = WiFiAPPlacementEnv(scenario)
    _, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    last_info: Dict[str, Any] = {"placed_aps": []}

    while not done:
        current_objective = final_objective(env.evaluate())
        best = None

        for action in valid_actions(env, candidate_limit=candidate_limit):
            trial_env = copy.deepcopy(env)
            _, reward, _, _, _ = trial_env.step(action)
            objective = final_objective(trial_env.evaluate())
            improvement = objective - current_objective
            if best is None or improvement > best["improvement"]:
                best = {
                    "action": action,
                    "reward": reward,
                    "objective": objective,
                    "improvement": improvement,
                }

        if best is None or best["improvement"] <= min_improvement:
            _, reward, terminated, truncated, last_info = env.step({"stop": True})
        else:
            _, reward, terminated, truncated, last_info = env.step(best["action"])

        total_reward += reward
        steps += 1
        done = terminated or truncated

    metrics = env.evaluate()
    return {
        "strategy": "greedy",
        "steps": steps,
        "total_reward": round(total_reward, 6),
        "objective": round(final_objective(metrics), 6),
        "metrics": metrics,
        "placed_aps": last_info.get("placed_aps", []),
    }


def run_baselines(
    scenario: Dict[str, Any],
    random_episodes: int = 100,
    seed: int = 42,
    greedy_candidate_limit: int = 80,
) -> Dict[str, Any]:
    """Run random and greedy baselines and return the best placement."""
    scenario = ensure_candidates(scenario)
    rng = random.Random(seed)

    random_results = [
        rollout_random(scenario, rng)
        for _ in range(max(0, random_episodes))
    ]
    greedy_result = rollout_greedy(scenario, candidate_limit=greedy_candidate_limit)
    all_results = random_results + [greedy_result]
    best = max(all_results, key=lambda result: result["objective"]) if all_results else greedy_result

    return {
        "scenario_name": scenario.get("name", "WiFi RL Scenario"),
        "candidate_count": len(scenario.get("candidate_positions", [])),
        "random_episodes": random_episodes,
        "seed": seed,
        "best": best,
        "greedy": greedy_result,
        "random_best": max(random_results, key=lambda result: result["objective"]) if random_results else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline AP placement search on an RL scenario JSON.")
    parser.add_argument("scenario", help="Path to RL scenario JSON exported from the planner")
    parser.add_argument("--episodes", type=int, default=100, help="Number of random rollouts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--candidate-limit", type=int, default=80, help="Top candidates considered by greedy search")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    with open(args.scenario, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    result = run_baselines(
        scenario,
        random_episodes=args.episodes,
        seed=args.seed,
        greedy_candidate_limit=args.candidate_limit,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
