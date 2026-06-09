"""Gymnasium-style RL environment for WiFi AP placement.

This environment intentionally has no hard dependency on gymnasium yet. It
implements the familiar ``reset`` and ``step`` API so the next phase can wrap it
with Gymnasium/Stable-Baselines when those packages are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, hypot
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .candidate_generator import distance_point_to_segment, point_in_polygon
from .room_profiles import (
    room_capacity_headroom,
    room_coverage_target,
    room_effective_priority,
    room_is_high_density,
    room_requires_service,
    room_type_profile,
)


Action = Union[Dict[str, Any], Sequence[int]]


@dataclass
class PlacedAP:
    candidate_index: int
    model_index: int
    power_index: int
    x: float
    y: float
    z: float
    model_id: str
    tx_power_dbm: float


def segments_intersect(a: Tuple[float, float], b: Tuple[float, float],
                       c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    """Return True when segment AB intersects segment CD."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d

    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(denom) < 1e-9:
        return False

    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    return 0 <= t <= 1 and 0 <= u <= 1


def segment_intersection_parameter(a: Tuple[float, float], b: Tuple[float, float],
                                   c: Tuple[float, float], d: Tuple[float, float]) -> Optional[float]:
    """Return ray position for an intersection, including wall endpoints."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(denom) < 1e-9:
        return None
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    epsilon = 1e-6
    return t if epsilon < t < 1.0 - epsilon and -epsilon <= u <= 1.0 + epsilon else None


class WiFiAPPlacementEnv:
    """Small, deterministic environment for AP placement research.

    Action format:
      - dict: ``{"candidate_index": int, "ap_model_index": int,
        "power_index": int, "stop": bool}``
      - tuple/list: ``(candidate_index, ap_model_index, power_index, stop)``

    Observation is a dictionary of scalar metrics plus selected AP count. The
    Gymnasium wrapper enriches this with candidate features and action masks for
    PPO-style training.
    """

    def __init__(self, scenario: Dict[str, Any], power_levels: Optional[List[float]] = None):
        self.scenario = scenario
        self.all_rooms = scenario.get("rooms", [])
        self.rooms = [room for room in self.all_rooms if room_requires_service(room)]
        self.walls = scenario.get("walls", [])
        self.candidates = scenario.get("candidate_positions", [])
        self.ap_models = (scenario.get("ap_catalog") or {}).get("models", [])
        self.constraints = scenario.get("constraints", {})
        self.targets = scenario.get("targets", {})
        self.reward_weights = scenario.get("reward_weights", {})
        self.propagation_model = scenario.get("propagation_model", {})
        self.power_levels = power_levels or [14.0, 17.0, 20.0, 23.0, 26.0]
        self.max_ap_count = int(self.constraints.get("max_ap_count") or 6)
        self.observation_version = int(self.constraints.get("observation_version") or 1)
        self.min_ap_count = max(
            0,
            min(self.max_ap_count, int(self.constraints.get("min_ap_count") or 0)),
        )
        self.min_ap_separation_m = float(self.constraints.get("min_ap_separation_m") or 5.0)
        # Distance below which two APs are considered wastefully clustered. Used for a
        # soft proximity penalty so the agent learns to spread APs out, not just respect
        # the hard min_ap_separation_m floor.
        self.comfortable_ap_spacing_m = float(
            self.constraints.get("comfortable_ap_spacing_m")
            or max(self.min_ap_separation_m * 1.8, 9.0)
        )
        self.room_sample_step_m = float(self.constraints.get("room_sample_step_m") or 1.2)
        self.max_samples_per_room = int(self.constraints.get("max_samples_per_room") or 20)
        self.min_room_sample_coverage = float(self.targets.get("min_room_sample_coverage_ratio") or 0.85)
        self.target_sample_coverage = float(self.targets.get("target_sample_coverage_ratio") or 0.95)
        self.good_signal_dbm = float(self.targets.get("good_signal_dbm") or self.targets.get("default_coverage_dbm") or -67)
        self.acceptable_signal_dbm = float(self.targets.get("acceptable_signal_dbm") or -70)
        self.frequency_ghz = float((scenario.get("floor_plan") or {}).get("selected_frequency_ghz") or 5.0)
        self.max_episode_steps = int(
            self.constraints.get("max_episode_steps") or max(20, self.max_ap_count)
        )
        self.room_samples = {
            str(room.get("id", index)): self.sample_room_points(room)
            for index, room in enumerate(self.rooms)
        }
        self.floor_samples = self.sample_floor_points()
        self._rssi_cache: Dict[Tuple[Any, ...], float] = {}
        self.placed_aps: List[PlacedAP] = []
        self.used_candidate_indices = set()
        self.last_metrics: Dict[str, float] = {}
        self.current_step = 0

    @property
    def action_space_shape(self) -> Tuple[int, int, int, int]:
        """Discrete action dimensions: candidate, model, power, stop flag."""
        return (len(self.candidates), len(self.ap_models), len(self.power_levels), 2)

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.placed_aps = []
        self.used_candidate_indices = set()
        self.current_step = 0
        self.last_metrics = self.evaluate()
        return self.get_observation(), {"metrics": self.last_metrics}

    def step(self, action: Action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        decoded = self.decode_action(action)
        previous_metrics = self.evaluate()
        reward = 0.0
        terminated = False
        truncated = False
        invalid_reason = None
        self.current_step += 1

        if decoded["stop"]:
            terminated = True
        else:
            invalid_reason = self.validate_action(decoded)
            if invalid_reason:
                reward += float(self.reward_weights.get("invalid_position", -2.0))
            else:
                self.place_ap(decoded)

        current_metrics = self.evaluate()
        reward += self.calculate_reward(previous_metrics, current_metrics)
        target_reached = not decoded["stop"] and invalid_reason is None and self.target_requirements_met(current_metrics)
        if target_reached:
            terminated = True

        if len(self.placed_aps) >= self.max_ap_count:
            truncated = True
        if self.current_step >= self.max_episode_steps:
            truncated = True
        if terminated or truncated:
            reward += self.calculate_terminal_reward(current_metrics, stopped=decoded["stop"] or target_reached)

        self.last_metrics = current_metrics
        return self.get_observation(), reward, terminated, truncated, {
            "metrics": current_metrics,
            "invalid_reason": invalid_reason,
            "target_reached": target_reached,
            "placed_aps": [ap.__dict__ for ap in self.placed_aps],
        }

    def decode_action(self, action: Action) -> Dict[str, Any]:
        if isinstance(action, dict):
            return {
                "candidate_index": int(action.get("candidate_index", 0)),
                "ap_model_index": int(action.get("ap_model_index", 0)),
                "power_index": int(action.get("power_index", 0)),
                "stop": bool(action.get("stop", False)),
            }

        values = list(action)
        return {
            "candidate_index": int(values[0]) if len(values) > 0 else 0,
            "ap_model_index": int(values[1]) if len(values) > 1 else 0,
            "power_index": int(values[2]) if len(values) > 2 else 0,
            "stop": bool(values[3]) if len(values) > 3 else False,
        }

    def validate_action(self, action: Dict[str, Any]) -> Optional[str]:
        candidate_index = action["candidate_index"]
        model_index = action["ap_model_index"]
        power_index = action["power_index"]

        if not 0 <= candidate_index < len(self.candidates):
            return "candidate_index_out_of_range"
        if not 0 <= model_index < len(self.ap_models):
            return "ap_model_index_out_of_range"
        if not 0 <= power_index < len(self.power_levels):
            return "power_index_out_of_range"
        if candidate_index in self.used_candidate_indices:
            return "candidate_already_used"

        candidate = self.candidates[candidate_index]
        if candidate.get("_action_padding"):
            return "candidate_is_action_padding"
        if self.candidate_room_limit_reached(candidate):
            return "room_ap_limit_reached"
        for ap in self.placed_aps:
            if hypot(candidate["x"] - ap.x, candidate["y"] - ap.y) < self.min_ap_separation_m:
                return "too_close_to_existing_ap"

        model = self.ap_models[model_index]
        band = "6" if self.frequency_ghz >= 6 else ("5" if self.frequency_ghz >= 5 else "2.4")
        if band not in [str(item) for item in model.get("bands", ["2.4", "5"])]:
            return "ap_model_does_not_support_band"

        return None

    def room_ap_limit(self, room: Dict[str, Any]) -> int:
        """Allow a second AP only for large or genuinely high-density rooms."""
        area_m2 = float(room.get("area_m2") or room.get("areaM2") or 0.0)
        clients = int(room.get("clients") or 0)
        room_type = str(room.get("type") or "").lower()
        return 2 if area_m2 >= 70.0 or clients >= 100 or room_type in {"auditorium", "high_density", "event", "hall"} else 1

    def candidate_room_limit_reached(self, candidate: Dict[str, Any]) -> bool:
        room_id = candidate.get("room_id")
        if room_id is None:
            return False
        room = next((item for item in self.rooms if str(item.get("id")) == str(room_id)), None)
        if room is None:
            return True
        count = 0
        for ap in self.placed_aps:
            placed_candidate = self.candidates[ap.candidate_index]
            if str(placed_candidate.get("room_id")) == str(room_id):
                count += 1
        return count >= self.room_ap_limit(room)

    def place_ap(self, action: Dict[str, Any]) -> None:
        candidate = self.candidates[action["candidate_index"]]
        model = self.ap_models[action["ap_model_index"]]
        max_power = float(model.get("maxPower", 26))
        tx_power = min(float(self.power_levels[action["power_index"]]), max_power)

        self.used_candidate_indices.add(action["candidate_index"])
        self.placed_aps.append(PlacedAP(
            candidate_index=action["candidate_index"],
            model_index=action["ap_model_index"],
            power_index=action["power_index"],
            x=float(candidate["x"]),
            y=float(candidate["y"]),
            z=float(candidate.get("z", 2.7)),
            model_id=str(model.get("id", f"model_{action['ap_model_index']}")),
            tx_power_dbm=tx_power,
        ))

    def get_observation(self) -> Dict[str, Any]:
        metrics = self.last_metrics or self.evaluate()
        # These deployment-audit metrics are used by runtime optimization but
        # intentionally stay outside the trained PPO observation contract.
        metrics = {
            key: value for key, value in metrics.items()
            if key not in {
                "capacity_shortfall_room_count",
                "high_density_room_without_local_ap_count",
                # Reward-shaping signal only; kept out of the PPO observation
                # contract so the observation space stays identical to v4.
                "ap_proximity_penalty",
            }
        }
        if self.observation_version < 2:
            metrics = {
                key: value for key, value in metrics.items()
                if key not in {"sinr_coverage_ratio", "average_sinr_db", "cochannel_interference_ratio"}
            }
        if self.observation_version < 3:
            metrics = {
                key: value for key, value in metrics.items()
                if key not in {
                    "floor_coverage_ratio",
                    "floor_acceptable_ratio",
                    "floor_blankspot_ratio",
                    "floor_sinr_coverage_ratio",
                }
            }
        observation = {
            "placed_ap_count": len(self.placed_aps),
            "remaining_ap_budget": max(0, self.max_ap_count - len(self.placed_aps)),
            "candidate_count": len(self.candidates),
            "room_count": len(self.rooms),
            "step_count": self.current_step,
            **metrics,
        }
        if self.observation_version >= 4:
            band = "6" if self.frequency_ghz >= 6 else ("5" if self.frequency_ghz >= 5 else "2.4")
            wall_losses = [
                float((wall.get("attenuation_db") or {}).get(band) or 0.0)
                for wall in self.walls
            ]
            propagation = (self.propagation_model.get("bands") or {}).get(band) or {}
            observation.update({
                "frequency_ghz": self.frequency_ghz,
                "wall_count": len(wall_losses),
                "wall_mean_attenuation_db": sum(wall_losses) / max(1, len(wall_losses)),
                "wall_max_attenuation_db": max(wall_losses, default=0.0),
                "wall_heavy_ratio": (
                    sum(1 for loss in wall_losses if loss >= 15.0) / max(1, len(wall_losses))
                ),
                "distance_loss_coefficient": float(
                    propagation.get("distance_loss_coefficient", 31.0)
                ),
                "shadow_margin_db": float(propagation.get("shadow_margin_db", 8.0)),
            })
        return observation

    def evaluate(self) -> Dict[str, float]:
        if not self.rooms:
            return {
                "coverage_ratio": 0.0,
                "priority_coverage_ratio": 0.0,
                "capacity_ratio": 0.0,
                "capacity_shortfall_room_count": 0.0,
                "high_density_room_without_local_ap_count": 0.0,
                "average_best_rssi_dbm": -100.0,
                "minimum_best_rssi_dbm": -100.0,
                "sample_coverage_ratio": 0.0,
                "acceptable_sample_ratio": 0.0,
                "blankspot_ratio": 1.0,
                "blankspot_percent": 100.0,
                "coverage_percent": 0.0,
                "acceptable_coverage_percent": 0.0,
                "average_room_sample_coverage": 0.0,
                "weak_room_count": 0.0,
                "uncovered_sample_count": 0.0,
                "cost": self.total_cost(),
                "ap_count": float(len(self.placed_aps)),
                "overlap_ratio": 0.0,
                "ap_proximity_penalty": self.ap_proximity_penalty(),
                "sinr_coverage_ratio": 0.0,
                "average_sinr_db": -20.0,
                "cochannel_interference_ratio": 0.0,
                "floor_coverage_ratio": 0.0,
                "floor_acceptable_ratio": 0.0,
                "floor_blankspot_ratio": 1.0,
                "floor_sinr_coverage_ratio": 0.0,
            }

        covered_rooms = 0.0
        weighted_covered = 0.0
        total_priority = 0.0
        room_capacity_records = []
        ap_supported_demands = [0.0 for _ in self.placed_aps]
        high_density_room_without_local_ap_count = 0
        best_rssi_values = []
        overlap_count = 0
        total_samples = 0
        covered_samples = 0
        acceptable_samples = 0
        total_room_sample_coverage = 0.0
        weak_room_count = 0
        sinr_covered_samples = 0
        interfered_samples = 0
        sinr_values = []
        channels = self.assigned_channels()

        for index, room in enumerate(self.rooms):
            target = room_coverage_target(room, self.good_signal_dbm)
            priority = room_effective_priority(room)
            clients = float(room.get("clients", 1))
            room_point = room.get("centroid") or {}
            x = float(room_point.get("x", 0))
            y = float(room_point.get("y", 0))
            samples = self.room_samples.get(str(room.get("id", index))) or [(x, y)]

            sample_best_signals = []
            ap_covered_samples = [0 for _ in self.placed_aps]
            for sample_x, sample_y in samples:
                sample_signals = [self.rssi_at_room(ap, sample_x, sample_y) for ap in self.placed_aps]
                best_signal = max(sample_signals) if sample_signals else -100.0
                sample_best_signals.append(best_signal)
                if sample_signals:
                    serving_index = max(range(len(sample_signals)), key=lambda ap_index: sample_signals[ap_index])
                    signal_mw = 10.0 ** (sample_signals[serving_index] / 10.0)
                    interference_mw = sum(
                        10.0 ** (signal / 10.0)
                        for ap_index, signal in enumerate(sample_signals)
                        if ap_index != serving_index and channels[ap_index] == channels[serving_index]
                    )
                    noise_mw = 10.0 ** (-95.0 / 10.0)
                    sinr_db = 10.0 * log10(signal_mw / max(noise_mw, interference_mw + noise_mw))
                    sinr_values.append(sinr_db)
                    if sinr_db >= float(self.targets.get("min_sinr_db") or 15.0):
                        sinr_covered_samples += 1
                    if sinr_db < float(self.targets.get("interference_sinr_db") or 20.0):
                        interfered_samples += 1
                for ap_index, signal in enumerate(sample_signals):
                    if signal >= target:
                        ap_covered_samples[ap_index] += 1

            covered_in_room = sum(1 for signal in sample_best_signals if signal >= target)
            acceptable_in_room = sum(1 for signal in sample_best_signals if signal >= self.acceptable_signal_dbm)
            room_sample_coverage = covered_in_room / max(1, len(samples))
            total_samples += len(samples)
            covered_samples += covered_in_room
            acceptable_samples += acceptable_in_room
            total_room_sample_coverage += room_sample_coverage
            best_rssi_values.extend(sample_best_signals)

            if room_sample_coverage >= self.min_room_sample_coverage:
                covered_rooms += 1
            else:
                weak_room_count += 1

            weighted_covered += priority * room_sample_coverage

            total_priority += priority
            supporting_indices = [
                ap_index for ap_index, covered_count in enumerate(ap_covered_samples)
                if covered_count / max(1, len(samples)) >= 0.5
            ]
            supporting_aps = [self.placed_aps[ap_index] for ap_index in supporting_indices]
            if len(supporting_aps) > 1:
                overlap_count += 1

            headroom = room_capacity_headroom(room, float(self.targets.get("capacity_headroom_ratio", 1.2)))
            demand = max(1.0, clients * headroom)
            for ap_index in supporting_indices:
                ap_supported_demands[ap_index] += demand
            local_ap_count = sum(
                1 for ap in self.placed_aps
                if point_in_polygon(ap.x, ap.y, room.get("polygon") or [])
            )
            if room_is_high_density(room) and local_ap_count <= 0:
                high_density_room_without_local_ap_count += 1
            room_capacity_records.append((demand, supporting_indices))

        capacity_scores = []
        for demand, supporting_indices in room_capacity_records:
            allocated_capacity = sum(
                float(
                    self.ap_models[self.placed_aps[ap_index].model_index].get("planningClients")
                    or self.ap_models[self.placed_aps[ap_index].model_index].get("recommendedClients", 40)
                )
                * demand / max(demand, ap_supported_demands[ap_index])
                for ap_index in supporting_indices
            )
            capacity_scores.append(min(1.0, allocated_capacity / demand))
        capacity_shortfall_room_count = sum(score < 0.95 for score in capacity_scores)

        floor_covered = 0
        floor_acceptable = 0
        floor_sinr_covered = 0
        for sample_x, sample_y in self.floor_samples:
            sample_signals = [self.rssi_at_room(ap, sample_x, sample_y) for ap in self.placed_aps]
            best_signal = max(sample_signals) if sample_signals else -100.0
            if best_signal >= self.good_signal_dbm:
                floor_covered += 1
            if best_signal >= self.acceptable_signal_dbm:
                floor_acceptable += 1
            if sample_signals:
                serving_index = max(range(len(sample_signals)), key=lambda ap_index: sample_signals[ap_index])
                signal_mw = 10.0 ** (sample_signals[serving_index] / 10.0)
                interference_mw = sum(
                    10.0 ** (signal / 10.0)
                    for ap_index, signal in enumerate(sample_signals)
                    if ap_index != serving_index and channels[ap_index] == channels[serving_index]
                )
                noise_mw = 10.0 ** (-95.0 / 10.0)
                sinr_db = 10.0 * log10(signal_mw / max(noise_mw, interference_mw + noise_mw))
                if sinr_db >= float(self.targets.get("min_sinr_db") or 15.0):
                    floor_sinr_covered += 1

        sample_coverage_ratio = covered_samples / max(1, total_samples)
        acceptable_sample_ratio = acceptable_samples / max(1, total_samples)
        blankspot_ratio = 1.0 - sample_coverage_ratio
        floor_coverage_ratio = floor_covered / max(1, len(self.floor_samples))
        floor_acceptable_ratio = floor_acceptable / max(1, len(self.floor_samples))
        floor_sinr_coverage_ratio = floor_sinr_covered / max(1, len(self.floor_samples))

        return {
            "coverage_ratio": covered_rooms / max(1, len(self.rooms)),
            "priority_coverage_ratio": weighted_covered / max(1.0, total_priority),
            "capacity_ratio": sum(capacity_scores) / max(1, len(capacity_scores)),
            "capacity_shortfall_room_count": float(capacity_shortfall_room_count),
            "high_density_room_without_local_ap_count": float(high_density_room_without_local_ap_count),
            "average_best_rssi_dbm": sum(best_rssi_values) / max(1, len(best_rssi_values)),
            "minimum_best_rssi_dbm": min(best_rssi_values) if best_rssi_values else -100.0,
            "sample_coverage_ratio": sample_coverage_ratio,
            "acceptable_sample_ratio": acceptable_sample_ratio,
            "blankspot_ratio": blankspot_ratio,
            "blankspot_percent": blankspot_ratio * 100.0,
            "coverage_percent": sample_coverage_ratio * 100.0,
            "acceptable_coverage_percent": acceptable_sample_ratio * 100.0,
            "average_room_sample_coverage": total_room_sample_coverage / max(1, len(self.rooms)),
            "weak_room_count": float(weak_room_count),
            "uncovered_sample_count": float(max(0, total_samples - covered_samples)),
            "cost": self.total_cost(),
            "ap_count": float(len(self.placed_aps)),
            "overlap_ratio": overlap_count / max(1, len(self.rooms)),
            "ap_proximity_penalty": self.ap_proximity_penalty(),
            "sinr_coverage_ratio": sinr_covered_samples / max(1, total_samples),
            "average_sinr_db": sum(sinr_values) / max(1, len(sinr_values)),
            "cochannel_interference_ratio": interfered_samples / max(1, total_samples),
            "floor_coverage_ratio": floor_coverage_ratio,
            "floor_acceptable_ratio": floor_acceptable_ratio,
            "floor_blankspot_ratio": 1.0 - floor_coverage_ratio,
            "floor_sinr_coverage_ratio": floor_sinr_coverage_ratio,
        }

    def calculate_reward(self, previous: Dict[str, float], current: Dict[str, float]) -> float:
        weights = self.reward_weights
        reward = float(weights.get("step", -0.01))
        reward += float(weights.get("coverage", 1.0)) * (current["coverage_ratio"] - previous["coverage_ratio"])
        reward += float(weights.get("sample_coverage", 1.2)) * (
            current.get("sample_coverage_ratio", 0.0) - previous.get("sample_coverage_ratio", 0.0)
        )
        reward += float(weights.get("blankspot", -0.8)) * max(
            0.0,
            current.get("blankspot_ratio", 1.0) - previous.get("blankspot_ratio", 1.0),
        )
        reward += float(weights.get("acceptable_coverage", 0.4)) * (
            current.get("acceptable_sample_ratio", 0.0) - previous.get("acceptable_sample_ratio", 0.0)
        )
        reward += float(weights.get("room_priority", 0.6)) * (
            current["priority_coverage_ratio"] - previous["priority_coverage_ratio"]
        )
        reward += float(weights.get("capacity", 0.5)) * (current["capacity_ratio"] - previous["capacity_ratio"])
        reward += float(weights.get("cost", -0.25)) * max(0.0, current["cost"] - previous["cost"]) / 1000.0
        reward += float(weights.get("ap_count", -0.5)) * max(0.0, current["ap_count"] - previous["ap_count"])
        reward += float(weights.get("overlap", -1.0)) * max(0.0, current["overlap_ratio"] - previous["overlap_ratio"])
        reward += float(weights.get("proximity", -1.5)) * max(
            0.0,
            current.get("ap_proximity_penalty", 0.0) - previous.get("ap_proximity_penalty", 0.0),
        )
        reward += float(weights.get("sinr_coverage", 1.2)) * (
            current.get("sinr_coverage_ratio", 0.0) - previous.get("sinr_coverage_ratio", 0.0)
        )
        reward += float(weights.get("interference", -1.0)) * max(
            0.0,
            current.get("cochannel_interference_ratio", 0.0)
            - previous.get("cochannel_interference_ratio", 0.0),
        )
        reward += float(weights.get("floor_coverage", 1.5)) * (
            current.get("floor_coverage_ratio", 0.0) - previous.get("floor_coverage_ratio", 0.0)
        )
        reward += float(weights.get("floor_sinr_coverage", 0.8)) * (
            current.get("floor_sinr_coverage_ratio", 0.0) - previous.get("floor_sinr_coverage_ratio", 0.0)
        )
        return reward

    def calculate_terminal_reward(self, metrics: Dict[str, float], stopped: bool) -> float:
        """Reward or penalize the final layout when an episode ends.

        Without a terminal objective, PPO can learn that stopping immediately is
        a safe zero-reward policy. This score keeps the incremental reward but
        also tells the agent whether the finished AP layout is actually useful.
        """
        weights = self.reward_weights
        sample_coverage = metrics.get("sample_coverage_ratio", metrics.get("coverage_ratio", 0.0))
        coverage_shortfall = max(0.0, self.target_sample_coverage - sample_coverage)
        weak_room_ratio = metrics.get("weak_room_count", 0.0) / max(1, len(self.rooms))

        reward = 0.0
        reward += float(weights.get("terminal_sample_coverage", 3.0)) * sample_coverage
        reward += float(weights.get("terminal_acceptable_coverage", 0.8)) * metrics.get("acceptable_sample_ratio", 0.0)
        reward += float(weights.get("terminal_coverage", 2.0)) * metrics.get("coverage_ratio", 0.0)
        reward += float(weights.get("terminal_priority", 1.5)) * metrics.get("priority_coverage_ratio", 0.0)
        reward += float(weights.get("terminal_capacity", 0.8)) * metrics.get("capacity_ratio", 0.0)
        reward -= float(weights.get("terminal_blankspot", 2.5)) * metrics.get("blankspot_ratio", 1.0)
        reward -= float(weights.get("terminal_shortfall", 4.0)) * coverage_shortfall
        reward -= float(weights.get("terminal_weak_room", 0.6)) * weak_room_ratio
        reward -= float(weights.get("terminal_overlap", 2.0)) * metrics.get("overlap_ratio", 0.0)
        reward -= float(weights.get("terminal_proximity", 3.0)) * metrics.get("ap_proximity_penalty", 0.0)
        reward += float(weights.get("terminal_sinr_coverage", 2.0)) * metrics.get("sinr_coverage_ratio", 0.0)
        reward -= float(weights.get("terminal_interference", 2.0)) * metrics.get("cochannel_interference_ratio", 0.0)
        reward += float(weights.get("terminal_floor_coverage", 3.0)) * metrics.get("floor_coverage_ratio", 0.0)
        reward -= float(weights.get("terminal_floor_blankspot", 3.0)) * metrics.get("floor_blankspot_ratio", 1.0)
        reward -= float(weights.get("terminal_ap_count", 0.8)) * metrics.get("ap_count", 0.0)
        reward -= float(weights.get("terminal_cost", 0.15)) * metrics.get("cost", 0.0) / 1000.0
        reward -= float(weights.get("terminal_min_ap_shortfall", 1.2)) * max(
            0.0,
            float(self.min_ap_count) - metrics.get("ap_count", 0.0),
        )

        if stopped and metrics.get("ap_count", 0.0) <= 0:
            reward += float(weights.get("stop_without_ap", -2.0))
        if stopped and coverage_shortfall > 0.0:
            reward += float(weights.get("early_stop_shortfall", -3.0)) * coverage_shortfall
        if stopped and metrics.get("ap_count", 0.0) < self.min_ap_count:
            reward += float(weights.get("early_stop_min_ap_shortfall", -2.0)) * (
                self.min_ap_count - metrics.get("ap_count", 0.0)
            )
        if stopped and coverage_shortfall <= 0.0:
            efficiency = 1.0 - metrics.get("ap_count", 0.0) / max(1.0, float(self.max_ap_count))
            reward += float(weights.get("stop_target_met", 1.5))
            reward += float(weights.get("stop_efficiency", 2.0)) * efficiency
        if not stopped and metrics.get("ap_count", 0.0) >= self.max_ap_count:
            reward += float(weights.get("max_ap_without_stop", -1.5))

        return reward

    def assigned_channels(self) -> List[int]:
        """Assign channels deterministically so placement reward reflects co-channel interference."""
        if not self.placed_aps:
            return []

        channels = [1, 6, 11] if self.frequency_ghz < 5 else [36, 40, 44, 48, 149, 153, 157, 161]
        avoid_radius = 14.0 if self.frequency_ghz < 5 else 8.0
        assigned: List[int] = []
        usage = {channel: 0 for channel in channels}

        for index, ap in enumerate(self.placed_aps):
            best_channel = channels[0]
            best_score = None
            for channel in channels:
                score = usage[channel] * 0.08
                for other_index, other_ap in enumerate(self.placed_aps[:index]):
                    if assigned[other_index] != channel:
                        continue
                    distance = hypot(ap.x - other_ap.x, ap.y - other_ap.y)
                    if distance < avoid_radius:
                        proximity = (avoid_radius - distance) / avoid_radius
                        score += 10.0 * proximity * proximity
                if best_score is None or score < best_score:
                    best_score = score
                    best_channel = channel
            assigned.append(best_channel)
            usage[best_channel] += 1

        return assigned

    def sample_room_points(self, room: Dict[str, Any]) -> List[Tuple[float, float]]:
        polygon = room.get("polygon") or []
        centroid = room.get("centroid") or {}
        fallback = (float(centroid.get("x", 0)), float(centroid.get("y", 0)))
        if len(polygon) < 3:
            return [fallback]

        xs = [float(point.get("x", 0)) for point in polygon]
        ys = [float(point.get("y", 0)) for point in polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return [fallback]

        step = max(0.5, self.room_sample_step_m)
        points: List[Tuple[float, float]] = []
        y = min_y + step / 2
        while y < max_y:
            x = min_x + step / 2
            while x < max_x:
                in_no_service_room = any(
                    not room_requires_service(room)
                    and point_in_polygon(x, y, room.get("polygon") or [])
                    for room in self.all_rooms
                )
                if point_in_polygon(x, y, polygon) and not in_no_service_room:
                    points.append((round(x, 3), round(y, 3)))
                x += step
            y += step

        if point_in_polygon(fallback[0], fallback[1], polygon):
            points.append(fallback)
        if not points:
            points = [fallback]

        # Prefer well-spread samples: corners/interior are more useful than many
        # dense neighboring points in large rectangular rooms.
        unique = list(dict.fromkeys(points))
        if len(unique) <= self.max_samples_per_room:
            return unique

        stride = max(1, len(unique) // self.max_samples_per_room)
        sampled = unique[::stride][: self.max_samples_per_room]
        if fallback not in sampled:
            sampled[-1] = fallback
        return sampled

    def sample_floor_points(self) -> List[Tuple[float, float]]:
        """Sample the complete building footprint so corridors and unlabeled areas count."""
        service_boundary = (self.scenario.get("floor_plan") or {}).get("service_boundary") or []
        boundary_points = [
            (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
            for point in service_boundary
            if isinstance(point, dict)
        ]
        wall_points = [
            (float(point[0]), float(point[1]))
            for wall in self.walls
            for point in (wall.get("points") or [])
            if len(point) >= 2
        ]
        room_points = [
            (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
            for room in self.rooms
            for point in (room.get("polygon") or [])
        ]
        source_points = boundary_points or wall_points or room_points
        if len(source_points) < 3:
            return []

        hull = source_points if len(boundary_points) >= 3 else self.convex_hull(source_points)
        if len(hull) < 3:
            return []
        polygon = [{"x": x, "y": y} for x, y in hull]
        min_x = min(x for x, _ in hull)
        max_x = max(x for x, _ in hull)
        min_y = min(y for _, y in hull)
        max_y = max(y for _, y in hull)
        step = max(0.8, float(self.constraints.get("floor_sample_step_m") or 1.2))
        points = []
        y = min_y + step / 2
        while y < max_y:
            x = min_x + step / 2
            while x < max_x:
                in_no_service_room = any(
                    not room_requires_service(room)
                    and point_in_polygon(x, y, room.get("polygon") or [])
                    for room in self.all_rooms
                )
                if point_in_polygon(x, y, polygon) and not in_no_service_room:
                    points.append((round(x, 3), round(y, 3)))
                x += step
            y += step
        max_samples = int(self.constraints.get("max_floor_samples") or 320)
        if len(points) <= max_samples:
            return points
        stride = max(1, len(points) // max_samples)
        return points[::stride][:max_samples]

    @staticmethod
    def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        unique = sorted(set(points))
        if len(unique) <= 2:
            return unique

        def cross(origin, a, b):
            return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def rssi_at_room(self, ap: PlacedAP, room_x: float, room_y: float) -> float:
        cache_key = (
            ap.candidate_index,
            ap.model_index,
            ap.power_index,
            round(ap.x, 3),
            round(ap.y, 3),
            round(ap.z, 3),
            round(ap.tx_power_dbm, 2),
            round(room_x, 3),
            round(room_y, 3),
        )
        cached = self._rssi_cache.get(cache_key)
        if cached is not None:
            return cached

        model = self.ap_models[ap.model_index]
        gain = self.gain_for_frequency(model)
        dz = ap.z - 1.0
        distance_m = max(0.5, (hypot(ap.x - room_x, ap.y - room_y) ** 2 + dz ** 2) ** 0.5)
        band = "6" if self.frequency_ghz >= 6 else ("5" if self.frequency_ghz >= 5 else "2.4")
        configured = (self.propagation_model.get("bands") or {}).get(band) or {}
        default_coefficient = 54.0 if self.frequency_ghz >= 5 else 48.0
        default_margin = 16.0 if self.frequency_ghz >= 5 else 14.0
        default_wall_multiplier = 1.28 if self.frequency_ghz >= 5 else 1.12
        distance_loss_coefficient = float(
            configured.get("distance_loss_coefficient", default_coefficient)
        )
        fade_margin_db = float(configured.get("shadow_margin_db", default_margin))
        wall_loss_multiplier = float(
            configured.get("wall_loss_multiplier", default_wall_multiplier)
        )
        indoor_loss = (
            20.0 * log10(self.frequency_ghz * 1000.0)
            + distance_loss_coefficient * log10(distance_m)
            - 28.0
        )
        attenuation = self.wall_attenuation_between(ap.x, ap.y, room_x, room_y) * wall_loss_multiplier
        signal = ap.tx_power_dbm + gain - indoor_loss - attenuation - fade_margin_db
        self._rssi_cache[cache_key] = signal
        return signal

    def gain_for_frequency(self, model: Dict[str, Any]) -> float:
        if self.frequency_ghz >= 6:
            return float(model.get("gain6") or model.get("gain5") or model.get("gain24") or 0)
        if self.frequency_ghz >= 5:
            return float(model.get("gain5") or model.get("gain24") or 0)
        return float(model.get("gain24") or model.get("gain5") or 0)

    def wall_attenuation_between(self, x1: float, y1: float, x2: float, y2: float) -> float:
        band = "6" if self.frequency_ghz >= 6 else ("5" if self.frequency_ghz >= 5 else "2.4")
        crossings: Dict[int, float] = {}
        for wall in self.walls:
            points = wall.get("points") or []
            if len(points) < 2:
                continue
            (wx1, wy1), (wx2, wy2) = points[0], points[1]
            t = segment_intersection_parameter(
                (x1, y1), (x2, y2), (float(wx1), float(wy1)), (float(wx2), float(wy2)),
            )
            if t is None:
                continue
            attenuation = wall.get("attenuation_db") or {}
            loss = float(attenuation.get(band) or attenuation.get("5") or attenuation.get("2.4") or 0)
            key = round(t * 100000)
            crossings[key] = max(crossings.get(key, 0.0), loss)
        return sum(crossings.values())

    def total_cost(self) -> float:
        return sum(float(self.ap_models[ap.model_index].get("cost", 100)) for ap in self.placed_aps)

    def ap_proximity_penalty(self) -> float:
        """Average 'how clustered' each AP is with its nearest neighbour, in [0, 1].

        For each AP we look at the distance to its closest other AP. Pairs sitting
        right on top of each other score ~1; pairs at or beyond comfortable_ap_spacing_m
        score 0. This is a soft signal (the hard floor stays min_ap_separation_m) that
        pushes the policy to spread APs out instead of crowding adjacent rooms.
        """
        if len(self.placed_aps) < 2:
            return 0.0
        spacing = max(1e-6, self.comfortable_ap_spacing_m)
        penalties = []
        for i, ap in enumerate(self.placed_aps):
            nearest = min(
                hypot(ap.x - other.x, ap.y - other.y)
                for j, other in enumerate(self.placed_aps)
                if j != i
            )
            penalties.append(max(0.0, 1.0 - nearest / spacing))
        return sum(penalties) / len(penalties)

    def candidate_action_mask(self) -> List[bool]:
        """Mask invalid candidate indices for the next placement step."""
        mask = []
        for index, candidate in enumerate(self.candidates):
            if candidate.get("_action_padding"):
                mask.append(False)
                continue
            if index in self.used_candidate_indices:
                mask.append(False)
                continue
            if self.candidate_room_limit_reached(candidate):
                mask.append(False)
                continue
            too_close = any(
                hypot(candidate["x"] - ap.x, candidate["y"] - ap.y) < self.min_ap_separation_m
                for ap in self.placed_aps
            )
            mask.append(not too_close)
        return mask

    def can_stop(self) -> bool:
        """Return whether a stop action should be available to the policy."""
        if not self.candidates:
            return True
        if not any(self.candidate_action_mask()):
            return True
        if len(self.placed_aps) >= self.max_ap_count:
            return True
        if not self.placed_aps:
            return False
        if len(self.placed_aps) < self.min_ap_count:
            return False
        return True

    def target_requirements_met(self, metrics: Dict[str, float]) -> bool:
        """Return whether the deployment already satisfies practical RF requirements."""
        max_weak_rooms = float(self.targets.get("max_weak_room_count", 1.0))
        return (
            metrics.get("sample_coverage_ratio", 0.0) >= self.target_sample_coverage
            and metrics.get("acceptable_sample_ratio", 0.0)
            >= float(self.targets.get("target_acceptable_coverage_ratio") or 0.97)
            and metrics.get("coverage_ratio", 0.0)
            >= float(self.targets.get("min_coverage_ratio") or 0.95)
            and metrics.get("capacity_ratio", 0.0)
            >= float(self.targets.get("min_capacity_ratio") or 0.95)
            and metrics.get("capacity_shortfall_room_count", 999.0)
            <= float(self.targets.get("max_capacity_shortfall_room_count", 0))
            and metrics.get("high_density_room_without_local_ap_count", 999.0) <= 0
            and metrics.get("weak_room_count", 999.0)
            <= max_weak_rooms
            and metrics.get("sinr_coverage_ratio", 0.0)
            >= float(self.targets.get("target_sinr_coverage_ratio") or 0.85)
            and metrics.get("floor_coverage_ratio", 0.0)
            >= float(self.targets.get("target_floor_coverage_ratio") or 0.97)
            and metrics.get("floor_sinr_coverage_ratio", 0.0)
            >= float(self.targets.get("target_floor_sinr_coverage_ratio") or 0.90)
        )

    def candidate_features(self) -> List[List[float]]:
        """Return per-candidate features used by the Gymnasium observation."""
        if not self.candidates:
            return []

        bounds = (self.scenario.get("floor_plan") or {}).get("bounds") or {}
        if bounds:
            min_x = float(bounds.get("min_x", 0.0))
            max_x = float(bounds.get("max_x", min_x + 1.0))
            min_y = float(bounds.get("min_y", 0.0))
            max_y = float(bounds.get("max_y", min_y + 1.0))
        else:
            xs = [float(candidate.get("x", 0.0)) for candidate in self.candidates]
            ys = [float(candidate.get("y", 0.0)) for candidate in self.candidates]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

        width = max(1e-6, max_x - min_x)
        height = max(1e-6, max_y - min_y)
        max_score = max(1.0, max(float(candidate.get("score", 0.0)) for candidate in self.candidates))
        room_priorities = {
            str(room.get("id")): room_effective_priority(room)
            for room in self.rooms
        }
        room_type_weights = {
            str(room.get("id")): float(room_type_profile(room.get("type")).get("candidate_score_multiplier", 1.0))
            for room in self.rooms
        }
        max_priority = max(1.0, max(room_priorities.values(), default=1.0))
        valid_mask = self.candidate_action_mask()

        features: List[List[float]] = []
        for index, candidate in enumerate(self.candidates):
            x = float(candidate.get("x", 0.0))
            y = float(candidate.get("y", 0.0))
            nearest_ap_distance = 1.0
            if self.placed_aps:
                nearest_ap_distance = min(hypot(x - ap.x, y - ap.y) for ap in self.placed_aps) / max(width, height)

            room_id = str(candidate.get("room_id"))
            features.append([
                (x - min_x) / width,
                (y - min_y) / height,
                float(candidate.get("score", 0.0)) / max_score,
                min(float(candidate.get("min_wall_distance_m", 0.0)), 10.0) / 10.0,
                1.0 if index in self.used_candidate_indices else 0.0,
                1.0 if valid_mask[index] else 0.0,
                min(1.0, nearest_ap_distance),
                room_priorities.get(room_id, 1.0) / max_priority,
                room_type_weights.get(room_id, 1.0),
            ])

        return features
