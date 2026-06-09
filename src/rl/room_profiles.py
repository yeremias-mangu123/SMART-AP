"""Room-type profiles used by RL AP placement.

The UI stores each room's type, priority, client count, target RSSI, and area.
These profiles add domain defaults on top of those user-editable values so
different room types influence training in a predictable way.
"""

from __future__ import annotations

from typing import Any, Dict


ROOM_TYPE_PROFILES: Dict[str, Dict[str, float]] = {
    "office": {
        "default_priority": 1.0,
        "priority_multiplier": 1.0,
        "default_coverage_dbm": -67.0,
        "capacity_headroom_ratio": 1.2,
        "candidate_score_multiplier": 1.0,
        "clients_per_m2": 0.12,
    },
    "meeting": {
        "default_priority": 2.0,
        "priority_multiplier": 1.25,
        "default_coverage_dbm": -67.0,
        "capacity_headroom_ratio": 1.45,
        "candidate_score_multiplier": 1.2,
        "clients_per_m2": 0.55,
    },
    "classroom": {
        "default_priority": 2.0,
        "priority_multiplier": 1.2,
        "default_coverage_dbm": -67.0,
        "capacity_headroom_ratio": 1.4,
        "candidate_score_multiplier": 1.15,
        "clients_per_m2": 0.65,
    },
    "corridor": {
        "default_priority": 0.8,
        "priority_multiplier": 0.75,
        "default_coverage_dbm": -70.0,
        "capacity_headroom_ratio": 1.0,
        "candidate_score_multiplier": 0.75,
        "clients_per_m2": 0.03,
    },
    "lobby": {
        "default_priority": 1.5,
        "priority_multiplier": 1.1,
        "default_coverage_dbm": -68.0,
        "capacity_headroom_ratio": 1.25,
        "candidate_score_multiplier": 1.05,
        "clients_per_m2": 0.2,
    },
    "server": {
        "default_priority": 3.0,
        "priority_multiplier": 2.0,
        "default_coverage_dbm": -65.0,
        "capacity_headroom_ratio": 1.6,
        "candidate_score_multiplier": 1.6,
        "clients_per_m2": 0.05,
    },
    "storage": {
        "default_priority": 0.5,
        "priority_multiplier": 0.55,
        "default_coverage_dbm": -72.0,
        "capacity_headroom_ratio": 0.8,
        "candidate_score_multiplier": 0.55,
        "clients_per_m2": 0.02,
    },
    "restroom": {
        "default_priority": 0.5,
        "priority_multiplier": 0.5,
        "default_coverage_dbm": -72.0,
        "capacity_headroom_ratio": 0.7,
        "candidate_score_multiplier": 0.5,
        "clients_per_m2": 0.03,
    },
    "stairs": {
        "default_priority": 0.0,
        "priority_multiplier": 0.0,
        "default_coverage_dbm": -80.0,
        "capacity_headroom_ratio": 0.0,
        "candidate_score_multiplier": 0.0,
        "clients_per_m2": 0.0,
    },
    "other": {
        "default_priority": 1.0,
        "priority_multiplier": 1.0,
        "default_coverage_dbm": -67.0,
        "capacity_headroom_ratio": 1.2,
        "candidate_score_multiplier": 1.0,
        "clients_per_m2": 0.1,
    },
}


def normalize_room_type(room_type: Any) -> str:
    """Normalize room type strings from the UI."""
    normalized = str(room_type or "other").strip().lower().replace(" ", "_")
    aliases = {
        "meeting_room": "meeting",
        "server_room": "server",
        "toilet": "restroom",
        "bathroom": "restroom",
        "wc": "restroom",
    }
    return aliases.get(normalized, normalized if normalized in ROOM_TYPE_PROFILES else "other")


def room_type_profile(room_type: Any) -> Dict[str, float]:
    """Return the configured profile for a room type."""
    return ROOM_TYPE_PROFILES[normalize_room_type(room_type)]


def room_coverage_target(room: Dict[str, Any], global_default: float = -67.0) -> float:
    """Return the effective per-room RSSI target."""
    if room.get("coverage_target_dbm") is not None:
        return float(room.get("coverage_target_dbm"))
    profile = room_type_profile(room.get("type"))
    return float(profile.get("default_coverage_dbm", global_default))


def room_effective_priority(room: Dict[str, Any]) -> float:
    """Return priority after applying the room-type multiplier."""
    profile = room_type_profile(room.get("type"))
    base_priority = float(room.get("priority", profile.get("default_priority", 1.0)))
    multiplier = float(room.get("priority_multiplier", profile.get("priority_multiplier", 1.0)))
    return max(0.0, base_priority * multiplier)


def room_capacity_headroom(room: Dict[str, Any], global_default: float = 1.2) -> float:
    """Return room-specific capacity headroom."""
    if room.get("capacity_headroom_ratio") is not None:
        return float(room.get("capacity_headroom_ratio"))
    profile = room_type_profile(room.get("type"))
    return float(profile.get("capacity_headroom_ratio", global_default))


def room_candidate_score_multiplier(room: Dict[str, Any]) -> float:
    """Return candidate score multiplier for the room type."""
    profile = room_type_profile(room.get("type"))
    return float(room.get("candidate_score_multiplier", profile.get("candidate_score_multiplier", 1.0)))


def estimated_room_clients(room_type: Any, area_m2: float) -> int:
    """Estimate client demand when no explicit room demand is available."""
    density = float(room_type_profile(room_type).get("clients_per_m2", 0.1))
    return 0 if density <= 0 else max(1, int(round(max(0.0, float(area_m2)) * density)))


def room_requires_service(room: Dict[str, Any]) -> bool:
    """Return whether a room should contribute to AP placement requirements."""
    if room.get("excluded") or room.get("service_excluded"):
        return False
    clients = room.get("clients")
    return clients is None or float(clients) > 0.0


def room_is_high_density(room: Dict[str, Any]) -> bool:
    """Return whether a room needs local capacity, not only usable RSSI."""
    clients = float(room.get("clients") or 0.0)
    area_m2 = float(room.get("area_m2") or room.get("areaM2") or 0.0)
    room_type = str(room.get("type") or room.get("roomType") or "").lower()
    density = clients / max(1.0, area_m2)
    return clients >= 80 or (clients >= 50 and density >= 3.0) or room_type in {
        "auditorium", "classroom", "high_density", "event", "hall",
    }
