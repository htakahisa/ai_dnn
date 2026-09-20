from __future__ import annotations

import os
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CarryRoutePriorityConfig:
    """Carry-first reward policy for GC v18+.

    Facing remains a secondary signal. The primary objective is Carry route
    reliability: site entry, plant progress, no-entry, timeout and route-stall
    penalties.
    """

    facing_weight: float = 0.10
    carry_site_entry_bonus: float = 5.0
    plant_progress_bonus: float = 4.0
    site_hold_bonus: float = 2.0
    carry_no_entry_penalty: float = 4.0
    timeout_penalty: float = 3.0
    spike_drop_penalty: float = 2.5
    route_stall_penalty: float = 2.0
    team_collision_penalty: float = 1.5
    escort_support_bonus: float = 0.5
    guard_suppression_factor: float = 0.5

    def reward(
        self,
        *,
        carry_site_entry: bool,
        plant_progress: float,
        site_hold: bool,
        carry_no_entry: bool,
        timeout: bool,
        spike_drop: bool,
        route_stall: bool,
        team_collision: bool,
        escort_support: bool,
        guard_active: bool,
    ) -> float:
        reward = 0.0

        if carry_site_entry:
            reward += self.carry_site_entry_bonus
        if plant_progress > 0.0:
            reward += self.plant_progress_bonus * plant_progress
        if site_hold:
            reward += self.site_hold_bonus
        if escort_support:
            reward += self.escort_support_bonus

        if carry_no_entry:
            reward -= self.carry_no_entry_penalty
        if timeout:
            reward -= self.timeout_penalty
        if spike_drop:
            reward -= self.spike_drop_penalty
        if route_stall:
            reward -= self.route_stall_penalty
        if team_collision:
            reward -= self.team_collision_penalty

        if guard_active:
            reward *= self.guard_suppression_factor

        return reward


def current_carry_route_priority() -> CarryRoutePriorityConfig:
    """Read stage-specific carry objectives from environment overrides.

    The GC training runs intentionally set these values via PowerShell before the
    trainer launches, so the selected carry policy can prioritize site entry and
    plant progress over generic traffic movement.
    """
    mode = os.environ.get("GC_TRAINING_MODE", "").strip().lower()
    if mode and mode != "carry_route_priority":
        return DEFAULT_CARRY_ROUTE_PRIORITY
    return CarryRoutePriorityConfig(
        facing_weight=_float_env("GC_FACING_WEIGHT", 0.10),
        carry_site_entry_bonus=_float_env("GC_CARRY_SITE_ENTRY_BONUS", 5.0),
        plant_progress_bonus=_float_env("GC_PLANT_PROGRESS_BONUS", 4.0),
        site_hold_bonus=_float_env("GC_SITE_HOLD_BONUS", 2.0),
        carry_no_entry_penalty=_float_env("GC_CARRY_NO_ENTRY_PENALTY", 4.0),
        timeout_penalty=_float_env("GC_TIMEOUT_PENALTY", 3.0),
        spike_drop_penalty=_float_env("GC_SPIKE_DROP_PENALTY", 2.5),
        route_stall_penalty=_float_env("GC_ROUTE_STALL_PENALTY", 2.0),
        team_collision_penalty=_float_env("GC_TEAM_COLLISION_PENALTY", 1.5),
        escort_support_bonus=_float_env("GC_ESCORT_SUPPORT_BONUS", 0.5),
        guard_suppression_factor=_float_env("GC_GUARD_SUPPRESSION_FACTOR", 0.5),
    )


DEFAULT_CARRY_ROUTE_PRIORITY = CarryRoutePriorityConfig()
