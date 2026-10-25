"""Training-only outcome, contribution and potential-difference rewards."""

from collections import Counter, defaultdict
import math
import numpy as np
from game_core import DEFUSE_REQUIRED_TICKS, PLANT_REQUIRED_TICKS, FACING_VECTORS
from grid_paths import distance_map
from .gc3_spatial_policy import parse_map, hint_fields, phase_settings
from .gc3_map_attacker_postplant import NEW_MAZE_STR as GUARD_MAP
from .gc3_map_defender_initial import DEFENDER_SETUP_POSITION_MAP_GC_STR as SETUP_MAP

# These weights change rewards only; they never choose or override an action.
REWARD_VERSION = "gc3-contribution-potential-v1"
WIN_REWARD = 10.0
LOSS_REWARD = -10.0
DAMAGE_DEALT_PER_HP = 0.005
DAMAGE_TAKEN_PER_HP = -0.005
KILL_REWARD = 0.5
DEATH_REWARD = -0.5
ASSIST_REWARD = 0.15
NEW_RECON_TARGET_REWARD = 0.15
NEW_SIGHT_TARGET_REWARD = 0.03
NEW_FLASH_TARGET_REWARD = 0.10
NEW_RAMP_TARGET_REWARD = 0.10
HEAL_PER_HP = 0.002
SMOKE_PROTECTION_REWARD = 0.10
SMOKE_PROTECTION_MAX_PER_CAST = 0.20
REJECTED_ABILITY_PENALTY = -0.03
DEPLOYMENT_POTENTIAL_WEIGHT = 0.15
WATCH_POTENTIAL_WEIGHT = 0.15
PREAIM_POTENTIAL_WEIGHT = 0.20
SPIKE_ACCESS_POTENTIAL_WEIGHT = 0.30
DEFUSE_PRESSURE_POTENTIAL_WEIGHT = 0.80
PLANT_PRESSURE_POTENTIAL_WEIGHT = 0.60
DISTANCE_SCALE = 12.0


def reward_settings():
    return {
        key: value
        for key, value in globals().items()
        if key == "REWARD_VERSION"
        or (key.isupper() and isinstance(value, (int, float)))
    }


def potential_difference(before, after, gamma, terminal):
    """Zero terminal potential prevents moving/turning cycles from farming reward."""
    return gamma * (0.0 if terminal else after) - before


def effective_hp(char):
    return max(0.0, float(char.hp)) + max(0.0, float(getattr(char, "shield_hp", 0.0)))


def alignment(char, point):
    dr, dc = point[0] - char.pos[0], point[1] - char.pos[1]
    length = math.hypot(dr, dc)
    if not length:
        return 0.0
    fx, fy = FACING_VECTORS.get(char.facing, (0.0, 0.0))
    # Engine diagonal vectors need not have unit length.
    norm = math.hypot(fx, fy)
    return float(max(0.0, (fx * dc + fy * dr) / (length * norm))) if norm else 0.0


class PhaseRewards:
    """Observe real game effects for reward only; keep privileged state out of policy inputs."""

    def __init__(self, game, phase, gamma):
        if phase not in ("search", "postplant", "preplant"):
            raise ValueError("Unknown reward phase")
        self.game, self.phase, self.gamma = game, phase, gamma
        self.team = "D" if phase == "search" else "A"
        self.own = {str(c.name): c for c in game.chars if c.team == self.team}
        self.enemies = {str(c.name): c for c in game.chars if c.team != self.team}
        self.seen_enemies = {
            name
            for name, c in self.enemies.items()
            if getattr(c, "los_revealed", False)
            or getattr(c, "reveal_remaining", 0) > 0
        }
        self.flash_targets, self.ramp_targets = set(), set()
        self.smoke_objects, self.smoke_pairs, self.smoke_totals = {}, set(), Counter()
        self.pending = defaultdict(Counter)
        self.damage_sources = defaultdict(Counter)
        if phase == "preplant":
            self.deployment = np.zeros(game.grid.shape, dtype=np.float32)
            self.watch_points, self.watch_distances = [], []
            self.spike_distances = None
        else:
            text = GUARD_MAP if phase == "postplant" else SETUP_MAP
            parsed = parse_map(text)
            radius = phase_settings(
                "postplant" if phase == "postplant" else "defender"
            )[3]
            self.deployment = hint_fields(text, radius)[0]
            self.watch_points = [tuple(map(int, p)) for p in np.argwhere(parsed == 6)]
            self.watch_distances = [
                distance_map(game.grid, point) for point in self.watch_points
            ]
            self.spike_distances = (
                distance_map(game.grid, tuple(game.planted_pos))
                if phase == "postplant"
                else None
            )

    def __enter__(self):
        # Patches affect this episode instance only; original engine methods run exactly once.
        self.original_contribution = self.game.analytics_tracker.record_contribution
        self.original_cast = self.game.execute_ai_ability
        self.game.analytics_tracker.record_contribution = self.contribution
        self.game.execute_ai_ability = self.cast
        return self

    def __exit__(self, *_):
        self.game.analytics_tracker.record_contribution = self.original_contribution
        self.game.execute_ai_ability = self.original_cast

    def contribution(self, owner, victim, tick, method):
        result = self.original_contribution(owner, victim, tick, method)
        if (
            owner is None
            or victim is None
            or str(owner.name) not in self.own
            or str(victim.name) not in self.enemies
        ):
            return result
        name, target = str(owner.name), str(victim.name)
        if method == "damage":
            self.damage_sources[target][name] += 1
        elif method in ("recon", "serenade") and target not in self.seen_enemies:
            self.pending[name]["recon_new_target"] += NEW_RECON_TARGET_REWARD
            self.seen_enemies.add(target)
        elif (
            method == "flash"
            and target not in self.flash_targets
            and self.before["enemies"][target]["blind"] <= 0
        ):
            self.pending[name]["flash_effect"] += NEW_FLASH_TARGET_REWARD
            self.flash_targets.add(target)
        elif method == "ramp" and target not in self.ramp_targets:
            self.pending[name]["ramp_effect"] += NEW_RAMP_TARGET_REWARD
            self.ramp_targets.add(target)
        return result

    def cast(self, owner, command):
        previous = {name: effective_hp(c) for name, c in self.own.items()}
        result = self.original_cast(owner, command)
        if str(owner.name) in self.own:
            if not result:
                self.pending[str(owner.name)][
                    "rejected_ability"
                ] += REJECTED_ABILITY_PENALTY
            elif command.get("ability") == "DANCE":
                healed = sum(
                    max(0.0, effective_hp(c) - previous[name])
                    for name, c in self.own.items()
                )
                self.pending[str(owner.name)]["healing"] += HEAL_PER_HP * healed
        return result

    def personal_potential(self, char):
        if not char.is_alive:
            return dict(deployment=0.0, watch=0.0, preaim=0.0, spike_access=0.0)
        origin = tuple(char.pos)
        watch = 0.0
        for point, distances in zip(self.watch_points, self.watch_distances):
            distance = int(distances[origin])
            if distance >= 0 and self.game.check_cell_line_of_sight(
                origin, point, block_smoke=False
            ):
                watch = max(
                    watch, alignment(char, point) / (1 + distance / DISTANCE_SCALE)
                )
        preaim = 0.0
        if not self.game.in_defender_setup_phase and not getattr(
            char, "blind_remaining", 0
        ):
            for enemy in self.enemies.values():
                # Reward an actual visible/revealed firing opportunity, never aim at an unseen enemy behind a wall.
                known = (
                    getattr(enemy, "los_revealed", False)
                    or getattr(enemy, "reveal_remaining", 0) > 0
                )
                if (
                    enemy.is_alive
                    and known
                    and self.game.check_shot_line_of_sight(char, enemy)
                ):
                    preaim = max(preaim, alignment(char, enemy.pos))
        distance = (
            int(self.spike_distances[origin])
            if self.spike_distances is not None
            else -1
        )
        return dict(
            deployment=DEPLOYMENT_POTENTIAL_WEIGHT * float(self.deployment[origin]),
            watch=WATCH_POTENTIAL_WEIGHT * watch,
            preaim=PREAIM_POTENTIAL_WEIGHT * preaim,
            spike_access=(
                SPIKE_ACCESS_POTENTIAL_WEIGHT / (1 + distance / DISTANCE_SCALE)
                if distance >= 0
                else 0.0
            ),
        )

    def objective_potential(self):
        if self.phase == "postplant":
            progress = max(
                (
                    float(c.defuse_timer) / DEFUSE_REQUIRED_TICKS
                    for c in self.enemies.values()
                    if c.is_alive
                ),
                default=0.0,
            )
            return {
                "defuse_pressure": -DEFUSE_PRESSURE_POTENTIAL_WEIGHT
                * min(1.0, progress)
            }
        progress = max(
            (
                float(c.plant_timer) / PLANT_REQUIRED_TICKS
                for c in self.enemies.values()
                if c.is_alive
            ),
            default=0.0,
        )
        return {"plant_pressure": -PLANT_PRESSURE_POTENTIAL_WEIGHT * min(1.0, progress)}

    def snapshot(self):
        def unit(c):
            stats = self.game.analytics_tracker.stats.get(str(c.name), {})
            return dict(
                hp=effective_hp(c),
                alive=bool(c.is_alive),
                kills=float(getattr(c, "kills", 0)),
                assists=float(stats.get("assists", 0)),
                blind=float(getattr(c, "blind_remaining", 0)),
            )

        return dict(
            own={name: unit(c) for name, c in self.own.items()},
            enemies={name: unit(c) for name, c in self.enemies.items()},
            potential={
                name: self.personal_potential(c) for name, c in self.own.items()
            },
            objective=self.objective_potential(),
        )

    def begin_tick(self):
        self.pending.clear()
        self.damage_sources.clear()
        self.before = self.snapshot()

    def smoke_protection(self):
        """Credit each own smoke at most twice for a genuinely blocked incoming firing line."""
        smokes = [s for s in self.game.smokes if s["remaining_ticks"] > 0]
        for smoke in smokes:
            owner = str(smoke.get("owner"))
            if owner not in self.own:
                continue
            identity = id(smoke)
            self.smoke_objects[identity] = (
                smoke  # Keep identity stable for this episode.
            )
            others = set().union(*(s["cells"] for s in smokes if s is not smoke))
            for ally in self.own.values():
                if not ally.is_alive:
                    continue
                for enemy in self.enemies.values():
                    pair = identity, str(ally.name), str(enemy.name)
                    if (
                        pair in self.smoke_pairs
                        or self.smoke_totals[identity] >= SMOKE_PROTECTION_MAX_PER_CAST
                    ):
                        continue
                    if (
                        not enemy.is_alive
                        or getattr(enemy, "blind_remaining", 0) > 0
                        or alignment(enemy, ally.pos) <= 0
                    ):
                        continue
                    if any(
                        getattr(enemy, field, 0) > 0
                        for field in (
                            "plant_timer",
                            "defuse_timer",
                            "orb_collect_timer",
                        )
                    ):
                        continue
                    if self.game.can_ignore_smoke_for_shot(enemy, ally):
                        continue
                    start, end = tuple(enemy.pos), tuple(ally.pos)
                    if not self.game.check_cell_line_of_sight(
                        start, end, block_smoke=False
                    ):
                        continue
                    line = self.game._line_cells(start, end)
                    if any(
                        c.is_alive
                        and c is not ally
                        and c is not enemy
                        and tuple(c.pos) in line[1:-1]
                        for c in self.game.chars
                    ):
                        continue
                    if self.game._smoke_allows_line(
                        line, others
                    ) and not self.game._smoke_allows_line(
                        line, others | smoke["cells"]
                    ):
                        amount = min(
                            SMOKE_PROTECTION_REWARD,
                            SMOKE_PROTECTION_MAX_PER_CAST - self.smoke_totals[identity],
                        )
                        self.pending[owner]["smoke_protection"] += amount
                        self.smoke_totals[identity] += amount
                        self.smoke_pairs.add(pair)

    def finish_tick(self, terminal, success):
        self.smoke_protection()
        after = self.snapshot()
        shared = {
            key: potential_difference(
                value, after["objective"][key], self.gamma, terminal
            )
            for key, value in self.before["objective"].items()
        }
        if terminal:
            shared["outcome"] = WIN_REWARD if success else LOSS_REWARD
        personal = {name: Counter(parts) for name, parts in self.pending.items()}
        # Nominal shot sizes are only attribution weights. Reward actual HP/shield loss, capped by available health.
        shot_damage, shot_count = defaultdict(Counter), defaultdict(Counter)
        for shot in self.game.last_shots or []:
            victim, owner = str(shot["target"].name), str(shot["shooter"].name)
            if victim in self.enemies and owner in self.own and shot["hit"]:
                shot_damage[victim][owner] += max(0.0, float(shot["damage"]))
                shot_count[victim][owner] += 1
        for victim in self.damage_sources.keys() | shot_damage.keys():
            sources = Counter(shot_damage[victim])
            for name, count in self.damage_sources[victim].items():
                # Ability damage has no amount in the engine's contribution event;
                # mixed sources use event counts for that part, rather than inventing damage.
                sources[name] += max(0.0, count - shot_count[victim][name])
            lost = max(
                0.0,
                self.before["enemies"][victim]["hp"] - after["enemies"][victim]["hp"],
            )
            total = sum(sources.values())
            if total:
                for name, weight in sources.items():
                    personal.setdefault(name, Counter())["damage_dealt"] += (
                        DAMAGE_DEALT_PER_HP * lost * weight / total
                    )
        for name, char in self.own.items():
            parts = personal.setdefault(name, Counter())
            before, current = self.before["own"][name], after["own"][name]
            parts["damage_taken"] += DAMAGE_TAKEN_PER_HP * max(
                0.0, before["hp"] - current["hp"]
            )
            parts["kill"] += KILL_REWARD * max(0.0, current["kills"] - before["kills"])
            parts["death"] += DEATH_REWARD * int(
                before["alive"] and not current["alive"]
            )
            parts["assist"] += ASSIST_REWARD * max(
                0.0, current["assists"] - before["assists"]
            )
            for key, value in self.before["potential"][name].items():
                parts[key] += potential_difference(
                    value, after["potential"][name][key], self.gamma, terminal
                )
        if not self.game.in_defender_setup_phase:
            for name, enemy in self.enemies.items():
                if (
                    enemy.is_alive
                    and getattr(enemy, "los_revealed", False)
                    and name not in self.seen_enemies
                ):
                    viewers = [
                        str(c.name)
                        for c in self.own.values()
                        if c.is_alive and self.game._can_reveal_by_sight(c, enemy)
                    ]
                    if viewers:
                        for viewer in viewers:
                            personal[viewer][
                                "new_sight"
                            ] += NEW_SIGHT_TARGET_REWARD / len(viewers)
                        self.seen_enemies.add(name)
        return shared, {name: dict(parts) for name, parts in personal.items()}
