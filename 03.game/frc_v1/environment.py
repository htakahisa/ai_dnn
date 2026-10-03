"""One-round rollouts using the real game, not a second combat simulator."""

import contextlib
import io
import random
from collections import deque
from dataclasses import dataclass
import numpy as np

from frc_v1 import ROSTER
from frc_v1.actions import FrcAction, TeamDecision, validate_action
from frc_v1.controller import FrcController
from frc_v1.model import CRITIC_SIZE

STAGES = ("threats", "support", "entry", "attack", "defense", "balemoon", "match")


def plant_distances(grid):
    """Shortest walkable distance to either plant site, ignoring current actors."""
    rows, columns = len(grid), len(grid[0])
    distances = np.full((rows, columns), -1, dtype=np.int16)
    pending = deque()
    for r in range(rows):
        for c in range(columns):
            if grid[r][c] == 2:
                distances[r, c] = 0
                pending.append((r, c))
    while pending:
        r, c = pending.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < columns and grid[nr][nc] != 1 and distances[nr, nc] < 0:
                distances[nr, nc] = distances[r, c] + 1
                pending.append((nr, nc))
    return distances


class ExternalActor:
    def act(self, observation, snapshot, belief):
        return TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))


def critic_observation(game, side):
    """Training-only state. It is never attached to an actor observation."""
    own = {str(c.base_name): c for c in game.chars if c.team == side}
    characters = [own[name] for name in ROSTER] + [c for c in game.chars if c.team != side]
    state = np.zeros(CRITIC_SIZE, np.float32)
    for i, c in enumerate(characters[:10]):
        state[i * 10:(i + 1) * 10] = (c.is_alive, c.team == side,
            c.pos[0] / max(1, game.height - 1), c.pos[1] / max(1, game.width - 1), c.hp / 100,
            c.max_hp / 100, c.ultimate_points / 10, c.blind_remaining / 256,
            c.reveal_remaining / 256, getattr(c, "life_contract_remaining", 0) / 10)
    return state


@dataclass(frozen=True)
class StepResult:
    observation: object
    reward: float
    terminated: bool
    metrics: dict


class FrcRoundEnvironment:
    def __init__(self, side, *, stage="match", seed=1, opponent="fnatic_v3", opponent_roster="Fnatic2023",
                 effects_mode="all", max_ticks=220):
        if side not in ("A", "D") or stage not in STAGES:
            raise ValueError("invalid FRC side/curriculum")
        self.side, self.stage, self.seed = side, stage, seed
        self.opponent, self.opponent_roster = opponent, opponent_roster
        self.effects_mode = effects_mode
        self.max_ticks = max_ticks
        self.episode = 0
        self.game = self.controller = None

    def reset(self):
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        seed = self.seed + self.episode * 997
        self.episode += 1
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        own = get_preset("Furina Classic")
        other = get_preset(self.opponent_roster)
        if other is None:
            raise ValueError("unknown opponent roster")
        self.controller = FrcController(self.side, actor=ExternalActor(), effects_mode=self.effects_mode)
        controlled = DualRoleTeamAI("FRC training", lambda: self.controller, lambda: self.controller)
        opponent = _build_team_ai(self.opponent)
        attacker, defender = (own, other) if self.side == "A" else (other, own)
        with contextlib.redirect_stdout(io.StringIO()):
            self.game = VisualFPSBattle(NEW_MAZE_STR,
                controlled if self.side == "A" else opponent, opponent if self.side == "A" else controlled,
                headless=True, attacker_roster=list(attacker.players), defender_roster=list(defender.players),
                spike_holder_name=attacker.spike_holder, defender_spike_holder_name=defender.spike_holder,
                attacker_igl_name=attacker.igl, defender_igl_name=defender.igl,
                attacker_team_name=attacker.name, defender_team_name=defender.name, disable_side_swap=True)
        self.game.stop_after_round = True
        self._plant_distances = plant_distances(self.game.grid)
        if self.stage != "match":
            self._configure_curriculum(random.Random(seed))
        self.steps = 0
        self.metrics = {"side": self.side, "stage": self.stage, "winner": None, "invalid_actions": 0,
            "plants": 0, "defuses": 0, "dance_casts": 0, "dance_effective_hp": 0.0,
            "ash_casts": 0, "balemoon_casts": 0, "balemoon_furina_deaths": 0,
            "lohen_first_contact": False, "first_contact_slot": None,
            "warning_exposures": 0, "contract_applications": 0, "overflow": 0}
        self.metrics.update(entry_progress=0.0, spike_progress=0.0, defense_progress=0.0)
        self._first_contact_recorded = False
        self._finished = False
        self.game._prepare_team_controllers_tick()
        return self.controller.observation

    def _configure_curriculum(self, rng):
        from game_core import NEON_WARNING_TICKS, TUNNEL_WARNING_TICKS, BALEMOON_WARNING_TICKS
        game = self.game
        game.defender_setup_phase.finish()
        own = {str(c.base_name): c for c in game.chars if c.team == self.side}
        allies = [own[name] for name in ROSTER]
        enemies = [c for c in game.chars if c.team != self.side]
        plant = list(zip(*np.where(game.grid == 2)))
        center = tuple(map(int, rng.choice(plant)))
        floor = [tuple(map(int, p)) for p in zip(*np.where(game.grid != 1))]
        nearby = sorted(floor, key=lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1]))
        selected = tuple(range(5))
        if self.stage == "threats":
            selected = ((self.episode - 1) % 5,)
        elif self.stage == "support":
            selected = ((0, 2), (2, 4), (1, 2, 3))[(self.episode - 1) % 3]
        elif self.stage == "entry":
            selected = (1, 2, 3, 4)
        if self.stage in ("threats", "support", "entry", "balemoon"):
            used = set()
            for slot, ally in enumerate(allies):
                ally.is_alive = slot in selected
                ally.has_spike = False
                if ally.is_alive:
                    pos = next(pos for pos in nearby[3:] if pos not in used)
                    ally.pos = list(pos)
                    used.add(pos)
            for index, enemy in enumerate(enemies):
                enemy.is_alive = index == 0 if self.stage != "balemoon" else index < 3
                enemy.has_spike = False
                if enemy.is_alive:
                    pos = next(pos for pos in nearby if pos not in used)
                    enemy.pos = list(pos)
                    used.add(pos)
            if self.side == "A":
                holder = allies[0] if allies[0].is_alive else allies[selected[0]]
                holder.has_spike = True
            else:
                enemies[0].has_spike = True
            game.target_plant_pos = center
        if self.stage == "threats":
            ally = allies[selected[0]]
            position = tuple(ally.pos)
            kind = ("NEON", "TUNNEL", "BALEMOON", "ASH", "FLASH", "RECON")[(self.episode - 1) % 6]
            owner = enemies[0]
            if kind in ("NEON", "TUNNEL", "BALEMOON"):
                cells = game._destruction_area_cells(position, radius=1 if kind == "TUNNEL" else 2)
                attr, ticks = {"NEON": ("neon_bursts", NEON_WARNING_TICKS),
                    "TUNNEL": ("tunnel_bursts", TUNNEL_WARNING_TICKS),
                    "BALEMOON": ("balemoon_warnings", BALEMOON_WARNING_TICKS)}[kind]
                getattr(game, attr).append({"pos": position, "cells": cells, "phase": "warning",
                    "remaining_ticks": rng.randint(0, ticks), "owner": owner.name, "team": owner.team})
            else:
                if kind == "ASH":
                    path = []
                    for pos in game._line_cells(tuple(owner.pos), position):
                        if game.grid[pos] == 1:
                            break
                        path.append(pos)
                else:
                    path = game._projectile_path(tuple(owner.pos), position)
                getattr(game, kind.lower() + "_projectiles").append({"path": path or [tuple(owner.pos)],
                    "progress": 0, "ticks_alive": 0, "owner": owner.name, "team": owner.team})
        if self.stage == "support":
            allies[2].hp = rng.randint(15, 65)
        if self.stage == "balemoon":
            allies[4].hp = rng.randint(15, 50)
            allies[4].ultimate_points = allies[4].ultimate_cost
            allies[0].dance_charges = rng.choice((0, 1, 3))
            allies[0].ultimate_points = rng.choice((0, 4))
            game.is_planted = rng.choice((False, True))
            if game.is_planted:
                game.planted_pos = center
                for c in game.chars:
                    c.has_spike = False
            elif self.side == "A":
                occupied = {tuple(c.pos) for c in game.chars if c.is_alive and c is not allies[0]}
                center = next(tuple(map(int, p)) for p in sorted(plant,
                    key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1])) if tuple(p) not in occupied)
                game.target_plant_pos = center
                allies[0].pos = list(center)
                allies[0].plant_timer = rng.randint(0, 3)
                allies[0].is_planting = allies[0].plant_timer > 0
        if self.stage == "defense" and self.side != "D":
            raise ValueError("defense curriculum needs side D")

    def critic_state(self):
        return critic_observation(self.game, self.side)

    def step(self, decision):
        if self._finished:
            raise RuntimeError("reset required after terminal round")
        controller, game = self.controller, self.game
        if len(decision.actions) != 5:
            raise ValueError("FRC rollout requires five actions")
        for slot, action in enumerate(decision.actions):
            validate_action(controller.snapshot, controller.observation.masks, slot, action)
        before = controller.snapshot
        controller.decision = decision
        in_setup = game.defender_setup_phase.active
        if in_setup:
            game._run_defender_setup_tick()
        else:
            game._prepare_team_controllers_tick()
            game._build_occupancy_counts()
            try:
                for c in game._move_order():
                    if c.is_alive:
                        game.move_character(c)
            finally:
                game._clear_occupancy_counts()
            game.process_battle()
            game._advance_combo_announcement()
        self.steps += 1
        terminated = bool(game.round_over)
        if self.steps >= self.max_ticks and not terminated:
            raise RuntimeError("real FRC round exceeded max_ticks; no synthetic victory assigned")
        game._prepare_team_controllers_tick()
        after = controller.snapshot
        reward = 0.0 if in_setup else -0.001
        if self.side == "A" and self.stage in ("attack", "match") and not in_setup and not before.is_planted:
            entry_before, entry_after = before.allies[2], after.allies[2]
            if entry_before.alive and entry_after.alive:
                delta = int(self._plant_distances[entry_before.position]) - int(
                    self._plant_distances[entry_after.position])
                self.metrics["entry_progress"] += delta
                reward += 0.01 * delta
            for old, new in zip(before.allies, after.allies):
                if old.has_spike and old.alive and new.alive and new.has_spike:
                    delta = int(self._plant_distances[old.position]) - int(
                        self._plant_distances[new.position])
                    self.metrics["spike_progress"] += delta
                    reward += 0.01 * delta
                    break
        if self.side == "D" and self.stage in ("defense", "match") and not in_setup and not before.is_planted:
            # Reward leaving spawn for a defendable site, then let the round
            # outcome and combat determine the final positions.
            for old, new in zip(before.allies, after.allies):
                if old.alive and new.alive:
                    old_distance = int(self._plant_distances[old.position])
                    new_distance = int(self._plant_distances[new.position])
                    delta = max(0, old_distance - 3) - max(0, new_distance - 3)
                    self.metrics["defense_progress"] += delta
                    reward += 0.003 * delta
        if not before.is_planted and after.is_planted:
            self.metrics["plants"] += 1
            reward += 0.2 if self.side == "A" else -0.2
        if game.is_defused:
            self.metrics["defuses"] = 1
        for slot, action in enumerate(decision.actions):
            a, b = before.allies[slot], after.allies[slot]
            if action.kind == "ABILITY" and b.charges < a.charges:
                if slot == 0:
                    target = action.ally_slot
                    healed = max(0.0, after.allies[target].hp - before.allies[target].hp)
                    self.metrics["dance_casts"] += 1
                    self.metrics["dance_effective_hp"] += healed
                    reward += 0.02 * min(50, healed) / 50
                elif slot == 4:
                    self.metrics["ash_casts"] += 1
            if slot == 4 and action.kind == "ULTIMATE" and b.points < a.points:
                self.metrics["balemoon_casts"] += 1
            if a.contract == 0 and b.contract > 0:
                self.metrics["contract_applications"] += 1
        if before.allies[0].alive and not after.allies[0].alive and any(e.kind == "BALEMOON" for e in before.effects):
            self.metrics["balemoon_furina_deaths"] += 1
        danger = {pos for e in after.effects if e.phase == "warning" for pos in e.cells}
        self.metrics["warning_exposures"] += sum(a.alive and a.position in danger for a in after.allies)
        self.metrics["overflow"] += controller.observation.overflow
        if not self._first_contact_recorded and not in_setup:
            participants = {ROSTER.index(str(c.base_name)) for pair in game.last_engagements for c in pair
                            if c.team == self.side}
            if participants:
                self.metrics["first_contact_slots"] = sorted(participants)
                self.metrics["first_contact_slot"] = min(participants)
                self.metrics["lohen_first_contact"] = 2 in participants
                self._first_contact_recorded = True
                if self.side == "A" and self.stage in ("entry", "attack", "match") and 2 in participants:
                    reward += 0.01  # one event per round, much smaller than a win.
        if terminated:
            winner = "A" if game.attacker_wins else "D"
            self.metrics["winner"] = winner
            self.metrics["steps"] = self.steps
            reward += 1.0 if winner == self.side else -1.0
        self._finished = terminated
        return StepResult(controller.observation, reward, terminated, dict(self.metrics))
