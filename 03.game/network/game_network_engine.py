"""Adapter between the authoritative game engine and network transport."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

GAME_ROOT = Path(__file__).resolve().parent.parent
if str(GAME_ROOT) not in sys.path:
    sys.path.insert(0, str(GAME_ROOT))

from iq_perception import IQPerceptionEngine
from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI
from game_core import FACING_VECTORS
try:
    from .game_network_controller import NetworkUserInputController
except ImportError:  # pragma: no cover - direct script execution
    from game_network_controller import NetworkUserInputController

NETWORK_MATCH_SEED = 20261025
NETWORK_TICK_TIME_MS = 1000


class NetworkGameAdapter:
    """Own a headless authoritative match and serialize private team views."""

    def __init__(self):
        random.seed(NETWORK_MATCH_SEED)
        np.random.seed(NETWORK_MATCH_SEED)
        self.team_a_attack_controller = NetworkUserInputController("A")
        self.team_a_defend_controller = NetworkUserInputController("D")
        self.team_d_attack_controller = NetworkUserInputController("A")
        self.team_d_defend_controller = NetworkUserInputController("D")
        self.team_a = DualRoleTeamAI(
            name="Network Team A",
            attacker_factory=lambda: self.team_a_attack_controller,
            defender_factory=lambda: self.team_a_defend_controller,
        )
        self.team_d = DualRoleTeamAI(
            name="Network Team D",
            attacker_factory=lambda: self.team_d_attack_controller,
            defender_factory=lambda: self.team_d_defend_controller,
        )
        self.game = VisualFPSBattle(
            NEW_MAZE_STR,
            self.team_a,
            self.team_d,
            headless=True,
            tick_time_ms=NETWORK_TICK_TIME_MS,
            attacker_team_name="Attackers",
            defender_team_name="Defenders",
        )
        self._pending_sound_events = []
        self._pending_visual_events = []
        self._last_visual_banner = None
        self._sound_event_sequence = 0
        self._attach_sound_event_sink()
        self.perception = IQPerceptionEngine()

    def _attach_sound_event_sink(self):
        self.game.network_sound_event_sink = self._queue_sound_event
        self.game.network_visual_event_sink = self._queue_visual_event

    def _queue_visual_event(self, event: dict) -> None:
        self._pending_visual_events.append(dict(event))

    def _queue_sound_event(self, filename: str, volume: float) -> None:
        self._sound_event_sequence += 1
        self._pending_sound_events.append({
            "id": self._sound_event_sequence,
            "file": str(filename),
            "volume": max(0.0, min(1.0, float(volume))),
        })

    def initial_observations(self) -> dict[str, dict]:
        observations = {team: self.observation(team) for team in ("A", "D")}
        self._pending_sound_events.clear()
        self._pending_visual_events.clear()
        return observations

    def configure_rosters(self, selections: dict[str, dict]) -> dict:
        """Rebuild the match from each connected team's selection screen."""
        selection_a = selections["A"]
        selection_d = selections["D"]
        random.seed(NETWORK_MATCH_SEED)
        np.random.seed(NETWORK_MATCH_SEED)
        self.game = VisualFPSBattle(
            NEW_MAZE_STR,
            self.team_a,
            self.team_d,
            headless=True,
            tick_time_ms=NETWORK_TICK_TIME_MS,
            attacker_roster=list(selection_a["roster"]),
            defender_roster=list(selection_d["roster"]),
            spike_holder_name=selection_a["spike_holder_name"],
            defender_spike_holder_name=selection_d["spike_holder_name"],
            attacker_igl_name=selection_a["igl_name"],
            defender_igl_name=selection_d["igl_name"],
            attacker_team_name=selection_a.get("team_name") or "Team A",
            defender_team_name=selection_d.get("team_name") or "Team D",
        )
        self._attach_sound_event_sink()
        self._last_visual_banner = None
        self._pending_visual_events.clear()
        self._pending_sound_events.clear()
        self.perception.clear_cache()
        return {
            "observations": self.initial_observations(),
            "rosters": {
                "A": list(selection_a["roster"]),
                "D": list(selection_d["roster"]),
            },
        }

    def submit_actions(self, identity: str, actions: list[dict]) -> None:
        role_controller = self._controller_for_identity(identity)
        movement_actions = [action for action in actions
                            if action["kind"] in {"move", "hold"}]
        role_controller.submit_actions(movement_actions, self.game.chars, self.game.grid)
        role = self._role_for_identity(identity)
        own_chars = {str(char.name): char for char in self.game.chars
                     if char.team == role}
        for action in actions:
            kind = action["kind"]
            if kind in {"move", "hold"}:
                continue
            char = own_chars.get(action["character"])
            if char is None:
                continue
            if kind == "ability":
                self.game.execute_ai_ability(char, action["payload"])
            elif kind == "ultimate":
                self.game.execute_ai_ultimate(char, action["payload"])
            elif kind == "facing":
                direction = action["direction"]
                if char.is_alive and direction in FACING_VECTORS:
                    role_controller.facings[char.name] = direction
                    if not char.facing_forced_this_tick:
                        char.facing = direction
            elif kind == "plant" and char.is_alive:
                char.is_planting = action["enabled"]
                char.plant_timer = 0
                if char.is_planting:
                    char.is_collecting_orb = False
                    char.orb_collect_timer = 0
                    char.collecting_orb_pos = None
            elif kind == "collect" and char.is_alive:
                char.is_collecting_orb = action["enabled"]
                if not char.is_collecting_orb:
                    char.orb_collect_timer = 0
                    char.collecting_orb_pos = None
                char.is_planting = False
                char.plant_timer = 0

    def advance_match(self) -> dict[str, dict]:
        if self.game.match_over:
            return self.initial_observations()
        before_omoko = {
            id(char): bool(getattr(char, "omoko_cornered_triggered", False))
            for char in self.game.chars
        }
        self.game.step_tick()
        from omoko_v1.ov1_ability_cornered_omoko import (
            OMOKO_CORNERED_NAME,
        )
        for char in self.game.chars:
            if (char.name == "おもこ"
                    and not before_omoko.get(id(char), False)
                    and bool(getattr(char, "omoko_cornered_triggered", False))):
                self.game._queue_personal_ability_effect(char, OMOKO_CORNERED_NAME)
        banner = getattr(self.game, "special_round_banner", None)
        signature = (self.game.current_round, banner.get("type"), banner.get("name")) if banner else None
        if banner and signature != self._last_visual_banner:
            self._queue_visual_event({key: value for key, value in banner.items()
                                      if key in {"type", "name", "base_name", "effect"}})
        self._last_visual_banner = signature
        return self.initial_observations()

    def _controller_for_identity(self, identity: str):
        is_attacking = (
            self.game.current_attacker_team_ai is self.team_a
        ) == (identity == "A")
        if identity == "A":
            return self.team_a_attack_controller if is_attacking else self.team_a_defend_controller
        return self.team_d_attack_controller if is_attacking else self.team_d_defend_controller

    def _role_for_identity(self, identity: str) -> str:
        is_attacking = (
            self.game.current_attacker_team_ai is self.team_a
        ) == (identity == "A")
        return "A" if is_attacking else "D"

    def observation(self, identity: str) -> dict:
        game = self.game
        role = self._role_for_identity(identity)
        own_chars = [char for char in game.chars if char.team == role]
        viewer = next((char for char in own_chars if char.is_alive), own_chars[0])
        # Setup ticks can share battle_tick=0; discard the perception cache so
        # the observation always reflects current setup positions.
        self.perception.clear_cache()
        view = self.perception.build_game_view(viewer=viewer, game=game)
        perceived = self.perception.build_perceived_state(
            viewer=viewer,
            game_state={
                "grid": game.grid,
                "chars": game.chars,
                "battle_tick": game.battle_tick,
                "is_planted": game.is_planted,
                "round_timer": game.round_timer,
                "detonate_timer": game.detonate_timer,
                "spike_pos": game.spike_pos,
                "planted_pos": game.planted_pos,
                "target_plant_pos": game.target_plant_pos,
            },
            game_view=view,
        )
        characters = []
        for real, proxy in zip(game.chars, perceived["chars"]):
            if real.team == role:
                characters.append({
                    "name": str(real.name), "team": role, "owner": identity,
                    "pos": [int(real.pos[0]), int(real.pos[1])],
                    "hp": int(real.hp), "alive": bool(real.is_alive),
                    "has_spike": bool(real.has_spike),
                    **_public_character_state(real),
                })
            elif (real.is_alive and bool(proxy.is_alive)
                    and getattr(proxy, "position_known", False)
                    and min(map(int, proxy.pos)) >= 0):
                characters.append({
                    "name": str(real.name), "team": real.team,
                    "owner": "D" if identity == "A" else "A",
                    "pos": [int(proxy.pos[0]), int(proxy.pos[1])],
                    "hp": int(proxy.hp), "alive": True,
                    "has_spike": False,
                })
        return {
            "team": identity,
            "role": role,
            "grid": [[int(cell) for cell in row] for row in game.grid],
            "sound_events": [dict(event) for event in self._pending_sound_events],
            "visual_events": [dict(event) for event in self._pending_visual_events],
            "available_orbs": [
                [int(row), int(col)] for row, col in sorted(game.available_orbs)
            ],
            "characters": characters,
            "effects": {
                # Smoke is a visible area effect. Projectile paths are sent
                # only for the viewer's team because they expose launch cells.
                "smokes": _safe_effects(game.smokes),
                "flash_projectiles": _safe_effects(game.flash_projectiles, role),
                "recon_projectiles": _safe_effects(game.recon_projectiles, role),
                "ash_projectiles": _safe_effects(game.ash_projectiles, role),
                "flash_bursts": _safe_effects(game.flash_bursts),
                "recon_bursts": _safe_effects(game.recon_bursts),
                "destruction_areas": _safe_effects(game.destruction_areas),
            },
            "tick": int(game.battle_tick),
            "round": int(game.current_round),
            "round_over": bool(game.round_over),
            "match_over": bool(game.match_over),
            "setup": bool(game.defender_setup_phase.active),
            "setup_ticks_remaining": int(game.defender_setup_phase.ticks_remaining),
            "score": {
                identity: int(game.attacker_wins if role == "A" else game.defender_wins),
                ("D" if identity == "A" else "A"): int(
                    game.defender_wins if role == "A" else game.attacker_wins
                ),
            },
            "is_planted": bool(game.is_planted),
            "spike_pos": _position(perceived.get("spike_pos")),
            "planted_pos": _position(perceived.get("planted_pos")),
            "target_plant_pos": _position(perceived.get("target_plant_pos")),
            "round_timer": int(perceived.get("round_timer", 0)),
            "detonate_timer": int(perceived.get("detonate_timer", 0)),
        }


def _position(value):
    if value is None:
        return None


def _public_character_state(char):
    """Serialize UI-relevant state for a player's own characters only."""
    fields = (
        "max_hp", "shield_hp", "max_shield_hp", "smoke_charges",
        "flash_charges", "recon_charges", "ramp_charges", "dance_charges",
        "ash_charges", "ultimate_name", "ultimate_points", "ultimate_cost",
        "facing", "is_planting", "plant_timer", "is_collecting_orb",
        "orb_collect_timer", "accuracy", "dodge_rate", "hs_rate", "iq",
        "reaction", "kills", "deaths", "round_kills", "blind_remaining",
        "electric_remaining", "ability_seal_remaining", "reveal_remaining",
        "defuse_timer", "erosion_curse", "fate_loom_remaining",
    )
    result = {}
    for field in fields:
        value = getattr(char, field, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[field] = value
    result["ability_name"] = getattr(char, "ability_name", "")
    return result


def _safe_effects(effects, own_team=None):
    result = []
    for effect in effects or ():
        if not isinstance(effect, dict):
            continue
        if own_team is not None and effect.get("team") != own_team:
            continue
        clean = {}
        for key, value in effect.items():
            if key in {"owner", "character", "source"}:
                continue
            if isinstance(value, (tuple, set, list)):
                clean[key] = _safe_position_list(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
        result.append(clean)
    return result


def _safe_position_list(value):
    result = []
    for item in value:
        if isinstance(item, (tuple, list, set)):
            result.append(_safe_position_list(item))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result.append(item)
    return result
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError, IndexError):
        return None

