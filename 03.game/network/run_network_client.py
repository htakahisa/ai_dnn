"""Network client using the same battle window and controls as run_game.py."""

from __future__ import annotations

import asyncio
import random
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path

import numpy as np

# This client is normally launched with ``network/`` as its working directory.
# Add the game root explicitly so it can import the same modules as run_game.py.
GAME_ROOT = Path(__file__).resolve().parent.parent
if str(GAME_ROOT) not in sys.path:
    sys.path.insert(0, str(GAME_ROOT))

from run_game import VisualFPSBattle, _build_team_ai
from map_data import NEW_MAZE_STR
from controllers import UserInputController
from game_core import Character, FACING_VECTORS
from character_stats import all_names
from battle_logic import _ensure_mixer_ready, pygame as pygame_audio
try:
    from .game_network_selection_ui import NetworkRosterSelectionWindow
except ImportError:  # pragma: no cover
    from game_network_selection_ui import NetworkRosterSelectionWindow

try:
    from .game_network_client import (
        NetworkMatchClient, SERVER_HOST, SERVER_PORT, REQUESTED_TEAM,
        get_private_lan_ipv4,
    )
except ImportError:  # pragma: no cover
    from game_network_client import (
        NetworkMatchClient, SERVER_HOST, SERVER_PORT, REQUESTED_TEAM,
        get_private_lan_ipv4,
    )

POLL_INTERVAL_MS = 40


class NetworkClientBattle(VisualFPSBattle):
    """Original run_game UI backed by the server's private observations."""

    def __init__(self, selection, host, port, requested_team):
        random.seed(20261025)
        np.random.seed(20261025)
        self.network_events = queue.Queue()
        self.selection = dict(selection)
        self.network_client = NetworkMatchClient(
            host, port, requested_team, selection=self.selection,
        )
        self.network_loop = None
        self.network_team = None
        self.network_ready = False
        self.network_observation = None
        self._tick_warning = None
        self._played_sound_event_ids = set()
        self._network_sound_cache = {}
        self.network_status = None
        self._network_send_lock = threading.Lock()
        own_roster = list(self.selection["roster"])
        filler_roster = [name for name in all_names() if name not in set(own_roster)][:5]
        if len(filler_roster) != 5:
            raise ValueError("The local character database does not contain five additional characters")
        local_attack_roster, local_defend_roster = (
            (filler_roster, own_roster) if requested_team == "D"
            else (own_roster, filler_roster)
        )
        super().__init__(
            NEW_MAZE_STR, _build_team_ai("user"), _build_team_ai("user"),
            attacker_roster=local_attack_roster,
            defender_roster=local_defend_roster,
            spike_holder_name=(filler_roster[0] if requested_team == "D"
                               else self.selection["spike_holder_name"]),
            defender_spike_holder_name=(self.selection["spike_holder_name"] if requested_team == "D"
                                        else filler_roster[0]),
            attacker_igl_name=(filler_roster[0] if requested_team == "D"
                               else self.selection["igl_name"]),
            defender_igl_name=(self.selection["igl_name"] if requested_team == "D"
                               else filler_roster[0]),
            attacker_team_name="Team A",
            defender_team_name="Team D",
        )
        self._network_all_characters = list(self.chars)
        self.record_replay = False
        self.root.title("Network Match | run_game")
        self.network_status.set("Connecting to match server...")
        self.root.after(POLL_INTERVAL_MS, self._poll_network)
        threading.Thread(target=self._network_thread, daemon=True).start()

    def _build_playback_controls(self):
        # Keep run_game's layout while removing controls that would simulate
        # locally and diverge from the server-authoritative match.
        bar = ttk.Frame(self.root, padding=5)
        bar.pack(side="top", fill="x", before=self.canvas)
        self.network_status = tk.StringVar(self.root, value="Connecting to match server...")
        ttk.Label(bar, textvariable=self.network_status).pack(side="left", padx=8)

    def get_user_controllers(self):
        if not isinstance(self.attacker_controller, UserInputController) or not isinstance(self.defender_controller, UserInputController):
            return []
        role = self._current_network_role()
        controller = self.attacker_controller if role == "A" else self.defender_controller
        return [(controller, role)]

    def _current_network_role(self):
        if self.network_observation and self.network_observation.get("role") in ("A", "D"):
            return self.network_observation["role"]
        if self.network_team in ("A", "D"):
            attacking = self.current_attacker_team_ai is self.team_a if hasattr(self, "team_a") else self.current_attacker_team_ai is self.initial_attacker_team_ai
            return "A" if (attacking == (self.network_team == "A")) else "D"
        return "A"

    def _controller_for_team(self, team):
        if team != self._current_network_role():
            return None
        return self.attacker_controller if team == "A" else self.defender_controller

    def is_visible_to_team(self, character, team):
        observation = self.network_observation
        return bool(observation and any(item.get("name") == str(character.name)
                                        for item in observation.get("characters", [])))

    def _network_thread(self):
        asyncio.run(self._network_loop())

    async def _network_loop(self):
        self.network_loop = asyncio.get_running_loop()
        try:
            self.network_events.put(await self.network_client.connect())
            while True:
                message = await self.network_client.receive()
                self.network_events.put(message)
                # Preserve a server rejection or peer-disconnect reason instead
                # of replacing it with the EOF message on the next read.
                if message.get("type") in {"error", "peer_disconnected"}:
                    break
        except Exception as exc:
            self.network_events.put({"type": "client_error", "message": str(exc)})

    def _emit_network_user_action(self, action):
        if not self.network_ready or self.network_loop is None:
            return True
        asyncio.run_coroutine_threadsafe(self.network_client.send_actions([action]), self.network_loop)
        return True

    def execute_ai_ability(self, owner, ability_action, *, during_battle=False):
        payload = dict(ability_action)
        if "target" in payload:
            payload["target"] = list(map(int, payload["target"]))
        return self._emit_network_user_action({"kind": "ability", "character": str(owner.name), "payload": payload})

    def execute_ai_ultimate(self, owner, ultimate_action, *, during_battle=False):
        payload = dict(ultimate_action)
        if "target" in payload:
            payload["target"] = list(map(int, payload["target"]))
        return self._emit_network_user_action({"kind": "ultimate", "character": str(owner.name), "payload": payload})

    def on_canvas_click(self, event):
        before = {}
        for controller, role in self.get_user_controllers():
            before.update(controller.targets)
        super().on_canvas_click(event)
        after = {}
        for controller, role in self.get_user_controllers():
            after.update(controller.targets)
        for name, target in after.items():
            if before.get(name) != target:
                self._emit_network_user_action({"kind": "move", "character": name, "target": list(target)})
        for name in before.keys() - after.keys():
            self._emit_network_user_action({"kind": "hold", "character": name})

    def on_canvas_right_click(self, event):
        role = self._current_network_role()
        controller = self._controller_for_team(role)
        super().on_canvas_right_click(event)
        if controller:
            for name, facing in list(controller.facings.items()):
                if facing in FACING_VECTORS:
                    self._emit_network_user_action({"kind": "facing", "character": name, "direction": facing})

    def _poll_network(self):
        changed = False
        while True:
            try:
                message = self.network_events.get_nowait()
            except queue.Empty:
                break
            kind = message.get("type")
            if kind == "joined":
                self.network_team = message.get("team")
                self.network_status.set(f"Connected as Team {self.network_team}; waiting for opponent")
            elif kind == "match_ready":
                self.network_ready = True
                self.network_status.set(f"Team {self.network_team} | Original run_game controls | server tick: 1 second")
            elif kind == "observation":
                self.network_observation = message.get("observation")
                if self.network_observation:
                    obs = self.network_observation
                    self.network_team = obs.get("team", self.network_team)
                    self._apply_observation(obs)
                    self._tick_warning = None
                    score = obs.get("score", {})
                    timer_label = (
                        f"Setup {int(obs.get('setup_ticks_remaining', 0))} ticks left"
                        if obs.get("setup") else
                        f"{int(obs.get('round_timer', 0))} ticks left"
                    )
                    self.network_status.set(
                        f"Team A {int(score.get('A', 0))} - {int(score.get('D', 0))} Team D"
                        f" | Round {obs['round']} | Tick {obs['tick']} | {timer_label}"
                    )
                    changed = True
            elif kind == "tick_warning":
                self._tick_warning = message.get("message", "A server tick was skipped")
                self.network_status.set("Network warning: " + self._tick_warning)
            elif kind == "match_over":
                self.network_ready = False
                self.network_status.set("Match over")
            elif kind in ("peer_disconnected", "client_error", "error"):
                self.network_ready = False
                self.network_status.set("Network: " + str(message.get("message", kind)))
        if changed:
            self.draw()
        banner = getattr(self, "_network_effect_banner", None)
        if banner is not None:
            elapsed = int((time.monotonic() - banner["started_at"]) * 1000)
            if elapsed >= banner["duration_ms"]:
                self._network_effect_banner = None
                self.special_round_banner = None
                self.personal_ability_effect = None
                self.draw()
            else:
                banner["value"]["animation_elapsed_ms"] = elapsed
                self.draw()
        self.root.after(POLL_INTERVAL_MS, self._poll_network)

    def _apply_observation(self, observation):
        self._play_network_sound_events(observation.get("sound_events", []))
        for event in observation.get("visual_events", []):
            effect = event.get("effect") if isinstance(event.get("effect"), dict) else {}
            try:
                duration_ms = max(1, int(effect.get("duration_ms", 2000)))
            except (TypeError, ValueError):
                duration_ms = 2000
            banner = {**event, "animation_elapsed_ms": 0}
            self._network_effect_banner = {
                "value": banner, "started_at": time.monotonic(),
                "duration_ms": duration_ms,
            }
            self.special_round_banner = banner if event.get("type") == "ACE" else None
            self.personal_ability_effect = banner if event.get("type") == "ABILITY" else None
        by_name = {str(char.name): char for char in self._network_all_characters}
        visible = []
        for item in observation.get("characters", []):
            char = by_name.get(item["name"])
            if char is None:
                team = item["team"]
                char = Character(
                    item["name"], team, item["pos"], "white",
                    "#c0392b" if team == "A" else "#27ae60",
                    has_spike=bool(item.get("has_spike", False)),
                )
                self._network_all_characters.append(char)
                by_name[item["name"]] = char
            char.team = item["team"]
            # Local placeholder characters may have been created on the
            # opposite side before the server assigned this client's role.
            char.bg_color = "#c0392b" if char.team == "A" else "#27ae60"
            char.pos = tuple(item["pos"])
            char.hp = int(item.get("hp", char.hp))
            char.is_alive = bool(item.get("alive", char.is_alive))
            char.has_spike = bool(item.get("has_spike", False))
            for field in ("max_hp", "shield_hp", "max_shield_hp", "smoke_charges",
                          "flash_charges", "recon_charges", "ramp_charges",
                          "dance_charges", "ash_charges", "ultimate_name",
                          "ultimate_points", "ultimate_cost", "facing", "is_planting",
                          "plant_timer", "is_collecting_orb", "orb_collect_timer",
                          "accuracy", "dodge_rate", "hs_rate", "iq", "reaction",
                          "kills", "deaths", "round_kills", "blind_remaining",
                          "electric_remaining", "ability_seal_remaining",
                          "reveal_remaining", "defuse_timer", "erosion_curse",
                          "fate_loom_remaining", "ability_name"):
                if field in item and hasattr(char, field):
                    try:
                        setattr(char, field, item[field])
                    except (AttributeError, TypeError):
                        pass
            # Hidden opponents are omitted by the server and therefore never drawn.
            visible.append(char)
        self.chars = visible
        self.grid = np.asarray(observation["grid"], dtype=np.int32)
        self.available_orbs = {
            tuple(map(int, pos)) for pos in observation.get("available_orbs", [])
        }
        self.battle_tick = int(observation.get("tick", self.battle_tick))
        self.current_round = int(observation.get("round", self.current_round))
        setup_phase = getattr(self, "defender_setup_phase", None)
        if setup_phase is not None:
            remaining = int(observation.get("setup_ticks_remaining", 0))
            setup_phase.state.active = bool(observation.get("setup", False))
            setup_phase.state.ticks_remaining = remaining
            setup_phase.state.ticks_elapsed = max(0, setup_phase.setup_ticks - remaining)
        self.round_over = bool(observation.get("round_over", self.round_over))
        self.match_over = bool(observation.get("match_over", self.match_over))
        self.round_timer = int(observation.get("round_timer", self.round_timer))
        self.detonate_timer = int(observation.get("detonate_timer", self.detonate_timer))
        self.is_planted = bool(observation.get("is_planted", False))
        self.spike_pos = tuple(observation["spike_pos"]) if observation.get("spike_pos") else None
        self.planted_pos = tuple(observation["planted_pos"]) if observation.get("planted_pos") else None
        effects = observation.get("effects", {})
        self.smokes = [{**item, "cells": {tuple(cell) for cell in item.get("cells", [])},
                        "center": tuple(item["center"]) if item.get("center") else None}
                       for item in effects.get("smokes", [])]
        self.flash_projectiles = [{**item, "path": [tuple(cell) for cell in item.get("path", [])]}
                                  for item in effects.get("flash_projectiles", [])]
        self.recon_projectiles = [{**item, "path": [tuple(cell) for cell in item.get("path", [])]}
                                  for item in effects.get("recon_projectiles", [])]
        self.ash_projectiles = [{**item, "path": [tuple(cell) for cell in item.get("path", [])]}
                                for item in effects.get("ash_projectiles", [])]
        self.flash_bursts = [{**item, "pos": tuple(item["pos"])}
                             for item in effects.get("flash_bursts", []) if item.get("pos")]
        self.recon_bursts = [{**item, "cells": {tuple(cell) for cell in item.get("cells", [])}}
                             for item in effects.get("recon_bursts", [])]
        self.destruction_areas = [
            {**item, "pos": tuple(item["pos"]),
             "cells": {tuple(cell) for cell in item.get("cells", [])}}
            for item in effects.get("destruction_areas", []) if item.get("pos")
        ]
        score = observation.get("score", {})
        self.attacker_wins = int(score.get("A", 0))
        self.defender_wins = int(score.get("D", 0))
        role = observation.get("role")
        self.active_user_team = role
        controllers = self._controller_for_team(role)
        if controllers and controllers.selected_char not in {str(c.name) for c in visible}:
            controllers.selected_char = None

    def _play_network_sound_events(self, events):
        if pygame_audio is None or not _ensure_mixer_ready():
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("id", ""))
            if not event_id or event_id in self._played_sound_event_ids:
                continue
            relative = Path(str(event.get("file", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "sound":
                continue
            path = GAME_ROOT / relative
            if not path.is_file():
                continue
            try:
                volume = max(0.0, min(1.0, float(event.get("volume", 1.0))))
                cache_key = (str(path), volume)
                sound = self._network_sound_cache.get(cache_key)
                if sound is None:
                    sound = pygame_audio.mixer.Sound(str(path))
                    sound.set_volume(volume)
                    self._network_sound_cache[cache_key] = sound
                sound.play()
                self._played_sound_event_ids.add(event_id)
            except Exception as exc:
                print(f"[NETWORK][SOUND_ERROR] {path}: {exc}", flush=True)

    def run(self, stop_event=None):
        self.draw()
        self.root.mainloop()


if __name__ == "__main__":
    def start_network_match(selection, server_host, server_port, requested_team):
        NetworkClientBattle(selection, server_host, server_port, requested_team).run()

    selector = NetworkRosterSelectionWindow(
        on_confirm=start_network_match,
        default_host=get_private_lan_ipv4() or SERVER_HOST,
        default_port=SERVER_PORT,
        default_team=REQUESTED_TEAM,
    )
    selector.run()
