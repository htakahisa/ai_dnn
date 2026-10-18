"""One paused round, driven by this chat through bounded JSON commands."""

from pathlib import Path
import contextlib
import io
import json
import random
import sys
import time

SEED = 20261018
ATTACKER_PRESET = "EG2023"
DEFENDER_PRESET = "Fnatic2023"
MAX_TICKS_PER_COMMAND = 10
MAX_TOTAL_STEPS = 220
COMMAND_WAIT_SECONDS = 300
MAX_JSON_BYTES = 5_000_000
SESSION_DIR = Path(__file__).resolve().parent / "logs" / "trial"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from controllers import BaseController, DefaultAttackerController, DefaultDefenderController
from iq_perception import IQPerceptionEngine
from map_data import NEW_MAZE_STR
from party_presets import get_preset
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Trial output exceeded its bounded JSON size")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def write_replay_html(public_observation, frames):
    template = Path(__file__).with_name("cp1_replay_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"grid": public_observation["observation"]["grid"], "frames": frames},
                         ensure_ascii=False).replace("<", "\\u003c")
    raw = template.replace("__REPLAY_DATA__", payload).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Trial HTML exceeded its bounded output size")
    (SESSION_DIR / "replay.html").write_bytes(raw)


def _position(value):
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def build_observation(char, state):
    """Serialize only IQ-filtered state for the manual chat trial."""
    allies, enemies = [], []
    for other in state.get("chars", ()):
        if not other.is_alive:
            continue
        pos = _position(other.pos)
        if other.team == char.team:
            allies.append({"name": str(other.name), "pos": pos, "hp": int(other.hp),
                           "has_spike": bool(getattr(other, "has_spike", False))})
        elif getattr(other, "position_known", False) and min(pos) >= 0:
            enemies.append({"name": str(other.name), "pos": pos, "hp": int(other.hp)})
    return {
        "team": char.team, "tick": int(state.get("battle_tick", 0)),
        "round_timer_ticks": int(state.get("round_timer", 0)),
        "is_planted": bool(state.get("is_planted", False)),
        "detonate_timer_ticks": int(state.get("detonate_timer", 0)),
        "spike_pos": _position(state.get("spike_pos")),
        "planted_pos": _position(state.get("planted_pos")),
        "target_plant_pos": _position(state.get("target_plant_pos")),
        "grid": [[int(cell) for cell in row] for row in state["grid"]],
        "allies": allies, "visible_enemies": enemies,
    }


class ChatController(BaseController):
    """Chat chooses destinations; existing rules execute the low-level actions."""

    def __init__(self):
        self.fallback = DefaultAttackerController()
        self.goals = {}

    def reset_round(self):
        self.goals = {}
        self.fallback.reset_round()

    def decide_move(self, char, state):
        r, c = map(int, char.pos)
        if char.has_spike and not state.get("is_planted") and state["grid"][r, c] == 2:
            return list(char.pos), "PLANT"
        goal = self.goals.get(str(char.name), "DEFAULT")
        if goal == "DEFAULT":
            return self.fallback.decide_move(char, state)
        ability = self.fallback._decide_ability(char, state)
        if ability:
            return list(char.pos), ability
        if goal == "HOLD":
            return list(char.pos)
        return self.move_towards_target(char.pos, goal, state["grid"],
                                       chars=state["chars"], moving_char=char)


def observation(game, sequence):
    allies = [char for char in game.chars if char.team == "A" and char.is_alive]
    state = {"sequence": sequence, "battle_tick": game.battle_tick,
             "round_over": game.round_over, "setup": game.defender_setup_phase.active,
             "score": {"A": game.attacker_wins, "D": game.defender_wins},
             "own_roster": [{"name": str(c.name), "alive": c.is_alive, "hp": c.hp}
                            for c in game.chars if c.team == "A"]}
    if allies:
        viewer = next((c for c in allies if c.name == game.attacker_igl_name), allies[0])
        engine = IQPerceptionEngine()
        view = engine.build_game_view(viewer=viewer, game=game)
        perceived = engine.build_perceived_state(viewer=viewer, game_state={
            "grid": game.grid, "chars": game.chars, "battle_tick": game.battle_tick,
            "is_planted": game.is_planted, "round_timer": game.round_timer,
            "detonate_timer": game.detonate_timer,
        }, game_view=view)
        state["observation"] = build_observation(view.perceived_character_for(viewer), perceived)
    return state


def main():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    command_path = SESSION_DIR / "command.json"
    if command_path.exists():
        command_path.unlink()
    random.seed(SEED)
    np.random.seed(SEED)
    controller = ChatController()
    attack, defense = get_preset(ATTACKER_PRESET), get_preset(DEFENDER_PRESET)
    with contextlib.redirect_stdout(io.StringIO()):
        game = VisualFPSBattle(
            NEW_MAZE_STR,
            DualRoleTeamAI("This chat", lambda: controller, DefaultDefenderController),
            _build_team_ai("default"), headless=True,
            attacker_roster=attack.players, defender_roster=defense.players,
            spike_holder_name=attack.spike_holder,
            attacker_igl_name=attack.igl, defender_igl_name=defense.igl,
            attacker_team_name="This chat (EG2023)",
            defender_team_name="Standard logic (Fnatic2023)",
            disable_side_swap=True, shield_abilities_enabled=False,
        )
    game.stop_after_round = True
    sequence = steps = 0
    trace = []
    initial_public = observation(game, sequence)
    while True:
        public = observation(game, sequence)
        write_json(SESSION_DIR / "observation.json", public)
        printable = dict(public)
        if sequence and "observation" in printable:
            printable["observation"] = {k: v for k, v in public["observation"].items() if k != "grid"}
        print(json.dumps(printable, ensure_ascii=True), flush=True)
        if game.round_over or game.match_over or steps >= MAX_TOTAL_STEPS:
            break
        deadline = time.monotonic() + COMMAND_WAIT_SECONDS
        command = None
        while time.monotonic() < deadline:
            if command_path.exists():
                if command_path.stat().st_size > 20_000:
                    raise ValueError("Trial command too large")
                candidate = json.loads(command_path.read_text(encoding="utf-8-sig"))
                if candidate.get("sequence") == sequence + 1:
                    command = candidate
                    break
            time.sleep(0.1)
        if command is None:
            raise TimeoutError("This chat did not supply another trial command")
        ticks = command.get("ticks", 1)
        if type(ticks) is not int or not 1 <= ticks <= MAX_TICKS_PER_COMMAND:
            raise ValueError("Invalid trial tick count")
        allowed_names = {str(c.name) for c in game.chars if c.team == "A"}
        goals = command.get("goals", {})
        for name, goal in goals.items():
            if name not in allowed_names:
                raise ValueError("Trial can only control its own team")
            if isinstance(goal, str) and goal in {"HOLD", "DEFAULT"}:
                continue
            if (not isinstance(goal, list) or len(goal) != 2
                or any(type(x) is not int for x in goal)
                or not controller._in_bounds(goal, game.grid) or game.grid[tuple(goal)] == 1):
                raise ValueError("Invalid trial destination")
        controller.goals = goals
        trace.append(command)
        for _ in range(ticks):
            with contextlib.redirect_stdout(io.StringIO()):
                advanced = game.step_tick()
            steps += 1
            if not advanced or game.round_over or steps >= MAX_TOTAL_STEPS:
                break
        sequence += 1
    result = {"controller": "This chat (manual commands)", "seed": SEED,
              "attackers": ATTACKER_PRESET, "defenders": DEFENDER_PRESET,
              "opponent_controller": "default", "steps": steps,
              "final": public, "commands": trace,
              "player_stats": {str(c.name): {"team": c.team, "kills": c.kills,
                                "deaths": c.deaths, "alive": c.is_alive} for c in game.chars}}
    write_json(SESSION_DIR / "result.json", result)
    # Full replay is an after-round artifact, never an input to chat decisions.
    write_json(SESSION_DIR / "replay.json", {"replay_frames": game.replay_frames})
    write_replay_html(initial_public, game.replay_frames)


if __name__ == "__main__":
    main()
