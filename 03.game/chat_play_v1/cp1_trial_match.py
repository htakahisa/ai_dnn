"""A paused match, driven by this chat through bounded JSON commands."""

from pathlib import Path
import contextlib
from datetime import datetime
import io
import json
import random
import sys
import time
import tkinter as tk
from tkinter import ttk

SEED = 20261018
DEFAULT_ATTACKER_PRESET = "EG2023"
DEFAULT_DEFENDER_PRESET = "Fnatic2023"
DEFAULT_OPPONENT_CONTROLLER = "Fnatic v1"
MAX_TICKS_PER_COMMAND = 10
MAX_TOTAL_STEPS = 5_000
COMMAND_WAIT_SECONDS = 300
MAX_JSON_BYTES = 50_000_000
FIRST_TO_ROUNDS = tuple(range(2, 14))
DEFAULT_FIRST_TO_ROUNDS = 13
SESSION_DIR = Path(__file__).resolve().parent / "logs" / "trial"
COMPETITION_RESULTS_DIR = Path(__file__).resolve().parent.parent / "competition_results"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from controllers import BaseController, DefaultAttackerController, DefaultDefenderController
from iq_perception import IQPerceptionEngine
from map_data import NEW_MAZE_STR
from party_presets import all_preset_names, get_preset
from roster_select import TEAM_AI_OPTIONS
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Trial output exceeded its bounded JSON size")
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(raw)
    temporary.replace(path)


def write_analytics_result(result, replay_frames):
    """Save the manual match in the existing analytics/replay input format."""
    team1 = f"This chat ({result['attackers']})"
    team2 = f"Standard logic ({result['defenders']})"
    score1 = int(result["final"]["score"]["chat"])
    score2 = int(result["final"]["score"]["opponent"])
    map_result = {
        "number": 1,
        "seed": int(result["seed"]),
        "team1": team1,
        "team2": team2,
        "score1": score1,
        "score2": score2,
        "winner": team1 if score1 > score2 else team2,
        "mvp1": "",
        "mvp2": "",
        "initial_attacker": team1,
        "overtime": bool(result["overtime"]),
        "player_stats": [
            {"name": name, "team": stats["team"],
             "kills": int(stats["kills"]), "deaths": int(stats["deaths"])}
            for name, stats in result["player_stats"].items()
        ],
        "total_rounds": len(result["rounds"]),
        "gunfights": [],
        "assist_events": [],
        "cover_events": [],
        "attacker_side_stats": {},
        "defender_side_stats": {},
        "round_records": result["tracker_rounds"],
        "replay_frames": replay_frames,
    }
    analytics_result = {
        "mode": "manual_chat_trial",
        "seed_mode": "fixed",
        "base_seed": int(result["seed"]),
        "team_controllers": {
            team1: "manual_chat",
            team2: result.get("opponent_controller_key", "default"),
        },
        "rating_enabled": False,
        "team1": team1,
        "team2": team2,
        "maps_to_win": 1,
        "first_to_rounds": int(result["first_to"]),
        "team1_wins": int(score1 > score2),
        "team2_wins": int(score2 > score1),
        "winner": team1 if score1 > score2 else team2,
        "loser": team2 if score1 > score2 else team1,
        "maps": [map_result],
        "total_rounds": len(result["rounds"]),
        "round_records": result["rounds"],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = COMPETITION_RESULTS_DIR / (
        f"series_This_chat_vs_Standard_logic_{score1}-{score2}_{timestamp}.json"
    )
    write_json(output_path, analytics_result)
    return output_path


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
        self.defender_fallback = DefaultDefenderController()
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
            fallback = self.fallback if char.team == "A" else self.defender_fallback
            return fallback.decide_move(char, state)
        ability = self.fallback._decide_ability(char, state)
        if ability:
            return list(char.pos), ability
        if goal == "HOLD":
            return list(char.pos)
        return self.move_towards_target(char.pos, goal, state["grid"],
                                       chars=state["chars"], moving_char=char)


def observation(game, sequence, own_names, own_igl, chat_team):
    own_names = set(map(str, own_names))
    allies = [char for char in game.chars if str(char.name) in own_names and char.is_alive]
    chat_side = "A" if game.current_attacker_team_ai is chat_team else "D"
    chat_score = game.attacker_wins if chat_side == "A" else game.defender_wins
    opponent_score = game.defender_wins if chat_side == "A" else game.attacker_wins
    state = {"sequence": sequence, "battle_tick": game.battle_tick,
             "round_over": game.round_over, "setup": game.defender_setup_phase.active,
             "round": game.current_round,
             "own_side": chat_side,
             "score": {"chat": chat_score, "opponent": opponent_score,
                       "A": game.attacker_wins, "D": game.defender_wins},
             "own_roster": [{"name": str(c.name), "alive": c.is_alive, "hp": c.hp}
                            for c in game.chars if str(c.name) in own_names]}
    if allies:
        viewer = next((c for c in allies if str(c.name) == str(own_igl)), allies[0])
        engine = IQPerceptionEngine()
        view = engine.build_game_view(viewer=viewer, game=game)
        perceived = engine.build_perceived_state(viewer=viewer, game_state={
            "grid": game.grid, "chars": game.chars, "battle_tick": game.battle_tick,
            "is_planted": game.is_planted, "round_timer": game.round_timer,
            "detonate_timer": game.detonate_timer,
        }, game_view=view)
        state["observation"] = build_observation(view.perceived_character_for(viewer), perceived)
    return state


def progress_text(game, chat_team):
    setup = game.defender_setup_phase
    if setup.active:
        total = setup.ticks_elapsed + setup.ticks_remaining
        phase = f"Setup {setup.ticks_elapsed}/{total}"
    else:
        phase = f"Battle tick {game.battle_tick}"
    chat_side = "A" if game.current_attacker_team_ai is chat_team else "D"
    chat_score = game.attacker_wins if chat_side == "A" else game.defender_wins
    opponent_score = game.defender_wins if chat_side == "A" else game.attacker_wins
    return f"Round {game.current_round} | {phase} | Chat {chat_score}-{opponent_score} | Chat side {chat_side}"


def select_match_setup():
    controller_options = {
        label: key for label, key in TEAM_AI_OPTIONS.items() if key != "user"
    }
    selected = {}
    root = tk.Tk()
    root.title("ChatGPT manual match setup")
    root.resizable(False, False)
    body = ttk.Frame(root, padding=18)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Match setup", font=("Segoe UI", 15, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
    )
    ttk.Label(body, text="Attacker team (this ChatGPT chat)").grid(
        row=1, column=0, sticky="w", padx=(0, 16), pady=5
    )
    attacker_var = tk.StringVar(value=DEFAULT_ATTACKER_PRESET)
    ttk.Combobox(
        body, textvariable=attacker_var, values=all_preset_names(),
        state="readonly", width=32,
    ).grid(row=1, column=1, sticky="ew", pady=5)
    ttk.Label(body, text="Defender team").grid(
        row=2, column=0, sticky="w", padx=(0, 16), pady=5
    )
    defender_var = tk.StringVar(value=DEFAULT_DEFENDER_PRESET)
    ttk.Combobox(
        body, textvariable=defender_var, values=all_preset_names(),
        state="readonly", width=32,
    ).grid(row=2, column=1, sticky="ew", pady=5)
    ttk.Label(body, text="Defender AI").grid(
        row=3, column=0, sticky="w", padx=(0, 16), pady=5
    )
    default_controller = (
        DEFAULT_OPPONENT_CONTROLLER
        if DEFAULT_OPPONENT_CONTROLLER in controller_options
        else next(iter(controller_options))
    )
    controller_var = tk.StringVar(value=default_controller)
    ttk.Combobox(
        body, textvariable=controller_var, values=list(controller_options),
        state="readonly", width=32,
    ).grid(row=3, column=1, sticky="ew", pady=5)
    ttk.Label(body, text="Rounds to win first").grid(
        row=4, column=0, sticky="w", padx=(0, 16), pady=5
    )
    first_to_var = tk.StringVar(value=str(DEFAULT_FIRST_TO_ROUNDS))
    ttk.Combobox(
        body, textvariable=first_to_var,
        values=[str(value) for value in FIRST_TO_ROUNDS],
        state="readonly", width=8,
    ).grid(row=4, column=1, sticky="w", pady=5)
    ttk.Label(
        body,
        text=(
            "先取ラウンド数を選択します。先取数 - 1 ラウンド終了後に攻守を交代します。\n"
            "同点の場合は延長戦に進み、2ラウンド差がつくまで続きます。"
        ),

        justify="left",
    ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 12))
    error_var = tk.StringVar(value="")
    ttk.Label(body, textvariable=error_var, foreground="#b91c1c").grid(
        row=6, column=0, columnspan=2, sticky="w"
    )

    def start_match():
        try:
            attacker = get_preset(attacker_var.get())
            defender = get_preset(defender_var.get())
            duplicates = set(attacker.players) & set(defender.players)
            if duplicates:
                raise ValueError(
                    "同じキャラクターが両チームにいます: "
                    + ", ".join(sorted(duplicates))
                )
            selected.update({
                "attacker_preset": attacker_var.get(),
                "defender_preset": defender_var.get(),
                "opponent_controller_label": controller_var.get(),
                "opponent_controller_key": controller_options[controller_var.get()],
                "first_to": int(first_to_var.get()),
            })
        except (KeyError, ValueError) as exc:
            error_var.set(str(exc))
            return
        root.quit()

    def cancel_setup():
        selected.clear()
        root.destroy()

    ttk.Button(body, text="Start match", command=start_match).grid(
        row=7, column=0, columnspan=2, sticky="e"
    )
    root.protocol("WM_DELETE_WINDOW", cancel_setup)
    root.mainloop()
    return (selected, root) if selected else (None, None)


def show_progress_view(root, match_setup, attack, defense):
    for widget in root.winfo_children():
        widget.destroy()
    root.title("ChatGPT manual match - progress")
    root.geometry("600x190")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", root.iconify)
    body = ttk.Frame(root, padding=18)
    body.pack(fill="both", expand=True)
    ttk.Label(
        body,
        text=(
            f"ChatGPT (this chat) / {attack.name}  vs  "
            f"{match_setup['opponent_controller_label']} / {defense.name} "
            f"(first to {match_setup['first_to']})"
        ),
        font=("Segoe UI", 11, "bold"),
        wraplength=560,
    ).pack(anchor="w", pady=(0, 14), fill="x")
    progress_var = tk.StringVar(value="Round 1 | Preparing")
    ttk.Label(body, textvariable=progress_var, font=("Consolas", 12)).pack(
        anchor="w", pady=4
    )
    status_var = tk.StringVar(value="試合を準備しています。")
    ttk.Label(body, textvariable=status_var, wraplength=560).pack(
        anchor="w", pady=4
    )
    close_button = ttk.Button(body, text="Close", command=root.destroy)
    return progress_var, status_var, close_button


def main():
    match_setup, root = select_match_setup()
    if match_setup is None:
        print("Match setup was cancelled.", flush=True)
        return

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    command_path = SESSION_DIR / "command.json"
    if command_path.exists():
        command_path.unlink()
    random.seed(SEED)
    np.random.seed(SEED)
    controller = ChatController()
    attack = get_preset(match_setup["attacker_preset"])
    defense = get_preset(match_setup["defender_preset"])
    progress_var, status_var, close_button = show_progress_view(
        root, match_setup, attack, defense
    )
    chat_team = DualRoleTeamAI("This chat", lambda: controller, lambda: controller)
    opponent_team = _build_team_ai(match_setup["opponent_controller_key"])
    print(
        f"Starting first-to-{match_setup['first_to']} match: "
        f"{attack.name} vs {match_setup['opponent_controller_label']} / {defense.name}",
        flush=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        game = VisualFPSBattle(
            NEW_MAZE_STR,
            chat_team,
            opponent_team,
            headless=True,
            attacker_roster=attack.players,
            defender_roster=defense.players,
            spike_holder_name=attack.spike_holder,
            attacker_igl_name=attack.igl,
            defender_igl_name=defense.igl,
            attacker_team_name=f"ChatGPT (this chat) - {attack.name}",
            defender_team_name=(
                f"{match_setup['opponent_controller_label']} - {defense.name}"
            ),
            disable_side_swap=False,
            shield_abilities_enabled=False,
        )
    game.winning_rounds = match_setup["first_to"]
    game.side_swap_after_rounds = match_setup["first_to"] - 1
    game.stop_after_round = True

    sequence = 0
    steps = 0
    trace = []
    rounds = []
    own_names = set(map(str, attack.players))

    def update_progress(status):
        progress_var.set(progress_text(game, chat_team))
        status_var.set(status)
        root.update_idletasks()
        root.update()

    while not game.match_over and steps < MAX_TOTAL_STEPS:
        round_number = game.current_round
        chat_side = "A" if game.current_attacker_team_ai is chat_team else "D"
        previous_chat_score = game.attacker_wins if chat_side == "A" else game.defender_wins
        previous_opponent_score = game.defender_wins if chat_side == "A" else game.attacker_wins

        while not game.round_over and not game.match_over and steps < MAX_TOTAL_STEPS:
            public = observation(game, sequence, own_names, attack.igl, chat_team)
            write_json(SESSION_DIR / "observation.json", public)
            printable = dict(public)
            if sequence and "observation" in printable:
                printable["observation"] = {
                    key: value
                    for key, value in public["observation"].items()
                    if key != "grid"
                }
            update_progress("Preparing the next chat command.")
            print(progress_text(game, chat_team), flush=True)
            print(json.dumps(printable, ensure_ascii=True), flush=True)

            deadline = time.monotonic() + COMMAND_WAIT_SECONDS
            command = None
            waiting_status = f"指示待ち: 次のコマンド sequence={sequence + 1}"
            chat_prompt = 'チャット対戦する場合は「Chat対戦お願いします」と送ってください。'
            waiting_message = waiting_status + chr(10) + chat_prompt
            update_progress(waiting_message)
            print(waiting_status, flush=True)
            print(chat_prompt, flush=True)
            while time.monotonic() < deadline:
                if command_path.exists():
                    if command_path.stat().st_size > 20_000:
                        raise ValueError("Trial command too large")
                    candidate = json.loads(command_path.read_text(encoding="utf-8-sig"))
                    if candidate.get("sequence") == sequence + 1:
                        command = candidate
                        break
                update_progress(waiting_message)
                time.sleep(0.1)
            if command is None:
                raise TimeoutError("This chat did not supply another trial command")

            ticks = command.get("ticks", 1)
            if type(ticks) is not int or not 1 <= ticks <= MAX_TICKS_PER_COMMAND:
                raise ValueError("Invalid trial tick count")
            goals = command.get("goals", {})
            for name, goal in goals.items():
                if name not in own_names:
                    raise ValueError("Trial can only control its own team")
                if isinstance(goal, str) and goal in {"HOLD", "DEFAULT"}:
                    continue
                if (
                    not isinstance(goal, list)
                    or len(goal) != 2
                    or any(type(value) is not int for value in goal)
                    or not controller._in_bounds(goal, game.grid)
                    or game.grid[tuple(goal)] == 1
                ):
                    raise ValueError("Invalid trial destination")

            controller.goals = goals
            trace.append({"round": round_number, **command})
            for _ in range(ticks):
                with contextlib.redirect_stdout(io.StringIO()):
                    advanced = game.step_tick()
                steps += 1
                update_progress("The match is progressing.")
                print(progress_text(game, chat_team), end="", flush=True)
                if not advanced or game.round_over or steps >= MAX_TOTAL_STEPS:
                    break
            print(flush=True)
            sequence += 1

        final_chat_side = "A" if game.current_attacker_team_ai is chat_team else "D"
        chat_score = game.attacker_wins if final_chat_side == "A" else game.defender_wins
        opponent_score = game.defender_wins if final_chat_side == "A" else game.attacker_wins
        rounds.append({
            "round": round_number,
            "chat_side": chat_side,
            "winner": (
                "chat" if chat_score > previous_chat_score
                else "opponent" if opponent_score > previous_opponent_score
                else "draw"
            ),
            "chat_score": chat_score,
            "opponent_score": opponent_score,
        })
        if game.match_over or steps >= MAX_TOTAL_STEPS:
            break

        game.current_round += 1
        with contextlib.redirect_stdout(io.StringIO()):
            game.init_round()

    public = observation(game, sequence, own_names, attack.igl, chat_team)
    result = {
        "controller": "This chat (manual commands)",
        "seed": SEED,
        "attackers": attack.name,
        "defenders": defense.name,
        "opponent_controller": match_setup["opponent_controller_label"],
        "opponent_controller_key": match_setup["opponent_controller_key"],
        "first_to": match_setup["first_to"],
        "overtime": game.overtime,
        "steps": steps,
        "final": public,
        "rounds": rounds,
        "tracker_rounds": game.analytics_tracker.round_records,
        "commands": trace,
        "player_stats": {
            str(char.name): {
                "team": char.team,
                "kills": char.kills,
                "deaths": char.deaths,
                "alive": char.is_alive,
            }
            for char in game.chars
        },
    }
    write_json(SESSION_DIR / "result.json", result)
    write_json(SESSION_DIR / "replay.json", {"replay_frames": game.replay_frames})
    analytics_path = write_analytics_result(result, game.replay_frames)
    completion = "Match complete." if game.match_over else "Tick safety limit reached before match completion."
    update_progress(f"{completion} Result JSON: {analytics_path}")
    close_button.pack(anchor="e", pady=(12, 0))
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    print(f"{completion} {progress_text(game, chat_team)} | Result JSON: {analytics_path}", flush=True)
    root.mainloop()

if __name__ == "__main__":
    main()
