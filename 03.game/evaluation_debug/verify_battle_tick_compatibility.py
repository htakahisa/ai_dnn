"""Compare the old and shared tick loops without changing production files.

Run from 03.game:
    python evaluation_debug/verify_battle_tick_compatibility.py

Only loop/run_headless_loop come from the pinned baseline commit. Controllers,
models, maps and all other game methods are identical in both runs, isolating
the battle_logic.py refactor from the accompanying ConCon training changes.
"""

import argparse
import ast
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = "a242c51f5aabc8134a09b65f745255fead89b96d"

with contextlib.redirect_stdout(io.StringIO()):
    import numpy as np
    import torch
    import battle_logic
    import ghost_champions_v1_macro
    from battle_logic import BattleLogicMixin
    from map_data import NEW_MAZE_STR
    from party_presets import get_preset
    from run_game import VisualFPSBattle, _build_team_ai


def old_loops():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:03.game/battle_logic.py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    klass = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == "BattleLogicMixin")
    namespace = vars(battle_logic).copy()
    methods = {}
    for node in klass.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("loop", "run_headless_loop"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<baseline>", "exec"),
                 namespace)
            methods[node.name] = namespace[node.name]
    assert len(methods) == 2
    return methods


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unserializable comparison value: {type(value)}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default,
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def probe(method, *, headless=False, setup=False, round_over=False,
          match_over=False, hp=2, counter=0, fail_move=False):
    events = []

    class Game(BattleLogicMixin):
        pass

    game = Game()
    carrier = SimpleNamespace(name="carrier", team="A", is_alive=True, has_spike=True,
                              hp=hp, pos=[1, 1], carnal_lust_syndicate_active=True)
    defender = SimpleNamespace(name="defender", team="D", is_alive=True,
                               has_spike=False, hp=100, pos=[3, 3])
    dead = SimpleNamespace(name="dead", team="D", is_alive=False, has_spike=False,
                          hp=0, pos=[4, 4], carnal_lust_syndicate_active=True)
    game.chars = [defender, dead, carrier]
    game.round_over, game.match_over, game.headless = round_over, match_over, headless
    game.defender_setup_phase = SimpleNamespace(active=setup)
    game._analytics_post_setup_ticks = counter
    game._analytics_initial_defender_positions = None
    game.root = SimpleNamespace(after=lambda delay, callback: events.append(("after", delay)))
    game._tick_delay_ms = lambda: 250
    game.draw = lambda: events.append("draw")
    game._run_defender_setup_tick = lambda: events.append("setup")
    game._prepare_team_controllers_tick = lambda: events.append("prepare")
    game._build_occupancy_counts = lambda: events.append("build")
    game._clear_occupancy_counts = lambda: events.append("clear")
    game._advance_combo_announcement = lambda: events.append("announcement")
    game._record_replay_frame = lambda: events.append("replay")

    def move(char):
        events.append(("move", char.name, char.hp))
        if fail_move:
            raise RuntimeError("movement failure")

    def combat():
        events.append("combat")
        game.match_over = True  # Stop headless probes after exactly one tick.

    game.move_character, game.process_battle = move, combat
    if headless and setup:
        def setup_tick():
            events.append("setup")
            game.match_over = True
        game._run_defender_setup_tick = setup_tick
    error = None
    try:
        method(game)
    except RuntimeError as exc:
        error = str(exc)
    return (events, vars(carrier), vars(dead), game._analytics_post_setup_ticks,
            game._analytics_initial_defender_positions, error)


def boundary_checks(legacy):
    count = 0
    for headless in (False, True):
        name = "run_headless_loop" if headless else "loop"
        cases = [{}, {"setup": True}, {"hp": 1}, {"counter": 8},
                 {"counter": 9}, {"counter": 10}, {"match_over": True},
                 {"fail_move": True}]
        if not headless:
            cases.append({"round_over": True})
        for case in cases:
            old = probe(legacy[name], headless=headless, **case)
            new = probe(getattr(BattleLogicMixin, name), headless=headless, **case)
            assert old == new, (name, case, old, new)
            count += 1
    # The old headless loop spun forever here. The new loop deliberately exits.
    game = SimpleNamespace(match_over=False, round_over=True,
                           step_tick=lambda: False)
    BattleLogicMixin.run_headless_loop(game)
    return count


def run_match(attacker_key, attacker_name, defender_key, defender_name, seed, legacy=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    traces, locations = [], []
    original_record = VisualFPSBattle._record_replay_frame
    original_open = builtins.open
    opening_class = ghost_champions_v1_macro.LearningDefenderOpeningMacroGCRuntime
    opening_init = opening_class.__init__ if opening_class is not None else None

    def seeded_opening_init(controller, *args, **kwargs):
        # This controller uses random.Random(None), independent of random.seed.
        # Give both sides of the experiment identical runtime variation.
        if kwargs.get("seed") is None:
            kwargs["seed"] = seed
        opening_init(controller, *args, **kwargs)

    def quiet_open(file, mode="r", *args, **kwargs):
        if (isinstance(file, (str, Path))
                and Path(file).name in {"attacker_guard_gc_debug.log",
                                        "defender_search_gc_debug.log"}
                and mode == "a"):
            return io.StringIO()
        return original_open(file, mode, *args, **kwargs)

    def record(game):
        original_record(game)
        if len(traces) >= 20000:
            raise AssertionError("Comparison match exceeded 20000 replay frames")
        state = {
            "replay": game.replay_frames[-1],
            "chars": [(char.name, char.kills, char.deaths, char.plant_timer,
                       char.defuse_timer, getattr(char, "just_died", False))
                      for char in game.chars],
            "analytics_ticks": game._analytics_post_setup_ticks,
            "initial_defenders": game._analytics_initial_defender_positions,
            "announcement_queue": game.announcement_queue,
            "announcement_index": game.combo_announcement_index,
            "announcement_ticks": game.combo_announcement_ticks_left,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state().tolist(),
        }
        traces.append(digest(state))
        locations.append((game.current_round, game.battle_tick))
        # Retain only the latest frame; complete frame hashes remain in traces.
        game.replay_frames[:] = game.replay_frames[-1:]

    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(builtins, "open", quiet_open))
        if opening_class is not None:
            stack.enter_context(patch.object(opening_class, "__init__", seeded_opening_init))
        stack.enter_context(patch.object(VisualFPSBattle, "_record_replay_frame", record))
        if legacy:
            for name, method in legacy.items():
                stack.enter_context(patch.object(BattleLogicMixin, name, method))
        attackers, defenders = get_preset(attacker_name), get_preset(defender_name)
        game = VisualFPSBattle(
            NEW_MAZE_STR, _build_team_ai(attacker_key), _build_team_ai(defender_key),
            headless=True, attacker_roster=attackers.players,
            defender_roster=defenders.players, spike_holder_name=attackers.spike_holder,
            defender_spike_holder_name=defenders.spike_holder,
            attacker_igl_name=attackers.igl, defender_igl_name=defenders.igl,
            attacker_team_name=attackers.name, defender_team_name=defenders.name,
        )
        game.run_headless_loop()
        assert game.match_over
        tracker = game.analytics_tracker
        summary = {
            "score": [game.attacker_wins, game.defender_wins],
            "round": game.current_round, "sides_swapped": game.sides_swapped,
            "overtime": game.overtime, "match_stats": game.match_stats,
            "round_records": tracker.round_records, "stats": tracker.stats,
            "side_stats": tracker.side_stats, "gunfights": tracker.gunfights,
            "assist_events": tracker.assist_events, "cover_events": tracker.cover_events,
        }
    return traces, locations, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(5),
                        help="Compare only one match case (0 through 4)")
    args = parser.parse_args()
    legacy = old_loops()
    source_hash = hashlib.sha256((ROOT / "battle_logic.py").read_bytes()).hexdigest()
    count = boundary_checks(legacy)
    print(f"PASS: {count} legacy/current boundary comparisons; terminal headless loop exits",
          flush=True)
    cases = [
        ("default", "EG2023", "default", "Fnatic2023", 0),
        ("default", "EG2023", "default", "Fnatic2023", 7),
        ("fnatic_v3", "Fnatic2023", "omoko_gaming_v1", "Omoko Gaming", 0),
        ("gc_v1", "Ghost Champions", "touyama_gaming_v2", "Touyama Gaming", 0),
        ("toru_ai_v3.1", "Team Elites", "fnatic_v3", "Fnatic2023", 0),
    ]
    if args.case is not None:
        cases = [cases[args.case]]
    results = []
    for case in cases:
        print(f"Comparing {case[0]} vs {case[2]}, seed={case[4]} ...", flush=True)
        old, locations, old_summary = run_match(*case, legacy=legacy)
        new, new_locations, new_summary = run_match(*case)
        for index, (old_hash, new_hash) in enumerate(zip(old, new)):
            assert old_hash == new_hash, (case, "frame mismatch", index, locations[index])
        assert len(old) == len(new), (case, "different frame counts", len(old), len(new))
        assert locations == new_locations
        assert digest(old_summary) == digest(new_summary), (case, "different analytics")
        result = {"attacker": case[0], "defender": case[2], "seed": case[4],
                  "frames_compared": len(old), "score": new_summary["score"],
                  "round": new_summary["round"],
                  "sides_swapped": new_summary["sides_swapped"],
                  "overtime": new_summary["overtime"], "identical": True}
        results.append(result)
        print(f"PASS: {len(old)} frames and final analytics match; {result['score']}", flush=True)
    assert source_hash == hashlib.sha256((ROOT / "battle_logic.py").read_bytes()).hexdigest(), \
        "battle_logic.py changed during verification; rerun against the final version"
    report = {"baseline_commit": BASELINE, "battle_logic_sha256": source_hash,
              "boundary_comparisons": count, "headless_terminal_exit": True,
              "private_gc_opening_rng_seeded": True,
              "scope": "Only battle_logic loop methods differ; other files held constant",
              "matches": results}
    suffix = "result" if args.case is None else f"case{args.case}"
    output = Path(__file__).with_name(f"battle_tick_compatibility_{suffix}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
