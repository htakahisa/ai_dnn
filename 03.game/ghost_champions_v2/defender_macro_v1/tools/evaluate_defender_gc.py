"""Compare GC defense on seeded round blocks or completed matches."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import contextlib
import importlib.util
import importlib
import types
import json
import os
import hashlib
import torch

from ghost_champions_v2.defender_macro_v1.analysis import AttackSiteAnalysis, ACTIVE_MODEL, Scenario, load_model
from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController
from ghost_champions_v2.defender_macro_v1.train_defender_analysis_gc import OPPONENTS, seed_all, seed_defender_opening, build_seeded_opponent, summarize, save_json


CANDIDATE_RUNTIMES = {
    "90": "gg_selective_facing", "45": "gg_selective_facing45",
    "adaptive": "gg_adaptive_facing", "stable": "gg_stable_facing",
    "search": "gg_search_facing", "covered": "gg_covered_retake",
    "covered_both": "gg_covered_both_facing", "balanced_covered": "gg_balanced_covered",
}


def frozen_module(name, *, previous=False, baseline=False, candidate=False):
    if previous or baseline or candidate:
        suffix = CANDIDATE_RUNTIMES[candidate] if candidate else "gg_baseline" if baseline else "omg"
        package = "ghost_champions_v2.defender_macro_v1.diagnostic_" + suffix + "_20261009"
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(HERE / ("data/diagnostic_runtime_" + suffix + "_20261009"))]
            sys.modules[package] = module
        return importlib.import_module(package + "." + name)
    path = HERE / "data/diagnostic_runtime_20261009" / (name + ".py")
    spec = importlib.util.spec_from_file_location("ghost_champions_v2.defender_macro_v1.diagnostic_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def play(opponent, seed, rounds, mode, directory, checkpoint=ACTIVE_MODEL, trace=False, candidate_version="90"):
    from run_game import VisualFPSBattle, _build_team_ai
    from party_presets import get_preset
    from controllers import DefaultAttackerController
    from team_ai import DualRoleTeamAI
    from ghost_champions_v1_macro import GhostChampionsV1DefenderController
    from simulation_runtime import cpu_inference
    from toruAI_v4.tv4_train_analysis import relocate_debug_logs
    scenario = Scenario()
    model, _ = load_model(checkpoint, scenario)
    seed_all(seed)
    analysis_cls, controller_cls = AttackSiteAnalysis, GhostChampionsV2DefenderController
    if mode == "before":
        analysis_cls = frozen_module("analysis").AttackSiteAnalysis
    if mode == "before":
        controller_cls = frozen_module("controller").GhostChampionsV2DefenderController
    if mode == "previous":
        analysis_cls = frozen_module("analysis", previous=True).AttackSiteAnalysis
        controller_cls = frozen_module("controller", previous=True).GhostChampionsV2DefenderController
    if mode == "baseline":
        analysis_cls = frozen_module("analysis", baseline=True).AttackSiteAnalysis
        controller_cls = frozen_module("controller", baseline=True).GhostChampionsV2DefenderController
    if mode == "candidate":
        analysis_cls = frozen_module("analysis", candidate=candidate_version).AttackSiteAnalysis
        controller_cls = frozen_module("controller", candidate=candidate_version).GhostChampionsV2DefenderController
    analysis = analysis_cls(scenario=scenario, model=model)
    defender = GhostChampionsV1DefenderController() if mode == "legacy" else controller_cls(analysis=analysis)
    decisions = []
    if trace:
        original_decide = defender.decide_move
        def traced_decide(char, state):
            facing_before = getattr(char, "facing", None)
            result = original_decide(char, state)
            if not state.get("defender_setup_active"):
                decisions.append(dict(tick=state.get("battle_tick", state.get("tick")),
                    name=str(char.name), pos=list(char.pos), hp=float(char.hp),
                    planted=bool(state.get("is_planted")), facing_before=facing_before,
                    facing_after=getattr(char, "facing", None),
                    forced_facing=bool(getattr(char, "facing_forced_this_tick", False)),
                    blind=int(getattr(char, "blind_remaining", 0)),
                    defuse=int(getattr(char, "defuse_timer", 0)), action=result,
                    known_enemies={str(e.name): list(e.pos) for e in state.get("chars", ())
                        if e.team != char.team and e.is_alive and getattr(e, "position_known", True)}))
            return result
        defender.decide_move = traced_decide
    seed_defender_opening(defender, seed)
    ai = DualRoleTeamAI("GC defender evaluation", DefaultAttackerController, lambda: defender)
    key, name = OPPONENTS[opponent]
    a, d = get_preset(name), get_preset("Ghost Champions")
    with (directory / "engine.log").open("a", encoding="utf-8") as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), cpu_inference():
        game = VisualFPSBattle(scenario.maze, build_seeded_opponent(opponent, seed), ai, headless=True,
            attacker_roster=list(a.players), defender_roster=list(d.players), spike_holder_name=a.spike_holder,
            defender_spike_holder_name=d.spike_holder, attacker_igl_name=a.igl, defender_igl_name=d.igl,
            attacker_team_name=a.name, defender_team_name=d.name, disable_side_swap=True)
        relocate_debug_logs(game.attacker_controller, directory)
        relocate_debug_logs(game.defender_controller, directory)
        game.stop_after_round = True
        game._record_replay_frame = lambda: None
        rows = []
        for number in range(1, rounds + 1):
            before = game.defender_wins
            planted = None
            steps = 0
            allocations, previous_allocation = [], None
            initial_deployment = None
            decisions.clear()
            trace_frames = []
            while not game.round_over and not game.match_over:
                game.step_tick()
                steps += 1
                if trace and game.is_planted:
                    trace_frames.append(dict(tick=game.battle_tick, timer=game.detonate_timer,
                        spike=list(game.planted_pos), units=[dict(name=str(c.name), team=c.team,
                            pos=list(c.pos), hp=float(c.hp), alive=bool(c.is_alive),
                            defuse=int(getattr(c, "defuse_timer", 0))) for c in game.chars]))
                if mode in ("current", "previous", "baseline", "candidate"):
                    macro = defender.macro
                    snapshot = macro.snapshot()
                    signature = tuple(sorted(snapshot["allocation"].items())), snapshot["allocation_source"]
                    if signature != previous_allocation and not game.is_planted:
                        allocations.append(dict(tick=game.battle_tick, setup=game.defender_setup_phase.active,
                            allocation=snapshot["allocation"], source=snapshot["allocation_source"],
                            history=snapshot["analysis"]["tendency"], contacts=snapshot["analysis"]["current_contacts"]))
                        previous_allocation = signature
                    if not game.defender_setup_phase.active and initial_deployment is None:
                        initial_deployment = dict(allocation=snapshot["allocation"], source=snapshot["allocation_source"],
                            goals=snapshot["goals"], actual_positions={str(c.name): list(c.pos) for c in game.chars if c.team == "D"},
                            arrived=sum(tuple(c.pos) == macro.goals.get(str(c.name)) for c in game.chars if c.team == "D"))
                if game.is_planted and planted is None:
                    planted = dict(tick=game.battle_tick, site=scenario.site_of(game.planted_pos),
                        defenders=sum(c.team == "D" and c.is_alive for c in game.chars),
                        attackers=sum(c.team == "A" and c.is_alive for c in game.chars))
                if steps > 500:
                    raise RuntimeError("Round exceeded 500 steps")
            if not game.round_over or game.current_round != number:
                raise RuntimeError("Unexpected round end")
            frames = analysis.frames if mode != "legacy" else []
            predicted = [f for f in frames if f["decision"]]
            decision = frames[-1]["decision"] if frames else None
            row = dict(opponent=opponent, seed=seed, round=number, mode=mode,
                defender_win=game.defender_wins > before, planted=planted is not None,
                site=planted["site"] if planted else None, plant=planted, decision=decision,
                correct=bool(planted and decision == {"L": "A", "R": "B"}[planted["site"]]),
                lead_ticks=None, first_decision_tick=predicted[0]["tick"] if predicted else None,
                kills=sum(c.round_kills for c in game.chars if c.team == "D"),
                initial_deployment=initial_deployment, allocation_changes=allocations)
            row["end"] = dict(tick=game.battle_tick, timer=game.detonate_timer,
                match_over=bool(game.match_over), defender_score=int(game.defender_wins),
                attacker_score=int(game.attacker_wins),
                defenders=sum(c.team == "D" and c.is_alive for c in game.chars),
                attackers=sum(c.team == "A" and c.is_alive for c in game.chars),
                max_defuse=max((getattr(c, "defuse_timer", 0) for c in game.chars if c.team == "D"), default=0))
            if mode in ("current", "candidate"):
                snapshot = defender.macro.snapshot()
                row["combat_facing_corrections"] = snapshot.get("combat_facing_corrections", 0)
                row["covered_retake_commits"] = snapshot.get("covered_retake_commits", 0)
            if trace:
                save_json(directory / f"{opponent}_round_{number:02d}_trace.json",
                    dict(decisions=list(decisions), frames=trace_frames))
            rows.append(row)
            if game.match_over:
                break
            if number < rounds:
                game.current_round += 1
                game.init_round()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("before", "previous", "baseline", "candidate", "current", "legacy"), default="current")
    p.add_argument("--trace", action="store_true", help="Record LIVE decisions and retake ground truth for diagnosis")
    p.add_argument("--candidate-version", choices=tuple(CANDIDATE_RUNTIMES), default="90")
    p.add_argument("--run", required=True)
    p.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=["GG", "OMG"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--checkpoint", type=Path, default=ACTIVE_MODEL)
    args = p.parse_args()
    if Path(args.run).name != args.run or args.run in (".", "..") or not 1 <= args.rounds <= 64:
        p.error("A single run name and 1-64 rounds are required")
    if args.seeds is not None and len(args.seeds) != len(args.opponents):
        p.error("Provide one seed per opponent")
    if len(set(args.opponents)) != len(args.opponents):
        p.error("Opponents must be unique")
    seeds = args.seeds or [4251007678 if o == "GG" else 4047630677 if o == "OMG" else 900000042 + list(OPPONENTS).index(o) * 100 for o in args.opponents]
    directory = HERE / "logs" / args.run
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_dir = HERE / "data/diagnostic_runtime_20261009" if args.mode == "before" else HERE / "data/diagnostic_runtime_omg_20261009" if args.mode == "previous" else HERE / "data/diagnostic_runtime_gg_baseline_20261009" if args.mode == "baseline" else HERE
    if args.mode == "candidate":
        source_dir = HERE / ("data/diagnostic_runtime_" + CANDIDATE_RUNTIMES[args.candidate_version] + "_20261009")
    runtime_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                      for name in ("analysis.py", "controller.py")}
    if args.mode in ("current", "previous", "baseline", "candidate"):
        runtime_hashes.update({name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                               for name in ("tendency.py", "deployment.py")})
    for name in ("combat.py", "retake.py"):
        if (source_dir / name).is_file():
            runtime_hashes[name] = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
    shared_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("character_stats.py", "player_combos.py", "awakening_events.py", "party_presets.py", "battle_logic.py", "game_core.py", "ghost_champions_v1_macro.py", "gc_v1/learning_defender_search_gc.py", "gc_v1/learning_defender_retake_gc.py")}
    torch.set_num_threads(1)
    old = Path.cwd()
    os.chdir(ROOT)
    rows = []
    try:
        for opponent, seed in zip(args.opponents, seeds):
            block = play(opponent, seed, args.rounds, args.mode, directory, checkpoint, args.trace, args.candidate_version)
            rows.extend(block)
            result = dict(mode=args.mode, opponent=opponent, seed=seed, **summarize(block))
            result["match_over"] = block[-1]["end"]["match_over"]
            result["score"] = dict(GC=block[-1]["end"]["defender_score"], opponent=block[-1]["end"]["attacker_score"])
            print(json.dumps(result), flush=True)
            save_json(directory / f"{opponent}.json", dict(summary=result, rounds=block))
        save_json(directory / "summary.json", dict(mode=args.mode, summary=summarize(rows), rounds=rows,
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
            runtime_hashes=runtime_hashes, shared_source_hashes=shared_hashes,
            python_hash_seed=os.environ.get("PYTHONHASHSEED"), opening_rng="match_seed_v1",
            opponent_rng="match_seed_concon_routes_guards_v1", runtime_directory=str(source_dir),
            candidate_version=args.candidate_version if args.mode == "candidate" else None))
    finally:
        os.chdir(old)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
