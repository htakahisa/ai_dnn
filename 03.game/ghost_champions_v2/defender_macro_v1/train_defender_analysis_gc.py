"""Train GC's shared attack-site analysis in real defender matches.

Run from ghost_champions_v2/defender_macro_v1/. All outputs stay below data/ and logs/.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from collections import deque
import contextlib
from datetime import datetime
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from toruAI_v4.tv4_model import SiteModel, optimize
from ghost_champions_v2.defender_macro_v1.analysis import AttackSiteAnalysis, GCFeatureHistory, Scenario, schema, load_model, ACTIVE_MODEL
from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController

OPPONENTS = {"TYG": ("touyama_gaming_v2", "Touyama Gaming"),
             "OMG": ("omoko_gaming_v1", "Omoko Gaming"), "FRC": ("frc_v1", "Furina Classic"),
             "FNC": ("fnatic_v3", "Fnatic2023"), "GG": ("concon_v1", "Gorigons"),
             "SPS": ("default", "SUPES")}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def seed_defender_opening(defender, seed):
    """GC Opening owns a Random(None) independent of the match/global RNG."""
    opening = getattr(defender, "opening_macro_controller", None)
    if opening is not None:
        opening.rng.seed(seed)


def build_seeded_opponent(opponent, seed):
    """Retain production opponents, including GG's private route/guard RNGs."""
    from run_game import _build_team_ai
    from functools import partial
    ai = _build_team_ai(OPPONENTS[opponent][0])
    if opponent == "GG":
        from concon_v1.co1_attacker_controller import ConconRoundAttackerController
        from concon_v1.co1_attacker_scenarios import CONCON_ATTACKER_MAP, CONCON_ATTACKER_POSTPLANT_MODELS
        def seeded_guard(factory, guard_seed):
            controller = factory()
            controller.rng.seed(guard_seed)
            return controller
        factories = {site: tuple(partial(seeded_guard, factory, seed + 1000 + index * 100 + i)
                                 for i, factory in enumerate(items))
                     for index, (site, items) in enumerate(sorted(CONCON_ATTACKER_POSTPLANT_MODELS.items()))}
        ai.attacker_factory = partial(ConconRoundAttackerController, map_names=CONCON_ATTACKER_MAP,
                                     postplant_factories=factories, seed=seed)
    return ai


def play_block(opponent, scenario, model, seed, rounds, log_dir, *, legacy=False, max_steps=500):
    """The predictor only gets IQ perception; truth is read afterwards as a label."""
    from party_presets import get_preset
    from run_game import VisualFPSBattle, _build_team_ai
    from controllers import DefaultAttackerController
    from team_ai import DualRoleTeamAI
    from ghost_champions_v1_macro import GhostChampionsV1DefenderController
    from simulation_runtime import cpu_inference
    analysis = AttackSiteAnalysis(scenario=scenario, model=model)
    attacker_key, preset_name = OPPONENTS[opponent]
    attacking, defending = get_preset(preset_name), get_preset("Ghost Champions")
    seed_all(seed)
    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        with (log_dir / "engine.log").open("a", encoding="utf-8") as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), cpu_inference():
            defender = GhostChampionsV1DefenderController() if legacy else GhostChampionsV2DefenderController(analysis=analysis)
            seed_defender_opening(defender, seed)
            ai = DualRoleTeamAI("GC defender macro v1", DefaultAttackerController, lambda: defender)
            game = VisualFPSBattle(scenario.maze, build_seeded_opponent(opponent, seed), ai, headless=True,
                attacker_roster=list(attacking.players), defender_roster=list(defending.players),
                spike_holder_name=attacking.spike_holder, defender_spike_holder_name=defending.spike_holder,
                attacker_igl_name=attacking.igl, defender_igl_name=defending.igl,
                attacker_team_name=attacking.name, defender_team_name=defending.name, disable_side_swap=True)
            game.stop_after_round = True
            game.analytics_tracker = None
            game._record_replay_frame = lambda: None
            # Old controllers sometimes have root-relative debug filenames.
            from toruAI_v4.tv4_train_analysis import relocate_debug_logs
            relocate_debug_logs(game.attacker_controller, log_dir)
            relocate_debug_logs(game.defender_controller, log_dir)
            results, samples = [], []
            for number in range(1, rounds + 1):
                before = game.defender_wins
                plant_site = plant_tick = None
                steps = 0
                while not game.round_over and not game.match_over:
                    game.step_tick()
                    if game.is_planted and plant_site is None:
                        # Label-only read, after preplant decisions were recorded.
                        plant_site = scenario.site_of(game.planted_pos)
                        plant_tick = int(game.battle_tick)
                    steps += 1
                    if steps > max_steps:
                        raise RuntimeError(f"{opponent} round {number} exceeded {max_steps} steps")
                if not game.round_over or game.current_round != number:
                    raise RuntimeError("Defender block ended before its scheduled round")
                frames = list(analysis.frames) if not legacy else []
                last = frames[-1] if frames else None
                correct = bool(plant_site and last and last["decision"] == {"L": "A", "R": "B"}[plant_site])
                stable_tick = None
                if correct:
                    for frame in reversed(frames):
                        if frame["decision"] != last["decision"]:
                            break
                        stable_tick = frame["tick"]
                result = dict(opponent=opponent, seed=seed, round=number, planted=bool(plant_site), site=plant_site,
                    defender_win=game.defender_wins > before, decision=last["decision"] if last else None,
                    correct=correct, lead_ticks=plant_tick - stable_tick if stable_tick is not None else None,
                    frames=len(frames), legacy=legacy)
                results.append(result)
                if plant_site and frames:
                    samples.append(dict(features=np.stack([f["features"] for f in frames]).astype(np.float16),
                                        label=int(plant_site == "R")))
                # No label is fed back as history. The live controller records
                # only planted positions it actually received through perception.
                if number < rounds:
                    game.current_round += 1
                    game.init_round()
            return results, samples
    finally:
        os.chdir(original_cwd)


def summarize(rows):
    planted = [r for r in rows if r["planted"]]
    decided = [r for r in planted if r["decision"] is not None]
    correct = [r for r in planted if r["correct"]]
    leads = [r["lead_ticks"] for r in correct if r["lead_ticks"] is not None]
    return dict(rounds=len(rows), plants=len(planted), decided=len(decided), correct=len(correct),
        accuracy=len(correct) / len(decided) if decided else None,
        coverage=len(decided) / len(planted) if planted else None,
        correct_all_plants=len(correct) / len(planted) if planted else None,
        mean_correct_lead=float(np.mean(leads)) if leads else None,
        defender_win_rate=sum(r["defender_win"] for r in rows) / len(rows) if rows else None)


def rank(evaluation):
    overall = evaluation["defender_win_rate"] or 0.
    opponent_rates = [row["defender_win_rate"] for row in evaluation.get("opponents", {}).values()
                      if row["defender_win_rate"] is not None]
    return (min(opponent_rates) if opponent_rates else overall, overall,
            evaluation["correct_all_plants"] or 0., evaluation["accuracy"] or 0.)


def save_checkpoint(path, model, scenario, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(schema=schema(scenario), model=model.state_dict(), **metadata), temporary)
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=datetime.now().strftime("analysis_%Y%m%d_%H%M%S"))
    p.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    p.add_argument("--sets", type=int, default=10, help="Additional training blocks per opponent")
    p.add_argument("--rounds-per-block", type=int, default=12)
    p.add_argument("--eval-rounds", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, help="Latest checkpoint including optimizer/replay")
    p.add_argument("--activate", action="store_true", help="Use this run's evaluated best in GC v2 matches")
    p.add_argument("--compare-legacy", action="store_true", help="Compare old GC defense on the final evaluation seeds")
    p.add_argument("--describe", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("--run must be a single directory name")
    if (any(getattr(args, k) < 1 for k in ("sets", "rounds_per_block", "eval_rounds", "eval_every", "updates", "batch_size"))
            or not 1 <= args.rounds_per_block <= 12 or not 1 <= args.eval_rounds <= 12
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or len(set(args.opponents)) != len(args.opponents)):
        raise ValueError("Counts/rate must be positive; blocks <=12 rounds; opponents must be unique")
    scenario = Scenario()
    if args.describe:
        print(json.dumps(schema(scenario), ensure_ascii=False, indent=2))
        return
    torch.set_num_threads(1)
    data_dir, log_dir = HERE / "data" / args.run, HERE / "logs" / args.run
    if data_dir.exists() or log_dir.exists():
        raise FileExistsError("Choose a new --run; existing run files are preserved")
    data_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    seed_all(args.seed)
    model = SiteModel(len(GCFeatureHistory(scenario).fields))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    replay = deque(maxlen=1200)
    rng = np.random.default_rng(args.seed)
    labeled_rounds = trained_rounds = completed = 0
    if args.resume:
        model, payload = load_model(args.resume.resolve(), scenario)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        replay.extend(dict(features=r["features"].numpy(), label=r["label"]) for r in payload["replay"])
        labeled_rounds, trained_rounds, completed = payload["labeled_rounds"], payload["trained_rounds"], payload["completed_sets"]
        rng.bit_generator.state = json.loads(payload["rng"])
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    save_json(data_dir / "manifest.json", dict(settings=settings, schema=schema(scenario),
        source_hashes={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                       for name in ("analysis.py", "controller.py", "tendency.py", "deployment.py", "train_defender_analysis_gc.py")},
        selection="worst_opponent_defender_win_rate_then_overall"))
    best_evaluation = None
    started = time.monotonic()
    try:
        for additional in range(1, args.sets + 1):
            set_number = completed + additional
            for i, opponent in enumerate(args.opponents):
                rows, samples = play_block(opponent, scenario, model,
                    args.seed + set_number * 10000 + i * 100, args.rounds_per_block, log_dir)
                replay.extend(samples)
                labeled_rounds += len(samples)
                trained_rounds += len(rows)
                loss = optimize(model, optimizer, replay, rng, args.updates, args.batch_size)
                progress = dict(phase="train", set=set_number, opponent=opponent, loss=loss,
                    labeled_rounds=labeled_rounds, trained_rounds=trained_rounds, **summarize(rows))
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                with (log_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
                with (log_dir / "rounds.jsonl").open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(dict(phase="train", set=set_number, **row)) + "\n")
            if not replay:
                raise RuntimeError("No planted rounds were observed; no learned checkpoint can be activated")
            evaluation = None
            if additional % args.eval_every == 0 or additional == args.sets:
                all_rows, by_opponent = [], {}
                for i, opponent in enumerate(args.opponents):
                    rows, _ = play_block(opponent, scenario, model,
                        args.seed + 900000000 + i * 100, args.eval_rounds, log_dir)
                    all_rows.extend(rows)
                    by_opponent[opponent] = summarize(rows)
                    print(json.dumps(dict(phase="eval_opponent", set=set_number, opponent=opponent,
                                          **by_opponent[opponent]), ensure_ascii=False), flush=True)
                evaluation = dict(**summarize(all_rows), opponents=by_opponent,
                                  seeds=[args.seed + 900000000 + i * 100 for i in range(len(args.opponents))])
                save_json(log_dir / f"evaluation_{set_number:04d}.json", dict(summary=evaluation, rounds=all_rows))
                print(json.dumps(dict(phase="eval", set=set_number, **evaluation), ensure_ascii=False), flush=True)
                if best_evaluation is None or rank(evaluation) > rank(best_evaluation):
                    best_evaluation = evaluation
                    save_checkpoint(data_dir / "best.pt", model, scenario, labeled_rounds=labeled_rounds,
                        trained_rounds=trained_rounds, completed_sets=set_number, evaluation=evaluation, run=args.run)
            save_checkpoint(data_dir / "latest.pt", model, scenario, labeled_rounds=labeled_rounds,
                trained_rounds=trained_rounds, completed_sets=set_number, evaluation=evaluation, run=args.run,
                optimizer=optimizer.state_dict(), rng=json.dumps(rng.bit_generator.state),
                replay=[dict(features=torch.from_numpy(r["features"]), label=r["label"]) for r in replay])
        if args.compare_legacy:
            rows = []
            for i, opponent in enumerate(args.opponents):
                current, _ = play_block(opponent, scenario, model,
                    args.seed + 900000000 + i * 100, args.eval_rounds, log_dir, legacy=True)
                rows.extend(current)
                print(json.dumps(dict(phase="legacy", opponent=opponent, **summarize(current)), ensure_ascii=False), flush=True)
            save_json(log_dir / "legacy_comparison.json", dict(summary=summarize(rows), rounds=rows))
        if args.activate:
            best, payload = load_model(data_dir / "best.pt", scenario)
            if not payload["evaluation"]["plants"]:
                raise RuntimeError("Evaluation contained no plants; model not activated")
            # Keep the previous active checkpoint available for rollback.
            if ACTIVE_MODEL.is_file():
                (data_dir / "previous_active.pt").write_bytes(ACTIVE_MODEL.read_bytes())
            save_checkpoint(ACTIVE_MODEL, best, scenario, **{k: v for k, v in payload.items() if k not in ("schema", "model")})
        save_json(log_dir / "status.json", dict(complete=True, active=args.activate,
            best_evaluation=best_evaluation, trained_rounds=trained_rounds,
            labeled_rounds=labeled_rounds, elapsed_seconds=time.monotonic() - started))
    except BaseException as error:
        save_json(log_dir / "status.json", dict(complete=False, error=str(error),
            trained_rounds=trained_rounds, labeled_rounds=labeled_rounds))
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
