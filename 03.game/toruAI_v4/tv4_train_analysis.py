"""Train six separate site predictors from real 12-round defender matches."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEST_DIRECTORY = HERE / "data" / "best"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import deque
import contextlib
import csv
import json
import logging
import os
import random
import time

import numpy as np
import torch

from toruAI_v4.tv4_scenario import OPPONENTS, Scenario
from toruAI_v4.tv4_observer import FeatureHistory, ObserverController
from toruAI_v4.tv4_model import VERSION, SiteModel, optimize



# 学習設定：相手AIごとの追加セット数。1セット = 12ラウンド。
# 通常はこちらを編集して実行してください。--sets は一時的な上書き用です。
TRAINING_SETS = 30
EVALUATION_INTERVAL = 10
EVALUATION_SEED_COUNT = 3  # 各seedで12ラウンド。合計36ラウンド。

@contextlib.contextmanager
def legacy_root():
    """Legacy attackers have project-relative weights; all new paths are absolute."""
    old = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relocate_debug_logs(controller, directory, seen=None):
    """Legacy debug outputs belong to this training run, not the project root."""
    seen = set() if seen is None else seen
    if id(controller) in seen:
        return
    seen.add(id(controller))
    if "_debug_log_path" in vars(controller):
        controller._debug_log_path = str(directory / Path(controller._debug_log_path).name) if directory else os.devnull
    for value in vars(controller).values():
        if hasattr(value, "__dict__") and callable(getattr(value, "decide_move", None)):
            relocate_debug_logs(value, directory, seen)


def round_metrics(controller, game, planted, plant_tick, site):
    frames = controller.frames
    last = frames[-1] if frames else None
    decision = last["decision"] if last else None
    decision_tick = last["decision_tick"] if last else None
    correct = None if site is None or decision is None else decision == site
    # Later outputs of the same side do not reset the decision time.
    lead = plant_tick - decision_tick if planted and decision_tick is not None else None
    stable_tick = None
    if correct:
        for frame in reversed(frames):
            if frame["decision"] != site:
                break
            stable_tick = frame["tick"]
    first = next((f["tick"] for f in frames if f["sightings"]), None)
    active_ticks = sum(f.get("active_defenders", 0) for f in frames)
    return dict(round=game.current_round, winner="A" if game.attacker_wins > controller.score_before else "D",
                site=site, plant_tick=plant_tick, decision=decision, decision_tick=decision_tick,
                correct=correct, lead_ticks=lead, correct_lead_ticks=lead if correct else None,
                stable_correct_lead_ticks=plant_tick - stable_tick if stable_tick is not None else None,
                changes=controller.gate.changes, first_sighting_tick=first,
                sightings=sum(len(f["sightings"]) for f in frames),
                retreats=sum(e["type"] == "retreat" for e in controller.events),
                spike_drops=sum(e["type"] == "spike_dropped" for e in controller.events),
                spike_disappearances=sum(e["type"] == "spike_disappeared" for e in controller.events),
                peeks=sum(e["type"] == "peek" for e in controller.events),
                movement_rate=sum(f.get("moving_defenders", 0) for f in frames) / active_ticks if active_ticks else None,
                defenders_at_plant=controller.defenders_at_plant,
                attackers_at_plant=controller.preplant_counts["attackers"] if planted else None,
                defenders_alive=sum(c.is_alive for c in game.chars if c.team == "D"),
                attackers_alive=sum(c.is_alive for c in game.chars if c.team == "A"),
                preplant_defenders_alive=controller.preplant_counts["defenders"],
                preplant_attackers_alive=controller.preplant_counts["attackers"],
                preplant_defender_losses=5 - controller.preplant_counts["defenders"],
                preplant_attackers_eliminated=5 - controller.preplant_counts["attackers"],
                preplant_defender_kills=controller.preplant_counts["defender_kills"],
                live_ticks=game.battle_tick, end_reason="planted" if planted else "attacker_eliminated"
                if not any(c.is_alive for c in game.chars if c.team == "A") else "defender_eliminated"
                if not any(c.is_alive for c in game.chars if c.team == "D") else "timeout")


def preplant_counts(game):
    """Outcome statistics only; never fed back into a prediction observation."""
    return {"defenders": sum(c.is_alive for c in game.chars if c.team == "D"),
            "attackers": sum(c.is_alive for c in game.chars if c.team == "A"),
            "defender_kills": sum(c.round_kills for c in game.chars if c.team == "D")}


def play_block(opponent, scenario, model, args, seed, on_round, engine_log):
    """Keep the same game/controller for all 12 rounds, preserving opponent history."""
    output = engine_log.open("a", encoding="utf-8") if engine_log else open(os.devnull, "w", encoding="utf-8")
    with legacy_root(), output as raw, contextlib.redirect_stdout(raw), contextlib.redirect_stderr(raw):
        from run_game import VisualFPSBattle, _build_team_ai
        from party_presets import get_preset
        from controllers import DefaultAttackerController
        from team_ai import DualRoleTeamAI
        from simulation_runtime import cpu_inference
        seed_all(seed)
        with cpu_inference(enabled=True):
            attackers = get_preset(OPPONENTS[opponent][1])
            defenders = get_preset(args.defender_preset)
            if set(attackers.players) & set(defenders.players):
                raise ValueError("Attacker and defender presets must use distinct player names")
            observer = ObserverController(scenario, model, threshold=args.threshold, confirm=args.confirm_ticks,
                                          rotate=args.rotate, peek_ticks=args.peek_ticks, hide_ticks=args.hide_ticks)
            ai = DualRoleTeamAI("Toru v4 observer", DefaultAttackerController, lambda: observer)
            game = VisualFPSBattle(scenario.maze, _build_team_ai(OPPONENTS[opponent][0]), ai,
                headless=True, attacker_roster=list(attackers.players), defender_roster=list(defenders.players),
                spike_holder_name=attackers.spike_holder, defender_spike_holder_name=defenders.spike_holder,
                attacker_igl_name=attackers.igl, defender_igl_name=defenders.igl,
                attacker_team_name=attackers.name, defender_team_name=defenders.name, disable_side_swap=True)
            game.stop_after_round, game.analytics_tracker = True, None
            relocate_debug_logs(game.attacker_controller, engine_log.parent if engine_log else None)
            # No full omniscient replay is needed for this training task.
            game._record_replay_frame = lambda: None
            history, results, samples = [], [], []
            for round_number in range(1, 13):
                observer.previous_rounds = list(history)
                observer.score_before = game.attacker_wins
                observer.defenders_at_plant = None
                observer.preplant_counts = None
                planted, plant_tick, site = False, None, None
                steps = 0
                while not game.round_over and not game.match_over:
                    was_live, tick_before = not game.defender_setup_phase.active, game.battle_tick
                    game.step_tick()
                    if was_live and observer.frames and observer.frames[-1]["tick"] == tick_before:
                        own = [c for c in game.chars if c.team == "D"]
                        active_names = {a["name"] for a in observer.frames[-1]["allies"] if a["alive"]}
                        observer.frames[-1]["moving_defenders"] = sum(bool(c.moved_this_tick) for c in own
                                                                     if str(getattr(c, "base_name", c.name)) in active_names)
                        observer.frames[-1]["active_defenders"] = observer.frames[-1]["allies_alive"]
                    steps += 1
                    if game.is_planted and not planted:
                        planted, plant_tick = True, int(game.battle_tick)
                        site = scenario.site_of(game.planted_pos)
                        observer.defenders_at_plant = sum(c.is_alive for c in game.chars if c.team == "D")
                        observer.preplant_counts = preplant_counts(game)
                    if steps > args.max_round_steps:
                        raise RuntimeError(f"{opponent} round {round_number} exceeded --max-round-steps; no fabricated label saved")
                if game.current_round != round_number or not game.round_over:
                    raise RuntimeError("12-round block ended unexpectedly")
                if observer.preplant_counts is None:
                    observer.preplant_counts = preplant_counts(game)
                result = round_metrics(observer, game, planted, plant_tick, site)
                results.append(result)
                # Plant label is accessed only after recording pre-plant inputs.
                if site is not None and observer.frames:
                    samples.append({"features": np.stack([f["features"] for f in observer.frames]).astype(np.float16),
                                    "label": int(site == "R"), "round": round_number, "plant_tick": plant_tick})
                on_round(result, observer.frames, observer.events)
                history.append({"site": site, "winner": result["winner"]})
                if round_number < 12:
                    game.current_round += 1
                    game.init_round()
            return results, samples


def summarize(results):
    planted = [r for r in results if r["site"] is not None]
    decided = [r for r in planted if r["decision"] is not None]
    correct = [r for r in decided if r["correct"]]
    leads = [r["correct_lead_ticks"] for r in correct]
    def mean(rows, key):
        known = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(known)) if known else None
    return {"rounds": len(results), "plants": len(planted), "left": sum(r["site"] == "L" for r in planted),
            "right": sum(r["site"] == "R" for r in planted), "decided": len(decided), "correct": len(correct),
            "accuracy": len(correct) / len(decided) if decided else None,
            "coverage": len(decided) / len(planted) if planted else None,
            "correct_all_plants": len(correct) / len(planted) if planted else None,
            "mean_correct_lead": float(np.mean(leads)) if leads else None,
            "median_correct_lead": float(np.median(leads)) if leads else None,
            "mean_alive": float(np.mean([r["defenders_alive"] for r in results])) if results else None,
            "mean_alive_at_plant": float(np.mean([r["defenders_at_plant"] for r in planted])) if planted else None,
            "mean_enemy_alive": mean(results, "attackers_alive"),
            "mean_enemy_alive_at_plant": mean(planted, "attackers_at_plant"),
            "mean_preplant_defenders_alive": mean(results, "preplant_defenders_alive"),
            "mean_preplant_attackers_alive": mean(results, "preplant_attackers_alive"),
            "mean_preplant_defender_losses": mean(results, "preplant_defender_losses"),
            "mean_preplant_attackers_eliminated": mean(results, "preplant_attackers_eliminated"),
            "mean_preplant_defender_kills": mean(results, "preplant_defender_kills"),
            "changes": sum(r["changes"] for r in results)}


class RunLog:
    def __init__(self, directory, detailed=False):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.detailed = detailed
        self.logger = logging.getLogger(str(directory))
        self.logger.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(), logging.FileHandler(directory / "training.log", mode="w", encoding="utf-8")):
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def jsonl(self, name, value):
        if not self.detailed:
            return
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def csv(self, name, row):
        if not self.detailed:
            return
        path = self.directory / name
        exists = path.exists()
        if exists:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                fieldnames = next(csv.reader(handle))
        else:
            fieldnames = list(row)
        with path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def record_round(self, identity, result, frames, events, trace):
        row = {**identity, **result}
        self.jsonl("rounds.jsonl", row)
        self.csv("rounds.csv", row)
        if trace:
            for frame in frames:
                self.jsonl("ticks.jsonl", {**identity, "round": result["round"],
                                          **{k: v for k, v in frame.items() if k != "features"}})
        for event in events:
            self.jsonl("events.jsonl", {**identity, "round": result["round"], **event})
        status = "NO_PLANT" if result["site"] is None else "WAIT" if result["decision"] is None else "OK" if result["correct"] else "MISS"
        self.logger.info("[%s][%s set=%03d R%02d] %-8s plant=%s@%s decision=%s@%s lead=%s plant_alive=D%s/A%s end_alive=D%d/A%d preplant_kills=%d peek=%d retreat=%d",
            identity["opponent"], identity["phase"], identity["set"], result["round"], status,
            result["site"] or "-", result["plant_tick"], result["decision"] or "-", result["decision_tick"],
            result["correct_lead_ticks"], result["defenders_at_plant"], result["attackers_at_plant"],
            result["defenders_alive"], result["attackers_alive"], result["preplant_defender_kills"], result["peeks"], result["retreats"])

    def write_summary(self, summary, checkpoint, best_path):
        summary_path = self.directory / "summary.json"
        if self.detailed:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            text = ["正解時の残りtick = プラント完了tick - 最後に寄り先を決定・変更したtick",
                    "平均・中央値は正解したラウンドのみで集計。誤判定・未判断・プラントなしは除外。", ""]
            text.extend(format_summary(row) for row in summary.values())
            (self.directory / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        for label, path in (("結果サマリ", self.directory / "summary.txt"),
                            ("セット別結果CSV", self.directory / "sets.csv"),
                            ("ラウンド別結果CSV", self.directory / "rounds.csv"),
                            ("集計JSON", summary_path), ("実行ログ", self.directory / "training.log"),
                            ("tick別の予測ログ", self.directory / "ticks.jsonl"),
                            ("覗き・退避・スパイクのイベント", self.directory / "events.jsonl"),
                            ("採用用bestモデル", best_path),
                            ("保存モデル", checkpoint)):
            if path.is_file():
                self.logger.info("[保存先] %s: %s", label, path.resolve())


def save_checkpoint(path, model, optimizer, replay, completed, opponent, scenario, fields, args, rng, train_rounds, training_plan=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ("seed", "threshold", "confirm_ticks", "rotate", "peek_ticks", "hide_ticks", "defender_preset")}
    state = {"version": VERSION, "opponent": opponent, "scenario": scenario.signature, "fields": fields,
             "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_sets": completed,
             "train_rounds": train_rounds, "rng": json.dumps(rng.bit_generator.state),
             "training_plan": training_plan,
             "replay": [{"features": torch.from_numpy(r["features"]), "label": r["label"]} for r in replay]}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def percent(value):
    return "-" if value is None else f"{value:.1%}"


BEST_RULE = "multi_seed_all_plants_correct_then_accuracy_then_lead_v2"


def best_rank(metrics):
    # Correctness across all plants also penalizes abstention. Only when
    # correctness ties do we reward more time to rotate.
    return (metrics["correct_all_plants"] or 0., metrics["accuracy"] or 0.,
            metrics["mean_correct_lead"] if metrics["mean_correct_lead"] is not None else -1.)


def evaluate_seeds(opponent, scenario, model, args, identity, seeds, log):
    results, seed_results = [], []
    started = time.perf_counter()
    for seed in seeds:
        seed_identity = {**{k: v for k, v in identity.items() if k != "replay_rounds"}, "seed": seed}
        callback = lambda r, f, e: log.record_round(seed_identity, r, f, e, args.trace_ticks)
        rounds, _ = play_block(opponent, scenario, model, args, seed, callback,
                               log.directory / "engine.log" if args.detailed_logs else None)
        metrics = summarize(rounds)
        seed_results.append({"seed": seed, **metrics})
        results.extend(rounds)
        log.logger.info("%s (seed=%d)", format_summary({**seed_identity, **metrics}), seed)
    # Pool rounds rather than averaging percentages from unequal plant counts.
    return {**identity, "seed": seeds[0], **summarize(results), "evaluation_seeds": list(seeds),
            "seed_results": seed_results, "loss": None, "replay_rounds": identity["replay_rounds"],
            "seconds": time.perf_counter() - started}


def consider_best(path, model, evaluation, opponent, scenario, fields, args, seeds, logger,
                  reevaluate=None, cache=None, previous_path=None):
    if not evaluation["plants"]:
        logger.info("[%s][best] 更新なし: 評価でプラントがなく比較できません", opponent)
        return False
    config = {k: getattr(args, k) for k in ("seed", "threshold", "confirm_ticks", "rotate", "peek_ticks", "hide_ticks", "defender_preset")}
    identity = {"version": VERSION, "opponent": opponent, "scenario": scenario.signature,
                "fields": fields, "config": config}
    old = None
    reference_path = path if path.exists() else previous_path
    if reference_path is not None and reference_path.exists():
        old = torch.load(reference_path, map_location="cpu", weights_only=True)
        if any(old.get(k) != value for k, value in identity.items()):
            raise ValueError(f"Best checkpoint evaluation conditions differ: {path}")
    if evaluation.get("evaluation_seeds") != list(seeds) or evaluation["phase"] != "eval":
        raise ValueError("Best selection requires the common independent evaluation seed group")
    previous = old["evaluation"] if old is not None else None
    if old is not None and (old.get("evaluation_seeds") != list(seeds) or old.get("selection_rule") != BEST_RULE):
        if reevaluate is None:
            raise ValueError("Existing best must be evaluated on the same seed group before comparison")
        stat = reference_path.stat()
        key = (stat.st_mtime_ns, stat.st_size, tuple(seeds))
        if cache is not None and key in cache:
            previous = cache[key]
        else:
            logger.info("[%s][best] 既存bestを共通の%dseedで再評価します（重みは変更しません）", opponent, len(seeds))
            previous = reevaluate(old)
            if previous.get("evaluation_seeds") != list(seeds):
                raise ValueError("Reference best evaluation seed group differs")
            if cache is not None:
                cache[key] = previous
    if previous is not None and best_rank(evaluation) <= best_rank(previous):
        logger.info("[%s][best] 更新なし: 今回=%s 既存=%s (全プラント正解割合, 正解率, 正解時の平均残りtick)",
                    opponent, best_rank(evaluation), best_rank(previous))
        logger.info("[%s][best] 採用モデル: %s", opponent, reference_path.resolve())
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({**identity, "evaluation_seeds": list(seeds), "selection_rule": BEST_RULE,
                "model": model.state_dict(), "completed_sets": evaluation["trained_rounds"] // 12,
                "train_rounds": evaluation["trained_rounds"], "evaluation": evaluation}, temporary)
    temporary.replace(path)
    logger.info("[%s][best] %s: %s", opponent, "初回保存" if old is None else "更新", path.resolve())
    logger.info("[%s][best] 全プラント正解割合=%s 正解率=%s 平均残りtick=%s (共通評価seeds=%s)",
                opponent, percent(evaluation["correct_all_plants"]), percent(evaluation["accuracy"]),
                "-" if evaluation["mean_correct_lead"] is None else f"{evaluation['mean_correct_lead']:.1f}", list(seeds))
    return True


def format_summary(row):
    def ticks(value):
        return "-（正解なし）" if value is None else f"{value:.1f}tick"
    phase = "学習後評価" if row["phase"] == "eval" else "既存bestの比較評価" if row["phase"] == "best_reference" else "学習前の予測"
    def people(key):
        value = row.get(key)
        return "未記録" if value is None else f"{value:.2f}人"
    return (f"[{row['opponent']}][{phase} set={row['set']:03d}] "
            f"正解率={percent(row['accuracy'])} ({row['correct']}/{row['decided']}) "
            f"判断率={percent(row['coverage'])} ({row['decided']}/{row['plants']}) "
            f"正解時の平均残りtick={ticks(row['mean_correct_lead'])} "
            f"中央値={ticks(row['median_correct_lead'])}\n"
            f"  プラント時の平均生存: 味方={people('mean_alive_at_plant')} 相手={people('mean_enemy_alive_at_plant')} "
            f"(プラントした{row['plants']}ラウンド)\n"
            f"  プラント前の平均撃破={people('mean_preplant_defender_kills')} 味方の平均損失={people('mean_preplant_defender_losses')} "
            f"(全{row['rounds']}ラウンド。プラントなしは終了時まで)\n"
            f"  ラウンド終了時の平均生存: 味方={people('mean_alive')} 相手={people('mean_enemy_alive')}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS), default=list(OPPONENTS))
    p.add_argument("--sets", type=int, default=TRAINING_SETS,
                   help="Override TRAINING_SETS (additional 12-round sets per opponent)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--defender-preset", default="Gorigons")
    p.add_argument("--posts", type=Path, help="JSON with five watch/retreat/alternate/look posts")
    p.add_argument("--threshold", type=float, default=.8)
    p.add_argument("--confirm-ticks", type=int, default=3)
    p.add_argument("--peek-ticks", type=int, default=4)
    p.add_argument("--hide-ticks", type=int, default=4)
    p.add_argument("--rotate", action="store_true", help="Move anchors/mid scout according to the prediction")
    p.add_argument("--updates", type=int, default=100, help="Optimizer updates after each 12-round set")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-rounds", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--eval-every", type=int, default=EVALUATION_INTERVAL, help="Multi-seed evaluation every N sets; final set always evaluated (0 disables intermediate evaluations)")
    p.add_argument("--eval-seeds", type=int, default=EVALUATION_SEED_COUNT, help="Number of common evaluation seeds, 12 rounds each")
    p.add_argument("--eval-only", action="store_true", help="Evaluate resumed models without updating weights or replay")
    p.add_argument("--resume", type=Path, help="Existing data run directory with opponent/latest.pt")
    p.add_argument("--data-dir", type=Path, default=HERE / "data" / "analysis", help="Analysis training output directory")
    p.add_argument("--log-dir", type=Path, default=HERE / "logs" / "analysis", help="Analysis logs; training.log is overwritten at startup")
    p.add_argument("--trace-ticks", action="store_true", help="Also save per-tick probabilities and sightings")
    p.add_argument("--detailed-logs", action="store_true", help="Save CSV/JSON summaries, events and engine debug logs (default: training.log only)")
    p.add_argument("--max-round-steps", type=int, default=400)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--describe", action="store_true", help="Print posts and feature schema without starting battles")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.detailed_logs = args.detailed_logs or args.trace_ticks
    positive = ("sets", "confirm_ticks", "peek_ticks", "hide_ticks", "updates", "batch_size", "replay_rounds", "max_round_steps", "torch_threads", "eval_seeds")
    if any(getattr(args, k) < 1 for k in positive) or not .5 < args.threshold < 1 or args.eval_every < 0 or args.learning_rate <= 0:
        raise ValueError("Counts must be positive; .5 < threshold < 1; eval-every >= 0; learning-rate > 0")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    if len(set(args.opponents)) != len(args.opponents):
        raise ValueError("Duplicate opponents are not allowed")
    scenario = Scenario(args.posts.resolve() if args.posts else None)
    fields = FeatureHistory(scenario).fields
    if args.describe:
        print(json.dumps({"scenario": scenario.metadata(), "features": fields, "feature_count": len(fields)}, ensure_ascii=False, indent=2))
        return
    torch.set_num_threads(args.torch_threads)
    data_dir = args.data_dir.resolve()
    resume_dir = args.resume.resolve() if args.resume else None
    if resume_dir is not None and not resume_dir.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {resume_dir}")
    log_dir = args.log_dir.resolve()
    log = RunLog(log_dir, detailed=args.detailed_logs)
    log.logger.info("12-round blocks; opponent-specific models; peek=%d hide=%d; rotation=%s", args.peek_ticks, args.hide_ticks, args.rotate)
    log.logger.info("models=%s logs=%s", data_dir, log_dir)
    log.logger.info("採用用bestの固定保存先=%s", BEST_DIRECTORY)
    log.logger.info("settings=%s", json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False))
    log.jsonl("config.jsonl", {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                              "scenario": scenario.metadata(), "features": fields, "data_dir": str(data_dir)})
    summary = {}
    try:
        for opponent in args.opponents:
            seed_all(args.seed + list(OPPONENTS).index(opponent))
            model = SiteModel(len(fields))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            replay = deque(maxlen=args.replay_rounds)
            rng = np.random.default_rng(args.seed + list(OPPONENTS).index(opponent))
            completed = train_rounds = 0
            checkpoint = data_dir / opponent / "latest.pt"
            best_path = BEST_DIRECTORY / opponent / "analysis_best.pt"
            previous_best_path = BEST_DIRECTORY / opponent / "best.pt"
            fixed_eval_seed = args.seed + 900_000_000 + list(OPPONENTS).index(opponent) * 1_000_000
            common_seeds = [fixed_eval_seed + i * 100 for i in range(args.eval_seeds)]
            best_cache = {}
            if not args.eval_only:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if args.resume:
                state = torch.load(resume_dir / opponent / "latest.pt", map_location="cpu", weights_only=True)
                config = {k: getattr(args, k) for k in state["config"]}
                if (state["version"] != VERSION or state["opponent"] != opponent or state["scenario"] != scenario.signature
                        or state["fields"] != fields or state["config"] != config):
                    raise ValueError(f"Checkpoint schema/config mismatch: {checkpoint}")
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
                replay.extend({"features": r["features"].numpy(), "label": r["label"]} for r in state["replay"])
                completed, train_rounds = state["completed_sets"], state["train_rounds"]
                rng.bit_generator.state = json.loads(state["rng"])
            model.eval()
            training_plan = {"default_training_sets": TRAINING_SETS, "sets_this_run": args.sets,
                             "start_completed_sets": completed, "target_completed_sets": completed + args.sets}
            log.logger.info("[%s][%s計画] 今回=%dセット 既存の学習完了=%dセット 対象=%d～%dセット (TRAINING_SETS=%d)",
                            opponent, "評価" if args.eval_only else "学習", args.sets, completed,
                            completed + 1, completed + args.sets, TRAINING_SETS)
            log.logger.info("[%s][評価計画] %dセットごとと最終セットに%dseed×12ラウンド seeds=%s",
                            opponent, args.eval_every, len(common_seeds), common_seeds)
            def reevaluate_best(saved):
                with torch.random.fork_rng(devices=[]):
                    reference = SiteModel(len(fields))
                reference.load_state_dict(saved["model"])
                reference.eval()
                reference_identity = {"opponent": opponent, "phase": "best_reference",
                                      "set": saved["completed_sets"], "trained_rounds": saved["train_rounds"],
                                      "sets_this_run": args.sets, "start_completed_sets": training_plan["start_completed_sets"],
                                      "replay_rounds": len(replay)}
                return evaluate_seeds(opponent, scenario, reference, args, reference_identity, common_seeds, log)
            for additional in range(1, args.sets + 1):
                set_number = completed + 1 if not args.eval_only else completed + additional
                phase = "eval" if args.eval_only else "train_preupdate"
                seed = (fixed_eval_seed + (additional - 1) * 100 if args.eval_only else
                        args.seed + list(OPPONENTS).index(opponent) * 1_000_000 + set_number * 100)
                identity = {"opponent": opponent, "phase": phase, "set": set_number, "seed": seed,
                            "trained_rounds": train_rounds, "sets_this_run": args.sets,
                            "start_completed_sets": training_plan["start_completed_sets"]}
                started = time.perf_counter()
                callback = lambda r, f, e: log.record_round(identity, r, f, e, args.trace_ticks)
                results, samples = play_block(opponent, scenario, model, args, seed, callback,
                                              log_dir / "engine.log" if args.detailed_logs else None)
                metrics = summarize(results)
                loss = None
                if not args.eval_only:
                    replay.extend(samples)
                    loss = optimize(model, optimizer, replay, rng, args.updates, args.batch_size)
                    completed += 1
                    train_rounds += 12
                    # A complete block is also retained separately as learning data.
                    arrays = {f"round_{i}_features": r["features"] for i, r in enumerate(samples)}
                    arrays["labels"] = np.asarray([r["label"] for r in samples], dtype=np.int8)
                    arrays["round_numbers"] = np.asarray([r["round"] for r in samples], dtype=np.int8)
                    arrays["plant_ticks"] = np.asarray([r["plant_tick"] for r in samples], dtype=np.int16)
                    np.savez_compressed(checkpoint.parent / f"set_{completed:06d}.npz", **arrays)
                    save_checkpoint(checkpoint, model, optimizer, replay, completed, opponent, scenario, fields, args, rng, train_rounds,
                                    training_plan=training_plan)
                block = {**identity, **metrics, "evaluation_seeds": [], "seed_results": [],
                         "loss": loss, "replay_rounds": len(replay), "seconds": time.perf_counter() - started}
                log.jsonl("sets.jsonl", block)
                log.csv("sets.csv", block)
                log.logger.info("%s L/R=%d/%d loss=%s replay=%d seconds=%.1f", format_summary(block),
                    metrics["left"], metrics["right"], "-" if loss is None else f"{loss:.4f}", len(replay), block["seconds"])
                summary[opponent] = block
                if not args.eval_only and ((args.eval_every and completed % args.eval_every == 0) or additional == args.sets):
                    # Always evaluate the final trained model, even for a short
                    # run or eval-every=0, so a usable best model can be selected.
                    identity = {**identity, "phase": "eval", "seed": fixed_eval_seed, "trained_rounds": train_rounds}
                    evaluation = evaluate_seeds(opponent, scenario, model, args,
                                                {**identity, "replay_rounds": len(replay)}, common_seeds, log)
                    log.jsonl("sets.jsonl", evaluation)
                    log.csv("sets.csv", evaluation)
                    log.logger.info("%s", format_summary(evaluation))
                    consider_best(best_path, model, evaluation, opponent, scenario, fields, args, common_seeds,
                                  log.logger, reevaluate=reevaluate_best, cache=best_cache, previous_path=previous_best_path)
                    summary[opponent] = evaluation
                log.write_summary(summary, checkpoint, best_path)
    except KeyboardInterrupt:
        log.logger.info("Interrupted. Completed training blocks are saved; use --resume %s", data_dir)
        return 130
    except Exception:
        log.logger.exception("Training failed; completed checkpoints are preserved. Use --detailed-logs to save opponent diagnostics.")
        raise
    finally:
        for handler in list(log.logger.handlers):
            handler.close()
            log.logger.removeHandler(handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
