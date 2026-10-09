"""Train opponent-specific preplant route analysis from real twelve-round matches."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 通常設定：ここを変更して、このディレクトリからオプションなしで実行します。
TRAINING_SETS = 60  # 相手AIごとの追加セット数。1セット = 12ラウンド。
MAX_PARALLEL_WORKERS = 6  # 相手AIごとの最大同時実行数。1で順次実行。

TARGET_OPPONENTS = ("gc_v1", "touyama_v2", "omoko_v1", "fnatic_v3", "frc_v1", "toru_ai_v3")
#TARGET_OPPONENTS = ("frc_v1",)  # 1種類でも末尾のカンマが必要。
TRAINING_PRESETS = ("Gorigons", "EG2023", "Vision Strikers", "Furina Classic", "Fnatic2023",
                    "Touyama Gaming", "Omoko Gaming", "Ghost Champions", "Team Elites")
EVALUATION_PRESETS = ("Eine Kleine", "SUPES", "BBL")
RANDOM_SEED = 42
RESUME_TRAINING = False
EVALUATION_ONLY = False
EVALUATION_INTERVAL = 10
EVALUATION_SEED_COUNT = 3
EXPLORATION_RATE = .8
OPTIMIZER_UPDATES = 100
BATCH_SIZE = 64
REPLAY_ROUNDS = 600
LEARNING_RATE = .001
MAX_ROUND_STEPS = 400
TORCH_THREADS = 1
DATA_DIRECTORY = HERE / "data" / "attacker_analysis"
LOG_DIRECTORY = HERE / "logs" / "attacker_analysis"
BEST_DIRECTORY = HERE / "data" / "best"
COLOR_CONSOLE = True
EVALUATION_CONSOLE_COLOR = "\033[1;32m"  # 実力評価は明るい緑色。

import argparse
from collections import deque
import json
import logging
import os

import numpy as np
import torch

from toruAI_v4.tv4_scenario import Scenario, OPPONENTS
from toruAI_v4.tv4_collect_attacker_analysis import play_block
from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel, optimize
from toruAI_v4.tv4_train_defender_analysis import seed_all
from toruAI_v4.tv4_attacker_rosters import eligible_attacker_presets


def console_supports_color(stream):
    if not COLOR_CONSOLE or not getattr(stream, "isatty", lambda: False)():
        return False
    if os.name != "nt":
        return True
    # Enable ANSI colors on the Windows console without a new dependency.
    try:
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleMode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetConsoleMode.restype = wintypes.BOOL
        kernel.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetConsoleMode.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = wintypes.DWORD()
        return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                    and kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError, ValueError):
        return False


class ConsoleFormatter(logging.Formatter):
    def __init__(self, fmt, *, color_enabled):
        super().__init__(fmt)
        self.color_enabled = color_enabled

    def format(self, record):
        rendered = super().format(record)
        if self.color_enabled and getattr(record, "highlight_evaluation", False):
            return f"{getattr(self, 'evaluation_color', EVALUATION_CONSOLE_COLOR)}{rendered}\033[0m"
        return rendered


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-workers", type=int, default=MAX_PARALLEL_WORKERS)
    p.add_argument("--sets", type=int, default=TRAINING_SETS)
    p.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS), default=list(TARGET_OPPONENTS))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--train-presets", nargs="+", default=list(TRAINING_PRESETS))
    p.add_argument("--eval-presets", nargs="+", default=list(EVALUATION_PRESETS))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME_TRAINING)
    p.add_argument("--eval-only", action=argparse.BooleanOptionalAction, default=EVALUATION_ONLY)
    p.add_argument("--eval-every", type=int, default=EVALUATION_INTERVAL)
    p.add_argument("--eval-seeds", type=int, default=EVALUATION_SEED_COUNT)
    p.add_argument("--exploration", type=float, default=EXPLORATION_RATE)
    p.add_argument("--updates", type=int, default=OPTIMIZER_UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--replay-rounds", type=int, default=REPLAY_ROUNDS)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--max-round-steps", type=int, default=MAX_ROUND_STEPS)
    p.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    p.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    p.add_argument("--log-dir", type=Path, default=LOG_DIRECTORY)
    p.add_argument("--best-dir", type=Path, default=BEST_DIRECTORY)
    p.add_argument("--describe", action="store_true")
    return p


def summarize(rounds):
    from toruAI_v4.tv4_attacker_combat import VIABLE_PLANT_SURVIVORS, VIABLE_PLANT_HP_FRACTION
    plants = [r for r in rounds if r["planted"]]
    initial_alive = sum(r["initial_alive"] for r in rounds)
    initial_enemy_alive = sum(r["initial_enemy_alive"] for r in rounds)
    initial_abilities = sum(r["initial_abilities"] for r in rounds)
    viable = [r for r in plants if r["alive"] >= VIABLE_PLANT_SURVIVORS and
              (not r.get("initial_team_hp") or r["remaining_hp"] / r["initial_team_hp"] >= VIABLE_PLANT_HP_FRACTION)]
    from collections import Counter
    combat_counts = Counter()
    for r in rounds:
        if r.get("combat_diagnostics"):
            combat_counts.update(r["combat_diagnostics"]["counts"])
    return {"rounds": len(rounds), "plants": len(plants),
        "combat_counts": dict(combat_counts),
        "viable_plants": len(viable), "viable_plant_rate": len(viable) / len(rounds) if rounds else 0.,
        "survivor_plant_rate": sum(r["alive"] >= VIABLE_PLANT_SURVIVORS for r in plants) / len(rounds) if rounds else 0.,
        "mean_alive_at_plant": float(np.mean([r["alive"] for r in plants])) if plants else None,
        "mean_remaining_hp": float(np.mean([r.get("remaining_hp", 0.) for r in rounds])) if rounds else 0.,
        "plant_rate": len(plants) / len(rounds) if rounds else 0.,
        "mean_alive": float(np.mean([r["alive"] for r in rounds])) if rounds else 0.,
        "mean_enemy_alive": float(np.mean([r["enemy_alive"] for r in rounds])) if rounds else 0.,
        "ally_survival_rate": sum(r["alive"] for r in rounds) / initial_alive if initial_alive else None,
        "enemy_survival_rate": sum(r["enemy_alive"] for r in rounds) / initial_enemy_alive if initial_enemy_alive else None,
        "ability_use_rate": sum(r["uses"] for r in rounds) / initial_abilities if initial_abilities else None,
        "ability_reserve_rate": sum(r["remaining_abilities_alive"] for r in rounds) / initial_abilities if initial_abilities else None,
        "mean_damage": float(np.mean([r["damage"] for r in rounds])) if rounds else 0.,
        "mean_ability_uses": float(np.mean([r["uses"] for r in rounds])) if rounds else 0.,
        "mean_plant_ticks": float(np.mean([r["tick"] for r in plants])) if plants else None,
        "mean_route_reviews": float(np.mean([len(r.get("route_decisions", [])) for r in rounds])) if rounds else 0.,
        "mean_site_switches": float(np.mean([sum(a["site"] != b["site"] for a, b in zip(
            r.get("route_decisions", []), r.get("route_decisions", [])[1:])) for r in rounds])) if rounds else 0.,
        "timeouts": sum(r["reason"] == "timeout" for r in rounds)}


def format_summary(metrics, plant_label="プラント成功率"):
    def percent(value):
        return "対象なし" if value is None else f"{value:.1%}"
    return (f"{plant_label}={percent(metrics['plant_rate'])} ({metrics['plants']}/{metrics['rounds']}) "
            f"味方生存率={percent(metrics['ally_survival_rate'])} (平均{metrics['mean_alive']:.2f}人) "
            f"敵生存率={percent(metrics['enemy_survival_rate'])} (平均{metrics['mean_enemy_alive']:.2f}人) "
            f"ability温存率={percent(metrics['ability_reserve_rate'])} "
            f"使用率={percent(metrics['ability_use_rate'])} (平均{metrics['mean_ability_uses']:.2f}回) "
            f"被害HP={metrics['mean_damage']:.1f} 残りHP={metrics.get('mean_remaining_hp', 0.):.1f} timeout={metrics['timeouts']} "
            f"経路再評価={metrics.get('mean_route_reviews', 0.):.1f}回/ラウンド "
            f"サイト変更={metrics.get('mean_site_switches', 0.):.1f}回/ラウンド "
            f"3人生存プラント率={metrics.get('survivor_plant_rate', 0.):.1%} "
            f"生存HP条件付きプラント率={metrics.get('viable_plant_rate', 0.):.1%}")


def format_last_evaluation(reference):
    if reference is None:
        return "未記録（次の評価で表示）"
    metrics = reference["evaluation"]["selected"]
    return f"{metrics['plant_rate']:.1%} ({metrics['plants']}/{metrics['rounds']}、set={reference['set']}の評価)"


def log_evaluation(logger, opponent, evaluation, completed):
    logger.info("[%s][実力評価 set=%d][解析選択100%%・ランダム探索なし] %s", opponent, completed,
                format_summary(evaluation["selected"]), extra={"highlight_evaluation": True})
    logger.info("[%s][比較評価 set=%d][ランダム探索100%%] %s", opponent, completed,
                format_summary(evaluation["random"], "ランダム経路の成功率"))
    logger.info("[%s][独立評価 set=%d][詳細] %s", opponent, completed, json.dumps(evaluation, ensure_ascii=False))


@torch.no_grad()
def prediction_metrics(model, samples, fields=None):
    """Holdout rounds weighted equally, rather than overweighting long failures."""
    rows = []
    for sample in samples:
        x = torch.tensor(sample["observations"].astype(np.float32))
        z = torch.tensor(sample["routes"].astype(np.float32))
        y = sample["outcomes"]
        placement, raw = model(x, z)
        predicted = raw.sigmoid().numpy()
        unseen_accuracy = None
        if fields is not None:
            current = sample["observations"][:, [fields.index(f"enemy_{i}_current") for i in range(5)]]
            unseen = (current < .5) & (sample["placements"] != model.regions)
            if unseen.any():
                unseen_accuracy = float((placement.argmax(-1).numpy() == sample["placements"])[unseen].mean())
        rows.append({"brier": float(np.mean((predicted[:, 0] - y[:, 0]) ** 2)),
            "placement_accuracy": float(np.mean(placement.argmax(-1).numpy() == sample["placements"])),
            "unseen_placement_accuracy": unseen_accuracy,
            "damage_mae": float(np.mean(abs(predicted[:, 1] - y[:, 1])) * 500),
            "losses_mae": float(np.mean(abs(predicted[:, 2] - y[:, 2])) * 5),
            "ability_mae": float(np.mean(abs(predicted[:, 3] - y[:, 3])) * 10),
            "plant_ticks_mae": float(np.mean(abs(predicted[:, 4] - y[:, 4])) * 100) if y[0, 0] else None})
    return {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            if any(r[key] is not None for r in rows) else None
            for key in ("brier", "placement_accuracy", "unseen_placement_accuracy", "damage_mae", "losses_mae", "ability_mae", "plant_ticks_mae")}


def evaluate(opponent, scenario, model, config, seeds):
    selected, random_rounds, random_samples = [], [], []
    presets = eligible_attacker_presets(config["eval_presets"], opponent)
    plan = []
    for i, seed in enumerate(seeds):
        preset = presets[i % len(presets)]
        rounds, _ = play_block(opponent, scenario, model, config, seed, exploration=0., preset_name=preset)
        selected.extend(rounds)
        rounds, samples = play_block(opponent, scenario, model, config, seed + 500_000, exploration=1., preset_name=preset)
        random_rounds.extend(rounds)
        random_samples.extend(samples)
        plan.append({"seed": seed, "preset": preset})
    return {"selected": summarize(selected), "random": summarize(random_rounds),
            "prediction": prediction_metrics(model, random_samples, AttackerEncoder(scenario).fields), "seeds": seeds, "roster_plan": plan}


def best_rank(evaluation):
    selected, prediction = evaluation["selected"], evaluation["prediction"]
    return (selected.get("viable_plant_rate", selected["plant_rate"]), selected["plant_rate"], selected["mean_alive"], -selected["mean_damage"],
            -selected["mean_ability_uses"], -(selected["mean_plant_ticks"] or 1000),
            -(prediction["brier"] if prediction["brier"] is not None else 1.))


def atomic_save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_latest(path, model, optimizer, replay, completed, opponent, schema, config, rng, last_evaluation=None):
    packed = [{k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in r.items()}
              for r in replay]
    atomic_save(path, {"schema": schema, "opponent": opponent, "config": config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_sets": completed,
        "trained_rounds": model.trained_rounds, "replay": packed, "rng": json.dumps(rng.bit_generator.state),
        "last_evaluation": last_evaluation})


def main(argv=None):
    args = parser().parse_args(argv)
    positive = ("sets", "eval_seeds", "updates", "batch_size", "replay_rounds", "max_round_steps", "torch_threads")
    if any(getattr(args, k) < 1 for k in positive) or args.eval_every < 0 or not 0 <= args.exploration <= 1 or args.learning_rate <= 0:
        raise ValueError("Counts must be positive, eval-every >= 0, exploration in [0,1], learning-rate > 0")
    if args.eval_only and not args.resume:
        raise ValueError("Evaluation-only requires RESUME_TRAINING=True or --resume")
    if len(args.opponents) != len(set(args.opponents)):
        raise ValueError("Duplicate opponents")
    torch.set_num_threads(args.torch_threads)
    scenario = Scenario()
    if set(args.train_presets) & set(args.eval_presets):
        raise ValueError("Generic analysis evaluation must use presets outside training")
    for opponent in args.opponents:
        eligible_attacker_presets(args.train_presets, opponent)
        eligible_attacker_presets(args.eval_presets, opponent)
    encoder = AttackerEncoder(scenario)
    schema = encoder.schema()
    if args.describe:
        from toruAI_v4.tv4_learn_attacker_analysis import candidate_routes
        origin = tuple(map(int, np.argwhere(scenario.grid == 3)[2]))
        print(json.dumps({"schema": schema, "routes": [{"site": r.site, "branches": r.branches,
            "steps": len(r.cells) - 1, "cells": r.cells} for r in candidate_routes(scenario, origin, 100)]},
            ensure_ascii=False, indent=2))
        return 0
    data_dir, log_dir, best_dir = args.data_dir.resolve(), args.log_dir.resolve(), args.best_dir.resolve()
    from toruAI_v4.tv4_training_parallel import run_opponent_workers
    status = run_opponent_workers(__file__, argv, args.opponents, args.max_workers, log_dir=log_dir)
    if status is not None:
        return status
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tv4.attacker_analysis")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")
    handlers = [console_handler, file_handler]
    console_handler.setFormatter(ConsoleFormatter("%(asctime)s %(message)s",
        color_enabled=console_supports_color(console_handler.stream)))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    for handler in handlers:
        logger.addHandler(handler)
    # Persist execution/evaluation conditions; cosmetic output locations are excluded.
    config = {k: getattr(args, k) for k in ("seed", "train_presets", "eval_presets", "exploration", "max_round_steps", "eval_seeds")}
    config["roster_scope"] = "generic_shared_players_multiple_presets"
    logger.info("attacker analysis: 12ラウンド/セット、相手別モデル、guard対象外")
    logger.info("成功率の見方: [実力評価]=解析で選んだ経路の性能（コンソールでは緑色）。[データ収集]はランダム探索を含む結果。")
    logger.info("設定=%s", json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False))
    try:
        for opponent in args.opponents:
            index = list(OPPONENTS).index(opponent)
            seed_all(args.seed + index)
            model = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(scenario.names))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            replay = deque(maxlen=args.replay_rounds)
            rng = np.random.default_rng(args.seed + index)
            completed = 0
            last_evaluation = None
            directory = data_dir / opponent
            latest = directory / "latest.pt"
            best_path = best_dir / opponent / "attacker_analysis_best.pt"
            if args.resume:
                saved = torch.load(latest, map_location="cpu", weights_only=True)
                if "attacker_preset" in saved.get("config", {}):
                    raise ValueError("This analysis checkpoint was trained with one roster. Multi-roster analysis requires a fresh run; existing files have not been changed.")
                if saved["schema"] != schema or saved["opponent"] != opponent or saved["config"] != config:
                    raise ValueError(f"Resume schema/config mismatch: {latest}")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
                replay.extend({k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in r.items()} for r in saved["replay"])
                completed, model.trained_rounds = saved["completed_sets"], saved["trained_rounds"]
                rng.bit_generator.state = json.loads(saved["rng"])
                last_evaluation = saved.get("last_evaluation")
            model.eval()
            seeds = [args.seed + 900_000_000 + index * 1_000_000 + i * 100 for i in range(args.eval_seeds)]
            old_evaluation = None
            if args.resume and best_path.exists():
                old = torch.load(best_path, map_location="cpu", weights_only=True)
                if old["schema"] != schema or old["config"] != config or old["evaluation"]["seeds"] != seeds:
                    raise ValueError(f"Best evaluation conditions differ: {best_path}")
                old_evaluation = old["evaluation"]
                logger.info("[%s][保存best] 採用set=%d プラント成功率=%.1f%% (%d/%d)", opponent,
                    old["completed_sets"], 100 * old_evaluation["selected"]["plant_rate"],
                    old_evaluation["selected"]["plants"], old_evaluation["selected"]["rounds"])
            if args.eval_only:
                logger.info("[%s] 評価専用 学習済み=%dセット モデル=%s", opponent, completed, latest)
                result = evaluate(opponent, scenario, model, config, seeds)
                log_evaluation(logger, opponent, result, completed)
                continue
            logger.info("[%s] 完了=%d 今回追加=%dセット 保存=%s", opponent, completed, args.sets, latest)
            for additional in range(1, args.sets + 1):
                set_number = completed + 1
                presets = eligible_attacker_presets(args.train_presets, opponent)
                preset = presets[completed % len(presets)]
                rounds, samples = play_block(opponent, scenario, model, config,
                    args.seed + index * 1_000_000 + set_number * 100, args.exploration, preset_name=preset)
                replay.extend(samples)
                loss = optimize(model, optimizer, replay, rng, args.updates, args.batch_size)
                completed += 1
                model.trained_rounds += 12
                directory.mkdir(parents=True, exist_ok=True)
                arrays = {f"round_{r['round']}_{key}": value for r in samples for key, value in r.items() if isinstance(value, np.ndarray)}
                np.savez_compressed(directory / f"set_{completed:06d}.npz", **arrays)
                save_latest(latest, model, optimizer, replay, completed, opponent, schema, config, rng, last_evaluation)
                metrics = summarize(rounds)
                logger.info("[%s][データ収集 set=%d][編成=%s][ランダム探索率=%.0f%%] 直近の実力評価=%s | %s loss=%s",
                    opponent, completed, preset, 100 * args.exploration, format_last_evaluation(last_evaluation),
                    format_summary(metrics, "収集プラント成功率"), loss)
                with (log_dir / f"{opponent}_rounds.jsonl").open("w" if additional == 1 else "a", encoding="utf-8") as output:
                    for row in rounds:
                        output.write(json.dumps({"set": completed, **row}, ensure_ascii=False) + "\n")
                if (args.eval_every and completed % args.eval_every == 0) or additional == args.sets:
                    result = evaluate(opponent, scenario, model, config, seeds)
                    last_evaluation = {"set": completed, "evaluation": result}
                    save_latest(latest, model, optimizer, replay, completed, opponent, schema, config, rng, last_evaluation)
                    log_evaluation(logger, opponent, result, completed)
                    if old_evaluation is None or best_rank(result) > best_rank(old_evaluation):
                        atomic_save(best_path, {"schema": schema, "opponent": opponent, "config": config,
                            "model": model.state_dict(), "trained_rounds": model.trained_rounds,
                            "completed_sets": completed, "evaluation": result,
                            "selection_rule": "viable_plant_then_plant_alive_damage_uses_time_brier_v2"})
                        old_evaluation = result
                        logger.info("[%s] best更新=%s", opponent, best_path)
                    else:
                        logger.info("[%s] best維持=%s", opponent, best_path)
    except KeyboardInterrupt:
        logger.info("中断。完了セットは保存済み。RESUME_TRAINING=Trueで再開できます。")
        return 130
    except Exception:
        logger.exception("学習失敗。完了セットのlatestは保持しています。")
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
