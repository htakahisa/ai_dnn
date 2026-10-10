"""Learn opponent-specific guards from real plant cases; evaluate from setup."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# plant完了 → 50件収集完了 → この学習。通常は定数を確認してオプションなし。
TRAINING_SETS = 30  # 1セット=相手AIの保存ケースをランダム順に各1回学習。
MAX_PARALLEL_WORKERS = 6  # 相手AIごとの最大同時実行数。1で順次実行。

TARGET_OPPONENTS = ("gc_v1", "concon_v1", "omoko_v1", "fnatic_v3", "frc_v1", "toru_ai_v4")
CASES_DIRECTORY = HERE / "data" / "attacker_guard_cases"
DATA_DIRECTORY = HERE / "data" / "attacker_guard"
BEST_DIRECTORY = HERE / "data" / "best"
LOG_DIRECTORY = HERE / "logs" / "attacker_guard"
RESUME_TRAINING = False
EVALUATION_ONLY = False
EVALUATION_INTERVAL = 5  # xセットごとの独立評価。最終セットも評価。
EVALUATION_SEED_COUNT = 3
RANDOM_SEED = 42
OPTIMIZER_UPDATES = 200
BATCH_SIZE = 64
REPLAY_SIZE = 30000
LEARNING_RATE = .001
EPSILON_START = .25
EPSILON_END = .03
TEACHER_START = .8
TEACHER_END = .1
EXPLORATION_DECAY_SETS = 20
DEMONSTRATION_WEIGHT = .05
MAX_ROUND_STEPS = 400
TORCH_THREADS = 1
COLOR_CONSOLE = True
EVALUATION_CONSOLE_COLOR = "\033[1;32m"
# guardの目的は解除を防いで設置後のラウンドに勝つこと。
GUARD_WIN_REWARD = 8.
GUARD_LOSS_REWARD = -8.
SURVIVOR_REWARD = .1
RESOURCE_REWARD = .05
TICK_PENALTY = .003
PROGRESS_REWARD = .02
DAMAGE_PENALTY = .006
DEATH_PENALTY = 1.
KILL_REWARD = .1  # 遅延・設置後勝利を優先し、撃破は補助。
TEAM_DAMAGE_REWARD = .001
ABILITY_USE_PENALTY = .01
DELAY_REWARD = .02
BLIND_COVER_REWARD = .05
CROSSFIRE_REWARD = .02
DEFUSE_APPROACH_REWARD = .1
DEFUSE_STALL_PENALTY = .05

import argparse
from collections import deque
import copy
import hashlib
import json
import logging
import numpy as np
import torch

from touyama_v3.tv3_scenario import Scenario, OPPONENTS
from touyama_v3.tv3_collect_defender_retake import read_rows
from touyama_v3.tv3_learn_attacker_guard import GuardDQN, guard_schema, learn_guard
from touyama_v3.tv3_guard_runtime import (
    CASE_FORMAT, load_sources, replay_case, play_block, summarize_guard, format_guard, guard_best_rank,
)
from touyama_v3.tv3_train_defender_analysis import seed_all
from touyama_v3.tv3_train_attacker_analysis import atomic_save, ConsoleFormatter, console_supports_color


def rewards_config():
    return {"win": GUARD_WIN_REWARD, "loss": GUARD_LOSS_REWARD, "survivor": SURVIVOR_REWARD,
        "reserve": RESOURCE_REWARD, "tick": TICK_PENALTY, "progress": PROGRESS_REWARD,
        "damage": DAMAGE_PENALTY, "death": DEATH_PENALTY, "kill": KILL_REWARD,
        "team_damage": TEAM_DAMAGE_REWARD, "ability_use": ABILITY_USE_PENALTY,
        "delay": DELAY_REWARD, "blind_cover": BLIND_COVER_REWARD, "crossfire": CROSSFIRE_REWARD,
        "defuse_approach": DEFUSE_APPROACH_REWARD, "defuse_stall": DEFUSE_STALL_PENALTY}


def load_dataset(directory, opponents, scenario):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "collection_summary.json").read_text(encoding="utf-8"))
    if manifest.get("format") != CASE_FORMAT or manifest["schema"] != guard_schema(scenario):
        raise ValueError("Guard collection map/schema mismatch")
    if not summary.get("complete"):
        raise ValueError("Guard collection is incomplete; finish collection before training")
    rows = read_rows(directory / "cases.jsonl")
    grouped, sources = {}, {}
    digest = hashlib.sha256((directory / "collection.json").read_bytes() + (directory / "cases.jsonl").read_bytes())
    # Resuming only selected opponents must retain the same dataset identity.
    for row in rows:
        if Path(row["file"]).name != row["file"] or row["source_hashes"] != manifest["source_hashes"].get(row["opponent"]):
            raise ValueError("Guard case source/path mismatch")
        actual_hash = hashlib.sha256((directory / row["file"]).read_bytes()).hexdigest()
        if actual_hash != row["sha256"]:
            raise ValueError(f"Guard case changed: {row['file']}")
        digest.update(actual_hash.encode())
    for opponent in opponents:
        if opponent not in manifest["opponents"]:
            raise ValueError(f"Opponent not in guard collection: {opponent}")
        grouped[opponent] = [row for row in rows if row["opponent"] == opponent]
        if len(grouped[opponent]) < summary["target_per_ai"]:
            raise ValueError(f"Missing guard cases: {opponent}")
        paths = manifest["source_paths"][opponent]
        source = load_sources(scenario, (opponent,), Path(paths["plant"]).parents[1], Path(paths["analysis"]).parents[1])[opponent]
        if not source["fixed_roster_training"]:
            raise ValueError(f"Touyama guard needs matching fixed-five plant data: {opponent}; roster scope mismatch")
        if source["hashes"] != manifest["source_hashes"][opponent]:
            raise ValueError(f"Plant/analysis best changed since collection: {opponent}; collect again after plant training")
        sources[opponent] = source
        for row in grouped[opponent]:
            if Path(row["file"]).name != row["file"] or row["source_hashes"] != source["hashes"]:
                raise ValueError("Guard case source/path mismatch")
    return grouped, sources, digest.hexdigest()


def evaluation(opponent, scenario, source, model, seeds, rewards, max_steps):
    records = []
    plan = []
    for i, seed in enumerate(seeds):
        preset = source["eval_presets"][i % len(source["eval_presets"])]
        rows, _ = play_block(opponent, scenario, source, model, seed, rewards=rewards, max_steps=max_steps,
                             preset_name=preset)
        records.extend(rows)
        plan.append({"seed": seed, "preset": preset})
    return {"metrics": summarize_guard(records), "seeds": list(seeds), "roster_plan": plan}


def log_evaluation(logger, opponent, result, completed):
    logger.info("[%s][実力評価 set=%d][通常setupから・教師/探索なし] %s", opponent, completed,
                format_guard(result["metrics"]), extra={"highlight_evaluation": True})
    for side, row in result["metrics"]["by_site"].items():
        if row["rounds"]:
            logger.info("[%s][サイト別 %s] guard勝率=%.1f%% (%d件)", opponent, side, 100 * row["win_rate"], row["rounds"])
        else:
            logger.info("[%s][サイト別 %s] 実際の設置なし（対象なし）", opponent, side)
    logger.info("[%s][評価詳細] %s", opponent, json.dumps(result, ensure_ascii=False))


def save_latest(path, model, target, optimizer, replay, rng, contract, completed, episodes, reference, schedule=None):
    packed = [[torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for v in row] for row in replay]
    atomic_save(path, {**contract, "model": model.state_dict(), "target": target.state_dict(),
        "optimizer": optimizer.state_dict(), "replay": packed, "rng": json.dumps(rng.bit_generator.state),
        "completed_sets": completed, "trained_guard_episodes": episodes, "last_evaluation": reference,
        "evaluation_schedule": schedule or {"interval": EVALUATION_INTERVAL, "unit": "sets"}})


def consider_best(path, model, contract, completed, episodes, result, previous=None, schedule=None):
    if not result["metrics"]["guard_rounds"]:
        return False
    if previous is not None and guard_best_rank(result["metrics"]) <= guard_best_rank(previous["metrics"]):
        return False
    atomic_save(path, {**contract, "model": model.state_dict(), "completed_sets": completed,
        "trained_guard_episodes": episodes, "evaluation": result,
        "evaluation_schedule": schedule or {"interval": EVALUATION_INTERVAL, "unit": "sets"},
        "selection_rule": "guard_win_full_win_alive_damage_reserve_uses_v1"})
    return True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-workers", type=int, default=MAX_PARALLEL_WORKERS)
    p.add_argument("--sets", type=int, default=TRAINING_SETS)
    p.add_argument("--opponents", nargs="+", choices=TARGET_OPPONENTS, default=list(TARGET_OPPONENTS))
    p.add_argument("--cases-dir", type=Path, default=CASES_DIRECTORY)
    p.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    p.add_argument("--best-dir", type=Path, default=BEST_DIRECTORY)
    p.add_argument("--log-dir", type=Path, default=LOG_DIRECTORY)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME_TRAINING)
    p.add_argument("--eval-only", action=argparse.BooleanOptionalAction, default=EVALUATION_ONLY)
    p.add_argument("--eval-every", type=int, default=EVALUATION_INTERVAL)
    p.add_argument("--eval-seeds", type=int, default=EVALUATION_SEED_COUNT)
    p.add_argument("--updates", type=int, default=OPTIMIZER_UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--replay-size", type=int, default=REPLAY_SIZE)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--max-round-steps", type=int, default=MAX_ROUND_STEPS)
    p.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    p.add_argument("--describe", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    positive = ("sets", "eval_seeds", "updates", "batch_size", "replay_size", "max_round_steps", "torch_threads")
    if min(getattr(args, k) for k in positive) < 1 or args.eval_every < 0 or args.learning_rate <= 0 or args.replay_size < args.batch_size:
        raise ValueError("Positive counts, replay >= batch, and nonnegative evaluation interval required")
    if args.eval_only and not args.resume:
        raise ValueError("EVALUATION_ONLY requires RESUME_TRAINING=True")
    if len(args.opponents) != len(set(args.opponents)) or EXPLORATION_DECAY_SETS < 1:
        raise ValueError("Invalid opponent list or exploration schedule")
    if not 0 <= EPSILON_END <= EPSILON_START <= 1 or not 0 <= TEACHER_END <= TEACHER_START <= 1:
        raise ValueError("Invalid exploration probabilities")
    torch.set_num_threads(args.torch_threads)
    scenario = Scenario()
    cases, sources, dataset_hash = load_dataset(args.cases_dir, args.opponents, scenario)
    schema = guard_schema(scenario)
    rewards = rewards_config()
    schedule = {"interval": args.eval_every, "unit": "sets", "set_unit": "one_pass_all_opponent_cases"}
    config = {"seed": args.seed, "eval_seeds": args.eval_seeds, "rewards": rewards,
        "schedule": [EPSILON_START, EPSILON_END, TEACHER_START, TEACHER_END, EXPLORATION_DECAY_SETS],
        "demonstration_weight": DEMONSTRATION_WEIGHT, "max_steps": args.max_round_steps,
        "evaluation_scope": "fresh_setup_with_fixed_plant_analysis"}
    if args.describe:
        print(json.dumps({"schema": schema, "cases": {k: len(v) for k, v in cases.items()},
                          "dataset_hash": dataset_hash, "source_hashes": {k: v["hashes"] for k, v in sources.items()}}, indent=2))
        return 0
    from touyama_v3.tv3_training_parallel import run_opponent_workers
    status = run_opponent_workers(__file__, argv, args.opponents, args.max_workers, log_dir=args.log_dir)
    if status is not None:
        return status
    args.log_dir.resolve().mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tv3.attacker_guard")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler()
    formatter = ConsoleFormatter("%(asctime)s %(message)s", color_enabled=COLOR_CONSOLE and console_supports_color(console.stream))
    formatter.evaluation_color = EVALUATION_CONSOLE_COLOR
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(args.log_dir.resolve() / "training.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    for handler in (console, file_handler):
        logger.addHandler(handler)
    logger.info("guard学習: 各AIの全保存ケース1巡=1セット、評価間隔=%dセット、左右の均等化なし", args.eval_every)
    logger.info("設定=%s", json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False))
    try:
        for opponent in args.opponents:
            index = list(OPPONENTS).index(opponent)
            seed_all(args.seed + index)
            model = GuardDQN()
            target = copy.deepcopy(model)
            target.eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            replay = deque(maxlen=args.replay_size)
            rng = np.random.default_rng(args.seed + index)
            contract = {"schema": schema, "opponent": opponent, "config": config,
                        "dataset_hash": dataset_hash, "source_hashes": sources[opponent]["hashes"]}
            completed = episodes = 0
            reference = previous = None
            latest = args.data_dir.resolve() / opponent / "latest.pt"
            best_path = args.best_dir.resolve() / opponent / "attacker_guard_best.pt"
            if args.resume:
                state = torch.load(latest, map_location="cpu", weights_only=True)
                if any(state.get(k) != v for k, v in contract.items()):
                    raise ValueError(f"Guard resume dataset/source/config mismatch: {latest}")
                model.load_state_dict(state["model"])
                target.load_state_dict(state["target"])
                optimizer.load_state_dict(state["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
                replay.extend([[v.numpy() if isinstance(v, torch.Tensor) else v for v in row] for row in state["replay"]])
                rng.bit_generator.state = json.loads(state["rng"])
                completed, episodes, reference = state["completed_sets"], state["trained_guard_episodes"], state.get("last_evaluation")
                if best_path.exists():
                    old = torch.load(best_path, map_location="cpu", weights_only=True)
                    if any(old.get(k) != v for k, v in contract.items()):
                        raise ValueError("Guard best conditions differ")
                    previous = old["evaluation"]
                    logger.info("[%s][保存best] 採用set=%d %s", opponent, old["completed_sets"], format_guard(previous["metrics"]))
            model.eval()
            seeds = [args.seed + 700_000_000 + index * 1_000_000 + i * 100 for i in range(args.eval_seeds)]
            if previous is not None and previous["seeds"] != seeds:
                raise ValueError("Guard best evaluation seeds differ")
            if args.eval_only:
                result = evaluation(opponent, scenario, sources[opponent], model, seeds, rewards, args.max_round_steps)
                log_evaluation(logger, opponent, result, completed)
                continue
            tensor_cache = {}
            for additional in range(1, args.sets + 1):
                fraction = min(1., completed / EXPLORATION_DECAY_SETS)
                epsilon = EPSILON_START + fraction * (EPSILON_END - EPSILON_START)
                teacher = TEACHER_START + fraction * (TEACHER_END - TEACHER_START)
                records = []
                for number, case_index in enumerate(rng.permutation(len(cases[opponent]))):
                    case = cases[opponent][int(case_index)]
                    rows, transitions = replay_case(args.cases_dir.resolve() / case["file"], scenario, model,
                        args.seed + index * 1_000_000 + (completed + 1) * 10000 + number,
                        rewards=rewards, epsilon=epsilon, teacher_probability=teacher,
                        tensor_cache=tensor_cache, max_steps=args.max_round_steps)
                    records.extend(rows)
                    replay.extend(transitions)
                    if (number + 1) % 10 == 0:
                        logger.info("[%s][学習 set=%d] ケース進捗=%d/%d", opponent, completed + 1, number + 1, len(cases[opponent]))
                loss = learn_guard(model, target, optimizer, replay, rng, args.updates, args.batch_size, DEMONSTRATION_WEIGHT)
                completed += 1
                episodes += len(records)
                save_latest(latest, model, target, optimizer, replay, rng, contract, completed, episodes, reference, schedule)
                last = "未記録" if reference is None else f"{reference['result']['metrics']['guard_win_rate']:.1%} (set={reference['set']})" if reference['result']['metrics']['guard_win_rate'] is not None else "対象なし"
                logger.info("[%s][学習 set=%d][教師=%.1f%% その他探索=%.1f%%] 直近実力=%s | 収集結果:%s loss=%s", opponent,
                    completed, 100 * teacher, 100 * epsilon, last, format_guard(summarize_guard(records)), loss)
                with (args.log_dir.resolve() / f"{opponent}_rounds.jsonl").open("w" if additional == 1 else "a", encoding="utf-8") as out:
                    for record in records:
                        out.write(json.dumps({"set": completed, **record}, ensure_ascii=False) + "\n")
                if (args.eval_every and completed % args.eval_every == 0) or additional == args.sets:
                    result = evaluation(opponent, scenario, sources[opponent], model, seeds, rewards, args.max_round_steps)
                    log_evaluation(logger, opponent, result, completed)
                    reference = {"set": completed, "result": result}
                    save_latest(latest, model, target, optimizer, replay, rng, contract, completed, episodes, reference, schedule)
                    if consider_best(best_path, model, contract, completed, episodes, result, previous, schedule):
                        previous = result
                        logger.info("[%s] best更新 採用set=%d 保存=%s", opponent, completed, best_path)
                    else:
                        logger.info("[%s] best維持（改善なし、またはguard評価対象なし）=%s", opponent, best_path)
            tensor_cache.clear()
    except KeyboardInterrupt:
        logger.info("中断。完了セットはlatest保存済み。RESUME_TRAINING=Trueで再開できます。")
        return 130
    except Exception:
        logger.exception("guard学習失敗。完了セットのlatestは保持しています。")
        raise
    finally:
        for handler in (console, file_handler):
            handler.close()
            logger.removeHandler(handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
