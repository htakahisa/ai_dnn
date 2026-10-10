"""Roster-independent retake policies specialized by attacker AI and actual site."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from collections import deque
import copy
import hashlib
import json
import logging
import re
import numpy as np
import torch

from touyama_v3.tv3_defender_policy import DefenderDQN, OBS_DIM, learn_dqn
from touyama_v3.tv3_defender_controller import load_policy, policy_metadata, DEFENDER_BEST
from touyama_v3.tv3_train_defender_search import (
    rollout, evaluation_plan, eligible_presets, summarize_defender, score_best, atomic_save, evaluation_summary,
    TRAINING_PRESETS as SEARCH_TRAINING_PRESETS, EVALUATION_PRESETS as SEARCH_EVALUATION_PRESETS,
)
from touyama_v3.tv3_scenario import OPPONENTS
from touyama_v3.tv3_train_defender_analysis import seed_all
from touyama_v3.tv3_retake_coordination import RETAKE_VERSION, retake_layout
from touyama_v3.tv3_retake_combat import RETAKE_OBS_DIM, LEGACY_RETAKE_OBS_DIM, initialize_retake
from touyama_v3.tv3_collect_site_sampling import SAMPLING_VERSION, site_sampling

RETAKE_REPLAY_SIZE = 10000  # Per attacker/site; twelve buffers remain bounded.
MAX_PARALLEL_WORKERS = 6  # 相手AIごとの最大同時実行数。左右モデルは同じworker。

# 通常はここを編集し、touyama_v3/ で python tv3_train_defender_retake.py を実行する。
# 1セット = 各相手AIの全保存ケースを1回ずつ学習（左右各50件なら各モデル50ラウンド）。
TRAINING_SETS = 10
TRAINING_OPPONENTS = tuple(OPPONENTS)
TRAINING_PRESETS = SEARCH_TRAINING_PRESETS
EVALUATION_PRESETS = SEARCH_EVALUATION_PRESETS
EVALUATION_INTERVAL = 1  # 毎セット、同じ評価条件で比較して最良モデルを保存する。
EVALUATION_SEED_COUNT = 6  # More natural plants for sites with sparse evaluation samples.
SITE_EVALUATION_SEED_COUNT = 2  # サイト指定収集のモデルは、独立seedの左右指定対戦でも評価。
SITE_EVALUATION_SEED_OFFSET = 500_000_000
SITE_EVALUATION_RIGHT_SEED_OFFSET = 100_000
UPDATES_PER_SET = 100
BATCH_SIZE = 64
RANDOM_SEED = 42
TEACHER_START_PROBABILITY = .95
TEACHER_MIN_PROBABILITY = .10
TEACHER_DECAY_SETS = 60
DEMONSTRATION_WEIGHT = .10
DEMONSTRATION_KIND_WEIGHT = .50  # Learn movement/defuse decisions separately from facing variants.
DEFUSER_SAMPLE_FRACTION = .50  # Mix designated-role examples with all-player replay.
TARGET_RETAKE_WIN_RATE = .50
BELOW_TARGET_UPDATE_MULTIPLIER = 3
TARGET_MIN_EVALUATION_ROUNDS = 12
# Training-only adjustments from per-model public action diagnostics.
MODEL_LEARNING_OVERRIDES = {
    ('frc_v1', 'L'): dict(kind_weight=1.0, defuser_sample_fraction=.75),
    ('frc_v1', 'R'): dict(kind_weight=.75, defuser_sample_fraction=.65),
    ('fnatic_v3', 'L'): dict(kind_weight=.75, defuser_sample_fraction=.65),
    ('fnatic_v3', 'R'): dict(kind_weight=.75, defuser_sample_fraction=.65),
    ('omoko_v1', 'L'): dict(kind_weight=.75, defuser_sample_fraction=.65),
    ('omoko_v1', 'R'): dict(kind_weight=.75, defuser_sample_fraction=.65),
}
COMBAT_ROUND_LOG_ENABLED = True
TRAINING_MODE = "fresh"  # "fresh": 新規学習 / "resume": latestから再開 / "eval": 評価のみ
CASES_DIRECTORY = HERE / "data" / "retake_cases"  # None にすると通常のsearchから学習
SEARCH_DIRECTORY = DEFENDER_BEST
DATA_DIRECTORY = HERE / "data" / "defender"
BEST_DIRECTORY = DEFENDER_BEST
LOG_DIRECTORY = HERE / "logs" / "defender"
CONSOLE_RESULT_COLORS = True  # コンソールの WIN=緑、LOSS=赤。ログファイルは通常の文字列。


class RetakeConsoleFormatter(logging.Formatter):
    def __init__(self, use_colors):
        super().__init__("%(asctime)s %(message)s", "%H:%M:%S")
        self.use_colors = use_colors

    def format(self, record):
        line = super().format(record)
        if self.use_colors:
            line = re.sub(r"\b(WIN|LOSS|LOSE)(?= plant=)",
                          lambda match: ("\033[32m" if match[0] == "WIN" else "\033[31m") + match[0] + "\033[0m",
                          line)
        return line


def enable_console_colors(stream):
    if not CONSOLE_RESULT_COLORS or not stream.isatty():
        return False
    import os
    if os.name == "nt":
        import ctypes
        import msvcrt
        try:
            handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
            mode = ctypes.c_ulong()
            kernel = ctypes.windll.kernel32
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
        except (OSError, ValueError):
            return False
    return True


def training_defaults():
    """Source-recorded settings; CLI flags are optional temporary overrides."""
    return dict(sets=TRAINING_SETS, opponents=list(TRAINING_OPPONENTS),
                train_presets=list(TRAINING_PRESETS), eval_presets=list(EVALUATION_PRESETS),
                eval_every=EVALUATION_INTERVAL, eval_seeds=EVALUATION_SEED_COUNT,
                updates=UPDATES_PER_SET, batch_size=BATCH_SIZE, seed=RANDOM_SEED,
                run_mode=TRAINING_MODE, max_workers=MAX_PARALLEL_WORKERS, cases_dir=CASES_DIRECTORY, search_dir=SEARCH_DIRECTORY,
                data_dir=DATA_DIRECTORY, best_dir=BEST_DIRECTORY, log_dir=LOG_DIRECTORY)


def evaluate_requested_site(opponent, side, opponent_plan, scenario, search, retakes, analyses,
                            *, logger=None, log_context="", round_callback=None):
    """Independent normal-start rounds with a recorded collection site condition."""
    records = []
    for _, preset, seed in opponent_plan[:SITE_EVALUATION_SEED_COUNT]:
        seed += SITE_EVALUATION_SEED_OFFSET + (SITE_EVALUATION_RIGHT_SEED_OFFSET if side == "R" else 0)
        with site_sampling(side, attacker=opponent):
            result, _ = rollout(opponent, preset, scenario, search, retakes, analyses, "retake", seed,
                logger=logger, log_context=log_context, round_callback=round_callback)
        records.extend(r for r in result if r["site"] == side)
    return records


def split_retakes(rounds, samples):
    return {side: ([r for r in rounds if r["site"] == side], [t for t in samples if t[6] == side])
            for side in ("L", "R")}


def train_retake(args, scenario, analyses, hashes):
    torch.set_num_threads(1)
    seed_all(args.seed)
    searches, search_sources, search_hashes = {}, {}, {}
    for opponent in args.opponents:
        source = (args.search_dir or args.best_dir).resolve() / opponent / "search_best.pt"
        search, search_state = load_policy(source, "search", scenario)
        if search_state.get("opponent") != opponent or search_state["analysis_hashes"].get(opponent) != hashes[opponent]:
            raise ValueError(f"Retake requires the corresponding opponent's search and analysis: {source}")
        searches[opponent], search_sources[opponent] = search, source
        search_hashes[opponent] = hashlib.sha256(source.read_bytes()).hexdigest()
    logger = logging.getLogger("touyama_v3_retake")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    directory = args.log_dir.resolve() / "retake"
    directory.mkdir(parents=True, exist_ok=True)
    combat_path = directory / 'combat_rounds.jsonl'
    if COMBAT_ROUND_LOG_ENABLED:
        combat_path.write_text('', encoding='utf8')
    def record_combat_round(record):
        if COMBAT_ROUND_LOG_ENABLED:
            with combat_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    console = logging.StreamHandler()
    console.setFormatter(RetakeConsoleFormatter(enable_console_colors(console.stream)))
    file_handler = logging.FileHandler(directory / "training.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)
    logger.addHandler(file_handler)
    rng = np.random.default_rng(args.seed)
    cases = None
    case_tensor_cache = {}
    dataset_hash = None
    targeted_cases = False
    if args.cases_dir is not None:
        from touyama_v3.tv3_collect_defender_retake import read_rows
        case_dir = args.cases_dir.resolve()
        collection = json.loads((case_dir / "collection.json").read_text(encoding="utf-8"))
        if collection.get("format") != "touyama_v3_plant_cases_v1" or collection.get("scenario") != scenario.signature:
            raise ValueError("Retake case format/map mismatch")
        targeted_cases = collection.get("site_sampling") == "targeted"
        if "sampling_version" in collection:
            summary = json.loads((case_dir / "collection_summary.json").read_text(encoding="utf-8"))
            if not summary.get("complete"):
                raise ValueError("Retake collection is incomplete; finish collection before training")
        rows = read_rows(case_dir / "cases.jsonl")
        cases = {opponent: [r for r in rows if r["opponent"] == opponent] for opponent in args.opponents}
        digest = hashlib.sha256((case_dir / "collection.json").read_bytes() + (case_dir / "cases.jsonl").read_bytes())
        for opponent, subset in cases.items():
            if collection["search_hashes"].get(opponent) != search_hashes[opponent] or collection["analysis_hashes"].get(opponent) != hashes[opponent]:
                raise ValueError(f"Retake case frozen search/analysis mismatch: {opponent}")
            if not subset:
                raise ValueError(f"No retake cases for {opponent}")
            for side in ("L", "R"):
                if not any(r["site"] == side for r in subset):
                    logger.info("[retake] 相手AI=%s サイト=%s 保存ケース0件: このサイトの学習をスキップ", opponent, side)
            for row in subset:
                if row["preset"] not in eligible_presets(args.train_presets, opponent):
                    raise ValueError(f"Case preset is not allowed for training: {row['preset']}")
                if row.get("search_hash") != search_hashes[opponent] or row.get("analysis_hash") != hashes[opponent]:
                    raise ValueError("Indexed case provenance differs from collection")
                path = case_dir / row["file"]
                if path.resolve().parent != case_dir:
                    raise ValueError("Case path must stay in the dataset")
        # Dataset identity must not depend on the worker's selected opponents.
        for row in rows:
            path = case_dir / row["file"]
            if path.resolve().parent != case_dir:
                raise ValueError("Case path must stay in the dataset")
            digest.update(path.read_bytes())
        dataset_hash = digest.hexdigest()
    models, targets, optimizers, replays, contracts, completed, sample_counts = {}, {}, {}, {}, {}, {}, {}
    last_evaluations = {}
    try:
        for opponent in args.opponents:
            search, search_hash = searches[opponent], search_hashes[opponent]
            models[opponent] = {}
            for side in ("L", "R"):
                key = opponent, side
                contract = dict(phase="retake", schema=policy_metadata(scenario, "retake"), opponent=opponent, site=side,
                                analysis_hashes={opponent: hashes[opponent]}, frozen_search_hash=search_hash,
                                training_presets=args.train_presets, evaluation_presets=args.eval_presets, seed=args.seed)
                contract.update(retake_version=RETAKE_VERSION, case_dataset_hash=dataset_hash,
                                retake_layout=retake_layout(scenario), demonstration=dict(start=TEACHER_START_PROBABILITY,
                                minimum=TEACHER_MIN_PROBABILITY, decay=TEACHER_DECAY_SETS, weight=DEMONSTRATION_WEIGHT,
                                kind_weight=MODEL_LEARNING_OVERRIDES.get(key, {}).get('kind_weight', DEMONSTRATION_KIND_WEIGHT),
                                defuser_sample_fraction=MODEL_LEARNING_OVERRIDES.get(key, {}).get('defuser_sample_fraction', DEFUSER_SAMPLE_FRACTION),
                                target=TARGET_RETAKE_WIN_RATE, target_min_rounds=TARGET_MIN_EVALUATION_ROUNDS,
                                extra_updates=BELOW_TARGET_UPDATE_MULTIPLIER))
                if targeted_cases:
                    contract["site_evaluation"] = dict(version=SAMPLING_VERSION, seeds=SITE_EVALUATION_SEED_COUNT,
                        seed_offset=SITE_EVALUATION_SEED_OFFSET, right_seed_offset=SITE_EVALUATION_RIGHT_SEED_OFFSET)
                contracts[key] = contract
                path = (args.best_dir.resolve() / opponent / f"retake_{side}_best.pt" if args.eval_only else
                        args.data_dir.resolve() / "retake" / opponent / f"{side}_latest.pt")
                model = DefenderDQN(RETAKE_OBS_DIM)
                # Transfer only from this project's newly trained generic search.
                initialize_retake(model, search)
                optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
                replay = deque(maxlen=RETAKE_REPLAY_SIZE)
                completed[key] = sample_counts[key] = 0
                if args.resume or args.eval_only:
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if any(saved.get(k) != v for k, v in contract.items()):
                        raise ValueError(f"Retake checkpoint conditions changed: {path}")
                    model.load_state_dict(saved["model"])
                    completed[key] = saved["completed_sets"]
                    sample_counts[key] = saved.get("sample_count", 0)
                    if saved.get('last_evaluation'):
                        last_evaluations[key] = saved['last_evaluation']
                    if not args.eval_only:
                        optimizer.load_state_dict(saved["optimizer"])
                        replay.extend([t[0].numpy(), t[1], t[2], t[3].numpy(), t[4].numpy(), t[5], t[6], t[7], t[8].numpy()] for t in saved["replay"])
                        if saved.get("rng"):
                            rng.bit_generator.state = json.loads(saved["rng"])
                models[opponent][side], targets[key], optimizers[key], replays[key] = model, copy.deepcopy(model), optimizer, replay
                if args.resume and saved.get("target") is not None:
                    targets[key].load_state_dict(saved["target"])
        logger.info("新規retake: 相手AIごとにL/Rの別モデル。任意の5人で重みを共有します")
        logger.info("学習モード=%s sets=%d", "評価のみ" if args.eval_only else "再開" if args.resume else "新規", args.sets)
        logger.info("相手別固定search=%s 学習編成=%s 評価編成=%s", search_sources, args.train_presets, args.eval_presets)
        if cases is not None:
            logger.info("保存状態から学習: data=%s 1セット=相手AIごとの全ケースを1回ずつ学習 件数=%s",
                        case_dir, {opponent: {side: sum(r['site'] == side for r in rows) for side in ('L', 'R')}
                                   for opponent, rows in cases.items()})
        logger.info("採用先=%s/<相手AI>/retake_L/R_best.pt ログ=%s", args.best_dir.resolve(), directory / "training.log")
        comparisons = {}
        target_sets = {opponent: max(completed[opponent, "L"], completed[opponent, "R"]) + args.sets for opponent in args.opponents}
        plan = evaluation_plan(args.opponents, args.eval_presets, args.eval_seeds, args.seed)
        logger.info("best選定: 評価間隔=%dセット、各回同じ編成・seed。リテイク勝率→解除回数→平均報酬で比較", args.eval_every)
        for additional in range(1, (1 if args.eval_only else args.sets) + 1):
            for opponent in args.opponents:
                search = searches[opponent]
                if not args.eval_only:
                    set_no = max(completed[opponent, "L"], completed[opponent, "R"]) + 1
                    names = eligible_presets(args.train_presets, opponent)
                    preset = names[int(rng.integers(len(names)))]
                    seed = args.seed + list(OPPONENTS).index(opponent) * 1000000 + set_no * 100
                    records, samples = [], []
                    if cases is None:
                        sources = [(preset, None)]
                    else:
                        sources = [(cases[opponent][i]["preset"], case_dir / cases[opponent][i]["file"])
                                   for i in rng.permutation(len(cases[opponent]))]
                    for case_no, (preset, path) in enumerate(sources):
                        result, transitions = rollout(opponent, preset, scenario, search, models[opponent], analyses,
                                                   "retake", seed + case_no, training=True,
                                                   epsilon=max(.03, .3 * np.exp(-set_no / 30)),
                                                   teacher_probability=max(TEACHER_MIN_PROBABILITY, TEACHER_START_PROBABILITY
                                                       -(TEACHER_START_PROBABILITY-TEACHER_MIN_PROBABILITY)*(set_no-1)/TEACHER_DECAY_SETS), logger=logger,
                                                   log_context=f"set={set_no}/{target_sets[opponent]} case={case_no + 1}/{len(sources)}",
                                                   round_callback=record_combat_round,
                                                   initial_case=path, case_tensor_cache=case_tensor_cache)
                        records.extend(result)
                        samples.extend(transitions)
                    split = split_retakes(records, samples)
                    for side in ("L", "R"):
                        key = opponent, side
                        if cases is not None and not any(r["site"] == side for r in cases[opponent]):
                            continue
                        subset, transitions = split[side]
                        replays[key].extend(transitions)
                        sample_counts[key] += len(transitions)
                        prior_metrics = last_evaluations.get(key)
                        below_target = prior_metrics is not None and (prior_metrics.get('retake_win_rate') or 0.) < TARGET_RETAKE_WIN_RATE
                        update_count = args.updates * (BELOW_TARGET_UPDATE_MULTIPLIER if below_target else 1)
                        loss = learn_dqn(models[opponent][side], targets[key], optimizers[key], replays[key],
                                         rng, update_count, args.batch_size, demonstration_weight=DEMONSTRATION_WEIGHT,
                                         mission_feature_index=LEGACY_RETAKE_OBS_DIM-1,
                                         mission_sample_fraction=contracts[key]['demonstration']['defuser_sample_fraction'],
                                         demonstration_kind_weight=contracts[key]['demonstration']['kind_weight']) if transitions else None
                        completed[key] = set_no
                        state = {**contracts[key], "model": models[opponent][side].state_dict(),
                                 "target": targets[key].state_dict(), "rng": json.dumps(rng.bit_generator.state),
                                 "optimizer": optimizers[key].state_dict(), "completed_sets": set_no,
                                 "last_evaluation": last_evaluations.get(key),
                                 "sample_count": sample_counts[key],
                                 "replay": [[torch.from_numpy(t[0]), int(t[1]), float(t[2]), torch.from_numpy(t[3]), torch.from_numpy(t[4]), float(t[5]), t[6], int(t[7]), torch.from_numpy(t[8])] for t in replays[key]]}
                        atomic_save(args.data_dir.resolve() / "retake" / opponent / f"{side}_latest.pt", state)
                        logger.info("[retake][学習 set=%d/%d] 相手AI=%s サイト=%s 学習編成=%s プラント=%d サンプル=%d 累計サンプル=%d loss=%s",
                                    set_no, target_sets[opponent], opponent, side, preset, len(subset), len(transitions), sample_counts[key],
                                    "-" if loss is None else f"{loss:.4f}")
                        logger.info("[retake][更新設定] 相手AI=%s サイト=%s updates=%d 前回評価%.0f%%未満=%s", opponent, side, update_count, TARGET_RETAKE_WIN_RATE*100, below_target)
                set_no = max(completed[opponent, "L"], completed[opponent, "R"])
                if args.eval_only or (args.eval_every and set_no % args.eval_every == 0) or additional == args.sets:
                    opponent_plan = [row for row in plan if row[0] == opponent]
                    records = []
                    for _, preset, seed in opponent_plan:
                        result, _ = rollout(opponent, preset, scenario, search, models[opponent], analyses, "retake", seed, logger=logger,
                                            log_context=f"set={set_no}/{target_sets[opponent]} seed={seed}", round_callback=record_combat_round)
                        records.extend(result)
                    for side in ("L", "R"):
                        key = opponent, side
                        subset = [r for r in records if r["site"] == side]
                        if targeted_cases:
                            logger.info("[retake][通常対戦評価] 相手AI=%s サイト=%s %s", opponent, side,
                                        evaluation_summary(summarize_defender(subset)))
                            subset = evaluate_requested_site(opponent, side, opponent_plan, scenario, search,
                                models[opponent], analyses, logger=logger,
                                log_context=f"サイト指定評価={side} set={set_no}", round_callback=record_combat_round)
                            logger.info("[retake][サイト指定評価・best選定] 相手AI=%s サイト=%s", opponent, side)
                        metrics = summarize_defender(subset)
                        if targeted_cases:
                            metrics["site_sampling"] = "targeted"
                        last_evaluations[key] = metrics
                        logger.info("[retake][目標判定] 相手AI=%s サイト=%s 目標=%.0f%% 評価数=%d 判定=%s", opponent, side,
                            TARGET_RETAKE_WIN_RATE*100, metrics['plants'],
                            '評価数不足' if metrics['plants'] < TARGET_MIN_EVALUATION_ROUNDS else
                            '達成' if (metrics.get('retake_win_rate') or 0.) >= TARGET_RETAKE_WIN_RATE else '未達')
                        logger.info("[retake][評価サマリ set=%d] 相手AI=%s サイト=%s %s", set_no, opponent, side, evaluation_summary(metrics))
                        if args.eval_only:
                            continue
                        path = args.best_dir.resolve() / opponent / f"retake_{side}_best.pt"
                        if not subset or sample_counts[key] == 0:
                            logger.info("[best] 相手AI=%s サイト=%s 保存なし: このサイトの実際のプラント/学習データがありません", opponent, side)
                            continue
                        if key not in comparisons and path.exists():
                            prior = torch.load(path, map_location="cpu", weights_only=True)
                            same_conditions = (all(prior.get(k) == v for k, v in contracts[key].items())
                                               and prior.get("evaluation_plan") == opponent_plan)
                            if same_conditions:
                                comparisons[key] = prior["evaluation"]
                            elif args.resume:
                                raise ValueError(f"Existing retake best conditions differ: {path}")
                            else:
                                logger.info("[best] 新規学習の条件に変更されたため、同じ保存先のbestを今回の評価で更新: %s", path)
                        previous = comparisons.get(key)
                        if previous is None or score_best(metrics, "retake") > score_best(previous, "retake"):
                            atomic_save(path, {**contracts[key], "model": models[opponent][side].state_dict(),
                                               "completed_sets": set_no, "sample_count": sample_counts[key],
                                               "evaluation_plan": opponent_plan, "evaluation": metrics})
                            comparisons[key] = metrics
                            logger.info("[best] 相手AI=%s サイト=%s 保存/更新: %s", opponent, side, path)
                        else:
                            logger.info("[best] 相手AI=%s サイト=%s 維持: %s", opponent, side, path)
        logger.info("retake学習/評価完了。相手別・左右別のbestを使用してください")
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    from touyama_v3.tv3_train_defender_search import main
    raise SystemExit(main(["--phase", "retake", *sys.argv[1:]]))
