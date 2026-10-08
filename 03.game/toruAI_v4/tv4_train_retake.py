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
import numpy as np
import torch

from toruAI_v4.tv4_defender_policy import DefenderDQN, OBS_DIM, learn_dqn
from toruAI_v4.tv4_defender_controller import load_policy, policy_metadata
from toruAI_v4.tv4_train_defender import (
    rollout, evaluation_plan, eligible_presets, summarize_defender, score_best, atomic_save, evaluation_summary,
)
from toruAI_v4.tv4_scenario import OPPONENTS
from toruAI_v4.tv4_train_analysis import seed_all

RETAKE_REPLAY_SIZE = 10000  # Per attacker/site; twelve buffers remain bounded.


def split_retakes(rounds, samples):
    return {side: ([r for r in rounds if r["site"] == side], [t for t in samples if t[6] == side])
            for side in ("L", "R")}


def train_retake(args, scenario, analyses, hashes):
    torch.set_num_threads(1)
    seed_all(args.seed)
    searches, search_sources, search_hashes = {}, {}, {}
    for opponent in args.opponents:
        source = args.best_dir.resolve() / opponent / "search_best.pt"
        search, search_state = load_policy(source, "search", scenario)
        if search_state.get("opponent") != opponent or search_state["analysis_hashes"].get(opponent) != hashes[opponent]:
            raise ValueError(f"Retake requires the corresponding opponent's search and analysis: {source}")
        searches[opponent], search_sources[opponent] = search, source
        search_hashes[opponent] = hashlib.sha256(source.read_bytes()).hexdigest()
    logger = logging.getLogger("toru_v4_retake")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    directory = args.log_dir.resolve() / "retake"
    directory.mkdir(parents=True, exist_ok=True)
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / "training.log", mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    rng = np.random.default_rng(args.seed)
    models, targets, optimizers, replays, contracts, completed, sample_counts = {}, {}, {}, {}, {}, {}, {}
    try:
        for opponent in args.opponents:
            search, search_hash = searches[opponent], search_hashes[opponent]
            models[opponent] = {}
            for side in ("L", "R"):
                key = opponent, side
                contract = dict(phase="retake", schema=policy_metadata(scenario), opponent=opponent, site=side,
                                analysis_hashes={opponent: hashes[opponent]}, frozen_search_hash=search_hash,
                                training_presets=args.train_presets, evaluation_presets=args.eval_presets, seed=args.seed)
                contracts[key] = contract
                path = (args.best_dir.resolve() / opponent / f"retake_{side}_best.pt" if args.eval_only else
                        args.data_dir.resolve() / "retake" / opponent / f"{side}_latest.pt")
                model = DefenderDQN(OBS_DIM)
                # Transfer only from this project's newly trained generic search.
                model.load_state_dict(search.state_dict())
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
                    if not args.eval_only:
                        optimizer.load_state_dict(saved["optimizer"])
                        replay.extend([t[0].numpy(), t[1], t[2], t[3].numpy(), t[4].numpy(), t[5], t[6]] for t in saved["replay"])
                        if saved.get("rng"):
                            rng.bit_generator.state = json.loads(saved["rng"])
                models[opponent][side], targets[key], optimizers[key], replays[key] = model, copy.deepcopy(model), optimizer, replay
                if args.resume and saved.get("target") is not None:
                    targets[key].load_state_dict(saved["target"])
        logger.info("新規retake: 相手AIごとにL/Rの別モデル。任意の5人で重みを共有します")
        logger.info("学習モード=%s sets=%d", "評価のみ" if args.eval_only else "再開" if args.resume else "新規", args.sets)
        logger.info("相手別固定search=%s 学習編成=%s 評価編成=%s", search_sources, args.train_presets, args.eval_presets)
        logger.info("採用先=%s/<相手AI>/retake_L/R_best.pt ログ=%s", args.best_dir.resolve(), directory / "training.log")
        comparisons = {}
        target_sets = {opponent: max(completed[opponent, "L"], completed[opponent, "R"]) + args.sets for opponent in args.opponents}
        plan = evaluation_plan(args.opponents, args.eval_presets, args.eval_seeds, args.seed)
        for additional in range(1, (1 if args.eval_only else args.sets) + 1):
            for opponent in args.opponents:
                search = searches[opponent]
                if not args.eval_only:
                    set_no = max(completed[opponent, "L"], completed[opponent, "R"]) + 1
                    names = eligible_presets(args.train_presets, opponent)
                    preset = names[int(rng.integers(len(names)))]
                    seed = args.seed + list(OPPONENTS).index(opponent) * 1000000 + set_no * 100
                    records, samples = rollout(opponent, preset, scenario, search, models[opponent], analyses,
                                               "retake", seed, training=True, epsilon=max(.03, .3 * np.exp(-set_no / 30)),
                                               teacher_probability=max(0., 1 - (set_no - 1) / 5), logger=logger,
                                               log_context=f"set={set_no}/{target_sets[opponent]}")
                    split = split_retakes(records, samples)
                    for side in ("L", "R"):
                        key = opponent, side
                        subset, transitions = split[side]
                        replays[key].extend(transitions)
                        sample_counts[key] += len(transitions)
                        loss = learn_dqn(models[opponent][side], targets[key], optimizers[key], replays[key],
                                         rng, args.updates, args.batch_size) if transitions else None
                        completed[key] = set_no
                        state = {**contracts[key], "model": models[opponent][side].state_dict(),
                                 "target": targets[key].state_dict(), "rng": json.dumps(rng.bit_generator.state),
                                 "optimizer": optimizers[key].state_dict(), "completed_sets": set_no,
                                 "sample_count": sample_counts[key],
                                 "replay": [[torch.from_numpy(t[0]), int(t[1]), float(t[2]), torch.from_numpy(t[3]), torch.from_numpy(t[4]), float(t[5]), t[6]] for t in replays[key]]}
                        atomic_save(args.data_dir.resolve() / "retake" / opponent / f"{side}_latest.pt", state)
                        logger.info("[retake][学習 set=%d/%d] 相手AI=%s サイト=%s 学習編成=%s プラント=%d サンプル=%d 累計サンプル=%d loss=%s",
                                    set_no, target_sets[opponent], opponent, side, preset, len(subset), len(transitions), sample_counts[key],
                                    "-" if loss is None else f"{loss:.4f}")
                set_no = max(completed[opponent, "L"], completed[opponent, "R"])
                if args.eval_only or (args.eval_every and set_no % args.eval_every == 0) or additional == args.sets:
                    opponent_plan = [row for row in plan if row[0] == opponent]
                    records = []
                    for _, preset, seed in opponent_plan:
                        result, _ = rollout(opponent, preset, scenario, search, models[opponent], analyses, "retake", seed, logger=logger,
                                            log_context=f"set={set_no}/{target_sets[opponent]} seed={seed}")
                        records.extend(result)
                    for side in ("L", "R"):
                        key = opponent, side
                        subset = [r for r in records if r["site"] == side]
                        metrics = summarize_defender(subset)
                        logger.info("[retake][評価サマリ set=%d] 相手AI=%s サイト=%s %s", set_no, opponent, side, evaluation_summary(metrics))
                        if args.eval_only:
                            continue
                        path = args.best_dir.resolve() / opponent / f"retake_{side}_best.pt"
                        if not subset or sample_counts[key] == 0:
                            logger.info("[best] 相手AI=%s サイト=%s 保存なし: このサイトの実際のプラント/学習データがありません", opponent, side)
                            continue
                        if key not in comparisons and path.exists():
                            _, prior = load_policy(path, "retake", scenario)
                            if any(prior.get(k) != v for k, v in contracts[key].items()):
                                raise ValueError(f"Existing retake best conditions differ: {path}")
                            if prior.get("evaluation_plan") != opponent_plan:
                                raise ValueError("Retake evaluation seed/preset plan changed; use a separate best directory")
                            comparisons[key] = prior["evaluation"]
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
    from toruAI_v4.tv4_train_defender import main
    main(["--phase", "retake", *sys.argv[1:]])
