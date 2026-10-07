"""Manually train retakes from real IQ rounds or collected plant cases."""

import argparse
from collections import deque
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from concon_v1.co1_battle_training import OPPONENTS
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_defender_retake_training import DefenderRetakeEnv
from concon_v1.co1_retake_common import RetakeDQN, GAMMA, ACTION_DIM, foundation_preservation_loss
from concon_v1.co1_retake_scenarios import get_scenario, validate_checkpoint, normalize_site_ability_distances, make_checkpoint
from concon_v1.co1_retake_config import DEFAULT_ABILITY_DISTANCES, add_ability_arguments, ability_distances_from_args
from concon_v1.co1_retake_logging import print_summary, print_retry_progress, print_evaluation_summary, format_elapsed_time
from concon_v1.co1_retake_training_schedule import iter_training_windows
from concon_v1.co1_retake_case_training import RetakeCaseDataset, iter_case_training_windows
from concon_v1.co1_retake_config import COORDINATION_VERSION

# 引数なしで実行する場合の学習条件。
DEFAULT_CASE_EPOCHS = 10  # USE_COLLECTED_CASES=True: データ件数×周回数。 500 データ件数 x 10 周回数 = 5000 episodes.
USE_COLLECTED_CASES = True  # True: 保存した設置直後から学習 / False: setup・searchから対戦。
RESUME_TRAINING = False  # True: 左右のlatest・optimizerを再開 / False: 基礎モデルから開始。


CASES_DIR = Path(__file__).resolve().parent / "data" / "defender_retake_cases"
DEFAULT_EPISODES = 2000  # USE_COLLECTED_CASES=False の場合の追加学習回数。
CHECKPOINT_INTERVAL = 100
DEFAULT_EVAL_ROUNDS = None  # Match the checkpoint interval across all opponents.
EVAL_START_EPSILON = 0.05  # 学習時のepsilonがこの値以下になってから評価を開始。
EPSILON_START = 0.15
EPSILON_END = 0.05
EPSILON_DECAY_RATIO = 0.7
FORCE_SAVE = False

BATCH_SIZE = 64
REPLAY_SIZE = 10000
LEARNING_RATE = 3e-4
FOUNDATION_LOSS_WEIGHT = 0.5  # 非交戦時に集合・移動・解除の優先順位を保つ補助損失。
FOUNDATION_MARGIN = 0.5
TARGET_UPDATE_INTERVAL = 1000
OPTIMIZE_INTERVAL = 4


def validate_epsilon_settings(start, end, decay_ratio):
    if (not all(math.isfinite(value) for value in (start, end, decay_ratio))
            or not 0 <= end <= start <= 1 or not 0 < decay_ratio <= 1):
        raise ValueError("epsilon must satisfy 0 <= end <= start <= 1 and 0 < decay ratio <= 1")


def epsilon_by_episode(episode, total=DEFAULT_EPISODES, start=EPSILON_START,
                       end=EPSILON_END, decay_ratio=EPSILON_DECAY_RATIO):
    validate_epsilon_settings(start, end, decay_ratio)
    fraction = min(1., max(0., (episode - 1) / max(1., total * decay_ratio - 1)))
    return end if fraction >= 1 else start + (end - start) * fraction


def optimize(model, target, optimizer, replay, batch_size=BATCH_SIZE):
    if len(replay) < batch_size:
        return None
    batch = random.sample(replay, batch_size)
    # Older in-memory/test transitions have no current-action mask. Their TD
    # update remains valid; do not guess legal actions for auxiliary learning.
    batch = [item if len(item) == 8 else (*item[:5], np.zeros(ACTION_DIM, dtype=bool), *item[5:]) for item in batch]
    obs, actions, rewards, next_obs, masks, current_masks, dones, durations = zip(*batch)
    device = next(model.parameters()).device
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    values = model(obs)
    selected = values.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        greedy = model(next_obs).masked_fill(~masks, -torch.inf).argmax(1)
        following = target(next_obs).gather(1, greedy[:, None]).squeeze(1)
        expected = torch.as_tensor(rewards, dtype=torch.float32, device=device) + (
            GAMMA ** torch.as_tensor(durations, dtype=torch.float32, device=device) * following
            * (1 - torch.as_tensor(dones, dtype=torch.float32, device=device)))
    loss = nn.functional.smooth_l1_loss(selected, expected)
    if model.foundation and FOUNDATION_LOSS_WEIGHT > 0:
        current_masks = torch.as_tensor(np.asarray(current_masks), dtype=torch.bool, device=device)
        loss = loss + FOUNDATION_LOSS_WEIGHT * foundation_preservation_loss(model, obs, values, current_masks, FOUNDATION_MARGIN)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.)
    optimizer.step()
    return float(loss.detach())


def load_training_model(site, directory, distances, resume_directory=None, device="cpu"):
    """Warm-start fresh runs, or add a validated foundation to old battle weights."""
    scenario = get_scenario(site)
    directory = Path(directory)
    source = ((Path(resume_directory) / scenario.model_path("latest").name) if resume_directory is not None
              else directory / scenario.model_path("foundation").name)
    if not source.is_file():
        raise FileNotFoundError(f"missing retake checkpoint: {source}; run co1_train_defender_retake_base.py first")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, scenario, distances)
    foundation = checkpoint
    if checkpoint.get("foundation_version") != 1:
        path = directory / scenario.model_path("foundation").name
        if not path.is_file():
            raise FileNotFoundError(f"old retake weights need a foundation: {path}; run co1_train_defender_retake_base.py first")
        foundation = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(foundation, scenario, distances)
    if foundation.get("foundation_version") != 1 or not (foundation.get("foundation_evaluation") or {}).get("passed"):
        raise ValueError(f"{site}: a validated movement/defuse foundation is required")
    model = RetakeDQN(scenario, foundation=True).to(device)
    model.load_state_dict(foundation["model_state_dict"])
    if foundation is not checkpoint:
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if incompatible.unexpected_keys or any(not key.startswith("foundation_") for key in incompatible.missing_keys):
            raise ValueError("legacy retake checkpoint has incompatible network weights")
        print(f"  {site}: added movement/defuse foundation to resumed battle network", flush=True)
    model.foundation_evaluation = foundation["foundation_evaluation"]
    model.foundation_values.weight.requires_grad_(False)
    print(f"  {site}: starting weights={source.resolve()} foundation_frozen=True", flush=True)
    return model, checkpoint


def summarize(records, opponents, counted_only=False):
    result = {}
    for site in ("L", "R"):
        details = {}
        for opponent in opponents:
            played = [record for record in records if record["opponent"] == opponent]
            retakes = [record for record in played if record["site"] == site and counts_as_retake(record)
                       and (not counted_only or record.get("counted_episode", True))]
            wins = sum(record["defused"] for record in retakes)
            details[opponent] = dict(rounds=len(played), retakes=len(retakes), defuses=wins,
                                     time_expired=sum(record.get("end_reason") in ("detonated", "timeout") for record in retakes),
                                     defender_eliminated=sum(record.get("end_reason") == "defender_eliminated" for record in retakes),
                                     defuse_rate=wins / len(retakes) if retakes else None,
                                     excluded_preplant=sum(not record["planted"] for record in played),
                                     excluded_no_retake=sum(not counts_as_retake(record) for record in played),
                                     excluded_training_quota=sum(record.get("excluded_training_quota", False) and record["site"] == site for record in played),
                                     excluded_other_site=sum(counts_as_retake(record) and record["site"] != site for record in played))
        rates = [item["defuse_rate"] for item in details.values() if item["retakes"]]
        retakes = [record for record in records if record["site"] == site and counts_as_retake(record)
                   and (not counted_only or record.get("counted_episode", True))]
        fire = sum(record["fire_decisions"] for record in retakes)
        moving = sum(record["moving_fire_decisions"] for record in retakes)
        result[site] = dict(opponents=details, retakes=len(retakes),
                            time_expired=sum(item["time_expired"] for item in details.values()),
                            defender_eliminated=sum(item["defender_eliminated"] for item in details.values()),
                            mean_defuse_rate=float(np.mean(rates)) if rates else None,
                            min_defuse_rate=min(rates) if rates else None,
                            moving_fire_rate=moving / fire if fire else 0.,
                            smoke_defuse_decisions=sum(record["smoke_defuse_decisions"] for record in retakes))
    return result


def counts_as_retake(record):
    return bool(record["planted"] and not record.get("excluded_from_retake", False))


def balanced_retake_schedule(opponents, episodes, checkpoint_interval, rng):
    """Balance counted retakes separately in every evaluation window."""
    if not opponents or min(episodes, checkpoint_interval) < 1:
        raise ValueError("a nonempty roster and positive episode/window counts are required")
    for start in range(0, episodes, checkpoint_interval):
        order = list(opponents)
        rng.shuffle(order)
        window = [order[index % len(order)] for index in range(min(checkpoint_interval, episodes - start))]
        rng.shuffle(window)
        yield from window


def iter_retake_rounds(env, opponent, required, epsilon=0., on_step=None, site=None):
    """Retry normal rounds against this opponent until its retake quota is met.

    Yield excluded attempts as well, for logging, without consuming the quota.
    No plant or survivor state is fabricated to make the quota reachable.
    """
    if required < 1:
        raise ValueError("required retake count must be positive")
    if site not in (None, "L", "R"):
        raise ValueError("evaluation site must be L or R")
    completed, attempts = 0, 0
    while completed < required:
        env.reset(opponent=opponent)
        ticks = 0
        while not env.done:
            transitions, _, _ = env.step(epsilon)
            ticks += 1
            if on_step is not None:
                on_step(transitions, ticks)
        record = env.result()
        attempts += 1
        counted = counts_as_retake(record) and (site is None or record["site"] == site)
        completed += int(counted)
        if not counted and attempts % 10 == 0:
            print_retry_progress(opponent, completed, required, attempts, site=site)
        yield dict(record, counted_episode=counted)


def evaluate(models, search_model, rounds=DEFAULT_EVAL_ROUNDS, seed=0, opponents=None, ability_distance=6):
    opponents = tuple(OPPONENTS if opponents is None else opponents)
    if not opponents:
        raise ValueError("evaluation requires a nonempty opponent roster")
    if rounds is None:
        if CHECKPOINT_INTERVAL % len(opponents):
            raise ValueError("specify evaluation rounds for a roster that cannot evenly divide the checkpoint interval")
        rounds = CHECKPOINT_INTERVAL // len(opponents)
    if rounds < 1:
        raise ValueError("evaluation rounds must be positive")
    print(f"Evaluating retake: total_retakes_per_opponent={rounds} site_distribution=natural epsilon=0.000", flush=True)
    env = DefenderRetakeEnv(models, search_model, seed, opponents, ability_distance)
    records = []
    for opponent in opponents:
        records.extend(iter_retake_rounds(env, opponent, rounds))
    result = summarize(records, opponents)
    for site in ("L", "R"):
        result[site].update(evaluation_scope="natural_site_distribution", required_retakes_per_opponent=rounds)
    return result


def train(episodes=None, seed=0, save_dir=None, search_model_path=None,
          resume_dir=None, opponents=None, eval_rounds=DEFAULT_EVAL_ROUNDS,
          checkpoint_interval=CHECKPOINT_INTERVAL, ability_distance=DEFAULT_ABILITY_DISTANCES,
          device="cpu", force_save=FORCE_SAVE, resume=False, epsilon_start=EPSILON_START,
          epsilon_end=EPSILON_END, epsilon_decay_ratio=EPSILON_DECAY_RATIO,
          cases_dir=None, case_epochs=None):
    started_at = time.perf_counter()
    validate_epsilon_settings(epsilon_start, epsilon_end, epsilon_decay_ratio)
    ability_distance = normalize_site_ability_distances(ability_distance)
    if checkpoint_interval < 1 or (eval_rounds is not None and eval_rounds < 1):
        raise ValueError("episodes, evaluation, checkpoint interval and ability distance must be positive")
    opponents = tuple(OPPONENTS if opponents is None else opponents)
    if not opponents or len(set(opponents)) != len(opponents) or any(name not in OPPONENTS for name in opponents):
        raise ValueError("select distinct known opponents")
    if case_epochs is not None and (cases_dir is None or episodes is not None or case_epochs < 1):
        raise ValueError("case epochs require --cases-dir, a positive count, and no explicit episodes")
    dataset = RetakeCaseDataset(cases_dir, opponents, seed) if cases_dir is not None else None
    if episodes is None:
        episodes = (dataset.size * (DEFAULT_CASE_EPOCHS if case_epochs is None else case_epochs)
                    if dataset is not None else DEFAULT_EPISODES)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if dataset is not None and (episodes % len(dataset.groups) or checkpoint_interval % len(dataset.groups)):
        raise ValueError("case episodes and checkpoint interval must be divisible by opponent count * 2 sites")
    if episodes % len(opponents) or checkpoint_interval % len(opponents):
        raise ValueError("episodes and checkpoint interval must be divisible by opponent count for equal team quotas")
    if eval_rounds is None:
        eval_rounds = checkpoint_interval // len(opponents)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Production search is frozen and never placed in an optimizer.
    search = ConconDefenderSearchController(model_path=search_model_path)
    search_path, search_model = search.model_path, search.model
    search_model.requires_grad_(False)
    if dataset is not None:
        dataset.validate_search(search_path)
    models, targets, optimizers, replays, samples, updates, starts, paths = {}, {}, {}, {}, {}, {}, {}, {}
    for site in ("L", "R"):
        scenario = get_scenario(site)
        starts[site], samples[site] = 0, 0
        directory = Path(save_dir) if save_dir is not None else scenario.save_dir
        source_directory = (Path(resume_dir) if resume_dir is not None else directory) if resume_dir is not None or resume else None
        models[site], checkpoint = load_training_model(site, directory, ability_distance[site], source_directory, device)
        optimizers[site] = torch.optim.Adam([parameter for parameter in models[site].parameters() if parameter.requires_grad], lr=LEARNING_RATE)
        if resume_dir is not None or resume:
            starts[site] = int(checkpoint["episode"])
            samples[site] = int(checkpoint.get("retake_transitions", 0))
            if "optimizer_state_dict" in checkpoint:
                optimizers[site].load_state_dict(checkpoint["optimizer_state_dict"])
        targets[site] = copy.deepcopy(models[site]).eval()
        replays[site], updates[site] = deque(maxlen=REPLAY_SIZE), 0
        directory.mkdir(parents=True, exist_ok=True)
        paths[site] = directory
    env = DefenderRetakeEnv(models, search_model, seed, opponents, ability_distance)
    epsilon_settings = dict(start=epsilon_start, end=epsilon_end, decay_ratio=epsilon_decay_ratio)
    print(f"Defender retake: coordination_version={COORDINATION_VERSION} additional_episodes={episodes} checkpoint_interval={checkpoint_interval} "
          f"retakes_per_team_per_window={checkpoint_interval // len(opponents)} "
          f"site_distribution={'balanced_cases' if dataset is not None else 'natural'} force_save={force_save}", flush=True)
    if dataset is not None:
        print(f"  cases: directory={dataset.directory} available={dataset.available_size} "
              f"selected={dataset.size} cases_per_team_per_site={dataset.per_group} "
              f"passes={episodes / dataset.size:g} index_fixed_at_start=True", flush=True)
        print(f"  available cases by opponent/site: {json.dumps(dataset.available_counts)}", flush=True)
        if dataset.available_size != dataset.size:
            print(f"  Using {dataset.per_group} cases per opponent/site for balance; "
                  f"{dataset.available_size - dataset.size} additional cases are outside this run.", flush=True)
    print(f"  epsilon: start={epsilon_start:.3f} end={epsilon_end:.3f} decay_ratio={epsilon_decay_ratio:.3f}", flush=True)
    print(f"  foundation preservation: loss_weight={FOUNDATION_LOSS_WEIGHT:g} margin={FOUNDATION_MARGIN:g} "
          "enabled_outside_firing_contact_including_assembly=True", flush=True)
    for site in ("L", "R"):
        limits = ability_distance[site]
        print(f"  {site} site ability BFS distance: flash={limits['FLASH']} recon={limits['RECON']} smoke={limits['SMOKE']}", flush=True)
    best = {site: None for site in models}
    window = []
    attempts = 0

    def learn_step(transitions, ticks):
        for site, transition in transitions:
            replays[site].append(transition)
            samples[site] += 1
        if transitions and ticks % OPTIMIZE_INTERVAL == 0:
            for site in {site for site, _ in transitions}:
                loss = optimize(models[site], targets[site], optimizers[site], replays[site])
                if loss is not None:
                    updates[site] += 1
                    if updates[site] % TARGET_UPDATE_INTERVAL == 0:
                        targets[site].load_state_dict(models[site].state_dict())

    def quota_progress(opponent, completed, required, attempted):
        print(f"  Waiting for training retakes: {opponent} retakes={completed}/{required} "
              f"attempted_rounds={attempted}", flush=True)

    epsilon_fn = lambda episode: epsilon_by_episode(episode, episodes, epsilon_start, epsilon_end, epsilon_decay_ratio)
    if dataset is None:
        schedule = iter_training_windows(env, opponents, episodes, checkpoint_interval, random.Random(seed),
            epsilon_fn, on_step=learn_step, on_progress=quota_progress)
    else:
        schedule = iter_case_training_windows(env, dataset, episodes, checkpoint_interval, random.Random(seed),
            epsilon_fn, on_step=learn_step)
    previous_boundary = 0
    window_started_at = time.perf_counter()
    for record, offset in schedule:
        now = time.perf_counter()
        record["elapsed_seconds"] = round(now - started_at, 3)
        if offset is not None:
            window_seconds = now - window_started_at
            record["window_training_seconds"] = round(window_seconds, 3)
        attempts = record["round"]
        window.append(record)
        for directory in set(paths.values()):
            with (directory / "retake_training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if offset is None:
            continue
        epsilon = epsilon_by_episode(offset, episodes, epsilon_start, epsilon_end, epsilon_decay_ratio)
        print_summary(summarize(window, opponents, counted_only=True),
                      f"Training summary: episodes={previous_boundary + 1}-{offset} "
                      f"attempted_rounds={len(window)} total_rounds={attempts} epsilon={epsilon:.3f} "
                      f"elapsed_time={format_elapsed_time(now - started_at)} "
                      f"training_time={format_elapsed_time(window_seconds)}")
        previous_boundary = offset
        window.clear()
        evaluation = None
        evaluation_seconds = None
        if epsilon <= EVAL_START_EPSILON:
            evaluation_started_at = time.perf_counter()
            evaluation = evaluate(models, search_model, eval_rounds, seed + 7919, opponents, ability_distance)
            evaluation_seconds = time.perf_counter() - evaluation_started_at
            print_evaluation_summary(evaluation)
            print(f"Evaluation timing: evaluation_time={format_elapsed_time(evaluation_seconds)} "
                  f"elapsed_time={format_elapsed_time(time.perf_counter() - started_at)}", flush=True)
        else:
            print(f"Evaluation skipped: training epsilon={epsilon:.6f} > {EVAL_START_EPSILON:.6f}; "
                  "latest checkpoints will be saved", flush=True)
        for site, model in models.items():
            scenario = get_scenario(site)
            checkpoint = make_checkpoint(model, site, starts[site] + offset, ability_distance,
                                         search_path, opponents, samples[site])
            checkpoint.update(optimizer_state_dict=optimizers[site].state_dict(),
                              evaluation=evaluation[site] if evaluation is not None else None,
                              evaluation_status="completed" if evaluation is not None else "waiting_for_epsilon",
                              eval_start_epsilon=EVAL_START_EPSILON,
                              foundation_behavior_version=3, foundation_loss_weight=FOUNDATION_LOSS_WEIGHT,
                              foundation_margin=FOUNDATION_MARGIN,
                              elapsed_seconds=round(time.perf_counter() - started_at, 3),
                              window_training_seconds=round(window_seconds, 3),
                              evaluation_seconds=round(evaluation_seconds, 3) if evaluation_seconds is not None else None,
                              epsilon=epsilon, epsilon_settings=epsilon_settings, force_save=force_save,
                              episode_counting="team_balanced_retake", retake_episodes_this_run=offset,
                              site_retake_episodes_this_run=record["training_episodes"][site],
                              attempted_rounds_this_run=attempts)
            if dataset is not None:
                checkpoint.update(training_source="collected_retake_cases", cases_directory=str(dataset.directory),
                                  cases_signature=dataset.signature, selected_cases=dataset.size,
                                  cases_per_team_per_site=dataset.per_group, case_passes_this_run=offset / dataset.size)
            torch.save(checkpoint, paths[site] / scenario.model_path("latest").name)
            if force_save:
                torch.save(checkpoint, paths[site] / scenario.model_path(f"episode_{starts[site] + offset}").name)
            if evaluation is None:
                continue
            metric = evaluation[site]
            # No fabricated episodes or score for a site never planted on.
            if not metric["retakes"]:
                print(f"  {site} best: skipped (no evaluation retakes)", flush=True)
                continue
            score = metric["mean_defuse_rate"], metric["min_defuse_rate"], -metric["moving_fire_rate"]
            if samples[site] and (best[site] is None or score > best[site]):
                best[site] = score
                torch.save(checkpoint, paths[site] / scenario.model_path().name)
                print(f"  {site} best: updated; {paths[site] / scenario.model_path().name}", flush=True)
            else:
                print(f"  {site} best: kept" if samples[site] else f"  {site} best: skipped (no training transitions)", flush=True)
        for site in ("L", "R"):
            print(f"Saved {site} latest: {(paths[site] / get_scenario(site).model_path('latest').name).resolve()}", flush=True)
        window_started_at = time.perf_counter()
    print(f"Training complete: episodes={episodes} "
          f"elapsed_time={format_elapsed_time(time.perf_counter() - started_at)}", flush=True)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--episodes", type=int,
                          help=f"additional retakes across both sites; default {DEFAULT_EPISODES} for live rounds, dataset size * case epochs for cases")
    duration.add_argument("--case-epochs", type=int,
                          help=f"passes over selected cases, requires --cases-dir (default {DEFAULT_CASE_EPOCHS})")
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR if USE_COLLECTED_CASES else None,
                        help="collected cases directory; default follows USE_COLLECTED_CASES and CASES_DIR")
    parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-end", type=float, default=EPSILON_END)
    parser.add_argument("--epsilon-decay-ratio", type=float, default=EPSILON_DECAY_RATIO,
                        help="fraction of counted retakes used to decay epsilon")
    parser.add_argument("--search-model", type=Path)
    resuming = parser.add_mutually_exclusive_group()
    resuming.add_argument("--resume-dir", type=Path, help="directory containing both L/R latest checkpoints")
    resuming.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME_TRAINING,
                          help="resume both sites from their save directories; default follows RESUME_TRAINING")
    parser.add_argument("--save-dir", type=Path, help="save both site checkpoints together; default: separate site data directories")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    add_ability_arguments(parser)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help=f"total retakes per balanced training window (default {CHECKPOINT_INTERVAL})")
    parser.add_argument("--eval-rounds", type=int, default=DEFAULT_EVAL_ROUNDS,
                        help="retakes per opponent in evaluation; default: checkpoint interval / opponent count; sites follow natural plants")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--force-save", action=argparse.BooleanOptionalAction, default=FORCE_SAVE,
                        help="keep numbered checkpoints at each interval (disable with --no-force-save)")
    args = parser.parse_args(argv)
    if args.case_epochs is not None and args.cases_dir is None:
        parser.error("--case-epochs requires --cases-dir")
    distances = ability_distances_from_args(args)
    torch.set_num_threads(1)
    train(episodes=args.episodes, seed=args.seed, save_dir=args.save_dir, search_model_path=args.search_model,
          resume_dir=args.resume_dir, opponents=args.opponents, eval_rounds=args.eval_rounds,
          checkpoint_interval=args.checkpoint_interval, ability_distance=distances,
          device=args.device, force_save=args.force_save, resume=args.resume,
          epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
          epsilon_decay_ratio=args.epsilon_decay_ratio, cases_dir=args.cases_dir, case_epochs=args.case_epochs)


if __name__ == "__main__":
    main()
