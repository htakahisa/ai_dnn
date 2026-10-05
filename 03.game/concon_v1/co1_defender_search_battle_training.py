"""Additional Double DQN search learning with a frozen positioning foundation."""

from collections import Counter, deque
from datetime import datetime
import io
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from game_core import FACING_DIRECTIONS, FACING_VECTORS
from concon_v1.co1_defender_common import DefenderSearchDQN

from concon_v1.co1_defender_scenario import get_scenario, validate_checkpoint
from concon_v1.co1_defender_search_common import (
    DefenderSearchBattleDQN, load_search_weights, observation_dim, ACTION_DIM, POLICY_TYPE, GORIGONS,
    SELF_FEATURES, TARGET_COUNT, ENEMY_FEATURES, LOCAL_FEATURES,
)
from concon_v1.co1_defender_search_training import DefenderSearchEnv, START_MODES, OPPONENTS
from concon_v1.co1_defender_search_rewards import GAMMA, REWARD_VERSION
from concon_v1.co1_defender_positioning import evaluate_positioning
from concon_v1.evaluate_co1_defender_search import evaluate, print_summary, survival_summary, survival_log

from concon_v1.co1_train_defender_search import (
    DEFAULT_EPISODES, CHECKPOINT_INTERVAL, DEFAULT_EVAL_ROUNDS, DEFAULT_SUPPORT_DISTANCE,
    EPSILON_START, EPSILON_END, EPSILON_DECAY_RATIO, TARGET_UPDATE_INTERVAL, REPLAY_SIZE,
    BATCH_SIZE, LEARNING_RATE, OPTIMIZE_INTERVAL, GRAD_CLIP_NORM,
    POSITIONING_CHECK_TRIALS, COMBAT_ACTION_MARGIN, COMBAT_LOSS_WEIGHT,
)


def epsilon_by_episode(episode, total=DEFAULT_EPISODES):
    fraction = min(1., max(0., episode / max(1, total * EPSILON_DECAY_RATIO)))
    return EPSILON_END if fraction >= 1 else EPSILON_START + (EPSILON_END - EPSILON_START) * fraction


def balanced_opponent_schedule(opponents, count, rng):
    """Shuffle complete roster passes; a partial pass differs by one game."""
    if not opponents or count < 0:
        raise ValueError("a nonempty opponent roster and nonnegative count are required")
    schedule = []
    while len(schedule) < count:
        roster = list(opponents)
        rng.shuffle(roster)
        schedule.extend(roster[:count - len(schedule)])
    return schedule


def stationary_combat_loss(model, observations, values):
    """Train stop-and-aim Q values above movement Q values at live contact.

    Labels use only existing IQ observations, including current movement
    legality. They never restrict actions or override runtime predictions.
    Blind targets retain the forward-movement exception.
    """
    tactical_start = model.basic_dim + 2 * model.height * model.width
    tactical = observations[:, tactical_start:]
    enemies_end = SELF_FEATURES + TARGET_COUNT * ENEMY_FEATURES
    enemies = tactical[:, SELF_FEATURES:enemies_end].reshape(-1, TARGET_COUNT, ENEMY_FEATURES)
    target_matches = (enemies[:, :, 3:5] - tactical[:, None, 13:15]).abs().amax(2) < 1e-5
    neutralized = (target_matches & (enemies[:, :, 17] > 0) & (enemies[:, :, 14] > 0)).any(1)
    legal = tactical[:, enemies_end:].reshape(-1, 5, LOCAL_FEATURES)[:, :, 0] > 0
    contact = (tactical[:, 15] > 0) & ~neutralized
    watch = (tactical[:, 15] == 0) & (tactical[:, 3] > 0)
    rows = (tactical[:, 0] > 0) & (contact | watch) & legal[:, 4] & legal[:, :4].any(1)
    if not rows.any():
        return values.sum() * 0
    source = observations[rows, model.map_size:model.map_size + 2]
    delta = (tactical[rows, 13:15] - source) * observations.new_tensor([model.height, model.width])
    delta = delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
    vectors = observations.new_tensor([FACING_VECTORS[direction] for direction in FACING_DIRECTIONS])
    alignment = delta[:, 1:2] * vectors[:, 0] + delta[:, 0:1] * vectors[:, 1]
    aimed = alignment >= .7
    valid = aimed.any(1)
    if not valid.any():
        return values.sum() * 0
    stopped = values[rows, 32:40].masked_fill(~aimed, -torch.inf).amax(1)[valid]
    movement_mask = legal[rows, :4].repeat_interleave(8, dim=1)
    moving = values[rows, :32].masked_fill(~movement_mask, -torch.inf).amax(1)[valid]
    return torch.relu(COMBAT_ACTION_MARGIN + moving - stopped).mean()


def memory_navigation_loss(model, observations, values):
    """Keep the learned foundation's movement during memory-only returns.

    This teacher is used only by the optimizer. Runtime still chooses freely
    from the search network's Q values, including every legal facing.
    """
    tactical = observations[:, model.basic_dim + 2 * model.height * model.width:]
    enemies_end = SELF_FEATURES + TARGET_COUNT * ENEMY_FEATURES
    enemies = tactical[:, SELF_FEATURES:enemies_end].reshape(-1, TARGET_COUNT, ENEMY_FEATURES)
    legal = tactical[:, enemies_end:].reshape(-1, 5, LOCAL_FEATURES)[:, :, 0] > 0
    rows = ((tactical[:, 0] > 0) & (tactical[:, 15] == 0) & (tactical[:, 3] == 0)
            & ~(enemies[:, :, 1] > 0).any(1) & (legal.sum(1) > 1))
    if not rows.any():
        return values.sum() * 0
    with torch.no_grad():
        foundation = DefenderSearchDQN.forward(model, observations[rows, :model.basic_dim])
        teacher = foundation.reshape(-1, 5, 8).amax(2).masked_fill(~legal[rows], -torch.inf).argmax(1)
    movement = values[rows, :40].reshape(-1, 5, 8).amax(2)
    desired = movement.gather(1, teacher[:, None]).squeeze(1)
    competitors = legal[rows].clone()
    competitors.scatter_(1, teacher[:, None], False)
    other = movement.masked_fill(~competitors, -torch.inf).amax(1)
    return torch.relu(COMBAT_ACTION_MARGIN + other - desired).mean()


def optimize(model, target, optimizer, replay, batch_size=BATCH_SIZE):
    if len(replay) < batch_size:
        return None
    obs, actions, rewards, next_obs, masks, dones, durations = zip(*random.sample(replay, batch_size))
    device = next(model.parameters()).device
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    masks = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    durations = torch.as_tensor(durations, dtype=torch.float32, device=device)
    values = model(obs)
    selected = values.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        greedy = model(next_obs).masked_fill(~masks, -torch.inf).argmax(1)
        following = target(next_obs).gather(1, greedy[:, None]).squeeze(1)
        expected = rewards + GAMMA ** durations * following * (1 - dones)
    loss = nn.functional.smooth_l1_loss(selected, expected)
    loss = loss + COMBAT_LOSS_WEIGHT * (stationary_combat_loss(model, obs, values)
                                      + memory_navigation_loss(model, obs, values))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], GRAD_CLIP_NORM)
    optimizer.step()
    return float(loss)


def make_battle_checkpoint(model, scenario, episode, run, opponents, positioning):
    return dict(policy_type=POLICY_TYPE, scenario_signature=scenario.signature, positioning_version=1,
        obs_dim=observation_dim(scenario), n_actions=ACTION_DIM, training_roster=list(GORIGONS.players),
        defender_perception="production_iq", phase_scope="setup_search_until_plant_or_elimination",
        reward_version=REWARD_VERSION, support_distance=run["support_distance"], episode=episode,
        battle_training_run=run, battle_episodes_this_run=episode - run["start_episode"],
        opponents=list(opponents), positioning=positioning,
        model_state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})


def qualifies_as_best(candidate, current, reasons=None):
    def decision(accepted, reason):
        if reasons is not None:
            reasons.append(reason)
        return accepted

    if not candidate["positioning"]["passed"]:
        return decision(False, "candidate positioning validation failed")
    if current is None:
        return decision(True, "no comparison evaluation exists; positioning passed")
    candidate_behavior, current_behavior = candidate["behavior"], current["behavior"]
    if (candidate_behavior.get("post_kill_decisions", 0) and current_behavior.get("post_kill_decisions", 0)
            and candidate_behavior["post_kill_hold_rate"] < current_behavior["post_kill_hold_rate"]):
        return decision(False, "post_kill_hold_rate decreased")
    if (candidate_behavior.get("memory_motion_decisions", 0) and current_behavior.get("memory_motion_decisions", 0)
            and candidate_behavior["memory_navigation_error_rate"] > current_behavior["memory_navigation_error_rate"]):
        return decision(False, "memory_navigation_error_rate increased")
    if (candidate["mean_search_score"] < current["mean_search_score"]
            or candidate["min_team_search_score"] < current["min_team_search_score"]):
        return decision(False, "mean_search_score or min_team_search_score decreased")
    if (candidate["mean_search_score"] > current["mean_search_score"]
            or candidate["min_team_search_score"] > current["min_team_search_score"]):
        return decision(True, "search score improved without regression in required metrics")
    accepted = candidate["behavior"]["behavior_error"] <= current["behavior"]["behavior_error"]
    return decision(accepted, "search scores tied; behavior_error " +
                    ("did not increase" if accepted else "increased"))


def comparison_identity(path, checkpoint, source):
    return dict(path=str(path.resolve()), source=source,
                episode=checkpoint.get("episode", 0), policy_type=checkpoint.get("policy_type", "unknown"),
                support_distance=checkpoint.get("support_distance", DEFAULT_SUPPORT_DISTANCE))


def log_best_comparison(candidate, current, candidate_identity, current_identity, conditions, log_path):
    if current is None:
        reasons = []
        accepted = qualifies_as_best(candidate, None, reasons)
        outcome = "CREATE first best from candidate" if accepted else "NO best saved; candidate rejected"
        print("Best model initialization for this training run (previous runs are not a selection baseline):", flush=True)
        print(f"  candidate: {json.dumps(candidate_identity, ensure_ascii=False)}", flush=True)
        print(f"  evaluation: {json.dumps(conditions, ensure_ascii=False)}", flush=True)
        print(f"  positioning.passed: candidate={candidate['positioning']['passed']}", flush=True)
        print(f"  result: {outcome}; reason={reasons[0]}", flush=True)
        record = dict(candidate=candidate_identity, current=None, evaluation_conditions=conditions,
                      candidate_evaluation={key: value for key, value in candidate.items() if key != "opponents"},
                      current_evaluation=None, accepted=accepted, outcome=outcome, reason=reasons[0])
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return accepted
    reasons = []
    accepted = qualifies_as_best(candidate, current, reasons)
    print("Best model comparison:", flush=True)
    print(f"  candidate: {json.dumps(candidate_identity, ensure_ascii=False)}", flush=True)
    print(f"  current:   {json.dumps(current_identity, ensure_ascii=False)}", flush=True)
    print(f"  evaluation: {json.dumps(conditions, ensure_ascii=False)}", flush=True)
    print(f"  positioning.passed: candidate={candidate['positioning']['passed']} "
          f"current={current['positioning']['passed']} (candidate must pass)", flush=True)
    for key in ("mean_search_score", "min_team_search_score"):
        print(f"  {key}: candidate={candidate[key]:.6f} current={current[key]:.6f} "
              f"delta={candidate[key] - current[key]:+.6f} (higher is better)", flush=True)
    cb, bb = candidate["behavior"], current["behavior"]
    for key, higher, count in (("moving_fire_rate", False, None),
                               ("normal_stationary_aligned_fire_rate", True, None),
                               ("post_kill_hold_rate", True, "post_kill_decisions"),
                               ("memory_navigation_error_rate", False, "memory_motion_decisions"),
                               ("behavior_error", False, None)):
        fallback = "stationary_aligned_fire_rate" if key == "normal_stationary_aligned_fire_rate" else key
        cv, bv = cb.get(key, cb.get(fallback, 0.)), bb.get(key, bb.get(fallback, 0.))
        rule = "higher is better" if higher else "lower is better"
        if key in ("moving_fire_rate", "normal_stationary_aligned_fire_rate"):
            rule = "diagnostic only; excluded from best selection"
        if count:
            rule += f"; samples candidate={cb.get(count, 0)} current={bb.get(count, 0)}"
            if not (cb.get(count, 0) and bb.get(count, 0)):
                rule += "; excluded from decision: one or both have no samples"
        if key == "behavior_error":
            rule += "; used only when both search scores tie"
        print(f"  {key}: candidate={cv:.6f} current={bv:.6f} delta={cv - bv:+.6f} ({rule})", flush=True)
    outcome = "UPDATE best with candidate" if accepted else "KEEP current comparison model; candidate rejected"
    print(f"  result: {outcome}; reason={reasons[0]}", flush=True)
    def evaluation_summary(result):
        return {**result, "opponents": {
            name: {key: value for key, value in metrics.items() if key != "details"}
            for name, metrics in result.get("opponents", {}).items()}}

    record = dict(candidate=candidate_identity, current=current_identity, evaluation_conditions=conditions,
                  candidate_evaluation=evaluation_summary(candidate), current_evaluation=evaluation_summary(current),
                  accepted=accepted, outcome=outcome, reason=reasons[0])
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return accepted


def comparison_checkpoint(best_path, resume_bytes, scenario):
    """Keep a valid best, or archive a best belonging to an old map/skill."""
    if not best_path.is_file():
        return resume_bytes
    comparison = best_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(comparison), map_location="cpu", weights_only=False)
    try:
        validate_checkpoint(checkpoint, scenario)
    except ValueError:
        backup = best_path.with_name(
            f"{best_path.stem}_incompatible_{datetime.now():%Y%m%d_%H%M%S_%f}{best_path.suffix}")
        best_path.rename(backup)
        print(f"Archived incompatible comparison model: {backup.resolve()}; "
              "the first validated evaluation candidate in this training run becomes best", flush=True)
        return resume_bytes
    return comparison


def train_battle(episodes=DEFAULT_EPISODES, seed=0, save_dir=None, resume=None, opponents=None,
                 eval_rounds=DEFAULT_EVAL_ROUNDS, checkpoint_interval=CHECKPOINT_INTERVAL,
                 support_distance=DEFAULT_SUPPORT_DISTANCE, device="cpu", start_modes=None, force_save=False):
    if min(episodes, eval_rounds, checkpoint_interval, support_distance) < 1:
        raise ValueError("episodes, evaluation, checkpoint interval and support distance must be positive")
    opponents = tuple(OPPONENTS if opponents is None else opponents)
    if (not opponents or len(set(opponents)) != len(opponents)
            or any(opponent not in OPPONENTS for opponent in opponents)):
        raise ValueError("select distinct known search opponents")
    if start_modes is not None and (not start_modes or any(mode not in START_MODES for mode in start_modes)):
        raise ValueError("select known start modes")
    scenario = get_scenario()
    resume = Path(resume) if resume is not None else scenario.model_path
    if not resume.is_file():
        raise FileNotFoundError(f"train the basic model first: {resume}")
    directory = Path(save_dir) if save_dir is not None else scenario.save_dir
    best_path = directory / scenario.battle_model_path("best").name
    latest_path = directory / scenario.battle_model_path("latest").name
    resume_bytes = resume.read_bytes()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    checkpoint = torch.load(io.BytesIO(resume_bytes), map_location="cpu", weights_only=False)
    model, target = DefenderSearchBattleDQN(scenario).to(device), DefenderSearchBattleDQN(scenario).to(device)
    load_search_weights(model, checkpoint, scenario)
    positioning = evaluate_positioning(model, scenario, seed=seed + 7919, trials=POSITIONING_CHECK_TRIALS)
    if not positioning["passed"]:
        raise RuntimeError("resumed defender basic positioning failed validation")
    # Validate the requested starting model first. An obsolete optional best
    # must not prevent training from a newly rebuilt, valid foundation.
    # Archive it so runtime also falls back to the current basic model until
    # a new best qualifies. Preserve its original bytes for recovery.
    comparison_checkpoint(best_path, resume_bytes, scenario)
    current_identity = None
    if best_path.is_file():
        print(f"Previous run best: {best_path.resolve()}; the first validated evaluation candidate "
              "in this training run will overwrite it", flush=True)
    print("Selecting best within this training run: the first evaluation candidate that passes positioning "
          "will be saved as best; subsequent candidates are compared with this run's best", flush=True)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=LEARNING_RATE)
    first_episode = int(checkpoint.get("episode", 0))
    run = dict(start_episode=first_episode, requested_additional_episodes=episodes,
               episode_start="normal_spawn_5v5_with_setup",
               opponent_sampling="balanced_per_checkpoint_window",
               seed=seed, support_distance=support_distance, foundation_source=str(resume.resolve()))
    print(f"Defender search: additional_episodes={episodes} start_episode={first_episode} "
          f"final_episode={first_episode + episodes} positioning_frozen=True "
          f"support_distance={support_distance} eval_rounds_per_opponent={eval_rounds}", flush=True)
    replay, records = deque(maxlen=REPLAY_SIZE), []
    env = DefenderSearchEnv(model, seed, opponents, support_distance)
    schedule_rng = random.Random(seed)
    opponent_schedule = []
    best_evaluation, global_step = None, 0
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for offset in range(1, episodes + 1):
        episode = first_episode + offset
        window_index = (offset - 1) % checkpoint_interval
        if window_index == 0:
            opponent_schedule = balanced_opponent_schedule(
                opponents, min(checkpoint_interval, episodes - offset + 1), schedule_rng)
        env.reset(opponent=opponent_schedule[window_index])
        epsilon = epsilon_by_episode(offset, episodes)
        reward_sum = 0.
        while not env.done:
            transitions, rewards, _ = env.step(epsilon)
            replay.extend(transitions)
            reward_sum += sum(rewards)
            global_step += 1
            if global_step % OPTIMIZE_INTERVAL == 0:
                optimize(model, target, optimizer, replay)
            if global_step % TARGET_UPDATE_INTERVAL == 0:
                target.load_state_dict(model.state_dict())
        record = dict(episode=episode, epsilon=epsilon, reward=reward_sum, **env.result())
        records.append(record)
        with (directory / "battle_training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if offset % checkpoint_interval != 0 and offset != episodes:
            continue
        print(f"Training summary: episodes={records[0]['episode']}-{episode} epsilon={epsilon:.3f} "
              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
        for opponent in opponents:
            team = [record for record in records if record["opponent"] == opponent]
            if team:
                wins = sum(record["winner"] == "D" for record in team)
                print(f"  {opponent}: defender wins {wins}/{len(team)} ({wins / len(team):.3f}) "
                      f"end={dict(Counter(record['end_reason'] for record in team))}", flush=True)
                survival = survival_summary(team)
                print(f"    {survival_log(survival)}", flush=True)
                print(f"    overall_search_score={survival['mean_search_score']:.3f}", flush=True)
        records.clear()
        candidate = make_battle_checkpoint(model, scenario, episode, run, opponents, positioning)
        candidate["epsilon"] = epsilon
        torch.save(candidate, latest_path)
        if force_save:
            torch.save(candidate, directory / scenario.battle_model_path(f"episode_{episode}").name)
        if epsilon > EPSILON_END:
            print(f"Best comparison skipped: candidate episode={episode} epsilon={epsilon:.6f} "
                  f"> {EPSILON_END}; latest saved: {latest_path.resolve()}", flush=True)
            continue
        candidate_identity = comparison_identity(latest_path, candidate, "current training candidate")
        conditions = dict(seed=seed, opponents=list(opponents), rounds_per_opponent=eval_rounds,
                          epsilon=0., episode_start="normal_spawn_5v5_with_setup")
        print(f"Evaluating best comparison: candidate={json.dumps(candidate_identity, ensure_ascii=False)} "
              f"current={json.dumps(current_identity, ensure_ascii=False)} "
              f"conditions={json.dumps(conditions, ensure_ascii=False)}", flush=True)
        if best_evaluation is not None:
            print("Reusing current comparison evaluation under the same evaluation conditions", flush=True)
        buffer = io.BytesIO()
        torch.save(candidate, buffer)
        result = evaluate(rounds=eval_rounds, seed=seed, opponents=opponents, frozen_checkpoint=buffer.getvalue())
        print("Candidate search policy:", flush=True)
        print_summary(result)
        candidate["evaluation"] = result
        torch.save(candidate, latest_path)
        if log_best_comparison(result, best_evaluation, candidate_identity, current_identity, conditions,
                               directory / "battle_best_comparison_log.jsonl"):
            best_evaluation = result
            torch.save(candidate, best_path)
            current_identity = comparison_identity(best_path, candidate, "best updated during this run")
            print(f"Saved search best: {best_path.resolve()}", flush=True)
    print(f"Saved search latest: {latest_path.resolve()} "
          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return model
