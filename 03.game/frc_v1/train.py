"""PPO training CLI with real-game curriculum and optional rule-teacher warmup."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from frc_v1.actions import KINDS, PHASES
from frc_v1 import FACING
from frc_v1.baseline import FrcBaseline
from frc_v1.environment import FrcRoundEnvironment, STAGES
from frc_v1.model import FrcPolicy


def action_record(decision, columns):
    return {"team": np.array((decision.site, PHASES.index(decision.phase), decision.entry, decision.follow,
                            decision.objective, *decision.intents), dtype=np.int64),
        "kind": np.array([KINDS.index(a.kind) for a in decision.actions], dtype=np.int64),
        "facing": np.array([FACING.index(a.facing) for a in decision.actions], dtype=np.int64),
        "target": np.array([a.ally_slot if a.ally_slot is not None else a.target[0] * columns + a.target[1]
                            if a.target is not None else -1 for a in decision.actions], dtype=np.int64)}


def gae(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards), np.float32)
    accumulator, next_value = 0.0, bootstrap
    for i in reversed(range(len(rewards))):
        active = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * next_value * active - values[i]
        accumulator = delta + gamma * lam * active * accumulator
        advantages[i] = accumulator
        next_value = values[i]
    return advantages, advantages + np.asarray(values, np.float32)


def train(args):
    if args.steps < 1 or args.rollout < 1 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("training sizes must be positive")
    if args.imitation_weight < 0:
        raise ValueError("imitation weight must be non-negative")
    if args.stage == "defense" and args.side != "D":
        raise ValueError("defense stage requires side D")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    env = FrcRoundEnvironment(args.side, stage=args.stage, seed=args.seed, opponent=args.opponent_ai,
                              opponent_roster=args.opponent, effects_mode=args.effects)
    observation = env.reset()
    policy = FrcPolicy.load(args.resume, side=args.side, device=args.device, deterministic=False) if args.resume else (
        FrcPolicy(env.controller.snapshot.grid, args.side, device=args.device, deterministic=False, effects_mode=args.effects))
    if policy.effects_mode != args.effects:
        raise ValueError("resume checkpoint effect mode differs; use a separate checkpoint for each ablation")
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.learning_rate)
    output = Path(args.output or f"frc_v1/checkpoints/{args.side}_policy.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    logfile = output.with_suffix(".jsonl")
    teacher = FrcBaseline()
    start, completed, rounds = time.monotonic(), 0, 0

    if args.teacher_steps:
        observations, records = [], []
        for _ in range(args.teacher_steps):
            decision = teacher.act(observation, env.controller.snapshot, env.controller.belief)
            observations.append(observation)
            records.append(action_record(decision, len(env.game.grid[0])))
            result = env.step(decision)
            observation = env.reset() if result.terminated else result.observation
        for _ in range(args.epochs):
            for offset in range(0, len(records), args.batch_size):
                _, log_prob, _, _ = policy.model.distribution(observations[offset:offset + args.batch_size],
                    records=records[offset:offset + args.batch_size])
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
                optimizer.step()
        print(json.dumps({"teacher_steps": args.teacher_steps, "imitation_loss": float(loss.detach())}), flush=True)

    while completed < args.steps:
        observations, critics, records, rewards, values, old_logs, dones = [], [], [], [], [], [], []
        teacher_records = []
        outcomes = []
        navigation_overrides = 0
        navigation_enabled = (args.side == "A" and args.stage in ("attack", "match") or
                              args.side == "D" and args.stage in ("defense", "match"))
        for _ in range(min(args.rollout, args.steps - completed)):
            full_state = env.critic_state()
            if args.imitation_weight:
                teacher_decision = teacher.act(observation, env.controller.snapshot, env.controller.belief)
                teacher_records.append(action_record(teacher_decision, len(env.game.grid[0])))
            # PPO records the sampled action. The same deterministic route
            # wrapper used in matches maps it to the action executed by the
            # environment, so its sampled log-probability remains valid.
            decision = policy.act(observation, env.controller.snapshot, env.controller.belief,
                                  critic=full_state, navigate=navigation_enabled)
            record, log_prob, value = policy.last_sample
            if navigation_enabled:
                navigation_overrides += sum(a.kind != KINDS[int(record["kind"][slot])]
                                            for slot, a in enumerate(decision.actions))
            observations.append(observation)
            critics.append(full_state)
            records.append(record)
            values.append(value)
            old_logs.append(log_prob)
            result = env.step(decision)
            rewards.append(result.reward)
            dones.append(result.terminated)
            completed += 1
            if result.terminated:
                rounds += 1
                outcomes.append(result.metrics)
                observation = env.reset()
            else:
                observation = result.observation
        bootstrap = 0.0
        if not dones[-1]:
            with torch.no_grad():
                feature = policy.model.features([observation])
                full_state = torch.as_tensor(env.critic_state()[None, :], device=feature.device)
                bootstrap = float(policy.model.critic(torch.cat((feature, full_state), dim=1))[0, 0])
        advantages, returns = gae(rewards, values, dones, bootstrap)
        advantages = (advantages - advantages.mean()) / max(1e-6, advantages.std()) if len(advantages) > 1 else advantages
        imitation_loss_value = None
        for _ in range(args.epochs):
            order = np.random.permutation(len(observations))
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                _, log_prob, entropy, predicted_value = policy.model.distribution(
                    [observations[i] for i in indices], records=[records[i] for i in indices], critic=[critics[i] for i in indices])
                device = log_prob.device
                old = torch.as_tensor(np.array(old_logs)[indices], dtype=torch.float32, device=device)
                advantage = torch.as_tensor(advantages[indices], device=device)
                target = torch.as_tensor(returns[indices], device=device)
                ratio = (log_prob - old).exp()
                actor_loss = -torch.min(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
                value_loss = (predicted_value - target).square().mean()
                loss = actor_loss + 0.5 * value_loss - 0.002 * entropy.mean()
                if args.imitation_weight:
                    _, teacher_log_prob, _, _ = policy.model.distribution(
                        [observations[i] for i in indices],
                        records=[teacher_records[i] for i in indices])
                    imitation_loss = -teacher_log_prob.mean()
                    imitation_loss_value = float(imitation_loss.detach())
                    loss = loss + args.imitation_weight * imitation_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite FRC training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
                optimizer.step()
        record = {"steps": completed, "rounds": rounds, "stage": args.stage, "side": args.side,
            "effects": args.effects, "seed": args.seed, "teacher_steps": args.teacher_steps,
            "imitation_weight": args.imitation_weight, "imitation_loss": imitation_loss_value,
            "navigation_enabled": navigation_enabled, "navigation_overrides": navigation_overrides,
            "loss": float(loss.detach()), "mean_reward": float(np.mean(rewards)),
            "elapsed_seconds": round(time.monotonic() - start, 2), "outcomes": outcomes}
        with logfile.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        policy.save(output, training={**record, "opponent": args.opponent, "opponent_ai": args.opponent_ai,
            "smoke_only": args.steps < 1000, "evaluated": False})
        print(json.dumps({key: value for key, value in record.items() if key != "outcomes"}), flush=True)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("A", "D"), required=True)
    parser.add_argument("--stage", choices=STAGES, default="match")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--teacher-steps", type=int, default=0)
    parser.add_argument("--imitation-weight", type=float, default=0.0,
                        help="keep matching the rule teacher on states visited by the policy during PPO")
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent", default="Fnatic2023")
    parser.add_argument("--opponent-ai", default="fnatic_v3")
    parser.add_argument("--effects", choices=("all", "none", "flight", "warning"), default="all")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
