from __future__ import annotations

import argparse
import copy
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from controllers import DefaultDefenderController
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI
from train_bc import set_global_seed

from ppo_actor_critic import (
    load_actor_critic_from_bc,
    save_ppo_checkpoint,
)
from ppo_attacker_controller import (
    PPOAttackerController,
    RolloutStep,
)


CHECKPOINT_DIR = Path("ppo_attacker_checkpoints")
BEST_MODEL = CHECKPOINT_DIR / "policy_fnatic_attacker_ppo_best.pt"
FINAL_MODEL = CHECKPOINT_DIR / "policy_fnatic_attacker_ppo_final.pt"
LOG_FILE = CHECKPOINT_DIR / "training_log.jsonl"


def make_game(controller: PPOAttackerController) -> VisualFPSBattle:
    attacker_roster, defender_roster = build_two_balanced_rosters()

    kwargs: dict[str, Any] = {
        "headless": True,
        "attacker_roster": attacker_roster,
        "defender_roster": defender_roster,
    }
    if "disable_side_swap" in inspect.signature(
        VisualFPSBattle.__init__
    ).parameters:
        kwargs["disable_side_swap"] = True

    attacker_team_ai = DualRoleTeamAI(
        name="PPO Attacker",
        attacker_factory=lambda: controller,
        defender_factory=lambda: DefaultDefenderController(),
        use_iq_perception=False,
    )
    defender_team_ai = DualRoleTeamAI(
        name="Logic Defender",
        attacker_factory=lambda: DefaultDefenderController(),
        defender_factory=lambda: DefaultDefenderController(),
        use_iq_perception=False,
    )

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_team_ai,
        defender_team_ai,
        **kwargs,
    )
    controller.set_game(game)
    return game


def compute_trajectory_gae(
    steps: list[RolloutStep],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """1選手の時系列だけでGAEを計算する。"""
    rewards = np.asarray([step.reward for step in steps], dtype=np.float32)
    values = np.asarray([step.value for step in steps], dtype=np.float32)
    dones = np.asarray([step.done for step in steps], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0

    for index in reversed(range(len(steps))):
        not_done = 1.0 - dones[index]
        delta = (
            rewards[index]
            + gamma * next_value * not_done
            - values[index]
        )
        last_gae = (
            delta
            + gamma * gae_lambda * not_done * last_gae
        )
        advantages[index] = last_gae
        next_value = values[index]

    returns = advantages + values
    return advantages, returns


def rollout_to_tensors(
    rollout: list[RolloutStep],
    device: torch.device,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> dict[str, torch.Tensor]:
    if not rollout:
        raise ValueError("rolloutが空です")

    # Characterごとに完全に分けてGAEを計算する。
    trajectories: dict[int, list[RolloutStep]] = defaultdict(list)
    for step in rollout:
        trajectories[int(step.trajectory_id)].append(step)

    advantage_by_step: dict[int, float] = {}
    return_by_step: dict[int, float] = {}

    for trajectory_steps in trajectories.values():
        advantages, returns = compute_trajectory_gae(
            trajectory_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for step, advantage, return_value in zip(
            trajectory_steps,
            advantages,
            returns,
        ):
            key = id(step)
            advantage_by_step[key] = float(advantage)
            return_by_step[key] = float(return_value)

    observations = torch.from_numpy(
        np.stack([step.obs for step in rollout]).astype(np.float32)
    ).to(device)
    actions = torch.tensor(
        [step.action for step in rollout],
        dtype=torch.long,
        device=device,
    )
    old_log_probs = torch.tensor(
        [step.old_log_prob for step in rollout],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.tensor(
        [return_by_step[id(step)] for step in rollout],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.tensor(
        [advantage_by_step[id(step)] for step in rollout],
        dtype=torch.float32,
        device=device,
    )
    masks = torch.from_numpy(
        np.stack([step.mask for step in rollout]).astype(np.bool_)
    ).to(device)

    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)

    return {
        "obs": observations,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "returns": returns,
        "advantages": advantages,
        "masks": masks,
        "trajectory_count": len(trajectories),
    }


def ppo_update(
    model,
    reference_model,
    optimizer,
    batch: dict[str, torch.Tensor],
    *,
    epochs: int = 3,
    minibatch_size: int = 512,
    clip_ratio: float = 0.10,
    value_coef: float = 0.5,
    entropy_coef: float = 0.002,
    reference_kl_coef: float = 0.35,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    model.train()
    reference_model.eval()

    total = int(batch["actions"].shape[0])
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "reference_kl": 0.0,
        "count": 0,
    }

    for _ in range(epochs):
        permutation = torch.randperm(
            total,
            device=batch["obs"].device,
        )
        for start in range(0, total, minibatch_size):
            indices = permutation[start:start + minibatch_size]

            observations = batch["obs"][indices]
            masks = batch["masks"][indices]
            actions = batch["actions"][indices]
            old_log_probs = batch["old_log_probs"][indices]
            advantages = batch["advantages"][indices]
            returns = batch["returns"][indices]

            logits, values = model(observations)
            masked_logits = logits.masked_fill(
                ~masks,
                torch.finfo(logits.dtype).min,
            )
            distribution = Categorical(logits=masked_logits)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            ) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)

            with torch.no_grad():
                reference_logits, _ = reference_model(observations)
                reference_masked = reference_logits.masked_fill(
                    ~masks,
                    torch.finfo(reference_logits.dtype).min,
                )
                reference_probs = torch.softmax(reference_masked, dim=1)
                reference_log_probs = torch.log_softmax(
                    reference_masked,
                    dim=1,
                )

            current_log_probs = torch.log_softmax(
                masked_logits,
                dim=1,
            )
            reference_kl = torch.sum(
                reference_probs
                * (reference_log_probs - current_log_probs),
                dim=1,
            ).mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + reference_kl_coef * reference_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
            optimizer.step()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["reference_kl"] += float(reference_kl.item())
            metrics["count"] += 1

    count = max(1, int(metrics.pop("count")))
    return {
        key: value / count
        for key, value in metrics.items()
    }


@torch.no_grad()
def evaluate(
    model,
    device: torch.device,
    *,
    matches: int,
    seed: int,
) -> dict[str, float]:
    model.eval()

    wins = 0
    attacker_rounds = 0
    defender_rounds = 0

    for index in range(matches):
        set_global_seed(seed + index)
        controller = PPOAttackerController(
            model,
            device,
            deterministic=True,
        )
        game = make_game(controller)
        game.run_headless_loop()

        attacker_score = int(getattr(game, "attacker_wins", 0))
        defender_score = int(getattr(game, "defender_wins", 0))
        wins += int(attacker_score > defender_score)
        attacker_rounds += attacker_score
        defender_rounds += defender_score

    return {
        "matches": float(matches),
        "win_rate": wins / matches if matches else 0.0,
        "avg_attacker_rounds": (
            attacker_rounds / matches if matches else 0.0
        ),
        "avg_defender_rounds": (
            defender_rounds / matches if matches else 0.0
        ),
        "avg_round_diff": (
            (attacker_rounds - defender_rounds) / matches
            if matches
            else 0.0
        ),
    }


def append_log(payload: dict[str, Any]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="安定化PPO追加学習（Attacker）"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("policy_fnatic_attacker_dagger_final.pt"),
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--eval-matches", type=int, default=20)
    parser.add_argument("--baseline-matches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollback-win-drop",
        type=float,
        default=0.10,
        help="採用済み基準からこの値以上勝率が落ちたらロールバック",
    )
    parser.add_argument(
        "--rollback-round-drop",
        type=float,
        default=1.5,
        help="採用済み基準から平均取得ラウンドがこの値以上落ちたらロールバック",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません")

    set_global_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_actor_critic_from_bc(args.model, device)
    reference_model = load_actor_critic_from_bc(
        args.model,
        device,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=1e-5,
    )

    print("=" * 76)
    print("Stable PPO Attacker Training")
    print("=" * 76)
    print(f"device          : {device}")
    print(f"initial model   : {args.model}")
    print(f"learning rate   : {args.learning_rate}")
    print("=" * 76)

    # 学習前に同じ初期方策を評価し、最低基準を確定する。
    baseline = evaluate(
        model,
        device,
        matches=args.baseline_matches,
        seed=args.seed + 500_000,
    )
    print(
        "[BASELINE] "
        f"win={baseline['win_rate']:.3f} "
        f"rounds={baseline['avg_attacker_rounds']:.2f}-"
        f"{baseline['avg_defender_rounds']:.2f} "
        f"diff={baseline['avg_round_diff']:+.2f}"
    )
    append_log({"type": "baseline", **baseline})

    accepted_model_state = copy.deepcopy(model.state_dict())
    accepted_optimizer_state = copy.deepcopy(optimizer.state_dict())
    accepted_result = dict(baseline)

    best_win_rate = baseline["win_rate"]
    best_rounds = baseline["avg_attacker_rounds"]
    update_number = 0
    rollout: list[RolloutStep] = []

    save_ppo_checkpoint(
        BEST_MODEL,
        model,
        optimizer,
        update=0,
        episodes=0,
        best_win_rate=best_win_rate,
        extra={"evaluation": baseline, "baseline": True},
    )

    for episode in range(1, args.episodes + 1):
        set_global_seed(args.seed + episode)

        # Rollout収集時は必ずevalモード。Dropoutを無効化。
        model.eval()

        controller = PPOAttackerController(
            model,
            device,
            deterministic=False,
        )
        game = make_game(controller)
        game.run_headless_loop()
        controller.finish_episode()
        rollout.extend(controller.rollout)

        if episode % args.episodes_per_update == 0:
            update_number += 1

            batch = rollout_to_tensors(
                rollout,
                device,
            )
            metrics = ppo_update(
                model,
                reference_model,
                optimizer,
                batch,
            )

            # 更新が終わったら、次のrollout前にevalへ戻す。
            model.eval()

            print(
                f"[UPDATE {update_number:04d}] "
                f"ep={episode} "
                f"steps={len(rollout)} "
                f"trajectories={batch['trajectory_count']} "
                f"reward={controller.episode_reward:.2f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"KL={metrics['reference_kl']:.5f}"
            )
            append_log(
                {
                    "type": "update",
                    "episode": episode,
                    "update": update_number,
                    "steps": len(rollout),
                    "trajectories": batch["trajectory_count"],
                    "episode_reward": controller.episode_reward,
                    **metrics,
                }
            )
            rollout.clear()

        if episode % args.eval_every == 0:
            result = evaluate(
                model,
                device,
                matches=args.eval_matches,
                seed=args.seed + 1_000_000 + episode,
            )

            win_drop = (
                accepted_result["win_rate"] - result["win_rate"]
            )
            round_drop = (
                accepted_result["avg_attacker_rounds"]
                - result["avg_attacker_rounds"]
            )
            rejected = (
                win_drop > args.rollback_win_drop
                or round_drop > args.rollback_round_drop
            )

            if rejected:
                model.load_state_dict(accepted_model_state)
                optimizer.load_state_dict(accepted_optimizer_state)
                model.eval()
                decision = "ROLLBACK"
            else:
                accepted_model_state = copy.deepcopy(
                    model.state_dict()
                )
                accepted_optimizer_state = copy.deepcopy(
                    optimizer.state_dict()
                )
                accepted_result = dict(result)
                decision = "ACCEPT"

                if (
                    result["win_rate"] > best_win_rate
                    or (
                        result["win_rate"] == best_win_rate
                        and result["avg_attacker_rounds"] > best_rounds
                    )
                ):
                    best_win_rate = result["win_rate"]
                    best_rounds = result["avg_attacker_rounds"]
                    save_ppo_checkpoint(
                        BEST_MODEL,
                        model,
                        optimizer,
                        update=update_number,
                        episodes=episode,
                        best_win_rate=best_win_rate,
                        extra={"evaluation": result},
                    )
                    print(f"[BEST] saved: {BEST_MODEL}")

            print(
                f"[EVAL/{decision}] ep={episode} "
                f"win={result['win_rate']:.3f} "
                f"rounds={result['avg_attacker_rounds']:.2f}-"
                f"{result['avg_defender_rounds']:.2f} "
                f"win_drop={win_drop:+.3f} "
                f"round_drop={round_drop:+.2f}"
            )
            append_log(
                {
                    "type": "evaluation",
                    "episode": episode,
                    "decision": decision,
                    "accepted_before": accepted_result
                    if rejected
                    else None,
                    "win_drop": win_drop,
                    "round_drop": round_drop,
                    **result,
                }
            )

            save_ppo_checkpoint(
                CHECKPOINT_DIR / f"ppo_attacker_ep{episode:06d}.pt",
                model,
                optimizer,
                update=update_number,
                episodes=episode,
                best_win_rate=best_win_rate,
                extra={
                    "evaluation": result,
                    "decision": decision,
                },
            )

    if rollout:
        update_number += 1
        batch = rollout_to_tensors(rollout, device)
        ppo_update(
            model,
            reference_model,
            optimizer,
            batch,
        )
        model.eval()

    # 最終出力は最後に採用された安全な状態。
    model.load_state_dict(accepted_model_state)
    optimizer.load_state_dict(accepted_optimizer_state)
    model.eval()

    save_ppo_checkpoint(
        FINAL_MODEL,
        model,
        optimizer,
        update=update_number,
        episodes=args.episodes,
        best_win_rate=best_win_rate,
        extra={
            "accepted_evaluation": accepted_result,
            "baseline_evaluation": baseline,
        },
    )
    print(f"saved: {FINAL_MODEL}")


if __name__ == "__main__":
    main()
