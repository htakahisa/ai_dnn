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
    is_ppo_checkpoint,
    load_actor_critic_from_bc,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)
from ppo_attacker_controller import (
    PPOAttackerController,
    RolloutStep,
)


CHECKPOINT_DIR = Path("ppo_attacker_checkpoints")
BEST_MODEL = CHECKPOINT_DIR / "policy_fnatic_attacker_ppo_best.pt"
FINAL_MODEL = CHECKPOINT_DIR / "policy_fnatic_attacker_ppo_final.pt"
LATEST_MODEL = CHECKPOINT_DIR / "policy_fnatic_attacker_ppo_latest.pt"
INTERRUPT_MODEL = CHECKPOINT_DIR / "policy_fnatic_attacker_ppo_interrupt.pt"
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
    epochs: int = 1,
    minibatch_size: int = 512,
    clip_ratio: float = 0.05,
    value_coef: float = 0.5,
    entropy_coef: float = 0.002,
    reference_kl_coef: float = 0.10,
    max_reference_kl: float = 0.15,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    # Rollout収集時と同じくDropoutを無効にしたまま更新する。
    # eval()でも勾配計算とoptimizer.step()は可能。
    model.eval()
    reference_model.eval()

    total = int(batch["actions"].shape[0])
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "reference_kl": 0.0,
        "ratio_mean": 0.0,
        "ratio_max": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "skipped_minibatches": 0.0,
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

            ratio_mean = float(ratio.mean().item())
            ratio_max = float(ratio.max().item())
            clip_fraction = float(
                (
                    (ratio < 1.0 - clip_ratio)
                    | (ratio > 1.0 + clip_ratio)
                )
                .float()
                .mean()
                .item()
            )

            # Referenceから離れすぎたミニバッチは更新しない。
            if (
                not torch.isfinite(reference_kl)
                or float(reference_kl.item()) > max_reference_kl
            ):
                metrics["skipped_minibatches"] += 1.0
                metrics["reference_kl"] += float(
                    reference_kl.item()
                    if torch.isfinite(reference_kl)
                    else max_reference_kl
                )
                metrics["ratio_mean"] += ratio_mean
                metrics["ratio_max"] = max(
                    metrics["ratio_max"],
                    ratio_max,
                )
                metrics["clip_fraction"] += clip_fraction
                metrics["count"] += 1
                continue

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + reference_kl_coef * reference_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
            optimizer.step()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["reference_kl"] += float(reference_kl.item())
            metrics["ratio_mean"] += ratio_mean
            metrics["ratio_max"] = max(
                metrics["ratio_max"],
                ratio_max,
            )
            metrics["clip_fraction"] += clip_fraction
            metrics["grad_norm"] += float(grad_norm)
            metrics["count"] += 1

    count = max(1, int(metrics.pop("count")))
    skipped = metrics["skipped_minibatches"]
    ratio_max = metrics["ratio_max"]

    return {
        "policy_loss": metrics["policy_loss"] / count,
        "value_loss": metrics["value_loss"] / count,
        "entropy": metrics["entropy"] / count,
        "reference_kl": metrics["reference_kl"] / count,
        "ratio_mean": metrics["ratio_mean"] / count,
        "ratio_max": ratio_max,
        "clip_fraction": metrics["clip_fraction"] / count,
        "grad_norm": metrics["grad_norm"] / count,
        "skipped_minibatches": skipped,
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


def save_runtime_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    update_number: int,
    total_episode: int,
    best_win_rate: float,
    best_rounds: float,
    accepted_result: dict[str, Any],
    rollout: list[RolloutStep],
    interrupted: bool = False,
    latest: bool = False,
) -> None:
    save_ppo_checkpoint(
        path,
        model,
        optimizer,
        update=update_number,
        episodes=total_episode,
        best_win_rate=best_win_rate,
        extra={
            "best_rounds": float(best_rounds),
            "accepted_result": dict(accepted_result),
            "rollout": list(rollout),
            "interrupted": bool(interrupted),
            "latest": bool(latest),
        },
    )



def checkpoint_result(
    metadata: dict[str, Any],
    *,
    fallback_win_rate: float = 0.0,
    fallback_rounds: float = 0.0,
) -> dict[str, float]:
    """チェックポイント内の評価情報を統一形式で取得する。"""
    for key in (
        "evaluation",
        "accepted_evaluation",
        "accepted_result",
        "best_evaluation",
        "baseline_evaluation",
    ):
        candidate = metadata.get(key)
        if isinstance(candidate, dict):
            return {
                "win_rate": float(
                    candidate.get("win_rate", fallback_win_rate)
                ),
                "avg_attacker_rounds": float(
                    candidate.get(
                        "avg_attacker_rounds",
                        fallback_rounds,
                    )
                ),
                "avg_defender_rounds": float(
                    candidate.get("avg_defender_rounds", 0.0)
                ),
                "avg_round_diff": float(
                    candidate.get("avg_round_diff", 0.0)
                ),
                "matches": float(candidate.get("matches", 0.0)),
            }

    return {
        "win_rate": float(
            metadata.get("best_win_rate", fallback_win_rate)
        ),
        "avg_attacker_rounds": float(
            metadata.get("best_rounds", fallback_rounds)
            or fallback_rounds
        ),
        "avg_defender_rounds": 0.0,
        "avg_round_diff": 0.0,
        "matches": 0.0,
    }


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
        "--save-every",
        type=int,
        default=5,
        help="latestチェックポイントの自動保存間隔。0以下で無効",
    )
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
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "PPOチェックポイントから再開する場合に"
            "optimizer状態も引き継ぐ"
        ),
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

    resume_metadata: dict[str, Any] = {}
    continuing_from_ppo = is_ppo_checkpoint(
        args.model,
        device,
    )

    if continuing_from_ppo:
        model, resume_metadata = load_ppo_checkpoint(
            args.model,
            device,
        )

        # PPO継続時のKL基準は、再開時点の方策を固定コピーする。
        reference_model = copy.deepcopy(model).to(device)
    else:
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

    if (
        continuing_from_ppo
        and args.resume_optimizer
        and resume_metadata.get("optimizer_state_dict") is not None
    ):
        try:
            optimizer.load_state_dict(
                resume_metadata["optimizer_state_dict"]
            )
            # CLIで指定した学習率を優先する。
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
            print("[RESUME] optimizer state restored")
        except Exception as exc:
            print(
                "[RESUME WARNING] optimizer stateを復元できません。"
                f"新規optimizerを使用します: {exc}"
            )

    print("=" * 76)
    print("Stable PPO Attacker Training")
    print("=" * 76)
    print(f"device          : {device}")
    print(f"initial model   : {args.model}")
    print(
        "resume mode     : "
        + ("PPO checkpoint" if continuing_from_ppo else "BC/DAgger initialization")
    )
    if continuing_from_ppo:
        print(
            f"previous episode: "
            f"{int(resume_metadata.get('episodes', 0))}"
        )
        print(
            f"previous update : "
            f"{int(resume_metadata.get('update', 0))}"
        )
    print(f"learning rate   : {args.learning_rate}")
    print("=" * 76)

    # 学習前に同じ初期方策を評価し、最低基準を確定する。
    # BASELINE中のCtrl+Cも捕捉し、再開可能なチェックポイントを残す。
    try:
        baseline = evaluate(
            model,
            device,
            matches=args.baseline_matches,
            seed=args.seed + 500_000,
        )
    except KeyboardInterrupt:
        previous_episodes_on_interrupt = (
            int(resume_metadata.get("episodes", 0))
            if continuing_from_ppo
            else 0
        )
        previous_updates_on_interrupt = (
            int(resume_metadata.get("update", 0))
            if continuing_from_ppo
            else 0
        )
        previous_best_win_rate = (
            float(resume_metadata.get("best_win_rate", 0.0))
            if continuing_from_ppo
            else 0.0
        )

        print(
            "\n[INTERRUPT] BASELINE評価中にCtrl+Cを受信しました。"
            "現在のモデルを保存しています..."
        )
        save_ppo_checkpoint(
            INTERRUPT_MODEL,
            model,
            optimizer,
            update=previous_updates_on_interrupt,
            episodes=previous_episodes_on_interrupt,
            best_win_rate=previous_best_win_rate,
            extra={
                "interrupted": True,
                "interrupt_stage": "baseline",
                "rollout": [],
            },
        )
        save_ppo_checkpoint(
            LATEST_MODEL,
            model,
            optimizer,
            update=previous_updates_on_interrupt,
            episodes=previous_episodes_on_interrupt,
            best_win_rate=previous_best_win_rate,
            extra={
                "interrupted": True,
                "latest": True,
                "interrupt_stage": "baseline",
                "rollout": [],
            },
        )
        append_log(
            {
                "type": "interrupt",
                "stage": "baseline",
                "episode": previous_episodes_on_interrupt,
                "update": previous_updates_on_interrupt,
                "checkpoint": str(INTERRUPT_MODEL),
            }
        )
        print(f"[INTERRUPT] saved: {INTERRUPT_MODEL}")
        print(f"[LATEST] saved: {LATEST_MODEL}")
        return

    print(
        "[BASELINE] "
        f"win={baseline['win_rate']:.3f} "
        f"rounds={baseline['avg_attacker_rounds']:.2f}-"
        f"{baseline['avg_defender_rounds']:.2f} "
        f"diff={baseline['avg_round_diff']:+.2f}"
    )
    append_log({"type": "baseline", **baseline})

    previous_episodes = (
        int(resume_metadata.get("episodes", 0))
        if continuing_from_ppo
        else 0
    )
    previous_updates = (
        int(resume_metadata.get("update", 0))
        if continuing_from_ppo
        else 0
    )
    update_number = previous_updates

    accepted_model_state = copy.deepcopy(model.state_dict())
    accepted_optimizer_state = copy.deepcopy(optimizer.state_dict())
    accepted_result = dict(baseline)

    # 現在モデルを暫定BESTにし、既存BESTがより良ければ読み込む。
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_result = dict(baseline)
    best_win_rate = float(baseline["win_rate"])
    best_rounds = float(baseline["avg_attacker_rounds"])
    best_checkpoint_loaded = False

    if BEST_MODEL.exists():
        try:
            stored_best_model, stored_best_metadata = load_ppo_checkpoint(
                BEST_MODEL,
                device,
            )
            if (
                stored_best_model.obs_size == model.obs_size
                and stored_best_model.num_actions == model.num_actions
            ):
                stored_result = checkpoint_result(
                    stored_best_metadata,
                    fallback_win_rate=float(
                        stored_best_metadata.get(
                            "best_win_rate",
                            0.0,
                        )
                    ),
                    fallback_rounds=float(
                        stored_best_metadata.get(
                            "best_rounds",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                stored_win = max(
                    float(stored_result["win_rate"]),
                    float(
                        stored_best_metadata.get(
                            "best_win_rate",
                            0.0,
                        )
                    ),
                )
                stored_rounds = float(
                    stored_result["avg_attacker_rounds"]
                )

                if (
                    stored_win > best_win_rate
                    or (
                        stored_win == best_win_rate
                        and stored_rounds > best_rounds
                    )
                ):
                    best_model_state = copy.deepcopy(
                        stored_best_model.state_dict()
                    )
                    best_result = dict(stored_result)
                    best_result["win_rate"] = stored_win
                    best_win_rate = stored_win
                    best_rounds = stored_rounds

                    stored_optimizer = stored_best_metadata.get(
                        "optimizer_state_dict"
                    )
                    if stored_optimizer is not None:
                        temp_optimizer = torch.optim.AdamW(
                            stored_best_model.parameters(),
                            lr=args.learning_rate,
                            eps=1e-5,
                            weight_decay=1e-5,
                        )
                        temp_optimizer.load_state_dict(
                            stored_optimizer
                        )
                        for group in temp_optimizer.param_groups:
                            group["lr"] = args.learning_rate
                        best_optimizer_state = copy.deepcopy(
                            temp_optimizer.state_dict()
                        )

                    best_checkpoint_loaded = True
                    print(
                        "[BEST RESTORE TARGET] "
                        f"win={best_win_rate:.3f} "
                        f"rounds={best_rounds:.2f} "
                        f"file={BEST_MODEL}"
                    )
        except Exception as exc:
            print(
                "[BEST WARNING] 既存BESTを読み込めません。"
                f"現在モデルを基準にします: {exc}"
            )

    baseline_best_win_drop = (
        best_win_rate - float(baseline["win_rate"])
    )
    baseline_best_round_drop = (
        best_rounds - float(baseline["avg_attacker_rounds"])
    )

    if best_checkpoint_loaded and (
        baseline_best_win_drop >= args.rollback_win_drop
        or baseline_best_round_drop >= args.rollback_round_drop
    ):
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        model.eval()
        reference_model.load_state_dict(best_model_state)
        reference_model.eval()

        accepted_model_state = copy.deepcopy(best_model_state)
        accepted_optimizer_state = copy.deepcopy(
            best_optimizer_state
        )
        accepted_result = dict(best_result)
        print(
            "[STARTUP ROLLBACK] 開始モデルがBEST基準を下回ったため、"
            "過去BESTへ復元しました。"
        )
    elif (
        baseline["win_rate"] > best_win_rate
        or (
            baseline["win_rate"] == best_win_rate
            and baseline["avg_attacker_rounds"] > best_rounds
        )
    ):
        best_win_rate = float(baseline["win_rate"])
        best_rounds = float(baseline["avg_attacker_rounds"])
        best_result = dict(baseline)
        best_model_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(
            optimizer.state_dict()
        )
        save_ppo_checkpoint(
            BEST_MODEL,
            model,
            optimizer,
            update=update_number,
            episodes=previous_episodes,
            best_win_rate=best_win_rate,
            extra={
                "evaluation": baseline,
                "best_rounds": best_rounds,
                "baseline": True,
            },
        )
        print(f"[BEST] baseline saved: {BEST_MODEL}")
    elif not BEST_MODEL.exists():
        save_ppo_checkpoint(
            BEST_MODEL,
            model,
            optimizer,
            update=update_number,
            episodes=previous_episodes,
            best_win_rate=best_win_rate,
            extra={
                "evaluation": baseline,
                "best_rounds": best_rounds,
                "baseline": True,
            },
        )
        print(f"[BEST] initial baseline saved: {BEST_MODEL}")

    restored_rollout = resume_metadata.get("rollout", [])
    rollout: list[RolloutStep] = (
        list(restored_rollout)
        if isinstance(restored_rollout, list)
        else []
    )
    if rollout:
        print(f"[RESUME] pending rollout restored: {len(rollout)} steps")

    update_reward_breakdown: dict[str, float] = defaultdict(float)
    update_reward_episode_count = 0

    current_total_episode = previous_episodes
    interrupted = False

    try:
        for episode in range(1, args.episodes + 1):
            total_episode = previous_episodes + episode
            current_total_episode = total_episode
            set_global_seed(args.seed + total_episode)
    
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
                    f"ep={total_episode} "
                    f"steps={len(rollout)} "
                    f"trajectories={batch['trajectory_count']} "
                    f"reward={controller.episode_reward:.2f} "
                    f"policy={metrics['policy_loss']:.4f} "
                    f"value={metrics['value_loss']:.4f} "
                    f"KL={metrics['reference_kl']:.5f} "
                    f"ratio={metrics['ratio_mean']:.4f}/"
                    f"{metrics['ratio_max']:.4f} "
                    f"clip={metrics['clip_fraction']:.3f} "
                    f"grad={metrics['grad_norm']:.3f} "
                    f"skip={int(metrics['skipped_minibatches'])}"
                )
                reward_divisor = max(1, update_reward_episode_count)
                reward_breakdown_avg = {
                    key: value / reward_divisor
                    for key, value in sorted(update_reward_breakdown.items())
                }
                nonzero_rewards = {
                    key: value
                    for key, value in reward_breakdown_avg.items()
                    if abs(value) >= 1e-9
                }
                if nonzero_rewards:
                    print(
                        "[REWARD/AVG] "
                        + " ".join(
                            f"{key}={value:+.2f}"
                            for key, value in nonzero_rewards.items()
                        )
                    )

                append_log(
                    {
                        "type": "update",
                        "episode": total_episode,
                        "update": update_number,
                        "steps": len(rollout),
                        "trajectories": batch["trajectory_count"],
                        "episode_reward": controller.episode_reward,
                        "reward_breakdown_avg": reward_breakdown_avg,
                        "reward_breakdown_episodes": update_reward_episode_count,
                        **metrics,
                    }
                )
                rollout.clear()
                update_reward_breakdown.clear()
                update_reward_episode_count = 0
    
            if episode % args.eval_every == 0:
                result = evaluate(
                    model,
                    device,
                    matches=args.eval_matches,
                    seed=args.seed + 1_000_000 + total_episode,
                )
    
                accepted_win_drop = (
                    float(accepted_result["win_rate"])
                    - float(result["win_rate"])
                )
                accepted_round_drop = (
                    float(
                        accepted_result["avg_attacker_rounds"]
                    )
                    - float(result["avg_attacker_rounds"])
                )
                best_win_drop = (
                    float(best_win_rate)
                    - float(result["win_rate"])
                )
                best_round_drop = (
                    float(best_rounds)
                    - float(result["avg_attacker_rounds"])
                )

                rejected = (
                    accepted_win_drop >= args.rollback_win_drop
                    or accepted_round_drop
                    >= args.rollback_round_drop
                    or best_win_drop >= args.rollback_win_drop
                    or best_round_drop
                    >= args.rollback_round_drop
                )

                if rejected:
                    model.load_state_dict(best_model_state)
                    optimizer.load_state_dict(
                        best_optimizer_state
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate
                    model.eval()
                    reference_model.load_state_dict(
                        best_model_state
                    )
                    reference_model.eval()

                    accepted_model_state = copy.deepcopy(
                        best_model_state
                    )
                    accepted_optimizer_state = copy.deepcopy(
                        best_optimizer_state
                    )
                    accepted_result = dict(best_result)
                    rollout.clear()
                    decision = "ROLLBACK_TO_BEST"
                else:
                    accepted_model_state = copy.deepcopy(
                        model.state_dict()
                    )
                    accepted_optimizer_state = copy.deepcopy(
                        optimizer.state_dict()
                    )
                    accepted_result = dict(result)
                    reference_model.load_state_dict(
                        model.state_dict()
                    )
                    reference_model.eval()
                    decision = "ACCEPT"

                    if (
                        result["win_rate"] > best_win_rate
                        or (
                            result["win_rate"] == best_win_rate
                            and result["avg_attacker_rounds"]
                            > best_rounds
                        )
                    ):
                        best_win_rate = float(
                            result["win_rate"]
                        )
                        best_rounds = float(
                            result["avg_attacker_rounds"]
                        )
                        best_result = dict(result)
                        best_model_state = copy.deepcopy(
                            model.state_dict()
                        )
                        best_optimizer_state = copy.deepcopy(
                            optimizer.state_dict()
                        )
                        save_ppo_checkpoint(
                            BEST_MODEL,
                            model,
                            optimizer,
                            update=update_number,
                            episodes=total_episode,
                            best_win_rate=best_win_rate,
                            extra={
                                "evaluation": result,
                                "best_rounds": best_rounds,
                            },
                        )
                        print(f"[BEST] saved: {BEST_MODEL}")

                print(
                    f"[EVAL/{decision}] ep={total_episode} "
                    f"win={result['win_rate']:.3f} "
                    f"rounds={result['avg_attacker_rounds']:.2f}-"
                    f"{result['avg_defender_rounds']:.2f} "
                    f"accepted_win_drop={accepted_win_drop:+.3f} "
                    f"best_win_drop={best_win_drop:+.3f} "
                    f"accepted_round_drop={accepted_round_drop:+.2f} "
                    f"best_round_drop={best_round_drop:+.2f}"
                )
                append_log(
                    {
                        "type": "evaluation",
                        "episode": total_episode,
                        "decision": decision,
                        "accepted_before": accepted_result
                        if rejected
                        else None,
                        "accepted_win_drop": accepted_win_drop,
                        "accepted_round_drop": accepted_round_drop,
                        "best_win_drop": best_win_drop,
                        "best_round_drop": best_round_drop,
                        **result,
                    }
                )
    
                save_ppo_checkpoint(
                    CHECKPOINT_DIR / f"ppo_attacker_ep{total_episode:06d}.pt",
                    model,
                    optimizer,
                    update=update_number,
                    episodes=total_episode,
                    best_win_rate=best_win_rate,
                    extra={
                        "evaluation": result,
                        "decision": decision,
                    },
                )
    

            if args.save_every > 0 and episode % args.save_every == 0:
                save_runtime_checkpoint(
                    LATEST_MODEL,
                    model=model,
                    optimizer=optimizer,
                    update_number=update_number,
                    total_episode=total_episode,
                    best_win_rate=best_win_rate,
                    best_rounds=best_rounds,
                    accepted_result=accepted_result,
                    rollout=rollout,
                    latest=True,
                )
                print(f"[LATEST] ep={total_episode} saved: {LATEST_MODEL}")

    except KeyboardInterrupt:
        interrupted = True
        model.eval()
        print("\n[INTERRUPT] Ctrl+Cを受信しました。保存しています...")
        save_runtime_checkpoint(
            INTERRUPT_MODEL,
            model=model,
            optimizer=optimizer,
            update_number=update_number,
            total_episode=current_total_episode,
            best_win_rate=best_win_rate,
            best_rounds=best_rounds,
            accepted_result=accepted_result,
            rollout=rollout,
            interrupted=True,
        )
        save_runtime_checkpoint(
            LATEST_MODEL,
            model=model,
            optimizer=optimizer,
            update_number=update_number,
            total_episode=current_total_episode,
            best_win_rate=best_win_rate,
            best_rounds=best_rounds,
            accepted_result=accepted_result,
            rollout=rollout,
            interrupted=True,
            latest=True,
        )
        append_log({
            "type": "interrupt",
            "episode": current_total_episode,
            "update": update_number,
            "rollout_steps": len(rollout),
            "checkpoint": str(INTERRUPT_MODEL),
        })
        print(f"[INTERRUPT] saved: {INTERRUPT_MODEL}")
        print(f"[LATEST] saved: {LATEST_MODEL}")

    if interrupted:
        return

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

    # 最終出力は過去最高の安全な状態。
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    model.eval()
    reference_model.load_state_dict(best_model_state)
    reference_model.eval()

    save_ppo_checkpoint(
        FINAL_MODEL,
        model,
        optimizer,
        update=update_number,
        episodes=previous_episodes + args.episodes,
        best_win_rate=best_win_rate,
        extra={
            "accepted_evaluation": accepted_result,
            "best_evaluation": best_result,
            "best_rounds": best_rounds,
            "baseline_evaluation": baseline,
        },
    )
    save_runtime_checkpoint(
        LATEST_MODEL,
        model=model,
        optimizer=optimizer,
        update_number=update_number,
        total_episode=previous_episodes + args.episodes,
        best_win_rate=best_win_rate,
        best_rounds=best_rounds,
        accepted_result=accepted_result,
        rollout=[],
        latest=True,
    )
    print(f"saved: {FINAL_MODEL}")
    print(f"latest: {LATEST_MODEL}")


if __name__ == "__main__":
    main()
