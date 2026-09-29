"""Actor-safe round-win policy gradients for Task 15 full matches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from party_presets import get_preset

from coach_v1.common.types import Side
from coach_v1.full_match import FullMatchResult, run_headless_full_match
from coach_v1.models.coach_model import INTENTS, MOVES, OBJECTIVES, legal_action_mask
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import CoachObservation
from coach_v1.opponent_pool import OpponentPool, OpponentSpec
from coach_v1.task15_self_play import build_opponent_team
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.opponent_pool_imitation import prepare_candidate_directory


PolicySample = tuple[np.ndarray, ...]


@dataclass(frozen=True)
class RewardTrainingMatch:
    opponent_id: str
    seed: int
    coach_started_as: str
    coach_score: int
    opponent_score: int
    attacker_rounds: int
    attacker_round_wins: int
    defender_rounds: int
    defender_round_wins: int
    attacker_loss: float
    defender_loss: float


class RoundRewardPolicy:
    """Sample movement from actor-safe observations and segment it by round."""

    def __init__(self, trainer: CoachTrainer, *, temperature: float = 0.7) -> None:
        if trainer.config.action_feedback:
            raise ValueError("Task 15 round-reward training does not support action feedback")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.temperature = float(temperature)
        self.hidden: torch.Tensor | None = None
        self.current: list[PolicySample] = []
        self.segments: list[list[PolicySample]] = []

    def reset_round(self) -> None:
        if self.current:
            self.segments.append(self.current)
            self.current = []
        self.hidden = None

    def finish(self) -> None:
        self.reset_round()

    def act(self, observation: CoachObservation) -> tuple[CoachInstruction, ...]:
        mask = legal_action_mask(observation)
        hidden = (torch.zeros((self.trainer.config.hidden_features,), dtype=torch.float32)
                  if self.hidden is None else self.hidden[0].detach().cpu())
        grid = torch.as_tensor(observation.grid.copy(), device=self.trainer.device)[None]
        vector = torch.as_tensor(observation.vector.copy(), device=self.trainer.device)[None]
        hidden_batch = hidden.to(self.trainer.device)[None]
        move_mask = torch.as_tensor(mask.movement.copy(), device=self.trainer.device)
        objective_mask = torch.as_tensor(mask.objective.copy(), device=self.trainer.device)
        self.trainer.actor.eval()
        with torch.no_grad():
            move_logits, intent_logits, objective_logits, next_hidden = self.trainer.actor(
                grid, vector, hidden_batch,
            )
            movement = torch.distributions.Categorical(
                logits=(move_logits[0] / self.temperature).masked_fill(
                    ~move_mask, -torch.inf,
                )
            ).sample()
            intent = intent_logits[0].argmax(-1)
            objective = objective_logits[0].masked_fill(
                ~objective_mask, -torch.inf,
            ).argmax(-1)
        self.current.append((
            observation.grid.copy(), observation.vector.copy(), hidden.numpy().copy(),
            mask.movement.copy(), movement.cpu().numpy().copy(),
        ))
        self.hidden = next_hidden.detach()
        return tuple(CoachInstruction(
            MOVES[int(movement[slot])], OBJECTIVES[int(objective[slot])],
            INTENTS[int(intent[slot])],
        ) for slot in range(5))


def _coach_round_results(
    result: FullMatchResult, *, coach_team_name: str,
) -> dict[Side, list[bool]]:
    by_side = {Side.ATTACKER: [], Side.DEFENDER: []}
    for record in result.round_records:
        coach_sides = {
            str(player.get("side"))
            for player in record.get("players", {}).values()
            if player.get("team") == coach_team_name
        }
        if len(coach_sides) != 1:
            raise ValueError("round record does not identify exactly one coach side")
        side = Side(next(iter(coach_sides)))
        by_side[side].append(str(record.get("winner")) == side.value)
    return by_side


def optimize_round_rewards(
    trainer: CoachTrainer,
    segments: Sequence[Sequence[PolicySample]],
    round_wins: Sequence[bool],
    *,
    temperature: float = 0.7,
    entropy_coefficient: float = 0.002,
) -> float:
    """One on-policy actor update with equal total weight per round."""

    if len(segments) != len(round_wins) or not segments or any(not item for item in segments):
        raise ValueError("one nonempty segment is required for every round result")
    if temperature <= 0 or entropy_coefficient < 0:
        raise ValueError("invalid policy-gradient hyperparameters")
    samples = [sample for segment in segments for sample in segment]
    weights = np.concatenate([
        np.full(len(segment), (1.0 if won else -1.0) / len(segment), dtype=np.float32)
        for segment, won in zip(segments, round_wins)
    ])
    tensors = [
        torch.as_tensor(np.stack([sample[index] for sample in samples]).copy(),
                        device=trainer.device)
        for index in range(5)
    ]
    grid, vector, hidden, move_mask, movement = tensors
    weight = torch.as_tensor(weights, device=trainer.device)
    alive = vector[:, -70:].reshape(-1, 5, 14)[:, :, 0]
    trainer.actor.train()
    move_logits, _, _, _ = trainer.actor(grid, vector, hidden)
    distribution = torch.distributions.Categorical(
        logits=(move_logits / temperature).masked_fill(~move_mask.bool(), -torch.inf)
    )
    log_probability = (distribution.log_prob(movement) * alive).sum(1)
    entropy = (distribution.entropy() * alive).sum(1).mean()
    loss = -(log_probability * weight).sum() / len(segments)
    loss = loss - entropy_coefficient * entropy
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(trainer.actor.parameters(), 1.0)
    trainer.optimizer.step()
    trainer.training_step += 1
    return float(loss.detach())


def train_reward_match(
    *,
    attacker_trainer: CoachTrainer,
    defender_trainer: CoachTrainer,
    opponent: OpponentSpec,
    seed: int,
    coach_starts_as: Side,
    temperature: float = 0.7,
) -> RewardTrainingMatch:
    coach_team_name = "coach_v1_reward_candidate"
    attacker_policy = RoundRewardPolicy(attacker_trainer, temperature=temperature)
    defender_policy = RoundRewardPolicy(defender_trainer, temperature=temperature)
    preset = get_preset(opponent.preset)
    result = run_headless_full_match(
        coach_team_ai=build_coach_v1_team(
            name=coach_team_name, attacker_coach=attacker_policy,
            defender_coach=defender_policy, device=attacker_trainer.device,
        ),
        opponent_team_ai=build_opponent_team(opponent, device=attacker_trainer.device),
        opponent_roster=preset.players,
        opponent_spike_holder=preset.spike_holder,
        opponent_igl=preset.igl,
        coach_starts_as=coach_starts_as,
        seed=seed,
        coach_team_name=coach_team_name,
        opponent_team_name=opponent.opponent_id,
        allow_mirrored_roster=opponent.mirrored_roster,
    )
    attacker_policy.finish()
    defender_policy.finish()
    outcomes = _coach_round_results(result, coach_team_name=coach_team_name)
    attacker_loss = optimize_round_rewards(
        attacker_trainer, attacker_policy.segments, outcomes[Side.ATTACKER],
        temperature=temperature,
    )
    defender_loss = optimize_round_rewards(
        defender_trainer, defender_policy.segments, outcomes[Side.DEFENDER],
        temperature=temperature,
    )
    attacker_trainer.episode += 1
    defender_trainer.episode += 1
    common = {
        "opponent": opponent.opponent_id,
        "match_seed": seed,
        "coach_started_as": coach_starts_as.value,
        "coach_score": result.summary.coach_score,
        "opponent_score": result.summary.opponent_score,
    }
    for trainer, side, loss in (
        (attacker_trainer, Side.ATTACKER, attacker_loss),
        (defender_trainer, Side.DEFENDER, defender_loss),
    ):
        trainer.history.append({
            "episode": trainer.episode,
            "stage": "task15_pool_round_reward",
            **common,
            "side_rounds": len(outcomes[side]),
            "side_round_wins": sum(outcomes[side]),
            "policy_loss": loss,
            "temperature": temperature,
        })
        trainer.save()
    return RewardTrainingMatch(
        opponent_id=opponent.opponent_id,
        seed=seed,
        coach_started_as=coach_starts_as.value,
        coach_score=result.summary.coach_score,
        opponent_score=result.summary.opponent_score,
        attacker_rounds=len(outcomes[Side.ATTACKER]),
        attacker_round_wins=sum(outcomes[Side.ATTACKER]),
        defender_rounds=len(outcomes[Side.DEFENDER]),
        defender_round_wins=sum(outcomes[Side.DEFENDER]),
        attacker_loss=attacker_loss,
        defender_loss=defender_loss,
    )


def train_reward_schedule(
    *,
    attacker_trainer: CoachTrainer,
    defender_trainer: CoachTrainer,
    pool: OpponentPool,
    opponent_ids: Sequence[str],
    seed: int,
    temperature: float = 0.7,
) -> tuple[RewardTrainingMatch, ...]:
    by_id = {item.opponent_id: item for item in pool.opponents}
    if not opponent_ids or any(item not in by_id for item in opponent_ids):
        raise ValueError("training schedule contains an unknown opponent")
    return tuple(train_reward_match(
        attacker_trainer=attacker_trainer,
        defender_trainer=defender_trainer,
        opponent=by_id[opponent_id],
        seed=seed + index,
        coach_starts_as=(Side.ATTACKER if index % 2 == 0 else Side.DEFENDER),
        temperature=temperature,
    ) for index, opponent_id in enumerate(opponent_ids))


def prepare_reward_trainers(
    *,
    root: Path,
    attacker_source: Path,
    defender_source: Path,
    seed: int,
    device: str,
) -> tuple[CoachTrainer, CoachTrainer]:
    attacker_dir, defender_dir = Path(root) / "attacker", Path(root) / "defender"
    prepare_candidate_directory(source=attacker_source, target=attacker_dir)
    prepare_candidate_directory(source=defender_source, target=defender_dir)
    attacker = CoachTrainer(Side.ATTACKER, seed=seed, directory=attacker_dir,
                            device=device)
    defender = CoachTrainer(Side.DEFENDER, seed=seed, directory=defender_dir,
                            device=device)
    attacker.resume()
    defender.resume()
    return attacker, defender
