"""Actor-safe DAgger collection from Task 15 full-match opponent states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from party_presets import get_preset

from coach_v1.common.types import ObjectiveAction, Side
from coach_v1.full_match import run_headless_full_match
from coach_v1.models.coach_model import (
    CoachPolicy, INTENTS, MOVES, OBJECTIVES, legal_action_mask,
)
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import COACH_VECTOR_FIELDS, CoachObservation
from coach_v1.opponent_pool import OpponentPool, OpponentSpec
from coach_v1.task15_self_play import build_opponent_team
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.training.attacker_teacher import teacher_actions as attacker_teacher
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_teacher import teacher_actions as defender_teacher


Sample = tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PoolTrainingMatch:
    opponent_id: str
    seed: int
    coach_started_as: str
    coach_score: int
    opponent_score: int
    attacker_samples: int
    defender_samples: int
    attacker_teacher_move_agreement: float
    defender_teacher_move_agreement: float
    attacker_loss: float
    defender_loss: float


class RecordingTeacherPolicy:
    """Run the current actor while retaining only actor-safe DAgger labels."""

    def __init__(self, trainer: CoachTrainer) -> None:
        if trainer.config.action_feedback:
            raise ValueError("Task 15 pool DAgger does not support action-feedback models")
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.policy = CoachPolicy(trainer.actor, trainer.encoder, device=trainer.device)
        self.samples: list[Sample] = []
        self.move_matches = 0
        self.move_labels = 0

    def reset_round(self) -> None:
        self.policy.reset_round()

    def act(self, observation: CoachObservation) -> tuple[CoachInstruction, ...]:
        mask = legal_action_mask(observation)
        hidden = (torch.zeros((self.trainer.config.hidden_features,), dtype=torch.float32)
                  if self.policy.hidden is None else self.policy.hidden[0].detach().cpu())
        teacher = (attacker_teacher if self.trainer.side is Side.ATTACKER
                   else defender_teacher)(observation, mask)
        played = self.policy.act(observation)
        alive = observation.vector[-70:].reshape(5, 14)[:, 0] > 0
        self.move_matches += sum(
            bool(alive[slot]) and played[slot].movement is teacher[slot].movement
            for slot in range(5)
        )
        self.move_labels += int(alive.sum())
        self.samples.append((
            observation.grid.copy(), observation.vector.copy(), hidden.numpy().copy(),
            mask.movement.copy(), mask.objective.copy(),
            np.asarray([MOVES.index(action.movement) for action in teacher]),
            np.asarray([INTENTS.index(action.intent) for action in teacher]),
            np.asarray([OBJECTIVES.index(action.objective) for action in teacher]),
        ))
        return played

    @property
    def move_agreement(self) -> float:
        return self.move_matches / max(1, self.move_labels)


def prepare_candidate_directory(*, source: Path, target: Path) -> None:
    """Copy one resumable side checkpoint without touching its official source."""

    source, target = Path(source), Path(target)
    required = ("latest.pt", "training_latest.pt")
    if target.exists() and any(target.iterdir()):
        if not all((target / name).is_file() for name in required):
            raise ValueError("nonempty candidate directory is not resumable")
        return
    target.mkdir(parents=True, exist_ok=True)
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, target / name)


def _bucket_name(sample: Sample, side: Side) -> str:
    vector = sample[1]
    fields = {name: index for index, name in enumerate(COACH_VECTOR_FIELDS)}
    situations = ("carry", "retrieve", "guard") if side is Side.ATTACKER else (
        "search", "retake",
    )
    situation = next(
        (name for name in situations if vector[fields[f"situation_{name}"]] > 0.5),
        "other",
    )
    objectives = sample[7]
    special = (ObjectiveAction.PLANT if side is Side.ATTACKER else ObjectiveAction.DEFUSE)
    if np.any(objectives == OBJECTIVES.index(special)):
        return f"{situation}_{special.value.lower()}"
    if vector[fields["defender_setup"]] > 0.5:
        return "setup"
    return situation


def balanced_pool_samples(
    samples: Sequence[Sample], *, side: Side, samples_per_bucket: int,
) -> tuple[list[Sample], dict[str, int]]:
    if not samples or samples_per_bucket <= 0:
        raise ValueError("nonempty samples and a positive bucket size required")
    buckets: dict[str, list[Sample]] = {}
    for sample in samples:
        buckets.setdefault(_bucket_name(sample, side), []).append(sample)
    selected = []
    for name in sorted(buckets):
        bucket = buckets[name]
        indices = np.random.choice(
            len(bucket), samples_per_bucket, replace=len(bucket) < samples_per_bucket,
        )
        selected.extend(bucket[int(index)] for index in indices)
    random.shuffle(selected)
    return selected, {name: len(bucket) for name, bucket in sorted(buckets.items())}


def optimize_pool_samples(
    trainer: CoachTrainer, samples: Sequence[Sample], *, epochs: int = 2,
) -> float:
    if not samples or epochs <= 0:
        raise ValueError("nonempty samples and positive epochs required")
    tensors = [torch.as_tensor(np.stack([sample[index] for sample in samples]).copy(),
                               device=trainer.device)
               for index in range(8)]
    grid, vector, hidden, move_mask, objective_mask, moves, intents, objectives = tensors
    alive = vector[:, -70:].reshape(-1, 5, 14)[:, :, 0]
    trainer.actor.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(samples), device=trainer.device).split(32):
            move_logits, intent_logits, objective_logits, _ = trainer.actor(
                grid[indices], vector[indices], hidden[indices])
            move_logits = move_logits.masked_fill(~move_mask[indices].bool(), -torch.inf)
            objective_logits = objective_logits.masked_fill(
                ~objective_mask[indices].bool(), -torch.inf)
            move_loss = nn.functional.cross_entropy(
                move_logits.flatten(0, 1), moves[indices].flatten(), reduction="none",
            ).reshape(-1, 5)
            intent_loss = nn.functional.cross_entropy(
                intent_logits.flatten(0, 1), intents[indices].flatten(), reduction="none",
            ).reshape(-1, 5)
            objective_loss = nn.functional.cross_entropy(
                objective_logits.flatten(0, 1), objectives[indices].flatten(), reduction="none",
            ).reshape(-1, 5)
            special = (ObjectiveAction.PLANT if trainer.side is Side.ATTACKER
                       else ObjectiveAction.DEFUSE)
            objective_weight = 1 + 19 * (
                objectives[indices] == OBJECTIVES.index(special)
            )
            loss = ((move_loss + 0.25 * intent_loss
                     + 0.5 * objective_loss * objective_weight)
                    * alive[indices]).sum() / alive[indices].sum().clamp_min(1)
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainer.actor.parameters(), 1.0)
            trainer.optimizer.step()
            trainer.training_step += 1
    return float(loss.detach())


def train_pool_match(
    *,
    attacker_trainer: CoachTrainer,
    defender_trainer: CoachTrainer,
    opponent: OpponentSpec,
    seed: int,
    coach_starts_as: Side,
    samples_per_bucket: int = 64,
    epochs: int = 2,
) -> PoolTrainingMatch:
    """Collect one real match, then imitate legal teacher actions on its states."""

    attacker_policy = RecordingTeacherPolicy(attacker_trainer)
    defender_policy = RecordingTeacherPolicy(defender_trainer)
    preset = get_preset(opponent.preset)
    result = run_headless_full_match(
        coach_team_ai=build_coach_v1_team(
            name="coach_v1_candidate", attacker_coach=attacker_policy,
            defender_coach=defender_policy, device=attacker_trainer.device,
        ),
        opponent_team_ai=build_opponent_team(opponent, device=attacker_trainer.device),
        opponent_roster=preset.players,
        opponent_spike_holder=preset.spike_holder,
        opponent_igl=preset.igl,
        coach_starts_as=coach_starts_as,
        seed=seed,
        coach_team_name="coach_v1_candidate",
        opponent_team_name=opponent.opponent_id,
        allow_mirrored_roster=opponent.mirrored_roster,
    )
    attacker_samples, attacker_buckets = balanced_pool_samples(
        attacker_policy.samples, side=Side.ATTACKER,
        samples_per_bucket=samples_per_bucket,
    )
    defender_samples, defender_buckets = balanced_pool_samples(
        defender_policy.samples, side=Side.DEFENDER,
        samples_per_bucket=samples_per_bucket,
    )
    attacker_loss = optimize_pool_samples(
        attacker_trainer, attacker_samples, epochs=epochs)
    defender_loss = optimize_pool_samples(
        defender_trainer, defender_samples, epochs=epochs)
    attacker_trainer.episode += 1
    defender_trainer.episode += 1
    common = {
        "opponent": opponent.opponent_id, "match_seed": seed,
        "coach_started_as": coach_starts_as.value,
        "coach_score": result.summary.coach_score,
        "opponent_score": result.summary.opponent_score,
    }
    attacker_trainer.history.append({
        "episode": attacker_trainer.episode, "stage": "task15_pool_dagger",
        **common, "raw_samples": len(attacker_policy.samples),
        "buckets": attacker_buckets, "balanced_samples": len(attacker_samples),
        "teacher_move_agreement": attacker_policy.move_agreement,
        "imitation_loss": attacker_loss,
    })
    defender_trainer.history.append({
        "episode": defender_trainer.episode, "stage": "task15_pool_dagger",
        **common, "raw_samples": len(defender_policy.samples),
        "buckets": defender_buckets, "balanced_samples": len(defender_samples),
        "teacher_move_agreement": defender_policy.move_agreement,
        "imitation_loss": defender_loss,
    })
    attacker_trainer.save()
    defender_trainer.save()
    return PoolTrainingMatch(
        opponent_id=opponent.opponent_id, seed=seed,
        coach_started_as=coach_starts_as.value,
        coach_score=result.summary.coach_score,
        opponent_score=result.summary.opponent_score,
        attacker_samples=len(attacker_policy.samples),
        defender_samples=len(defender_policy.samples),
        attacker_teacher_move_agreement=attacker_policy.move_agreement,
        defender_teacher_move_agreement=defender_policy.move_agreement,
        attacker_loss=attacker_loss, defender_loss=defender_loss,
    )


def train_pool_schedule(
    *,
    attacker_trainer: CoachTrainer,
    defender_trainer: CoachTrainer,
    pool: OpponentPool,
    opponent_ids: Sequence[str],
    seed: int,
    samples_per_bucket: int = 64,
    epochs: int = 2,
) -> tuple[PoolTrainingMatch, ...]:
    by_id = {item.opponent_id: item for item in pool.opponents}
    if not opponent_ids or any(item not in by_id for item in opponent_ids):
        raise ValueError("training schedule contains an unknown opponent")
    results = []
    for index, opponent_id in enumerate(opponent_ids):
        results.append(train_pool_match(
            attacker_trainer=attacker_trainer,
            defender_trainer=defender_trainer,
            opponent=by_id[opponent_id], seed=seed + index,
            coach_starts_as=(Side.ATTACKER if index % 2 == 0 else Side.DEFENDER),
            samples_per_bucket=samples_per_bucket, epochs=epochs,
        ))
    return tuple(results)
