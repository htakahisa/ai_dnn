"""Behavior cloning for the defender coach from actor-safe demonstrations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from coach_v1.common.types import ObjectiveAction, Side
from coach_v1.models.coach_model import CoachPolicy, INTENTS, MOVES, OBJECTIVES
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import COACH_VECTOR_FIELDS
from coach_v1.training.coach_environment import DEFENDER_STAGES, CoachTrainingEnvironment
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_teacher import teacher_actions


Sample = tuple[np.ndarray, ...]


def fit_imitation(trainer: CoachTrainer, *, episodes: int, max_ticks: int,
                  stage: str, on_policy: bool = False) -> list[dict]:
    if (trainer.side is not Side.DEFENDER or episodes <= 0
            or stage not in DEFENDER_STAGES or max_ticks <= 0):
        raise ValueError("defender trainer, stage and positive episode/tick counts required")
    environment = CoachTrainingEnvironment(Side.DEFENDER, seed=trainer.seed,
                                            stage=stage, max_ticks=max_ticks)
    prefix = "aggregation_" if on_policy else "imitation_"
    label = prefix + stage
    for _ in range(episodes):
        samples, totals = _collect_demonstration(
            trainer, environment, episode=trainer.episode, on_policy=on_policy,
        )
        loss = _optimize_samples(trainer, samples)
        trainer.episode += 1
        trainer.history.append({"episode": trainer.episode, "stage": label,
                                "ticks": len(samples), **totals,
                                "imitation_loss": float(loss.detach())})
        trainer.save()
    return trainer.history


def fit_balanced_imitation(
    trainer: CoachTrainer,
    *,
    cycles: int,
    stage_ticks: Mapping[str, int],
    samples_per_bucket: int = 24,
    epochs: int = 3,
) -> list[dict]:
    """Mix every requested curriculum stage in each optimization batch.

    Full-round samples are split into public pre/post-plant buckets.  The
    full-round trajectory is actor-driven (DAgger-style); bounded curriculum
    stages remain teacher-driven so rare DEFUSE and setup labels are retained.
    """
    if (trainer.side is not Side.DEFENDER or cycles <= 0
            or samples_per_bucket <= 0 or epochs <= 0
            or not stage_ticks
            or any(stage not in DEFENDER_STAGES or ticks <= 0
                   for stage, ticks in stage_ticks.items())):
        raise ValueError("defender trainer, valid stages and positive balance settings required")
    environments = {
        stage: CoachTrainingEnvironment(
            Side.DEFENDER, seed=trainer.seed, stage=stage, max_ticks=ticks,
        )
        for stage, ticks in stage_ticks.items()
    }
    planted_index = tuple(COACH_VECTOR_FIELDS).index("spike_planted")
    for _ in range(cycles):
        buckets: dict[str, list[Sample]] = defaultdict(list)
        totals: dict[str, float] = {"reward": 0.0}
        ticks = 0
        for stage, environment in environments.items():
            samples, metrics = _collect_demonstration(
                trainer, environment, episode=trainer.episode,
                on_policy=stage == "full_round",
            )
            trainer.episode += 1
            ticks += len(samples)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            for sample in samples:
                if stage == "full_round":
                    phase = "postplant" if sample[1][planted_index] else "preplant"
                    bucket = f"full_round_{phase}"
                else:
                    bucket = stage
                buckets[bucket].append(sample)
        selected = _balanced_samples(buckets, samples_per_bucket)
        loss = _optimize_samples(trainer, selected, epochs=epochs)
        trainer.history.append({
            "episode": trainer.episode,
            "stage": "balanced_imitation",
            "ticks": ticks,
            **totals,
            "bucket_counts": {name: len(samples) for name, samples in buckets.items()},
            "samples_per_bucket": samples_per_bucket,
            "balanced_samples": len(selected),
            "imitation_loss": float(loss.detach()),
        })
        trainer.save()
    return trainer.history


def validate_defender_actor(
    trainer: CoachTrainer,
    *,
    seeds: Sequence[int],
    max_ticks: int,
) -> dict[str, object]:
    """Evaluate only the actor on fixed full-round seeds for best selection."""
    if (trainer.side is not Side.DEFENDER or not seeds or max_ticks <= 0
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)):
        raise ValueError("defender trainer, integer seeds and positive max_ticks required")
    policy = CoachPolicy(trainer.actor, trainer.encoder, device=trainer.device)
    totals: dict[str, float] = {"reward": 0.0}
    for seed in seeds:
        environment = CoachTrainingEnvironment(
            Side.DEFENDER, seed=seed, stage="full_round", max_ticks=max_ticks,
            observation_version=trainer.encoder.version,
        )
        state = environment.reset()
        policy.reset_round()
        while state is not None:
            transition = environment.step(policy.act(state.observation))
            totals["reward"] += transition.reward
            for name, value in transition.metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            state = transition.next_state
    count = float(len(seeds))
    retake_rate = totals.get("retake_success", 0.0) / max(
        1.0, totals.get("retake_opportunity", 0.0))
    plant_prevention = totals.get("plant_prevention_win", 0.0) / count
    overrotation = totals.get("overrotation_ticks", 0.0) / max(
        1.0, totals.get("no_sighting_ticks", 0.0))
    invalid = totals.get("invalid_moves", 0.0) / max(
        1.0, totals.get("move_requests", 0.0))
    round_win = totals.get("round_win", 0.0) / count
    score = (round_win + 0.25 * retake_rate + 0.1 * plant_prevention
             - 0.05 * overrotation - 0.05 * invalid)
    return {
        "seeds": list(seeds),
        "reward": totals["reward"] / count,
        "round_win": round_win,
        "plant_prevention_rate": plant_prevention,
        "retake_success_rate": retake_rate,
        "overrotation_rate": overrotation,
        "invalid_move_rate": invalid,
        "selection_score": score,
    }


def _collect_demonstration(
    trainer: CoachTrainer,
    environment: CoachTrainingEnvironment,
    *,
    episode: int,
    on_policy: bool,
) -> tuple[list[Sample], dict[str, float]]:
    state = environment.reset(episode=episode)
    hidden = torch.zeros((1, trainer.config.hidden_features), device=trainer.device)
    samples: list[Sample] = []
    totals: dict[str, float] = {"reward": 0.0}
    trainer.actor.eval()
    while state is not None:
        observation, mask = state.observation, state.mask
        actions = teacher_actions(observation, mask)
        with torch.no_grad():
            move_logits, intent_logits, objective_logits, next_hidden = trainer.actor(
                torch.as_tensor(observation.grid.copy(), device=trainer.device)[None],
                torch.as_tensor(observation.vector.copy(), device=trainer.device)[None],
                hidden,
            )
        samples.append((
            observation.grid.copy(), observation.vector.copy(),
            hidden[0].cpu().numpy().copy(), mask.movement.copy(),
            mask.objective.copy(),
            np.asarray([MOVES.index(action.movement) for action in actions]),
            np.asarray([INTENTS.index(action.intent) for action in actions]),
            np.asarray([OBJECTIVES.index(action.objective) for action in actions]),
        ))
        if on_policy:
            movement = move_logits[0].masked_fill(
                ~torch.as_tensor(mask.movement.copy(), device=trainer.device), -torch.inf)
            objective = objective_logits[0].masked_fill(
                ~torch.as_tensor(mask.objective.copy(), device=trainer.device), -torch.inf)
            played = tuple(CoachInstruction(
                MOVES[int(movement[slot].argmax())],
                OBJECTIVES[int(objective[slot].argmax())],
                INTENTS[int(intent_logits[0, slot].argmax())],
            ) for slot in range(5))
        else:
            played = actions
        transition = environment.step(played)
        totals["reward"] += transition.reward
        for name, value in transition.metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        hidden = next_hidden
        state = transition.next_state
    return samples, totals


def _balanced_samples(
    buckets: Mapping[str, Sequence[Sample]], samples_per_bucket: int,
) -> list[Sample]:
    if samples_per_bucket <= 0 or not buckets or any(not samples for samples in buckets.values()):
        raise ValueError("nonempty sample buckets and a positive sample count required")
    selected = []
    for name in sorted(buckets):
        samples = buckets[name]
        indices = np.random.choice(
            len(samples), samples_per_bucket, replace=len(samples) < samples_per_bucket,
        )
        selected.extend(samples[int(index)] for index in indices)
    return selected


def _optimize_samples(
    trainer: CoachTrainer, samples: Sequence[Sample], *, epochs: int = 3,
) -> torch.Tensor:
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
                move_logits.flatten(0, 1), moves[indices].flatten(), reduction="none"
            ).reshape(-1, 5)
            intent_loss = nn.functional.cross_entropy(
                intent_logits.flatten(0, 1), intents[indices].flatten(), reduction="none"
            ).reshape(-1, 5)
            objective_loss = nn.functional.cross_entropy(
                objective_logits.flatten(0, 1), objectives[indices].flatten(),
                reduction="none",
            ).reshape(-1, 5)
            defuse_weight = 1 + 19 * (
                objectives[indices] == OBJECTIVES.index(ObjectiveAction.DEFUSE)
            )
            loss = ((move_loss + 0.25 * intent_loss
                     + 0.5 * objective_loss * defuse_weight)
                    * alive[indices]).sum() / alive[indices].sum().clamp_min(1)
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainer.actor.parameters(), 1.0)
            trainer.optimizer.step()
            trainer.training_step += 1
    return loss
