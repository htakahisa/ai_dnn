"""Behavior cloning for the defender coach from actor-safe demonstrations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from coach_v1.common.types import ObjectiveAction, Side
from coach_v1.models.coach_model import INTENTS, MOVES, OBJECTIVES
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.training.coach_environment import DEFENDER_STAGES, CoachTrainingEnvironment
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_teacher import teacher_actions


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
        state = environment.reset(episode=trainer.episode)
        hidden = torch.zeros((1, trainer.config.hidden_features), device=trainer.device)
        samples = []
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

        tensors = [torch.as_tensor(np.stack([sample[index] for sample in samples]).copy(),
                                   device=trainer.device)
                   for index in range(8)]
        grid, vector, hidden, move_mask, objective_mask, moves, intents, objectives = tensors
        alive = vector[:, -70:].reshape(-1, 5, 14)[:, :, 0]
        trainer.actor.train()
        for _ in range(3):
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
        trainer.episode += 1
        trainer.history.append({"episode": trainer.episode, "stage": label,
                                "ticks": len(samples), **totals,
                                "imitation_loss": float(loss.detach())})
        trainer.save()
    return trainer.history
