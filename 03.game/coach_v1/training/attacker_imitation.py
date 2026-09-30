"""Optional training-only behavior cloning from legal attacker demonstrations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from coach_v1.common.types import ObjectiveAction, Side
from coach_v1.models.coach_model import INTENTS, MOVES, OBJECTIVES
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.training.attacker_teacher import teacher_actions
from coach_v1.training.coach_environment import ATTACKER_STAGES, CoachTrainingEnvironment
from coach_v1.training.coach_trainer import CoachTrainer


def fit_imitation(trainer: CoachTrainer, *, episodes: int, max_ticks: int = 160,
                  on_policy: bool = False, stage: str = "full_round",
                  carrier_move_weight: float = 1.0) -> list[dict]:
    if (trainer.side is not Side.ATTACKER or episodes <= 0
            or stage not in ATTACKER_STAGES or carrier_move_weight < 1):
        raise ValueError("attacker trainer and positive episodes required")
    environment = CoachTrainingEnvironment(Side.ATTACKER, seed=trainer.seed,
                                            stage=stage, max_ticks=max_ticks)
    for _ in range(episodes):
        state = environment.reset(episode=trainer.episode)
        hidden = torch.zeros((1, trainer.config.hidden_features), device=trainer.device)
        samples = []
        totals = {"reward": 0.0, "plant": 0.0, "attacker_win": 0.0}
        trainer.actor.eval()
        while state is not None:
            obs, mask = state.observation, state.mask
            actions = teacher_actions(obs, mask)
            with torch.no_grad():
                move_logits, intent_logits, objective_logits, next_hidden = trainer.actor(
                    torch.as_tensor(obs.grid.copy(), device=trainer.device)[None],
                    torch.as_tensor(obs.vector.copy(), device=trainer.device)[None], hidden)
            samples.append((obs.grid.copy(), obs.vector.copy(), hidden[0].cpu().numpy().copy(),
                            mask.movement.copy(), mask.objective.copy(),
                            np.asarray([MOVES.index(a.movement) for a in actions]),
                            np.asarray([OBJECTIVES.index(a.objective) for a in actions])))
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
            totals["plant"] += transition.metrics["plant"]
            totals["attacker_win"] += transition.metrics["attacker_win"]
            hidden = next_hidden
            state = transition.next_state
        grid = torch.as_tensor(np.stack([s[0] for s in samples]).copy(), device=trainer.device)
        vector = torch.as_tensor(np.stack([s[1] for s in samples]).copy(), device=trainer.device)
        hidden = torch.as_tensor(np.stack([s[2] for s in samples]).copy(), device=trainer.device)
        move_mask = torch.as_tensor(np.stack([s[3] for s in samples]).copy(), device=trainer.device)
        objective_mask = torch.as_tensor(np.stack([s[4] for s in samples]).copy(), device=trainer.device)
        moves = torch.as_tensor(np.stack([s[5] for s in samples]).copy(), device=trainer.device)
        objectives = torch.as_tensor(np.stack([s[6] for s in samples]).copy(), device=trainer.device)
        alive = vector[:, -70:].reshape(-1, 5, 14)[:, :, 0]
        trainer.actor.train()
        for _ in range(3):
            for indices in torch.randperm(len(samples), device=trainer.device).split(32):
                move_logits, _, objective_logits, _ = trainer.actor(
                    grid[indices], vector[indices], hidden[indices])
                move_logits = move_logits.masked_fill(~move_mask[indices], -torch.inf)
                objective_logits = objective_logits.masked_fill(~objective_mask[indices], -torch.inf)
                move_loss = nn.functional.cross_entropy(
                    move_logits.flatten(0, 1), moves[indices].flatten(), reduction="none"
                ).reshape(-1, 5)
                objective_loss = nn.functional.cross_entropy(
                    objective_logits.flatten(0, 1), objectives[indices].flatten(),
                    reduction="none"
                ).reshape(-1, 5)
                plant_weight = 1 + 19 * (objectives[indices] ==
                                         OBJECTIVES.index(ObjectiveAction.PLANT))
                carrier = vector[indices, -70:].reshape(-1, 5, 14)[:, :, 3]
                move_weight = 1 + (carrier_move_weight - 1) * carrier
                loss = ((move_loss * move_weight + 0.5 * objective_loss * plant_weight)
                        * alive[indices]).sum() / alive[indices].sum().clamp_min(1)
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainer.actor.parameters(), 1.0)
                trainer.optimizer.step()
                trainer.training_step += 1
        trainer.episode += 1
        label = ("aggregation_" if on_policy else "imitation_") + stage
        trainer.history.append({"episode": trainer.episode, "stage": label,
                                "ticks": len(samples), **totals, "imitation_loss": float(loss.detach())})
        trainer.save()
    return trainer.history
