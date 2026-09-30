"""On-policy coach training, rollout persistence and side-specific resume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from coach_v1.common.checkpoint import (
    CheckpointMetadata, build_checkpoint_metadata, build_checkpoint_payload,
    validate_checkpoint_compatibility,
)
from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import ModelTarget, Side
from coach_v1.models.coach_model import (
    INTENTS, MOVES, OBJECTIVES, CoachActorModel, CoachCritic, CoachModelConfig,
    action_feedback,
)
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import CoachObservationEncoder
from coach_v1.common.versions import COACH_OBSERVATION_VERSION
from coach_v1.training.coach_environment import CoachTrainingEnvironment


@dataclass(frozen=True)
class RolloutStep:
    grid: np.ndarray
    vector: np.ndarray
    enemy_truth: np.ndarray
    movement_mask: np.ndarray
    objective_mask: np.ndarray
    hidden: np.ndarray
    movement: np.ndarray
    intent: np.ndarray
    objective: np.ndarray
    log_probability: float
    value: float
    reward: float
    done: bool
    feedback: np.ndarray | None = None


class CoachTrainer:
    def __init__(self, side: Side, *, seed: int,
                 config: CoachModelConfig = CoachModelConfig(),
                 directory: Path | None = None, device: str = "cpu",
                 learning_rate: float = 3e-4,
                 observation_version: str = COACH_OBSERVATION_VERSION) -> None:
        if not isinstance(side, Side) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("side and integer seed required")
        self.side, self.seed, self.config, self.device = side, seed, config, device
        # Keep v1 production checkpoints intact until a v2 policy is trained
        # and explicitly promoted by the later round curriculum.
        self.directory = (Path(directory) if directory
                          else COACH_CHECKPOINT_PATHS[side.value] / "observation_v2")
        self.encoder = CoachObservationEncoder(version=observation_version)
        if (config.grid_channels != len(self.encoder.grid_channels)
                or config.vector_features != len(self.encoder.vector_fields)):
            raise ValueError("coach config and observation version disagree")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.actor = CoachActorModel(config).to(device)
        self.critic = CoachCritic(config).to(device)
        self.optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate,
        )
        self.training_step = 0
        self.episode = 0
        self.history: list[dict] = []

    def _metadata(self):
        return build_checkpoint_metadata(
            target=ModelTarget.coach(self.side), map_hash=self.encoder.map_hash,
            watch_points_hash=self.encoder.watch_points_hash,
            model_config=self.config.to_dict(), training_seed=self.seed,
            training_step=self.training_step,
            observation_version=self.encoder.version,
        )

    def collect(self, environment: CoachTrainingEnvironment, *, episode: int,
                rollout_path: Path | None = None) -> tuple[list[RolloutStep], dict]:
        if environment.side is not self.side:
            raise ValueError("environment side mismatch")
        state = environment.reset(episode=episode)
        hidden = torch.zeros((1, self.config.hidden_features), device=self.device)
        previous_observation = None
        previous_actions = None
        trajectory: list[RolloutStep] = []
        totals = {"reward": 0.0, "new_clear_cells": 0.0,
                  "invalid_moves": 0.0, "round_win": 0.0}
        self.actor.eval()
        self.critic.eval()
        while state is not None:
            obs, mask = state.observation, state.mask
            grid = torch.as_tensor(obs.grid.copy(), device=self.device)[None]
            vector = torch.as_tensor(obs.vector.copy(), device=self.device)[None]
            truth = torch.as_tensor(state.critic_enemy_truth.copy(), device=self.device)[None]
            move_mask = torch.as_tensor(mask.movement.copy(), device=self.device)
            objective_mask = torch.as_tensor(mask.objective.copy(), device=self.device)
            feedback = (action_feedback(obs, previous_observation, previous_actions)
                        if self.config.action_feedback else None)
            feedback_tensor = (torch.as_tensor(feedback, device=self.device)[None]
                               if feedback is not None else None)
            with torch.no_grad():
                move_logits, intent_logits, objective_logits, next_hidden = self.actor(
                    grid, vector, hidden, feedback=feedback_tensor)
                value = self.critic(grid, vector, truth)[0]
                move_dist = torch.distributions.Categorical(logits=move_logits[0].masked_fill(~move_mask, -torch.inf))
                intent_dist = torch.distributions.Categorical(logits=intent_logits[0])
                objective_dist = torch.distributions.Categorical(logits=objective_logits[0].masked_fill(~objective_mask, -torch.inf))
                movement = move_dist.sample()
                intent = intent_dist.sample()
                objective = objective_dist.sample()
                alive = torch.as_tensor(vector[0, -70:].reshape(5, 14)[:, 0] > 0)
                logp = (move_dist.log_prob(movement) + objective_dist.log_prob(objective)
                        + intent_dist.log_prob(intent) * alive).sum()
            actions = tuple(CoachInstruction(MOVES[int(movement[i])], OBJECTIVES[int(objective[i])],
                                             INTENTS[int(intent[i])]) for i in range(5))
            transition = environment.step(actions)
            trajectory.append(RolloutStep(
                obs.grid.copy(), obs.vector.copy(), state.critic_enemy_truth.copy(),
                mask.movement.copy(), mask.objective.copy(), hidden[0].cpu().numpy().copy(),
                movement.cpu().numpy().copy(), intent.cpu().numpy().copy(),
                objective.cpu().numpy().copy(), float(logp), float(value),
                transition.reward, transition.done, feedback.copy() if feedback is not None else None,
            ))
            totals["reward"] += transition.reward
            for name, metric in transition.metrics.items():
                totals[name] = totals.get(name, 0.0) + metric
            hidden = next_hidden.detach()
            previous_observation, previous_actions = obs, actions
            state = transition.next_state
        if rollout_path is not None:
            path = Path(rollout_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"metadata": self._metadata().to_dict(), "stage": environment.stage,
                        "episode": episode, "steps": trajectory, "metrics": totals}, path)
        totals["ticks"] = len(trajectory)
        return trajectory, totals

    def update(self, trajectory: list[RolloutStep], *, gamma: float = 0.99,
               clip: float = 0.2, epochs: int = 2) -> dict:
        if not trajectory or epochs <= 0:
            raise ValueError("nonempty rollout and positive epochs required")
        # Monte Carlo return uses only rewards from the current episode.
        returns = []
        running = 0.0
        for item in reversed(trajectory):
            running = item.reward + gamma * running * (not item.done)
            returns.append(running)
        returns.reverse()
        device = self.device
        grids = torch.as_tensor(np.stack([r.grid for r in trajectory]).copy(), device=device)
        vectors = torch.as_tensor(np.stack([r.vector for r in trajectory]).copy(), device=device)
        truth = torch.as_tensor(np.stack([r.enemy_truth for r in trajectory]).copy(), device=device)
        hidden = torch.as_tensor(np.stack([r.hidden for r in trajectory]).copy(), device=device)
        move_mask = torch.as_tensor(np.stack([r.movement_mask for r in trajectory]).copy(), device=device)
        objective_mask = torch.as_tensor(np.stack([r.objective_mask for r in trajectory]).copy(), device=device)
        movement = torch.as_tensor(np.stack([r.movement for r in trajectory]).copy(), device=device)
        intent = torch.as_tensor(np.stack([r.intent for r in trajectory]).copy(), device=device)
        objective = torch.as_tensor(np.stack([r.objective for r in trajectory]).copy(), device=device)
        feedback = (torch.as_tensor(np.stack([r.feedback for r in trajectory]).copy(), device=device)
                    if self.config.action_feedback else None)
        old_logp = torch.tensor([r.log_probability for r in trajectory], device=device)
        target = torch.tensor(returns, device=device, dtype=torch.float32)
        baseline = torch.tensor([r.value for r in trajectory], device=device)
        advantage = target - baseline
        if len(trajectory) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
        alive = vectors[:, -70:].reshape(-1, 5, 14)[:, :, 0]
        self.actor.train()
        self.critic.train()
        for _ in range(epochs):
            move, intents, objectives, _ = self.actor(grids, vectors, hidden,
                                                     feedback=feedback)
            move_dist = torch.distributions.Categorical(logits=move.masked_fill(~move_mask, -torch.inf))
            intent_dist = torch.distributions.Categorical(logits=intents)
            objective_dist = torch.distributions.Categorical(logits=objectives.masked_fill(~objective_mask, -torch.inf))
            logp = (move_dist.log_prob(movement) + objective_dist.log_prob(objective)
                    + intent_dist.log_prob(intent) * alive).sum(1)
            ratio = (logp - old_logp).exp()
            policy_loss = -torch.minimum(ratio * advantage,
                                         ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
            values = self.critic(grids, vectors, truth)
            value_loss = nn.functional.smooth_l1_loss(values, target)
            entropy = (move_dist.entropy() + objective_dist.entropy()
                       + intent_dist.entropy() * alive).sum(1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.002 * entropy
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
            self.optimizer.step()
            self.training_step += 1
        return {"policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach())}

    def fit(self, *, episodes: int, stage: str = "2v1", max_ticks: int = 40,
            save_rollouts: bool = True, collision_penalty: float = 0.01) -> list[dict]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        environment = CoachTrainingEnvironment(self.side, seed=self.seed,
                                                stage=stage, max_ticks=max_ticks,
                                                collision_penalty=collision_penalty,
                                                observation_version=self.encoder.version)
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(episodes):
            rollout_path = self.directory / "rollouts" / f"episode_{self.episode:06d}.pt" if save_rollouts else None
            trajectory, metrics = self.collect(environment, episode=self.episode,
                                               rollout_path=rollout_path)
            losses = self.update(trajectory)
            self.episode += 1
            self.history.append({"episode": self.episode, "stage": stage,
                                 "collision_penalty": collision_penalty,
                                 **metrics, **losses})
            self.save()
        return self.history

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        actor_path = self.directory / "latest.pt"
        if actor_path.exists():
            existing = torch.load(actor_path, map_location="cpu", weights_only=False)
            existing_version = CheckpointMetadata.from_dict(
                existing["metadata"]
            ).observation_version
            if existing_version != self.encoder.version:
                raise ValueError("refusing to overwrite a coach checkpoint with a different observation version")
        # Actor file contains neither critic weights nor optimizer/training truth.
        actor = build_checkpoint_payload(metadata=self._metadata(),
                                         model_state_dict=self.actor.state_dict(),
                                         optimizer_state_dict=None)
        temporary = actor_path.with_suffix(".tmp")
        torch.save(actor, temporary)
        temporary.replace(actor_path)
        training = {"metadata": self._metadata().to_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "episode": self.episode, "history": self.history,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        train_path = self.directory / "training_latest.pt"
        temporary = train_path.with_suffix(".tmp")
        torch.save(training, temporary)
        temporary.replace(train_path)
        return actor_path

    def resume(self) -> None:
        actor = torch.load(self.directory / "latest.pt", map_location=self.device, weights_only=False)
        training = torch.load(self.directory / "training_latest.pt", map_location=self.device, weights_only=False)
        actor_meta = CheckpointMetadata.from_dict(actor["metadata"])
        training_meta = CheckpointMetadata.from_dict(training["metadata"])
        validate_checkpoint_compatibility(actor_meta, self._metadata())
        validate_checkpoint_compatibility(training_meta, self._metadata())
        self.encoder.validate_checkpoint(actor_meta, side=self.side)
        if actor_meta.training_seed != self.seed or actor_meta.training_step != training_meta.training_step:
            raise ValueError("coach resume provenance mismatch")
        self.actor.load_state_dict(actor["model_state_dict"], strict=True)
        self.critic.load_state_dict(training["critic_state_dict"], strict=True)
        self.optimizer.load_state_dict(training["optimizer_state_dict"])
        self.training_step = actor_meta.training_step
        self.episode = int(training["episode"])
        self.history = list(training["history"])
        torch.set_rng_state(training["torch_rng"])
        np.random.set_state(training["numpy_rng"])
        random.setstate(training["python_rng"])
