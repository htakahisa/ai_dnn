from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from controllers import DefaultAttackerController
from policy_attacker_controller import (
    build_action_mask,
    decode_action,
    get_game_observation,
)
from ppo_actor_critic import PPOActorCritic
from train_bc import observation_to_vector


@dataclass
class PendingStep:
    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    mask: np.ndarray
    snapshot: dict[str, float]
    char_name: str


@dataclass
class RolloutStep:
    trajectory_id: int
    obs: np.ndarray
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    mask: np.ndarray


def _alive(char: Any) -> bool:
    return bool(getattr(char, "is_alive", True))


class PPOAttackerController:
    """PPO rollout収集用Controller。方策は共有し、軌跡は選手ごとに分離する。"""

    def __init__(
        self,
        model: PPOActorCritic,
        device: str | torch.device,
        deterministic: bool = False,
    ):
        self.model = model
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)

        self.game: Any | None = None
        self.helper = DefaultAttackerController()

        # key=id(Character)。同名選手がいても混同しない。
        self.pending: dict[int, PendingStep] = {}
        self.rollout: list[RolloutStep] = []

        self.episode_reward = 0.0
        self.action_counts = np.zeros(model.num_actions, dtype=np.int64)
        self.forced_action_count = 0

        self.reward_breakdown: dict[str, float] = {
            "ROUND_WIN": 0.0, "ROUND_LOSS": 0.0,
            "KILL": 0.0, "DEATH": 0.0,
            "TEAM_KILL": 0.0, "TEAM_DEATH": 0.0,
            "PLANT": 0.0, "PLANT_PROGRESS": 0.0,
            "SPIKE_RECOVERY": 0.0, "DAMAGE_TAKEN": 0.0,
            "SITE_PROGRESS": 0.0, "SITE_ENTRY": 0.0,
            "OSCILLATION": 0.0,
            "MATCH_RESULT": 0.0,
        }
        self.best_site_distance: dict[int, float] = {}
        self.site_entry_awarded: set[int] = set()
        self.position_history: dict[int, deque[tuple[int, int]]] = {}

    def set_game(self, game: Any) -> None:
        self.game = game

    def reset_round(self) -> None:
        if self.game is not None and self.pending:
            current_attackers = {
                str(getattr(char, "name", "")): char
                for char in getattr(self.game, "chars", [])
                if getattr(char, "team", None) == "A"
            }
            for trajectory_id in list(self.pending):
                pending = self.pending.get(trajectory_id)
                if pending is None:
                    continue
                current_char = current_attackers.get(pending.char_name)
                if current_char is None:
                    self.pending.pop(trajectory_id, None)
                    self.best_site_distance.pop(trajectory_id, None)
                    self.site_entry_awarded.discard(trajectory_id)
                    self.position_history.pop(trajectory_id, None)
                    continue
                self._close(
                    current_char,
                    done=True,
                    trajectory_id_override=trajectory_id,
                )

        if hasattr(self.helper, "reset_round"):
            self.helper.reset_round()

    def _site_distance(self, char: Any) -> float:
        if self.game is None:
            return 0.0
        target = getattr(self.game, "target_plant_pos", None)
        pos = getattr(char, "pos", None)
        if target is None or pos is None or len(pos) != 2:
            return 0.0
        return float(
            abs(int(pos[0]) - int(target[0]))
            + abs(int(pos[1]) - int(target[1]))
        )

    def _snapshot(self, char: Any) -> dict[str, float]:
        if self.game is None:
            raise RuntimeError("set_game(game)が未実行です")

        attackers = [
            c for c in self.game.chars if getattr(c, "team", None) == "A"
        ]
        defenders = [
            c for c in self.game.chars if getattr(c, "team", None) == "D"
        ]
        return {
            "self_k": float(getattr(char, "kills", 0)),
            "self_d": float(getattr(char, "deaths", 0)),
            "alive_a": float(sum(_alive(c) for c in attackers)),
            "alive_d": float(sum(_alive(c) for c in defenders)),
            "attacker_rounds": float(getattr(self.game, "attacker_wins", 0)),
            "defender_rounds": float(getattr(self.game, "defender_wins", 0)),
            "planted": float(bool(getattr(self.game, "is_planted", False))),
            "has_spike": float(bool(getattr(char, "has_spike", False))),
            "plant_timer": float(getattr(char, "plant_timer", 0)),
            "hp": float(getattr(char, "hp", 0)),
            "on_site": float(
                bool(
                    0 <= int(char.pos[0]) < self.game.grid.shape[0]
                    and 0 <= int(char.pos[1]) < self.game.grid.shape[1]
                    and int(self.game.grid[int(char.pos[0]), int(char.pos[1])]) == 2
                )
            ),
            "site_distance": self._site_distance(char),
            "pos_r": float(int(char.pos[0])),
            "pos_c": float(int(char.pos[1])),
        }

    @staticmethod
    def _reward_components(
        previous: dict[str, float],
        current: dict[str, float],
    ) -> dict[str, float]:
        components = {
            "ROUND_WIN": 0.0, "ROUND_LOSS": 0.0,
            "KILL": 0.0, "DEATH": 0.0,
            "TEAM_KILL": 0.0, "TEAM_DEATH": 0.0,
            "PLANT": 0.0, "PLANT_PROGRESS": 0.0,
            "SPIKE_RECOVERY": 0.0, "DAMAGE_TAKEN": 0.0,
        }

        components["ROUND_WIN"] += 12.0 * max(
            0.0, current["attacker_rounds"] - previous["attacker_rounds"]
        )
        components["ROUND_LOSS"] -= 12.0 * max(
            0.0, current["defender_rounds"] - previous["defender_rounds"]
        )
        components["KILL"] += 1.25 * max(
            0.0, current["self_k"] - previous["self_k"]
        )
        components["DEATH"] -= 1.50 * max(
            0.0, current["self_d"] - previous["self_d"]
        )
        components["TEAM_KILL"] += 0.15 * max(
            0.0, previous["alive_d"] - current["alive_d"]
        )
        components["TEAM_DEATH"] -= 0.15 * max(
            0.0, previous["alive_a"] - current["alive_a"]
        )

        if previous["planted"] < 0.5 <= current["planted"]:
            components["PLANT"] += 5.0

        components["PLANT_PROGRESS"] += 0.05 * max(
            0.0, current["plant_timer"] - previous["plant_timer"]
        )

        if previous["has_spike"] < 0.5 <= current["has_spike"]:
            components["SPIKE_RECOVERY"] += 0.50

        hp_lost = max(0.0, previous["hp"] - current["hp"])
        components["DAMAGE_TAKEN"] -= hp_lost * 0.0015
        return components


    def _close(
        self,
        char: Any,
        *,
        done: bool = False,
        terminal_bonus: float = 0.0,
        trajectory_id_override: int | None = None,
    ) -> None:
        trajectory_id = (
            int(trajectory_id_override)
            if trajectory_id_override is not None
            else id(char)
        )
        pending = self.pending.pop(trajectory_id, None)
        if pending is None:
            return

        current_snapshot = self._snapshot(char)
        components = self._reward_components(
            pending.snapshot,
            current_snapshot,
        )

        previous_best = self.best_site_distance.get(
            trajectory_id,
            float(pending.snapshot.get("site_distance", 0.0)),
        )
        current_distance = float(
            current_snapshot.get("site_distance", previous_best)
        )

        if (
            not done
            and current_snapshot.get("planted", 0.0) < 0.5
            and current_distance < previous_best
        ):
            improvement = previous_best - current_distance
            components["SITE_PROGRESS"] = min(0.30, 0.06 * improvement)
            self.best_site_distance[trajectory_id] = current_distance
        else:
            components["SITE_PROGRESS"] = 0.0

        if (
            not done
            and trajectory_id not in self.site_entry_awarded
            and pending.snapshot.get("on_site", 0.0) < 0.5
            and current_snapshot.get("on_site", 0.0) >= 0.5
        ):
            components["SITE_ENTRY"] = 0.75
            self.site_entry_awarded.add(trajectory_id)
        else:
            components["SITE_ENTRY"] = 0.0

        current_pos = (
            int(current_snapshot.get("pos_r", 0.0)),
            int(current_snapshot.get("pos_c", 0.0)),
        )
        history = self.position_history.setdefault(
            trajectory_id,
            deque(maxlen=3),
        )

        if not history:
            previous_pos = (
                int(pending.snapshot.get("pos_r", current_pos[0])),
                int(pending.snapshot.get("pos_c", current_pos[1])),
            )
            history.append(previous_pos)

        history.append(current_pos)

        oscillation_penalty = 0.0
        if (
            not done
            and current_snapshot.get("planted", 0.0) < 0.5
            and len(history) >= 3
        ):
            pos_a, pos_b, pos_c = list(history)[-3:]
            if pos_a == pos_c and pos_a != pos_b:
                oscillation_penalty = -0.20

        components["OSCILLATION"] = oscillation_penalty

        components["MATCH_RESULT"] = float(terminal_bonus)
        reward = float(sum(components.values()))

        for key, value in components.items():
            self.reward_breakdown[key] = (
                self.reward_breakdown.get(key, 0.0) + float(value)
            )

        self.rollout.append(
            RolloutStep(
                trajectory_id=trajectory_id,
                obs=pending.obs,
                action=pending.action,
                old_log_prob=pending.log_prob,
                value=pending.value,
                reward=reward,
                done=bool(done),
                mask=pending.mask,
            )
        )
        self.episode_reward += reward

        if done:
            self.best_site_distance.pop(trajectory_id, None)
            self.site_entry_awarded.discard(trajectory_id)
            self.position_history.pop(trajectory_id, None)


    def _shortest_path_distance(self, start: Any, target: Any, grid) -> int:
        method = getattr(self.helper, "shortest_path_distance", None)
        if callable(method):
            try:
                return int(method(start, target, grid))
            except (TypeError, ValueError):
                pass

        start_pos = (int(start[0]), int(start[1]))
        target_pos = (int(target[0]), int(target[1]))
        if start_pos == target_pos:
            return 0

        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        queue = deque([(start_pos, 0)])
        visited = {start_pos}

        while queue:
            (row, col), distance = queue.popleft()
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                position = (nr, nc)
                if position in visited:
                    continue
                if not (
                    0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                ):
                    continue
                if int(grid[nr, nc]) == 1:
                    continue
                if position == target_pos:
                    return distance + 1
                visited.add(position)
                queue.append((position, distance + 1))

        return 10**9

    def _objective_override(
        self,
        char: Any,
        game_state: dict,
    ) -> int | None:
        """既存Fnatic v1と同じ設置・落下スパイク回収の安全装置。"""
        grid = game_state["grid"]
        row, col = int(char.pos[0]), int(char.pos[1])

        if (
            getattr(char, "has_spike", False)
            and int(grid[row, col]) == 2
            and not bool(game_state.get("is_planted", False))
        ):
            return 8

        spike_pos = game_state.get("spike_pos")
        holder = next(
            (
                other
                for other in game_state.get("chars", [])
                if _alive(other)
                and getattr(other, "team", None) == char.team
                and getattr(other, "has_spike", False)
            ),
            None,
        )
        if holder is not None or spike_pos is None:
            return None

        alive_attackers = [
            other
            for other in game_state.get("chars", [])
            if _alive(other)
            and getattr(other, "team", None) == char.team
        ]
        if not alive_attackers:
            return None

        retriever = min(
            alive_attackers,
            key=lambda other: (
                self._shortest_path_distance(
                    other.pos,
                    spike_pos,
                    grid,
                ),
                str(getattr(other, "name", "")),
            ),
        )
        if retriever is not char:
            return None

        next_pos = self.helper.move_towards_target(
            char.pos,
            spike_pos,
            grid,
            chars=game_state.get("chars", []),
            moving_char=char,
        )
        dr = int(next_pos[0]) - row
        dc = int(next_pos[1]) - col
        return {
            (-1, 0): 0,
            (1, 0): 1,
            (0, -1): 2,
            (0, 1): 3,
        }.get((dr, dc))

    def decide_move(self, char: Any, game_state: dict) -> Any:
        if self.game is None:
            raise RuntimeError("set_game(game)が未実行です")

        # 同じ選手の前回遷移だけを閉じる。
        self._close(char)

        observation = get_game_observation(self.game, char)
        obs = observation_to_vector(observation)
        if len(obs) != self.model.obs_size:
            raise ValueError(
                f"obs mismatch {len(obs)} != {self.model.obs_size}"
            )

        # Rollout中は常にeval。Dropoutを無効にする。
        self.model.eval()

        tensor = (
            torch.from_numpy(obs)
            .to(dtype=torch.float32, device=self.device)
            .unsqueeze(0)
        )
        mask = build_action_mask(self.game, char, game_state)
        mask_device = mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(tensor)
            masked_logits = logits.masked_fill(
                ~mask_device,
                torch.finfo(logits.dtype).min,
            )
            distribution = Categorical(logits=masked_logits)

            if self.deterministic:
                action_tensor = masked_logits.argmax(dim=1)
            else:
                action_tensor = distribution.sample()

            log_prob = distribution.log_prob(action_tensor)

        sampled_action = int(action_tensor.item())
        forced_action = self._objective_override(char, game_state)
        executed_action = (
            int(forced_action)
            if forced_action is not None
            else sampled_action
        )
        self.action_counts[executed_action] += 1

        # 強制行動は方策からsampleされたものではないため学習対象外。
        if forced_action is None:
            snapshot = self._snapshot(char)
            trajectory_id = id(char)
            self.pending[trajectory_id] = PendingStep(
                obs=obs.copy(),
                action=sampled_action,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                mask=mask.cpu().numpy().astype(np.bool_),
                snapshot=snapshot,
                char_name=str(getattr(char, "name", "")),
            )
            self.best_site_distance.setdefault(
                trajectory_id,
                float(snapshot.get("site_distance", 0.0)),
            )
            self.position_history.setdefault(
                trajectory_id,
                deque(
                    [
                        (
                            int(snapshot.get("pos_r", 0.0)),
                            int(snapshot.get("pos_c", 0.0)),
                        )
                    ],
                    maxlen=3,
                ),
            )
        else:
            self.forced_action_count += 1

        helper_result = self.helper.decide_move(char, game_state)
        return decode_action(
            executed_action,
            char,
            game_state,
            helper_result,
        )

    def finish_episode(self) -> None:
        if self.game is None:
            return

        attacker_score = int(getattr(self.game, "attacker_wins", 0))
        defender_score = int(getattr(self.game, "defender_wins", 0))
        if attacker_score > defender_score:
            terminal_bonus = 12.0
        elif defender_score > attacker_score:
            terminal_bonus = -12.0
        else:
            terminal_bonus = 0.0

        chars_by_id = {id(char): char for char in self.game.chars}
        for trajectory_id in list(self.pending):
            char = chars_by_id.get(trajectory_id)
            if char is not None:
                self._close(
                    char,
                    done=True,
                    terminal_bonus=terminal_bonus,
                )
            else:
                self.pending.pop(trajectory_id, None)
