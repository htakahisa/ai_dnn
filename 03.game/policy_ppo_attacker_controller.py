from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch

from controllers import DefaultAttackerController
from policy_attacker_controller import (
    build_action_mask,
    decode_action,
    get_game_observation,
)
from ppo_actor_critic import PPOActorCritic
from train_bc import NUM_ACTIONS, observation_to_vector


def resolve_device(requested: str) -> torch.device:
    normalized = str(requested or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"deviceは auto / cpu / cuda のいずれかです: {requested}")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。device='cpu'を使用してください。")
    return torch.device(normalized)


def load_ppo_policy(model_path: str | Path, device: torch.device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PPO学習済みモデルが見つかりません: {model_path}")

    loaded = torch.load(model_path, map_location=device)
    if not isinstance(loaded, dict):
        raise TypeError("PPOモデルファイルが辞書形式ではありません")

    state_dict = loaded.get("model_state_dict", loaded)
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }

    first_weight = state_dict.get("encoder.0.weight")
    policy_weight = state_dict.get("policy_head.weight")
    if first_weight is None or policy_weight is None:
        raise KeyError("PPO checkpointの構造が不正です")

    obs_size = int(loaded.get("obs_size", first_weight.shape[1]))
    num_actions = int(loaded.get("num_actions", policy_weight.shape[0]))
    if num_actions != NUM_ACTIONS:
        raise ValueError(
            f"アクション数不一致: model={num_actions}, code={NUM_ACTIONS}"
        )

    model = PPOActorCritic(obs_size, num_actions).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, obs_size


def _alive(char: Any) -> bool:
    return bool(getattr(char, "is_alive", True))


class PolicyPPOAttackerController:
    def __init__(
        self,
        model_path: str | Path = (
            "ppo_attacker_checkpoints/"
            "policy_fnatic_attacker_ppo_best.pt"
        ),
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.device = resolve_device(device)
        self.model, self.obs_size = load_ppo_policy(
            self.model_path,
            self.device,
        )
        self.game = None
        self.target_helper = DefaultAttackerController()

        print(f"Fnatic PPO attacker model : {self.model_path}")
        print(f"Fnatic PPO attacker device: {self.device}")

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        if hasattr(self.target_helper, "reset_round"):
            self.target_helper.reset_round()

    def _shortest_path_distance(self, start, target, grid):
        method = getattr(self.target_helper, "shortest_path_distance", None)
        if callable(method):
            try:
                return int(method(start, target, grid))
            except (TypeError, ValueError):
                pass

        start_pos = (int(start[0]), int(start[1]))
        target_pos = (int(target[0]), int(target[1]))
        if start_pos == target_pos:
            return 0

        queue = deque([(start_pos, 0)])
        visited = {start_pos}
        for_pos = ((-1, 0), (1, 0), (0, -1), (0, 1))

        while queue:
            (row, col), distance = queue.popleft()
            for dr, dc in for_pos:
                nr, nc = row + dr, col + dc
                pos = (nr, nc)
                if pos in visited:
                    continue
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    continue
                if int(grid[nr, nc]) == 1:
                    continue
                if pos == target_pos:
                    return distance + 1
                visited.add(pos)
                queue.append((pos, distance + 1))
        return 10**9

    def _objective_override(self, char, game_state):
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
                self._shortest_path_distance(other.pos, spike_pos, grid),
                str(getattr(other, "name", "")),
            ),
        )
        if retriever is not char:
            return None

        next_pos = self.target_helper.move_towards_target(
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

    def decide_move(self, char, game_state):
        if self.game is None:
            raise RuntimeError("PolicyPPOAttackerController.set_game(game)が未実行です")

        observation = get_game_observation(self.game, char)
        obs_vector = observation_to_vector(observation)
        if int(obs_vector.shape[0]) != self.obs_size:
            raise ValueError(
                f"観測次元不一致: observation={obs_vector.shape[0]}, "
                f"model={self.obs_size}"
            )

        action_mask = build_action_mask(self.game, char, game_state)

        self.model.eval()
        with torch.inference_mode():
            obs_tensor = (
                torch.from_numpy(obs_vector)
                .to(dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
            logits, _ = self.model(obs_tensor)
            mask = action_mask.to(self.device).unsqueeze(0)
            masked_logits = logits.masked_fill(
                ~mask,
                torch.finfo(logits.dtype).min,
            )
            action_index = int(masked_logits.argmax(dim=1).item())

        forced_action = self._objective_override(char, game_state)
        if forced_action is not None:
            action_index = int(forced_action)

        helper_result = self.target_helper.decide_move(char, game_state)
        return decode_action(action_index, char, game_state, helper_result)
