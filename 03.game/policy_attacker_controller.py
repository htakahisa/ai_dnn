"""
Fnatic v1 用 学習済みポリシーController。

run_game.py から直接読み込み、policy_fnatic_attacker_dagger_final.pt で
アタッカーを操作する。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers import DefaultAttackerController
from train_bc import NUM_ACTIONS, PolicyNetwork, observation_to_vector

DIRECTION_BY_ACTION = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
ABILITY_BY_ACTION = {4: "SMOKE", 5: "FLASH", 6: "RECON"}

LOCAL_MAP_RADIUS = 3
LOCAL_EMPTY = 0
LOCAL_WALL = 1
LOCAL_SITE = 2
LOCAL_ALLY = 3
LOCAL_ENEMY = 4
LOCAL_SELF = 5
LOCAL_OUT_OF_MAP = 6
LOCAL_SPIKE = 7


def character_is_alive(char: Any) -> bool:
    return bool(getattr(char, "is_alive", True))


def resolve_device(requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"deviceは auto / cpu / cuda のいずれかです: {requested}")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。device='cpu'を使用してください。")
    return normalized


def occupied_positions(
    game: Any, viewer: Any | None = None
) -> dict[tuple[int, int], Any]:
    occupied: dict[tuple[int, int], Any] = {}
    for char in getattr(game, "chars", []):
        if not character_is_alive(char) or char is viewer:
            continue
        pos = getattr(char, "pos", None)
        if pos is None or len(pos) != 2:
            continue
        occupied[(int(pos[0]), int(pos[1]))] = char
    return occupied


def is_valid_destination(game: Any, viewer: Any, row: int, col: int) -> bool:
    row = int(row)
    col = int(col)
    grid = game.grid
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return False
    if grid[row, col] == 1:
        return False
    return (row, col) not in occupied_positions(game, viewer)


def build_valid_move_mask(game: Any, viewer: Any) -> list[int]:
    row = int(viewer.pos[0])
    col = int(viewer.pos[1])
    return [
        int(is_valid_destination(game, viewer, row + dr, col + dc))
        for dr, dc in DIRECTION_BY_ACTION.values()
    ]


def build_action_mask(game: Any, char: Any, game_state: dict) -> torch.Tensor:
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    move_mask = build_valid_move_mask(game, char)
    for action_idx in range(4):
        mask[action_idx] = bool(move_mask[action_idx])

    mask[4] = getattr(char, "smoke_charges", 0) > 0
    mask[5] = getattr(char, "flash_charges", 0) > 0
    mask[6] = getattr(char, "recon_charges", 0) > 0
    mask[7] = True

    row = int(char.pos[0])
    col = int(char.pos[1])
    grid = game_state["grid"]
    mask[8] = bool(
        getattr(char, "has_spike", False)
        and grid[row, col] == 2
        and not bool(game_state.get("is_planted", False))
    )
    return mask


def masked_action_probabilities(
    logits: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"expected logits shape [1, N], got {tuple(logits.shape)}")
    if action_mask.numel() != logits.shape[1]:
        raise ValueError(
            f"action mask size mismatch: mask={action_mask.numel()}, logits={logits.shape[1]}"
        )
    mask = action_mask.to(logits.device).unsqueeze(0)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked_logits, dim=1)


def build_local_map(
    game: Any, viewer: Any, radius: int = LOCAL_MAP_RADIUS
) -> list[list[int]]:
    grid = game.grid
    height, width = grid.shape
    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])
    viewer_team = viewer.team

    characters: dict[tuple[int, int], Any] = {}
    for char in getattr(game, "chars", []):
        if character_is_alive(char):
            characters[(int(char.pos[0]), int(char.pos[1]))] = char

    spike = getattr(game, "spike_pos", None)
    spike_position = (int(spike[0]), int(spike[1])) if spike is not None else None

    local_map: list[list[int]] = []
    for dr in range(-radius, radius + 1):
        row_values: list[int] = []
        for dc in range(-radius, radius + 1):
            row = viewer_row + dr
            col = viewer_col + dc
            position = (row, col)

            if not (0 <= row < height and 0 <= col < width):
                value = LOCAL_OUT_OF_MAP
            elif position == (viewer_row, viewer_col):
                value = LOCAL_SELF
            elif position in characters:
                other = characters[position]
                value = (
                    LOCAL_ALLY
                    if getattr(other, "team", None) == viewer_team
                    else LOCAL_ENEMY
                )
            elif spike_position is not None and position == spike_position:
                value = LOCAL_SPIKE
            elif grid[row, col] == 1:
                value = LOCAL_WALL
            elif grid[row, col] == 2:
                value = LOCAL_SITE
            else:
                value = LOCAL_EMPTY
            row_values.append(int(value))
        local_map.append(row_values)
    return local_map


def get_game_observation(game: Any, viewer: Any) -> dict:
    if viewer is None:
        raise ValueError("viewer must not be None")

    viewer_team = viewer.team
    enemy_team = "D" if viewer_team == "A" else "A"
    viewer_row = int(viewer.pos[0])
    viewer_col = int(viewer.pos[1])

    obs = {
        "grid": game.grid.flatten().tolist(),
        "allies": [],
        "visible_enemies": [],
        "game_state": [],
        "spike_pos": [0, 0],
        "target_plant_pos": [0, 0],
        "visible_enemy_count": 0,
        "distance_to_site": 0.0,
        "viewer_pos": [viewer_row, viewer_col],
        "local_map": build_local_map(game, viewer),
        "valid_move_mask": build_valid_move_mask(game, viewer),
        "spike_on_ground": 1 if getattr(game, "spike_pos", None) is not None else 0,
        "ally_has_spike": (
            1
            if any(
                getattr(other, "team", None) == viewer_team
                and character_is_alive(other)
                and getattr(other, "has_spike", False)
                for other in game.chars
            )
            else 0
        ),
        "viewer_has_spike": 1 if getattr(viewer, "has_spike", False) else 0,
    }

    allies = [viewer] + [
        char for char in game.chars if char.team == viewer_team and char is not viewer
    ]
    for char in allies:
        obs["allies"].append(
            {
                "name": char.name,
                "pos": [int(char.pos[0]), int(char.pos[1])],
                "rel_pos": [
                    int(char.pos[0]) - viewer_row,
                    int(char.pos[1]) - viewer_col,
                ],
                "hp": int(char.hp),
                "is_alive": 1 if character_is_alive(char) else 0,
                "has_spike": 1 if getattr(char, "has_spike", False) else 0,
                "recon_cd": 1 if getattr(char, "recon_charges", 0) > 0 else 0,
                "flash_cd": 1 if getattr(char, "flash_charges", 0) > 0 else 0,
                "smoke_cd": 1 if getattr(char, "smoke_charges", 0) > 0 else 0,
            }
        )

    for char in game.chars:
        if char.team == enemy_team and character_is_alive(char):
            obs["visible_enemies"].append(
                {
                    "name": char.name,
                    "rel_pos": [
                        int(char.pos[0]) - viewer_row,
                        int(char.pos[1]) - viewer_col,
                    ],
                    "hp": int(char.hp),
                }
            )

    obs["visible_enemy_count"] = len(obs["visible_enemies"])

    spike_pos = getattr(game, "spike_pos", None)
    if spike_pos is not None:
        obs["spike_pos"] = [
            int(spike_pos[0]) - viewer_row,
            int(spike_pos[1]) - viewer_col,
        ]

    target = getattr(game, "target_plant_pos", None)
    if target is not None:
        plant_dr = int(target[0]) - viewer_row
        plant_dc = int(target[1]) - viewer_col
        obs["target_plant_pos"] = [plant_dr, plant_dc]
        obs["distance_to_site"] = float(abs(plant_dr) + abs(plant_dc))

    obs["game_state"] = [
        float(getattr(game, "round_timer", 0.0)) / 100.0,
        1.0 if bool(getattr(game, "is_planted", False)) else 0.0,
    ]
    return obs


def choose_ability_target(
    ability_name: str, char: Any, game_state: dict, helper_result: Any
) -> tuple[int, int]:
    if (
        isinstance(helper_result, tuple)
        and len(helper_result) == 2
        and isinstance(helper_result[1], dict)
        and str(helper_result[1].get("ability", "")).upper() == ability_name
    ):
        target = helper_result[1].get("target")
        if target is not None:
            return int(target[0]), int(target[1])

    enemies = [
        other
        for other in game_state.get("chars", [])
        if character_is_alive(other) and getattr(other, "team", None) != char.team
    ]
    if ability_name in {"SMOKE", "FLASH"} and enemies:
        closest = min(
            enemies,
            key=lambda enemy: max(
                abs(int(enemy.pos[0]) - int(char.pos[0])),
                abs(int(enemy.pos[1]) - int(char.pos[1])),
            ),
        )
        return int(closest.pos[0]), int(closest.pos[1])

    if ability_name == "RECON":
        target = game_state.get("planted_pos") or game_state.get("target_plant_pos")
        if target is not None:
            return int(target[0]), int(target[1])

    return int(char.pos[0]), int(char.pos[1])


def load_policy(model_path: str | Path, device: str) -> tuple[PolicyNetwork, int]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

    loaded = torch.load(model_path, map_location=device)
    if not isinstance(loaded, dict):
        raise TypeError("モデルファイルが辞書形式ではありません")

    if isinstance(loaded.get("model_state_dict"), dict):
        state_dict = loaded["model_state_dict"]
        obs_size = loaded.get("obs_size")
        num_actions = int(loaded.get("num_actions", NUM_ACTIONS))
    elif isinstance(loaded.get("state_dict"), dict):
        state_dict = loaded["state_dict"]
        obs_size = loaded.get("obs_size")
        num_actions = int(loaded.get("num_actions", NUM_ACTIONS))
    elif isinstance(loaded.get("policy_state_dict"), dict):
        state_dict = loaded["policy_state_dict"]
        obs_size = loaded.get("obs_size")
        num_actions = int(loaded.get("num_actions", NUM_ACTIONS))
    else:
        state_dict = loaded
        obs_size = None
        num_actions = NUM_ACTIONS

    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("module."): value for key, value in state_dict.items()
        }

    first_weight = state_dict.get("net.0.weight")
    if first_weight is None:
        raise KeyError("モデル内に net.0.weight がありません")

    inferred_obs_size = int(first_weight.shape[1])
    obs_size = inferred_obs_size if obs_size is None else int(obs_size)

    if num_actions != NUM_ACTIONS:
        raise ValueError(
            f"モデルと現在コードのアクション数が一致しません: model={num_actions}, code={NUM_ACTIONS}"
        )

    model = PolicyNetwork(obs_size=obs_size, num_actions=num_actions).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, obs_size


def decode_action(
    action_idx: int, char: Any, game_state: dict, helper_result: Any
) -> Any:
    grid = game_state["grid"]

    if action_idx in DIRECTION_BY_ACTION:
        dr, dc = DIRECTION_BY_ACTION[action_idx]
        nr = int(char.pos[0]) + dr
        nc = int(char.pos[1]) + dc
        valid = (
            0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != 1
        )
        if valid:
            for other in game_state.get("chars", []):
                if other is char or not character_is_alive(other):
                    continue
                if int(other.pos[0]) == nr and int(other.pos[1]) == nc:
                    valid = False
                    break
        return [nr, nc] if valid else list(char.pos)

    if action_idx in ABILITY_BY_ACTION:
        ability_name = ABILITY_BY_ACTION[action_idx]
        charge_name = {
            "SMOKE": "smoke_charges",
            "FLASH": "flash_charges",
            "RECON": "recon_charges",
        }[ability_name]
        if getattr(char, charge_name, 0) <= 0:
            return list(char.pos)
        target = choose_ability_target(ability_name, char, game_state, helper_result)
        return list(char.pos), {"ability": ability_name, "target": target}

    if action_idx == 8:
        row = int(char.pos[0])
        col = int(char.pos[1])
        if (
            getattr(char, "has_spike", False)
            and grid[row, col] == 2
            and not bool(game_state.get("is_planted", False))
        ):
            return list(char.pos), "PLANT"

    return list(char.pos)


def _fallback_shortest_path_distance(start: Any, target: Any, grid: np.ndarray) -> int:
    start_pos = (int(start[0]), int(start[1]))
    target_pos = (int(target[0]), int(target[1]))
    if start_pos == target_pos:
        return 0

    queue = deque([(start_pos, 0)])
    visited = {start_pos}
    while queue:
        (row, col), distance = queue.popleft()
        for dr, dc in DIRECTION_BY_ACTION.values():
            nr = row + dr
            nc = col + dc
            next_pos = (nr, nc)
            if next_pos in visited:
                continue
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if grid[nr, nc] == 1:
                continue
            if next_pos == target_pos:
                return distance + 1
            visited.add(next_pos)
            queue.append((next_pos, distance + 1))
    return 10**9


class PolicyAttackerController:
    """Fnatic v1 学習済みアタッカーController。"""

    def __init__(
        self,
        model_path: str | Path = "policy_fnatic_attacker_dagger_final.pt",
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = resolve_device(device)
        self.model, self.obs_size = load_policy(self.model_path, self.device)
        self.game: Any | None = None
        self.target_helper = DefaultAttackerController()

        print(f"Fnatic v1 model : {self.model_path}")
        print(f"Fnatic v1 device: {self.device}")

    def set_game(self, game: Any) -> None:
        self.game = game

    def reset_round(self) -> None:
        if hasattr(self.target_helper, "reset_round"):
            self.target_helper.reset_round()

    def _shortest_path_distance(self, start: Any, target: Any, grid: np.ndarray) -> int:
        method = getattr(self.target_helper, "shortest_path_distance", None)
        if callable(method):
            try:
                return int(method(start, target, grid))
            except (TypeError, ValueError):
                pass
        return _fallback_shortest_path_distance(start, target, grid)

    def _objective_override(self, char: Any, game_state: dict) -> int | None:
        grid = game_state["grid"]
        row = int(char.pos[0])
        col = int(char.pos[1])

        if (
            getattr(char, "has_spike", False)
            and grid[row, col] == 2
            and not bool(game_state.get("is_planted", False))
        ):
            return 8

        spike_pos = game_state.get("spike_pos")
        holder = next(
            (
                other
                for other in game_state.get("chars", [])
                if character_is_alive(other)
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
            if character_is_alive(other) and getattr(other, "team", None) == char.team
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
        return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get((dr, dc))

    def decide_move(self, char: Any, game_state: dict) -> Any:
        if self.game is None:
            raise RuntimeError("PolicyAttackerController.set_game(game)が未実行です")

        observation = get_game_observation(self.game, char)
        obs_vec = observation_to_vector(observation)
        if int(obs_vec.shape[0]) != self.obs_size:
            raise ValueError(
                f"観測次元がモデルと一致しません: observation={obs_vec.shape[0]}, model={self.obs_size}"
            )

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)
            logits = self.model(obs_tensor)
            action_mask = build_action_mask(self.game, char, game_state)
            probabilities = masked_action_probabilities(logits, action_mask)
            action_idx = int(probabilities.argmax(dim=1).item())

        forced_action = self._objective_override(char, game_state)
        if forced_action is not None:
            action_idx = forced_action

        helper_result = self.target_helper.decide_move(char, game_state)
        return decode_action(action_idx, char, game_state, helper_result)
