# learning_attacker_ai_v2.py
"""
学習済み Attacker AI ver2.0 を実戦ゲームへ接続するコントローラー。

重要:
- train_attacker_ai_v2.py と同じ125次元観測・13行動を使用する。
- 5人全員が同じモデルを共有する。
- decide_ability()、collision_safe_step()、FixedEscortControllerは使わない。
- AIが選んだ不正移動を別ルートへ自動修正しない。
- MOVE / PLANT / ABILITY を battle_logic.py が理解できる形式で返す。

推奨モデル:
    attacker_ai_v2_data/dqn_attacker_ai_v2_best.pt
または
    attacker_ai_v2_data/dqn_attacker_ai_v2_final.pt

battle_logic.py の game_state に次の項目があると、学習環境との一致度が上がる:
    "smokes"
    "battle_tick"
    "spike_timer"
    "active_defuser_name"

無い場合もゼロ値で動作する。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from controllers import BaseController
from train_attacker_ai_v2 import (
    ABILITY_ACTIONS,
    ABILITY_HUNT,
    ABILITY_NONE,
    ABILITY_TO_INDEX,
    ACTION_ABILITY_DOWN_FAR,
    ACTION_ABILITY_DOWN_NEAR,
    ACTION_ABILITY_LEFT_FAR,
    ACTION_ABILITY_LEFT_NEAR,
    ACTION_ABILITY_RIGHT_FAR,
    ACTION_ABILITY_RIGHT_NEAR,
    ACTION_ABILITY_UP_FAR,
    ACTION_ABILITY_UP_NEAR,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_WAIT_OR_PLANT,
    BLIND_DURATION_TICKS,
    DEFUSE_REQUIRED_TICKS,
    DuelingQNetwork,
    ENEMY_MEMORY_TICKS,
    ENEMY_SLOT_DIM,
    K_ENEMIES,
    MOVE_DELTAS,
    N_ACTIONS,
    N_ATTACKERS,
    OBS_DIM,
    PLANT_REQUIRED_TICKS,
    REVEAL_DURATION_TICKS,
    ROUND_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
    TEAM_ATTACKER,
    TEAM_DEFENDER,
    TEAMMATE_SLOT_DIM,
    bresenham_cells,
    safe_normalized_distance,
)


Position = Tuple[int, int]


@dataclass
class RuntimeEnemyMemoryEntry:
    pos: Position
    hp: int
    age: int
    visible: bool
    revealed: bool


@dataclass
class RequestedMove:
    origin: Position
    target: Position
    action: int
    tick_id: Optional[int]


class RuntimeEnemyMemory:
    """実戦用Characterを対象にした敵記憶。"""

    def __init__(
        self,
        memory_ticks: int = ENEMY_MEMORY_TICKS,
        k_enemies: int = K_ENEMIES,
    ) -> None:
        self.memory_ticks = int(memory_ticks)
        self.k_enemies = int(k_enemies)
        self.entries: Dict[str, RuntimeEnemyMemoryEntry] = {}

    def reset(self) -> None:
        self.entries.clear()

    def update(
        self,
        observer,
        enemies: Sequence,
        grid: np.ndarray,
        smoke_cells: Set[Position],
    ) -> Set[str]:
        visible_names: Set[str] = set()

        for enemy in enemies:
            name = str(enemy.name)
            if not _is_alive(enemy):
                self.entries.pop(name, None)
                continue

            revealed = _reveal_ticks(enemy) > 0
            visible = (
                _is_alive(observer)
                and _blind_ticks(observer) <= 0
                and (
                    revealed
                    or has_runtime_line_of_sight(
                        tuple(observer.pos),
                        tuple(enemy.pos),
                        grid,
                        smoke_cells,
                    )
                )
            )

            if visible:
                self.entries[name] = RuntimeEnemyMemoryEntry(
                    pos=tuple(enemy.pos),
                    hp=int(getattr(enemy, "hp", 0)),
                    age=0,
                    visible=True,
                    revealed=revealed,
                )
                visible_names.add(name)

        for name in list(self.entries):
            if name in visible_names:
                continue

            entry = self.entries[name]
            entry.age += 1
            entry.visible = False
            entry.revealed = False
            if entry.age > self.memory_ticks:
                del self.entries[name]

        return visible_names

    def build_features(
        self,
        observer_pos: Position,
        height: int,
        width: int,
    ) -> List[float]:
        pr, pc = observer_pos
        ordered = sorted(
            self.entries.items(),
            key=lambda item: max(
                abs(item[1].pos[0] - pr),
                abs(item[1].pos[1] - pc),
            ),
        )

        features: List[float] = []
        for slot in range(self.k_enemies):
            if slot >= len(ordered):
                features.extend([0.0] * ENEMY_SLOT_DIM)
                continue

            _, entry = ordered[slot]
            er, ec = entry.pos
            features.extend(
                [
                    1.0,
                    1.0 if entry.visible else 0.0,
                    0.0 if entry.visible else 1.0,
                    (er - pr) / max(height - 1, 1),
                    (ec - pc) / max(width - 1, 1),
                    max(entry.hp, 0) / max(_runtime_max_hp_value(), 1),
                    min(entry.age / max(self.memory_ticks, 1), 1.0),
                    1.0 if entry.revealed else 0.0,
                ]
            )

        return features


def _runtime_max_hp_value() -> int:
    return 100


def _is_alive(char) -> bool:
    return bool(getattr(char, "is_alive", getattr(char, "alive", False)))


def _blind_ticks(char) -> float:
    return float(
        getattr(
            char,
            "blind_remaining",
            getattr(char, "blind_ticks", 0.0),
        )
    )


def _reveal_ticks(char) -> float:
    return float(
        getattr(
            char,
            "reveal_remaining",
            getattr(char, "reveal_ticks", 0.0),
        )
    )


def _ability_charges(char) -> int:
    ability = str(getattr(char, "ability_name", ABILITY_NONE)).upper()
    if ability == "SMOKE":
        return int(getattr(char, "smoke_charges", 0))
    if ability == "FLASH":
        return int(getattr(char, "flash_charges", 0))
    if ability == "RECON":
        return int(getattr(char, "recon_charges", 0))
    return 0


def _inside(pos: Position, grid: np.ndarray) -> bool:
    return 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]


def _is_walkable(pos: Position, grid: np.ndarray) -> bool:
    return _inside(pos, grid) and grid[pos] != 1


def _smoke_cells_from_game_state(game_state: dict) -> Set[Position]:
    result: Set[Position] = set()
    smokes = game_state.get("smokes", [])

    for smoke in smokes:
        if isinstance(smoke, dict):
            cells = smoke.get("cells")
            if cells is not None:
                for cell in cells:
                    if isinstance(cell, (list, tuple)) and len(cell) == 2:
                        result.add((int(cell[0]), int(cell[1])))
                continue

            center = smoke.get("center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                cr, cc = int(center[0]), int(center[1])
                radius = int(smoke.get("radius", 1))
                for r in range(cr - radius, cr + radius + 1):
                    for c in range(cc - radius, cc + radius + 1):
                        result.add((r, c))
        else:
            center = getattr(smoke, "center", None)
            if isinstance(center, (list, tuple)) and len(center) == 2:
                cr, cc = int(center[0]), int(center[1])
                radius = int(getattr(smoke, "radius", 1))
                for r in range(cr - radius, cr + radius + 1):
                    for c in range(cc - radius, cc + radius + 1):
                        result.add((r, c))

    return result


def has_runtime_line_of_sight(
    start: Position,
    end: Position,
    grid: np.ndarray,
    smoke_cells: Set[Position],
) -> bool:
    if start == end:
        return True

    cells = bresenham_cells(start, end)
    for index, cell in enumerate(cells):
        if not _inside(cell, grid):
            return False
        if grid[cell] == 1:
            return False
        if index > 0 and cell in smoke_cells:
            return False
    return True


def _distance_map(goal: Position, grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    distances = np.full((height, width), np.inf, dtype=np.float32)

    if not _is_walkable(goal, grid):
        return distances

    queue: List[Position] = [goal]
    distances[goal] = 0.0
    head = 0

    while head < len(queue):
        r, c = queue[head]
        head += 1
        next_distance = distances[r, c] + 1.0

        for dr, dc in MOVE_DELTAS.values():
            nr, nc = r + dr, c + dc
            if not _is_walkable((nr, nc), grid):
                continue
            if next_distance < distances[nr, nc]:
                distances[nr, nc] = next_distance
                queue.append((nr, nc))

    return distances


class LearningAttackerAIv2Controller(BaseController):
    """
    アタッカー5人共通モデル用コントローラー。

    battle_logic.py は各キャラクターについて decide_move() を呼ぶ。
    モデル自体は1個だけ保持し、観測履歴・敵記憶はキャラクター名ごとに分離する。
    """

    def __init__(
        self,
        model_path: str = "attacker_ai_v2_data/dqn_attacker_ai_v2_best.pt",
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        greedy: bool = True,
        temperature: float = 0.35,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if obs_dim != OBS_DIM:
            raise ValueError(
                f"obs_dimは学習側と同じ{OBS_DIM}である必要があります: {obs_dim}"
            )
        if n_actions != N_ACTIONS:
            raise ValueError(
                f"n_actionsは学習側と同じ{N_ACTIONS}である必要があります: {n_actions}"
            )
        if temperature <= 0:
            raise ValueError("temperatureは0より大きくしてください。")

        if device is None:
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device

        self.device = torch.device(selected_device)
        self.greedy = bool(greedy)
        self.temperature = float(temperature)
        self.verbose = bool(verbose)

        model_path_obj = Path(model_path)
        self.model_path = (
            model_path_obj
            if model_path_obj.is_absolute()
            else Path(__file__).resolve().parent / model_path_obj
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "AI ver2.0のモデルが見つかりません。\n"
                f"探した場所: {self.model_path}\n"
                "train_attacker_ai_v2.pyで学習後、"
                "dqn_attacker_ai_v2_best.pt または final.pt を指定してください。"
            )

        self.model = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self._load_model(self.model_path)
        self.model.eval()

        self.last_actions: Dict[str, int] = {}
        self.enemy_memories: Dict[str, RuntimeEnemyMemory] = {}
        self.requested_moves: Dict[str, RequestedMove] = {}
        self.last_move_failed: Dict[str, bool] = {}
        self.last_move_fail_reason: Dict[str, str] = {}

        self._distance_cache_key: Optional[Tuple[int, int, int]] = None
        self._distance_cache: Optional[np.ndarray] = None
        self._fallback_tick = 0
        self._last_seen_game_tick: Optional[int] = None

        print(
            f"[AI v2.0] model={self.model_path} "
            f"device={self.device} greedy={self.greedy}"
        )

    def _load_model(self, path: Path) -> None:
        loaded = torch.load(str(path), map_location=self.device)

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "モデル構造がtrain_attacker_ai_v2.pyと一致しません。\n"
                f"期待: OBS_DIM={OBS_DIM}, N_ACTIONS={N_ACTIONS}\n"
                f"model: {path}"
            ) from exc

    def reset_round(self) -> None:
        self.last_actions.clear()
        self.enemy_memories.clear()
        self.requested_moves.clear()
        self.last_move_failed.clear()
        self.last_move_fail_reason.clear()
        self._distance_cache_key = None
        self._distance_cache = None
        self._fallback_tick = 0
        self._last_seen_game_tick = None

    def _memory_for(self, char_name: str) -> RuntimeEnemyMemory:
        if char_name not in self.enemy_memories:
            self.enemy_memories[char_name] = RuntimeEnemyMemory()
        return self.enemy_memories[char_name]

    def decide_move(self, char, game_state: dict):
        """
        戻り値:
            MOVE:
                ([row, col], "MOVE")

            PLANT:
                ([row, col], "PLANT")

            ABILITY:
                ([row, col], "ABILITY", {
                    "ability": "SMOKE"/"FLASH"/"RECON",
                    "target": [row, col],
                })
        """
        grid = np.asarray(game_state["grid"])
        chars = list(game_state["chars"])
        tick_id = self._get_tick_id(game_state)

        self._update_previous_move_result(char, grid, chars, tick_id)

        observation = self._make_observation(
            char=char,
            game_state=game_state,
            grid=grid,
            chars=chars,
        )
        action, q_values = self._select_action(observation)

        self.last_actions[str(char.name)] = action

        command = self._action_to_game_command(
            char=char,
            action=action,
            grid=grid,
            game_state=game_state,
            tick_id=tick_id,
        )

        if self.verbose:
            print(
                f"[AI v2.0] tick={tick_id} char={char.name} "
                f"action={action}:{ACTION_NAMES.get(action)} "
                f"q={np.round(q_values, 3).tolist()} "
                f"command={command}"
            )
        print(
            action,
            ACTION_NAMES.get(action),
            char.pos,
        )

        objective = self._objective_for(
        char,
        game_state,
        chars,
        grid,
        )

        dr = objective[0] - int(char.pos[0])
        dc = objective[1] - int(char.pos[1])

        print(
             action,
             ACTION_NAMES.get(action, str(action)),
            "pos=", list(char.pos),
            "objective=", objective,
            "delta=", (dr, dc),
            "has_spike=", bool(getattr(char, "has_spike", False)),
        )
        return command

    def _select_action(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state).squeeze(0).cpu().numpy()

        if self.greedy:
            action = int(np.argmax(q_values))
        else:
            centered = q_values - np.max(q_values)
            logits = centered / self.temperature
            probabilities = np.exp(logits)
            probability_sum = probabilities.sum()
            if not np.isfinite(probability_sum) or probability_sum <= 0:
                action = random.randrange(N_ACTIONS)
            else:
                probabilities /= probability_sum
                action = int(np.random.choice(N_ACTIONS, p=probabilities))

        return action, q_values

    def _action_to_game_command(
        self,
        char,
        action: int,
        grid: np.ndarray,
        game_state: dict,
        tick_id: Optional[int],
    ):
        current = (int(char.pos[0]), int(char.pos[1]))

        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            target = (current[0] + dr, current[1] + dc)

            # ここでは壁・味方を理由に別方向へ変えない。
            # battle_logic.pyがゲームルールとして成功/失敗を決める。
            self.requested_moves[str(char.name)] = RequestedMove(
                origin=current,
                target=target,
                action=action,
                tick_id=tick_id,
            )
            return [target[0], target[1]], "MOVE"

        if action == ACTION_WAIT_OR_PLANT:
            self.requested_moves.pop(str(char.name), None)

            if (
                bool(getattr(char, "has_spike", False))
                and not bool(game_state.get("is_planted", False))
                and self._is_runtime_plant_position(char, game_state, grid)
            ):
                return list(current), "PLANT"

            return list(current), "MOVE"

        if action in ABILITY_ACTIONS:
            self.requested_moves.pop(str(char.name), None)
            target = self._ability_target(current, action, grid)
            payload = {
                "ability": str(getattr(char, "ability_name", "")).upper(),
                "target": [target[0], target[1]],
            }
            return list(current), "ABILITY", payload

        self.requested_moves.pop(str(char.name), None)
        return list(current), "MOVE"

    def _ability_target(
        self,
        origin: Position,
        action: int,
        grid: np.ndarray,
    ) -> Position:
        spec = ABILITY_ACTIONS.get(action)
        if spec is None:
            return origin

        dr, dc, max_distance = spec
        current = origin

        for _ in range(max_distance):
            candidate = (current[0] + dr, current[1] + dc)
            if not _inside(candidate, grid):
                break
            if grid[candidate] == 1:
                break
            current = candidate

        return current

    def _is_runtime_plant_position(
        self,
        char,
        game_state: dict,
        grid: np.ndarray,
    ) -> bool:
        current = tuple(char.pos)
        target = game_state.get("target_plant_pos")

        if isinstance(target, (list, tuple)) and len(target) == 2:
            return current == (int(target[0]), int(target[1]))

        return _inside(current, grid) and int(grid[current]) == 2

    def _update_previous_move_result(
        self,
        char,
        grid: np.ndarray,
        chars: Sequence,
        tick_id: Optional[int],
    ) -> None:
        name = str(char.name)
        request = self.requested_moves.pop(name, None)
        if request is None:
            return

        current = tuple(char.pos)
        if current == request.target:
            self.last_move_failed[name] = False
            self.last_move_fail_reason[name] = ""
            return

        self.last_move_failed[name] = True

        if not _inside(request.target, grid):
            reason = "OUT_OF_BOUNDS"
        elif grid[request.target] == 1:
            reason = "WALL"
        elif any(
            other is not char
            and _is_alive(other)
            and tuple(other.pos) == request.target
            for other in chars
        ):
            reason = "ALLY"
        else:
            reason = "CONFLICT"

        self.last_move_fail_reason[name] = reason

    def _make_observation(
        self,
        char,
        game_state: dict,
        grid: np.ndarray,
        chars: Sequence,
    ) -> np.ndarray:
        height, width = grid.shape
        r, c = int(char.pos[0]), int(char.pos[1])
        current = (r, c)

        teammates = [
            other
            for other in chars
            if other is not char and other.team == TEAM_ATTACKER
        ]
        enemies = [
            other
            for other in chars
            if other.team == TEAM_DEFENDER
        ]

        objective = self._objective_for(char, game_state, chars, grid)
        objective_distance = self._distance_to_objective(current, objective, grid)

        max_hp = max(int(getattr(char, "max_hp", 100)), 1)
        ability_name = str(
            getattr(char, "ability_name", ABILITY_NONE)
        ).upper()

        position_features = [
            r / max(height - 1, 1),
            c / max(width - 1, 1),
        ]

        self_features = [
            max(float(getattr(char, "hp", 0)), 0.0) / max_hp,
            1.0 if _is_alive(char) else 0.0,
            1.0 if bool(getattr(char, "moved_this_tick", False)) else 0.0,
            min(_blind_ticks(char) / max(BLIND_DURATION_TICKS, 1), 1.0),
            min(_reveal_ticks(char) / max(REVEAL_DURATION_TICKS, 1), 1.0),
            1.0 if bool(getattr(char, "has_spike", False)) else 0.0,
            1.0 if bool(getattr(char, "is_planting", False)) else 0.0,
            min(
                float(getattr(char, "plant_timer", 0))
                / max(PLANT_REQUIRED_TICKS, 1),
                1.0,
            ),
        ]

        ability_onehot = [0.0] * len(ABILITY_TO_INDEX)
        ability_onehot[ABILITY_TO_INDEX.get(ability_name, 0)] = 1.0
        charge_features = [float(min(max(_ability_charges(char), 0), 1))]

        name = str(char.name)
        fail_reason = self.last_move_fail_reason.get(name, "")
        failure_features = [
            1.0 if self.last_move_failed.get(name, False) else 0.0,
            1.0 if fail_reason == "OUT_OF_BOUNDS" else 0.0,
            1.0 if fail_reason == "WALL" else 0.0,
            1.0 if fail_reason in {"ALLY", "CONFLICT"} else 0.0,
        ]

        last_action_onehot = [0.0] * N_ACTIONS
        last_action = self.last_actions.get(name)
        if last_action is not None and 0 <= last_action < N_ACTIONS:
            last_action_onehot[last_action] = 1.0

        occupied = {
            tuple(other.pos)
            for other in chars
            if other is not char and _is_alive(other)
        }
        adjacent_blocked: List[float] = []
        for dr, dc in MOVE_DELTAS.values():
            candidate = (r + dr, c + dc)
            adjacent_blocked.append(
                1.0
                if not _is_walkable(candidate, grid) or candidate in occupied
                else 0.0
            )

        objective_features = [
            (objective[0] - r) / max(height - 1, 1),
            (objective[1] - c) / max(width - 1, 1),
            safe_normalized_distance(
                objective_distance,
                height + width,
            ),
        ]

        is_planted = bool(game_state.get("is_planted", False))
        spike_pos = game_state.get("spike_pos")
        phase_features = [
            0.0 if is_planted else 1.0,
            1.0 if is_planted else 0.0,
            1.0 if spike_pos is not None and not is_planted else 0.0,
        ]

        battle_tick = float(game_state.get("battle_tick", self._fallback_tick))
        spike_timer = float(game_state.get("spike_timer", 0.0))
        timer_features = [
            min(battle_tick / max(ROUND_DURATION_TICKS, 1), 1.0),
            (
                min(spike_timer / max(SPIKE_DETONATION_TICKS, 1), 1.0)
                if is_planted
                else 0.0
            ),
        ]

        teammate_features: List[float] = []
        teammates = sorted(
            teammates,
            key=lambda teammate: max(
                abs(int(teammate.pos[0]) - r),
                abs(int(teammate.pos[1]) - c),
            ),
        )

        for teammate in teammates[: N_ATTACKERS - 1]:
            teammate_max_hp = max(
                int(getattr(teammate, "max_hp", 100)),
                1,
            )
            teammate_features.extend(
                [
                    1.0 if _is_alive(teammate) else 0.0,
                    (int(teammate.pos[0]) - r) / max(height - 1, 1),
                    (int(teammate.pos[1]) - c) / max(width - 1, 1),
                    max(float(getattr(teammate, "hp", 0)), 0.0)
                    / teammate_max_hp,
                    1.0 if bool(getattr(teammate, "has_spike", False)) else 0.0,
                    1.0
                    if bool(getattr(teammate, "moved_this_tick", False))
                    else 0.0,
                ]
            )

        while len(teammate_features) < (N_ATTACKERS - 1) * TEAMMATE_SLOT_DIM:
            teammate_features.extend([0.0] * TEAMMATE_SLOT_DIM)

        smoke_cells = _smoke_cells_from_game_state(game_state)
        memory = self._memory_for(name)
        memory.update(char, enemies, grid, smoke_cells)
        enemy_features = memory.build_features(current, height, width)

        utility_features = self._smoke_features(
            current,
            smoke_cells,
            height,
            width,
        )
        defuse_features = self._defuse_features(
            char,
            enemies,
            game_state,
            height,
            width,
        )

        values: List[float] = []

        values.extend(position_features)
        values.extend(self_features)
        values.extend(ability_onehot)
        values.extend(charge_features)
        values.extend(failure_features)
        values.extend(last_action_onehot)
        values.extend(adjacent_blocked)
        values.extend(objective_features)
        values.extend(phase_features)
        values.extend(timer_features)
        values.extend(teammate_features)

        # 学習側の
        # while len(out) < 76:
        #     out += [0.0]
        # と完全に同じ位置へパディングを入れる。
        while len(values) < 76:
            values.append(0.0)

        if len(values) != 76:
            raise RuntimeError(
                "AI ver2.0の敵情報前の観測次元が一致しません。 "
                f"expected=76, actual={len(values)}"
            )

        # 学習側でも、76要素へ調整した後に敵情報を追加している
        values.extend(enemy_features)
        values.extend(utility_features)
        values.extend(defuse_features)

        observation = np.asarray(values, dtype=np.float32)

        if observation.shape != (OBS_DIM,):
            raise RuntimeError(
                "AI ver2.0の観測次元が学習側と一致しません。 "
                f"expected={(OBS_DIM,)}, actual={observation.shape}"
            )

        return observation

    def _objective_for(
        self,
        char,
        game_state: dict,
        chars: Sequence,
        grid: np.ndarray,
    ) -> Position:
        if bool(game_state.get("is_planted", False)):
            planted = game_state.get("planted_pos")
            if isinstance(planted, (list, tuple)) and len(planted) == 2:
                return int(planted[0]), int(planted[1])

        if bool(getattr(char, "has_spike", False)):
            target = game_state.get("target_plant_pos")
            if isinstance(target, (list, tuple)) and len(target) == 2:
                return int(target[0]), int(target[1])

        dropped = game_state.get("spike_pos")
        if isinstance(dropped, (list, tuple)) and len(dropped) == 2:
            return int(dropped[0]), int(dropped[1])

        holder = next(
            (
                other
                for other in chars
                if other.team == TEAM_ATTACKER
                and _is_alive(other)
                and bool(getattr(other, "has_spike", False))
            ),
            None,
        )
        if holder is not None:
            return int(holder.pos[0]), int(holder.pos[1])

        target = game_state.get("target_plant_pos")
        if isinstance(target, (list, tuple)) and len(target) == 2:
            return int(target[0]), int(target[1])

        plant_cells = np.argwhere(grid == 2)
        if len(plant_cells) > 0:
            return int(plant_cells[0][0]), int(plant_cells[0][1])

        return int(char.pos[0]), int(char.pos[1])

    def _distance_to_objective(
        self,
        start: Position,
        objective: Position,
        grid: np.ndarray,
    ) -> float:
        key = (
            int(objective[0]),
            int(objective[1]),
            hash(grid.tobytes()),
        )
        if self._distance_cache_key != key:
            self._distance_cache_key = key
            self._distance_cache = _distance_map(objective, grid)

        if self._distance_cache is None:
            return float("inf")
        return float(self._distance_cache[start])

    def _smoke_features(
        self,
        pos: Position,
        smoke_cells: Set[Position],
        height: int,
        width: int,
    ) -> List[float]:
        in_smoke = 1.0 if pos in smoke_cells else 0.0
        direction_features: List[float] = []
        max_range = max(height, width)

        for dr, dc in MOVE_DELTAS.values():
            nearest: Optional[int] = None
            for distance in range(1, max_range + 1):
                candidate = (
                    pos[0] + dr * distance,
                    pos[1] + dc * distance,
                )
                if not (0 <= candidate[0] < height and 0 <= candidate[1] < width):
                    break
                if candidate in smoke_cells:
                    nearest = distance
                    break

            direction_features.append(
                0.0
                if nearest is None
                else 1.0 - min(nearest / max(max_range, 1), 1.0)
            )

        return [in_smoke] + direction_features

    def _defuse_features(
        self,
        observer,
        enemies: Sequence,
        game_state: dict,
        height: int,
        width: int,
    ) -> List[float]:
        active_name = game_state.get("active_defuser_name")

        active_defusers = [
            enemy
            for enemy in enemies
            if _is_alive(enemy)
            and (
                float(getattr(enemy, "defuse_timer", 0)) > 0
                or (
                    active_name is not None
                    and str(enemy.name) == str(active_name)
                )
            )
        ]

        if not active_defusers:
            return [0.0, 0.0, 0.0, 0.0]

        defuser = max(
            active_defusers,
            key=lambda enemy: float(getattr(enemy, "defuse_timer", 0)),
        )
        observer_r, observer_c = int(observer.pos[0]), int(observer.pos[1])

        return [
            1.0,
            min(
                float(getattr(defuser, "defuse_timer", 0))
                / max(DEFUSE_REQUIRED_TICKS, 1),
                1.0,
            ),
            (int(defuser.pos[0]) - observer_r) / max(height - 1, 1),
            (int(defuser.pos[1]) - observer_c) / max(width - 1, 1),
        ]

    def _get_tick_id(self, game_state: dict) -> Optional[int]:
        raw = game_state.get("battle_tick")
        if raw is not None:
            try:
                tick = int(raw)
            except (TypeError, ValueError):
                tick = self._fallback_tick

            if self._last_seen_game_tick != tick:
                self._last_seen_game_tick = tick
                self._fallback_tick = tick
            return tick

        # battle_tickが未提供でも観測構築自体は可能。
        self._fallback_tick += 1
        return None


# 短い別名。run_game.py側で好みの名前を使える。
LearningAttackerV2Controller = LearningAttackerAIv2Controller


__all__ = [
    "LearningAttackerAIv2Controller",
    "LearningAttackerV2Controller",
]