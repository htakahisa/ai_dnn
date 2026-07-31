# attacker_v2/learning_attacker_retrieve.py
"""
train_attacker_retrieve.py で学習したスパイク回収専用モデルを実ゲームで動かすコントローラー。
"""

import sys
from pathlib import Path
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controllers import BaseController
# 💡train_attacker_retrieve.py から必要な定数と関数をインポート
from train_attacker_retrieve import (
    DuelingQNetwork, OBS_DIM, N_ACTIONS, ABILITY_TYPES,
    bfs_distances
)


class LearningAttackerRetrieveController(BaseController):
    def __init__(self, model_path="data_temp/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt", greedy=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path_obj = Path(model_path)
        full_path = model_path_obj if model_path_obj.is_absolute() else _ROOT_DIR / model_path_obj

        self.model = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(self.device)
        self.model.load_state_dict(torch.load(str(full_path), map_location=self.device))
        self.model.eval()

        self.greedy = greedy
        self.last_actions = {}

        self._cached_grid_id = None
        self._cached_spike_pos = None
        self.dist_map = None

    def reset_round(self):
        self.last_actions.clear()

    def _ensure_spike_map(self, grid, spike_pos):
        spike_pos = tuple(spike_pos)
        if self._cached_spike_pos == spike_pos and self._cached_grid_id == id(grid):
            return
        self._cached_spike_pos = spike_pos
        self._cached_grid_id = id(grid)
        self.dist_map = bfs_distances(spike_pos, grid)

    def _is_walkable(self, r, c, grid):
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w and grid[r, c] != 1

    def _find_tracked_enemy(self, char, game_state):
        chars = game_state.get("chars", [])
        defenders = [d for d in chars if d.is_alive and d.team == "D"]
        if not defenders:
            return None
        pr, pc = char.pos
        return min(defenders, key=lambda d: max(abs(d.pos[0] - pr), abs(d.pos[1] - pc)))

    def _occupied_cells(self, char, game_state):
        return {
            tuple(o.pos) for o in game_state["chars"]
            if o is not char and o.is_alive
        }

    def _get_spike_position(self, game_state):
        """
        💡実ゲームの仕様に合わせて、落ちているスパイクの位置を取得するロジックを実装してください。
        ここでは仮に game_state["spike_pos"] に座標が入っているものとします。
        """
        return game_state.get("spike_pos")

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        r, c = char.pos

        spike_pos = self._get_spike_position(game_state)
        if spike_pos is None:
            # 💡スパイクが見つからない（すでに誰かが拾った等）場合はその場に留まる
            return char.pos, "MOVE"

        self._ensure_spike_map(grid, spike_pos)

        tracked_enemy = self._find_tracked_enemy(char, game_state)
        occupied_cells = self._occupied_cells(char, game_state)   # 💡追加

        obs = self._make_observation(char, grid, tracked_enemy, spike_pos, occupied_cells)   # 💡修正
        mask = self._get_action_mask(char, grid, occupied_cells)                              # 💡修正

        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0).cpu().numpy()

        if self.greedy:
            action = self._masked_argmax(q_values, mask)
        else:
            action = self._masked_softmax(q_values, mask, temperature=0.5)

        self.last_actions[char.name] = action

        if action == 4:
            aim_cell = self._get_aim_cell(char, grid, tracked_enemy)
            return aim_cell, "ABILITY"

        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = [r + moves[action][0], c + moves[action][1]]
        height, width = grid.shape
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width and grid[next_pos[0], next_pos[1]] != 1:
            return next_pos, "MOVE"
        return char.pos, "MOVE"

    def _get_aim_cell(self, char, grid, tracked_enemy):
        pr, pc = char.pos
        if tracked_enemy is not None:
            visible = self.has_line_of_sight(char.pos, tracked_enemy.pos, grid)
            revealed = tracked_enemy.reveal_remaining > 0
            if visible or revealed:
                return tuple(tracked_enemy.pos)

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        last_act = self.last_actions.get(char.name)
        if last_act in moves:
            dr, dc = moves[last_act]
            return (pr + dr, pc + dc)

        best_dir, best_dist = None, self.dist_map[pr, pc]
        for dr, dc in moves.values():
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc, grid):
                d = self.dist_map[nr, nc]
                if np.isfinite(d) and d < best_dist:
                    best_dist, best_dir = d, (dr, dc)
        if best_dir is None:
            best_dir = (0, 1)
        return (pr + best_dir[0], pc + best_dir[1])

    def _make_observation(self, char, grid, tracked_enemy, spike_pos, occupied_cells):   # 💡修正: 引数追加
        pr, pc = char.pos
        sr, sc = spike_pos
        height, width = grid.shape

        base = [pr / (height - 1), pc / (width - 1), sr / (height - 1), sc / (width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_act = self.last_actions.get(char.name, None)
        last_onehot = [0.0] * N_ACTIONS
        if last_act is not None:
            last_onehot[last_act] = 1.0

        max_dist = max(height, width) * 2
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc, grid):
                d = self.dist_map[nr, nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)

        ability_onehot = [1.0 if char.ability_type == t else 0.0 for t in ABILITY_TYPES]
        has_charge = (char.flash_charges > 0) or (char.smoke_charges > 0) or (char.recon_charges > 0)
        ability_charge = [1.0 if has_charge else 0.0]

        enemy_blinded_flag = [1.0 if (tracked_enemy is not None and tracked_enemy.blind_remaining > 0) else 0.0]
        enemy = [0.0, 0.0, 0.0]
        if tracked_enemy is not None:
            visible = self.has_line_of_sight(char.pos, tracked_enemy.pos, grid)
            revealed = tracked_enemy.reveal_remaining > 0
            if visible or revealed:
                er, ec = tracked_enemy.pos
                enemy = [1.0, (er - pr) / height, (ec - pc) / width]

        # 💡追加: 学習時のteammate_infoと同じ形式
        others = [p for p in occupied_cells if p != (pr, pc)]
        teammate_info = [0.0, 0.0, 0.0]
        if others:
            nearest = min(others, key=lambda p: max(abs(p[0]-pr), abs(p[1]-pc)))
            dist = max(abs(nearest[0]-pr), abs(nearest[1]-pc))
            max_d = max(height, width)
            teammate_info = [
                1.0 - min(dist / max_d, 1.0),
                (nearest[0]-pr) / height,
                (nearest[1]-pc) / width,
            ]

        return np.array(
            base + walls + last_onehot + dists + ability_onehot + ability_charge + enemy_blinded_flag + enemy + teammate_info,   # 💡追加
            dtype=np.float32
        )

    def _get_action_mask(self, char, grid, occupied_cells):   # 💡修正: 引数追加
        r, c = char.pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            free = self._is_walkable(nr, nc, grid) and (nr, nc) not in occupied_cells   # 💡変更
            mask[a] = 1.0 if free else 0.0
        has_charge = (char.flash_charges > 0) or (char.smoke_charges > 0) or (char.recon_charges > 0)
        mask[4] = 1.0 if has_charge else 0.0
        return mask

    @staticmethod
    def _masked_argmax(q_values, mask):
        if mask.sum() == 0:
            return int(np.argmax(q_values))
        masked = np.where(mask > 0, q_values, -np.inf)
        return int(np.argmax(masked))

    @staticmethod
    def _masked_softmax(q_values, mask, temperature=0.5):
        if mask.sum() == 0:
            masked = q_values
        else:
            masked = np.where(mask > 0, q_values, -np.inf)
        probs = np.exp((masked - np.max(masked)) / temperature)
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))