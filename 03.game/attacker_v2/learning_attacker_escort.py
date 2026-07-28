# attacker_v2/learning_attacker_escort.py
"""
train_attacker_escort.py で学習した護衛専用モデルを実ゲームで動かすコントローラー。
「味方が既にサイト内でアビリティ使用済みか」は、外部(MultiRoleAttackerController)から
site_ability_used_by_teammate 属性に都度セットしてもらう想定(このクラス単体では判定できない)。
"""

import sys
from pathlib import Path
from collections import deque
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controllers import BaseController
from train_attacker_escort import (
    DuelingQNetwork, OBS_DIM, N_ACTIONS, ABILITY_TYPES,
    split_site_components, multi_source_bfs, SITE_APPROACH_DIST_THRESHOLD,
)


class LearningAttackerEscortController(BaseController):
    def __init__(self, model_path="data_temp/attacker_escort_data/dqn_attacker_escort_best_by_eval.pt", greedy=False):
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
        self.site_components = None
        self.site_maps = None
        self.site_cell_sets = None

        # 💡外部(MultiRoleAttackerController)からラウンド中随時更新される想定の共有フラグ
        self.site_ability_used_by_teammate = False

    def reset_round(self):
        self.last_actions.clear()
        self.site_ability_used_by_teammate = False

    def _ensure_site_maps(self, grid):
        if self._cached_grid_id == id(grid):
            return
        self._cached_grid_id = id(grid)
        site_cells = list(zip(*np.where(grid == 2)))
        self.site_components = split_site_components(site_cells)
        self.site_maps = [multi_source_bfs(comp, grid) for comp in self.site_components]
        self.site_cell_sets = [set(comp) for comp in self.site_components]

    def _select_site_for_target(self, target_plant_pos):
        target_plant_pos = tuple(target_plant_pos)
        for cells, (dist_map, label_map) in zip(self.site_cell_sets, self.site_maps):
            if target_plant_pos in cells:
                return dist_map, label_map, cells
        dist_map, label_map = self.site_maps[0]
        return dist_map, label_map, self.site_cell_sets[0]

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

    def _find_carrier(self, char, game_state):
        """キャリアー(スパイク保持中の味方)を探す。自分自身は対象外。"""
        chars = game_state.get("chars", [])
        return next(
            (c for c in chars if c.is_alive and c.team == "A" and c.has_spike and c.name != char.name),
            None
        )

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        self._ensure_site_maps(grid)

        r, c = char.pos

        carrier = self._find_carrier(char, game_state)
        if carrier is None:
            # 💡キャリアーが見つからない(誰もスパイクを持っていない等)場合はその場に留まる。
            # retrieve/guardフェーズへの遷移はMultiRoleAttackerController側で振り分けられる想定。
            return char.pos, "MOVE"

        target_plant_pos = game_state.get("target_plant_pos")
        if target_plant_pos is not None:
            self.dist_map, self.label_map, self.site_cells = self._select_site_for_target(target_plant_pos)
        elif not hasattr(self, "site_cells"):
            self.dist_map, self.label_map = self.site_maps[0]
            self.site_cells = self.site_cell_sets[0]

        tracked_enemy = self._find_tracked_enemy(char, game_state)

        obs = self._make_observation(char, grid, tracked_enemy, carrier)
        mask = self._get_action_mask(char, grid)

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

    def _make_observation(self, char, grid, tracked_enemy, carrier):
        pr, pc = char.pos
        height, width = grid.shape

        base = [pr / (height - 1), pc / (width - 1)]

        cr, cc = carrier.pos
        max_dist = max(height, width)
        carrier_dist = max(abs(cr - pr), abs(cc - pc))
        carrier_rel = [carrier_dist / max_dist, (cr - pr) / height, (cc - pc) / width]

        walls = [0.0 if self._is_walkable(pr + dr, pc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_act = self.last_actions.get(char.name, None)
        last_onehot = [0.0] * N_ACTIONS
        if last_act is not None:
            last_onehot[last_act] = 1.0

        site_max_dist = max(height, width) * 2
        site_dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc, grid):
                d = self.dist_map[nr, nc]
                site_dists.append(d / site_max_dist if np.isfinite(d) else 1.0)
            else:
                site_dists.append(1.0)

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

        own_site_dist = self.dist_map[pr, pc]
        own_site_dist_norm = [own_site_dist / site_max_dist if np.isfinite(own_site_dist) else 1.0]

        teammate_used_flag = [1.0 if self.site_ability_used_by_teammate else 0.0]

        return np.array(
            base + carrier_rel + walls + last_onehot + site_dists +
            ability_onehot + ability_charge + enemy_blinded_flag + enemy +
            own_site_dist_norm + teammate_used_flag,
            dtype=np.float32
        )

    def _get_action_mask(self, char, grid):
        r, c = char.pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            mask[a] = 1.0 if self._is_walkable(r + dr, c + dc, grid) else 0.0
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