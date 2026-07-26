# learning_attacker_carry.py
"""
train_attacker_carry.py で学習したcarry専用モデルを、実際のゲーム(run_game.py)で
動かすためのコントローラー。CarryOnlyEnv の観測構築ロジック・行動マスクと
完全に一致させる必要がある(ズレるとモデルの性能が出ない)。
"""

from pathlib import Path
from collections import deque
import numpy as np
import torch
import sys

# 💡追加: 自分のディレクトリ(attacker_v2)と、1階層上(project root)の両方をsys.pathに追加。
# root側の追加は controllers.py を見つけるため、
# 自分のディレクトリ側の追加は train_attacker_carry.py を見つけるため。
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controllers import BaseController
from train_attacker_carry import (
    DuelingQNetwork, OBS_DIM, N_ACTIONS, ABILITY_TYPES,
    split_site_components, multi_source_bfs,
)

class LearningAttackerCarryController(BaseController):
    """carryフェーズ専用のAIコントローラー。retrieve/guardのロジックは持たない
    (has_spikeでないキャラが来た場合はその場に留まるだけのフォールバックとする)。"""

    def __init__(self, model_path="data_temp/attacker_carry_data/dqn_attacker_carry_best_by_eval.pt", greedy=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path_obj = Path(model_path)
        # 💡変更: 基準をこのファイルのディレクトリ(attacker_v2)ではなく _ROOT_DIR に変更
        full_path = model_path_obj if model_path_obj.is_absolute() else _ROOT_DIR / model_path_obj

        self.model = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(self.device)
        self.model.load_state_dict(torch.load(str(full_path), map_location=self.device))
        self.model.eval()

        self.greedy = greedy
        self.last_actions = {}

        # 💡 マップ依存のサイト別BFS(site_maps)はgrid確定後に一度だけ計算する
        self._cached_grid_id = None
        self.site_components = None
        self.site_maps = None
        self.site_cell_sets = None

    def reset_round(self):
        self.last_actions.clear()

    # -----------------------------------------------------------------
    def _ensure_site_maps(self, grid):
        # 💡 gridは固定マップなので、id()が変わらない限り再計算しない
        if self._cached_grid_id == id(grid):
            return
        self._cached_grid_id = id(grid)
        site_cells = list(zip(*np.where(grid == 2)))
        self.site_components = split_site_components(site_cells)
        self.site_maps = [multi_source_bfs(comp, grid) for comp in self.site_components]
        self.site_cell_sets = [set(comp) for comp in self.site_components]

    def _select_site_for_target(self, target_plant_pos):
        """target_plant_pos(このラウンドで狙うサイト)がどの成分に属するかを特定し、
        そのサイト用のdist_map/label_map/site_cellsを選択する。"""
        target_plant_pos = tuple(target_plant_pos)
        for cells, (dist_map, label_map) in zip(self.site_cell_sets, self.site_maps):
            if target_plant_pos in cells:
                return dist_map, label_map, cells
        # 異常系フォールバック: 見つからなければ最初のサイトを使う
        dist_map, label_map = self.site_maps[0]
        return dist_map, label_map, self.site_cell_sets[0]

    def _is_walkable(self, r, c, grid):
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w and grid[r, c] != 1

    # -----------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        self._ensure_site_maps(grid)

        r, c = char.pos

        if not char.has_spike:
            return char.pos, "MOVE"

        # 💡追加: このラウンドで狙うべきサイト(target_plant_pos)に対応するBFSマップを選択する。
        # これをしないと「一番近いサイト」に誘導されてしまい、実際に指定されたサイト
        # (オレンジ表示)を無視してしまう。
        target_plant_pos = game_state.get("target_plant_pos")
        if target_plant_pos is not None:
            self.dist_map, self.label_map, self.site_cells = self._select_site_for_target(target_plant_pos)
        elif not hasattr(self, "site_cells"):
            self.dist_map, self.label_map = self.site_maps[0]
            self.site_cells = self.site_cell_sets[0]

        obs = self._make_observation(char, grid)
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
            return char.pos, "PLANT"

        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = [r + moves[action][0], c + moves[action][1]]
        height, width = grid.shape
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width and grid[next_pos[0], next_pos[1]] != 1:
            return next_pos, "MOVE"
        return char.pos, "MOVE"

    # -----------------------------------------------------------------
    def _make_observation(self, char, grid):
        pr, pc = char.pos
        gr, gc = self.label_map[pr][pc]
        height, width = grid.shape

        base = [pr / (height - 1), pc / (width - 1), gr / (height - 1), gc / (width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_act = self.last_actions.get(char.name, None)
        last_onehot = [0.0] * 5
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

        # 💡注意: 学習時(CarryOnlyEnv)はability_onehotのみ毎エピソードランダムで、
        # ability_charge=1.0固定, blind_flag=0.0固定だった。分布を合わせるため、
        # ここでも charge/blind は学習時と同じ固定値にしている(実キャラの状態は反映しない)。
        ability_onehot = [1.0 if char.ability_type == t else 0.0 for t in ABILITY_TYPES]
        ability_charge = [1.0]
        blind_flag = [0.0]
        enemy = [0.0, 0.0, 0.0]  # 学習時と同じダミー

        return np.array(
            base + walls + last_onehot + dists + ability_onehot + ability_charge + blind_flag + enemy,
            dtype=np.float32
        )

    def _get_action_mask(self, char, grid):
        r, c = char.pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            mask[a] = 1.0 if self._is_walkable(r + dr, c + dc, grid) else 0.0
        mask[4] = 1.0 if (r, c) in self.site_cells else 0.0
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