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

    def _occupied_cells(self, char, game_state):
        return {
            tuple(o.pos) for o in game_state["chars"]
            if o is not char and o.is_alive
        }

    # -----------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        self._ensure_site_maps(grid)
        occupied_cells = self._occupied_cells(char, game_state)   # 💡追加

        r, c = char.pos

        if not char.has_spike:
            return char.pos, "MOVE"

        target_plant_pos = game_state.get("target_plant_pos")
        if target_plant_pos is not None:
            self.dist_map, self.label_map, self.site_cells = self._select_site_for_target(target_plant_pos)
        elif not hasattr(self, "site_cells"):
            self.dist_map, self.label_map = self.site_maps[0]
            self.site_cells = self.site_cell_sets[0]

        # 💡追加: 学習環境の「単一の敵bot」に相当する、最も近い生存defenderを追跡対象にする
        tracked_enemy = self._find_tracked_enemy(char, game_state)

        obs = self._make_observation(char, grid, tracked_enemy, occupied_cells)
        mask = self._get_action_mask(char, grid, occupied_cells)

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

        if action == 5:
            # 💡追加: アビリティ発動。狙うマスは学習時と同じ優先順位で決める。
            # 実際の着弾判定・効果適用(blind_remaining/reveal_remaining更新)はabilities.py側が行う。
            aim_cell = self._get_aim_cell(char, grid, tracked_enemy)
            ability_name = char.ability_type.upper()
            return char.pos, {"ability": ability_name, "target": tuple(aim_cell)}

        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = [r + moves[action][0], c + moves[action][1]]
        height, width = grid.shape
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width and grid[next_pos[0], next_pos[1]] != 1:
            return next_pos, "MOVE"
        return char.pos, "MOVE"
    
    def _find_tracked_enemy(self, char, game_state):
        """学習環境の単一bot相当として、最も近い生存defenderを1体選ぶ。
        いなければNone。"""
        chars = game_state.get("chars", [])
        defenders = [d for d in chars if d.is_alive and d.team == "D"]
        if not defenders:
            return None
        pr, pc = char.pos
        return min(
            defenders,
            key=lambda d: max(abs(d.pos[0] - pr), abs(d.pos[1] - pc))
        )

    def _get_aim_cell(self, char, grid, tracked_enemy):
        """狙うマスを決める。優先順位は学習時のCarryOnlyEnv._get_aim_directionと同じ:
        1) 敵が見えている/リコンで察知中ならその方向 2) 直前の移動方向
        3) どちらもなければBFS勾配上の最善方向。"""
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

    # -----------------------------------------------------------------
    def _make_observation(self, char, grid, tracked_enemy, occupied_cells): 
        pr, pc = char.pos
        gr, gc = self.label_map[pr][pc]
        height, width = grid.shape

        base = [pr / (height - 1), pc / (width - 1), gr / (height - 1), gc / (width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_act = self.last_actions.get(char.name, None)
        # 💡変更: action=5(アビリティ使用)を追加したため6次元に拡張
        last_onehot = [0.0] * 6
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
        # 💡変更: 実際のチャージ状態を反映(flash/smoke/reconのいずれかが1つ立っている)
        has_charge = (char.flash_charges > 0) or (char.smoke_charges > 0) or (char.recon_charges > 0)
        ability_charge = [1.0 if has_charge else 0.0]

        # 💡変更: 「敵が怯んでいるか」は追跡中のdefenderのblind_remainingをそのまま見る
        enemy_blinded_flag = [1.0 if (tracked_enemy is not None and tracked_enemy.blind_remaining > 0) else 0.0]

        # 💡変更: LOSが通っているか、リコンで察知中(reveal_remaining>0)なら見える
        enemy = [0.0, 0.0, 0.0]
        if tracked_enemy is not None:
            visible = self.has_line_of_sight(char.pos, tracked_enemy.pos, grid)
            revealed = tracked_enemy.reveal_remaining > 0
            if visible or revealed:
                er, ec = tracked_enemy.pos
                enemy = [1.0, (er - pr) / height, (ec - pc) / width]

        # 💡追加: 学習時のteammate_infoと同じ形式
        pr, pc = char.pos
        others = [p for p in occupied_cells if p != (pr, pc)]
        teammate_info = [0.0, 0.0, 0.0]
        if others:
            nearest = min(others, key=lambda p: max(abs(p[0]-pr), abs(p[1]-pc)))
            dist = max(abs(nearest[0]-pr), abs(nearest[1]-pc))
            max_dist = max(grid.shape)
            teammate_info = [
                1.0 - min(dist / max_dist, 1.0),
                (nearest[0]-pr) / grid.shape[0],
                (nearest[1]-pc) / grid.shape[1],
            ]

        return np.array(
            base + walls + last_onehot + dists + ability_onehot +
            ability_charge + enemy_blinded_flag + enemy + teammate_info,
            dtype=np.float32
        )

    def _get_action_mask(self, char, grid, occupied_cells):
        r, c = char.pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            free = self._is_walkable(nr, nc, grid) and (nr, nc) not in occupied_cells
            mask[a] = 1.0 if free else 0.0
        mask[4] = 1.0 if (r, c) in self.site_cells else 0.0
        mask[5] = 1.0 if (char.flash_charges > 0 or char.smoke_charges > 0 or char.recon_charges > 0) else 0.0
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