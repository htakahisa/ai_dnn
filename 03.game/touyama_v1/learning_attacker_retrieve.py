"""learning_attacker_retrieve.py

Retrieve フェーズ（落下スパイク回収）用の推論コントローラー。
train_attacker_retrieve.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。controllers.py / battle_logic.py / abilities_los.py への
依存はなく、必要なLOS・BFS計算はこのファイル内に複製する。
game_core からは定数のみ参照する(ロジックは参照しない)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "ABILITY"]
N_ACTIONS = len(ACTIONS)
ROLES = ["FLASH", "SMOKE", "RECON"]

OBS_DIM = 20

DEFAULT_MODEL_PATH = "attacker_retrieve_best.pt"


# ---------------------------------------------------------------------------
# Dueling DQN (train_attacker_retrieve.py と同一構造)
# ---------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# 補助関数(LOS / BFS)。abilities_los.py 等とは独立した複製実装。
# ---------------------------------------------------------------------------
def _line_cells(p1, p2):
    y0, x0 = int(p1[0]), int(p1[1])
    y1, x1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _has_los(grid, p1, p2):
    for r, c in _line_cells(p1, p2):
        if grid[r, c] == 1:
            return False
    return True


def _bfs_distance_map(grid, goal):
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _ability_charge(char):
    """ロールに対応する残チャージ数を取得する。HUNT(アビリティ無し)は常に0。"""
    return {
        "FLASH": getattr(char, "flash_charges", 0),
        "SMOKE": getattr(char, "smoke_charges", 0),
        "RECON": getattr(char, "recon_charges", 0),
    }.get(char.ability_name, 0)


def _role_onehot(char):
    return [1.0 if char.ability_name == role else 0.0 for role in ROLES]


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningAttackerRetrieveController:
    """落下スパイク回収フェーズ専用のAI。

    decide_move(char, game_state) は run_game.py / battle_logic.py の
    既存インタフェースに合わせて以下のいずれかを返す:
        - next_pos (list[int,int])                       : 通常移動
        - (next_pos, {"ability": name, "target": (r,c)})  : アビリティ使用
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self.model = DuelingQNet(OBS_DIM, N_ACTIONS).to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            if verbose:
                print(f"[LearningAttackerRetrieveController] loaded: {model_path}")
        except Exception as exc:
            print(f"[LOAD ERROR] retrieve model '{model_path}' の読込に失敗: {exc}")
        self.model.eval()

    # -- 観測構築 -------------------------------------------------------
    def _build_observation(self, char, grid, spike_pos, chars):
        height, width = grid.shape
        r, c = int(char.pos[0]), int(char.pos[1])

        def is_wall(rr, cc):
            if not (0 <= rr < height and 0 <= cc < width):
                return 1.0
            return 1.0 if grid[rr, cc] == 1 else 0.0

        wall_up = is_wall(r - 1, c)
        wall_down = is_wall(r + 1, c)
        wall_left = is_wall(r, c - 1)
        wall_right = is_wall(r, c + 1)

        dist_map = _bfs_distance_map(grid, spike_pos)
        max_dist_scale = float(height + width)

        # 生存中の味方が占有しているセルを取得し、壁と同様に「進めない」扱いにする。
        # train_attacker_retrieve.py の ally_blocked_cell と対応させるための変更。
        ally_occupied = {
            tuple(map(int, other.pos))
            for other in chars
            if other is not char
            and getattr(other, "is_alive", True)
            and getattr(other, "team", None) == char.team
        }

        neighbor_dists = []
        for (dr_, dc_), is_wall_flag in zip(
            CARDINAL, [wall_up, wall_down, wall_left, wall_right]
        ):
            nr, nc = r + dr_, c + dc_
            in_bounds = 0 <= nr < height and 0 <= nc < width
            blocked = (
                is_wall_flag
                or not in_bounds
                or (in_bounds and (nr, nc) in ally_occupied)
            )
            if blocked:
                neighbor_dists.append(1.0)
            else:
                nd = dist_map[nr, nc]
                neighbor_dists.append(1.0 if nd < 0 else min(1.0, nd / max_dist_scale))

        raw_dist = dist_map[r, c]
        dist_norm = min(1.0, raw_dist / max_dist_scale) if raw_dist >= 0 else 1.0

        role_onehot = _role_onehot(char)
        charge = float(_ability_charge(char))

        visible_enemy = self._nearest_visible_enemy(char, grid, chars)
        if visible_enemy is not None:
            er, ec = visible_enemy.pos
            edr = float(np.clip((er - r) / height, -1, 1))
            edc = float(np.clip((ec - c) / width, -1, 1))
            e_present = 1.0
            e_blind = 1.0 if getattr(visible_enemy, "blind_remaining", 0) > 0 else 0.0
            e_reveal = 1.0 if getattr(visible_enemy, "reveal_remaining", 0) > 0 else 0.0
        else:
            edr = edc = 0.0
            e_present = e_blind = e_reveal = 0.0

        obs = [
            r / height, c / width,
            wall_up, wall_down, wall_left, wall_right,
            *neighbor_dists,
            dist_norm,
            *role_onehot,
            charge,
            e_present, edr, edc, e_blind, e_reveal,
        ]
        return np.array(obs, dtype=np.float32), dist_map

    def _nearest_visible_enemy(self, char, grid, chars):
        best = None
        best_dist = None
        for other in chars:
            if not getattr(other, "is_alive", True):
                continue
            if getattr(other, "team", None) == char.team:
                continue
            if not _has_los(grid, tuple(char.pos), tuple(other.pos)):
                continue
            dist = max(abs(other.pos[0] - char.pos[0]), abs(other.pos[1] - char.pos[1]))
            if best is None or dist < best_dist:
                best = other
                best_dist = dist
        return best

    def _action_mask(self, char, grid):
        height, width = grid.shape
        r, c = int(char.pos[0]), int(char.pos[1])
        mask = [True] * N_ACTIONS
        for i, (dr, dc) in enumerate(CARDINAL):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width) or grid[nr, nc] == 1:
                mask[i] = False
        if _ability_charge(char) <= 0:
            mask[5] = False
        return np.array(mask, dtype=bool)

    # -- メイン ----------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])

        # スパイクの目標地点を決める。落下中(spike_pos)を最優先、
        # 念のため無い場合は現在地に留まる。
        spike_pos = game_state.get("spike_pos")
        if spike_pos is None:
            return list(char.pos)

        obs, dist_map = self._build_observation(char, grid, spike_pos, chars)
        mask = self._action_mask(char, grid)

        state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action = int(torch.argmax(q_values).item())

        if self.verbose:
            print(f"[RETRIEVE] {char.name} pos={tuple(char.pos)} action={ACTIONS[action]}")

        if action < 4:
            dr, dc = CARDINAL[action]
            next_pos = [char.pos[0] + dr, char.pos[1] + dc]
            return next_pos

        if action == 4:
            return list(char.pos)

        # action == 5: ABILITY
        target_char = self._nearest_visible_enemy(char, grid, chars)
        if target_char is None:
            # マスク上は許容されていても学習が不十分で見えない敵に撃とうとした場合の
            # 安全策。チャージを無駄にしないよう移動に切り替える。
            return list(char.pos)

        return list(char.pos), {
            "ability": char.ability_name,
            "target": (int(target_char.pos[0]), int(target_char.pos[1])),
        }

    def reset_round(self):
        """ラウンド開始時に呼ばれる。内部状態を持たないため何もしない。"""
        pass