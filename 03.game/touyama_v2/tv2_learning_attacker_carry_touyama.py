"""touyama_v2/learning_attacker_carry_touyama.py

固定チーム(Tortlilyan/いぐるん/ろびぃな/夢の街/えんぺん)専用の
Attacker「carry phase」推論コントローラー(簡略版)。

train_attacker_carry.py で学習したDQNモデルをロードし、スパイク保持者
(has_spike==Trueのキャラ)のアビリティ使用判断・明示PLANT判断のみを担当する。
移動は決定的BFS経路探索(中継地点→優先設置場所)で処理し、DQNの行動空間には
含まれない。エスコート4人はヒューリスティックで動かす。

completely self-contained: run_game.py / controllers.py / battle_logic.py /
abilities_los.py は一切importしない。run_game.py / controllers.py は変更しない。

【重要: サイト選択・target】
AI_CONTROLLED_SITE_SELECTION=True の場合、run_game.py init_round()が決めた
target_plant_pos(オレンジマス)は無視し、このコントローラー自身が
SITE_SELECTION_WEIGHTSの比率で左右サイトを選び、そのサイトの優先設置場所
(priority_cells)からtargetを選ぶ。選んだtargetはself._active_target として
保持し(game_state["target_plant_pos"]は参照しない)、表示整合のため
game.target_plant_posへも書き込む。

【重要: IQAwareControllerとの相性】
team_ai.py の IQAwareController.decide_move() は毎tick set_game(view) を
呼び出すため、self.gameは使い捨てのview用オブジェクトに上書きされ続ける。
target_plant_posの上書き(reset_round())は本物のgameに対して行う必要が
あるため、最初に渡された(=起動時の本物の)gameだけを self._real_game に
別途保持する。

観測ベクトル・行動空間はtrain_attacker_carry.pyのbuild_observation() /
decode_action() / build_action_mask()と完全に一致させる必要がある。
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 設定(train_attacker_carry.pyと一致させる)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "touyama_v2/data/attacker_carry_touyama_data/dqn_attacker_carry_touyama_best_by_eval.pt"
DEBUG_LOG_PATH = "attacker_carry_touyama_debug.log"

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OBS_DIM = 34
# 移動はBFS決定的なため行動空間に含めない。ここに含めるのはアビリティ使用判断
# (NONE/ABILITY)・明示PLANTの3値と、向き(facing)選択の直積のみ。
# tv2_train_attacker_carry.py / tv2_train_defender_search.pyと同一規約:
# action_idx = base_idx(0-2)*8 + facing_idx(0-7)。
BASE_ACTION_DIM = 3
ACTION_NONE = 0
ACTION_ABILITY = 1
ACTION_PLANT = 2
FACING_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_DIM = BASE_ACTION_DIM * len(FACING_DIRS)  # 24

SITE_VALUES = frozenset({2, 5})
# スモーク定点(lineup)はチェックポイントから読み込む(SMOKE_LINEUP_CELLS_BY_SITE)。

ABILITY_RANGE = 8
SIGHTING_STALENESS_CAP = 20

# train_attacker_carry.py の game_core定数と一致させる(自己完結ルールのため複製)
ROUND_DURATION_TICKS = 100
PLANT_REQUIRED_TICKS = 4

# --- 本番プレイ時のサイト選択・target選定をAI側に委ねる ---------------------
AI_CONTROLLED_SITE_SELECTION = True    # Trueのとき、下の確率で選択する
SITE_SELECTION_WEIGHTS = {"left": 0.4, "right": 0.6}


# ============================================================================
# ネットワーク(train_attacker_carry.py と同一構造)
# ============================================================================

class AttackerCarryDuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, action_dim))

    def forward(self, x):
        f = self.feature(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + (a - a.mean(dim=1, keepdim=True))


# ============================================================================
# LOS・BFS
# ============================================================================

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


def _has_los(grid, p1, p2, smoke_cells=None):
    cells = _line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells and len(cells) > 2:
        if any(cell in smoke_cells for cell in cells):
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


def _bfs_next_step(grid, start, goal, occupied, allow_adjacent_goal=True):
    height, width = grid.shape
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return start

    candidate_goals = []
    if grid[goal[0], goal[1]] != 1 and goal not in occupied:
        candidate_goals.append(goal)
    if allow_adjacent_goal or goal in occupied:
        for dr, dc in CARDINAL:
            adj = (goal[0] + dr, goal[1] + dc)
            if 0 <= adj[0] < height and 0 <= adj[1] < width and grid[adj[0], adj[1]] != 1 and adj not in occupied:
                candidate_goals.append(adj)
    candidate_goals = list(dict.fromkeys(candidate_goals))
    if not candidate_goals:
        return _random_step(grid, start, occupied)

    candidate_set = set(candidate_goals)
    queue = deque([start])
    parent = {start: None}
    reached = None
    while queue:
        cur = queue.popleft()
        if cur in candidate_set:
            reached = cur
            break
        r, c = cur
        for dr, dc in CARDINAL:
            nxt = (r + dr, c + dc)
            if nxt in parent:
                continue
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if grid[nxt[0], nxt[1]] == 1 or nxt in occupied:
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if reached is None:
        return _random_step(grid, start, occupied)

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return start
    return step


def _random_step(grid, pos, occupied):
    height, width = grid.shape
    r, c = pos
    valid = [
        (r + dr, c + dc) for dr, dc in CARDINAL
        if 0 <= r + dr < height and 0 <= c + dc < width
        and grid[r + dr, c + dc] != 1 and (r + dr, c + dc) not in occupied
    ]
    return random.choice(valid) if valid else pos


def _ray_openness(grid, pos, direction, max_range=12):
    """壁のみに基づく直線方向の見通し距離(0〜1に正規化)。
    tv2_train_attacker_carry.pyの_ray_opennessと同一ロジック。"""
    height, width = grid.shape
    r, c = int(pos[0]), int(pos[1])
    dr, dc = direction
    dist = 0
    while dist < max_range:
        nr, nc = r + dr * (dist + 1), c + dc * (dist + 1)
        if not (0 <= nr < height and 0 <= nc < width) or grid[nr, nc] == 1:
            break
        dist += 1
    return dist / max_range


# ============================================================================
# コントローラー本体
# ============================================================================

class LearningAttackerCarryTouyamaController:
    """スパイク保持者(has_spike==True)のみDQNで操作する(アビリティ/PLANTのみ)。
    それ以外のtouyama_v2メンバー(エスコート)はヒューリスティックで動かす。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="auto", greedy=True, epsilon=0.0, debug=False):
        self.greedy = greedy
        self.epsilon = epsilon
        self.debug = debug

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Carryモデルが見つかりません: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        ckpt_obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
        ckpt_n_actions = int(checkpoint.get("n_actions", ACTION_DIM))
        if ckpt_obs_dim != OBS_DIM or ckpt_n_actions != ACTION_DIM:
            raise ValueError(
                f"チェックポイントの観測/行動空間がこのコントローラーと不一致です: "
                f"obs_dim={ckpt_obs_dim}(期待値{OBS_DIM}) n_actions={ckpt_n_actions}(期待値{ACTION_DIM})。"
                f"train_attacker_carry.pyのバージョンが古い可能性があります。"
            )

        self.policy_net = AttackerCarryDuelingDQN(obs_dim=ckpt_obs_dim, action_dim=ckpt_n_actions).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()

        self.priority_cells = [tuple(map(int, cell)) for cell in (checkpoint.get("priority_cells") or [])]
        self.has_priority_cells = bool(checkpoint.get("has_priority_cells", False)) and bool(self.priority_cells)

        raw_waypoint_cells = checkpoint.get("waypoint_cells") or {}
        self.waypoint_cells = {
            str(site): (int(cell[0]), int(cell[1])) for site, cell in raw_waypoint_cells.items()
        }

        raw_lineup_cells = checkpoint.get("smoke_lineup_cells_by_site") or {}
        self.smoke_lineup_cells_by_site = {
            str(site): [(int(cell[0]), int(cell[1])) for cell in cells]
            for site, cells in raw_lineup_cells.items()
        }

        self._log(
            f"[LOAD] model={model_path} episode={checkpoint.get('episode')} "
            f"success_rate={checkpoint.get('success_rate')} "
            f"has_priority_cells={self.has_priority_cells} priority_cells={self.priority_cells} "
            f"waypoint_cells={self.waypoint_cells}"
        )

        self.game = None
        # IQAwareController経由だと毎tick set_game() が呼ばれ、self.gameは
        # 使い捨てのview用オブジェクトに上書きされ続ける。target_plant_posの
        # 上書き(reset_round())は本物のgameに対して行う必要があるため、
        # 最初に渡された(=起動時の本物の)gameだけを別途保持する。
        self._real_game = None

        self._waypoint_dist_maps = {}     # site -> dist_map(waypoint_cells由来)
        self._plant_cells = None          # SITE_VALUESの全マス(アビリティ目標フォールバック用)
        self._priority_cells_by_site = {"left": [], "right": []}
        self._plant_cells_by_site = {"left": [], "right": []}   # 構造的フォールバック用

        # ラウンド単位の状態
        self._sighting = None
        self._active_site = None
        self._active_target = None        # このラウンドのナビゲーション最終目標(優先設置場所)
        self._reached_waypoint = True
        self._target_dist_map = None
        self._cached_target_pos = None
        self._cell_openness = {}

    # -- run_game.py 側フック(hasattr判定で自動呼び出しされる) -------------
    def set_game(self, game):
        if self._real_game is None:
            self._real_game = game
        self.game = game
        grid = game.grid

        self._plant_cells = [
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if int(grid[r, c]) in SITE_VALUES
        ]

        self._waypoint_dist_maps = {
            site: _bfs_distance_map(grid, cell) for site, cell in self.waypoint_cells.items()
        }

        # 進行方向先読み用の開放度マップ(壁情報のみ、敵の実位置は使わない)。
        # マップはラウンド中不変なのでset_game時に1回だけ計算する。
        walkable = [
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if grid[r, c] != 1
        ]
        self._cell_openness = {
            cell: [_ray_openness(grid, cell, d) for d in CARDINAL]
            for cell in walkable
        }

        width = grid.shape[1]
        self._plant_cells_by_site = {"left": [], "right": []}
        for cell in self._plant_cells:
            site_key = "left" if cell[1] < width // 2 else "right"
            self._plant_cells_by_site[site_key].append(cell)

        self._priority_cells_by_site = {"left": [], "right": []}
        if self.has_priority_cells:
            for cell in self.priority_cells:
                site_key = "left" if cell[1] < width // 2 else "right"
                self._priority_cells_by_site[site_key].append(cell)

    def reset_round(self):
        self._sighting = None
        self._active_site = None
        self._active_target = None
        self._reached_waypoint = True
        self._target_dist_map = None
        self._cached_target_pos = None

        if AI_CONTROLLED_SITE_SELECTION and self._real_game is not None:
            chosen_site = self._choose_weighted_site()

            candidates = self._priority_cells_by_site.get(chosen_site) or []
            if not candidates:
                candidates = self._plant_cells_by_site.get(chosen_site) or []
            if not candidates:
                fallback_site = "right" if chosen_site == "left" else "left"
                candidates = (
                    self._priority_cells_by_site.get(fallback_site)
                    or self._plant_cells_by_site.get(fallback_site)
                    or self._plant_cells
                )
                chosen_site = fallback_site

            if candidates:
                target = random.choice(candidates)
                self._active_site = chosen_site
                self._active_target = target
                # 表示整合のため本物のgameにも書き込む(内部ロジックはself._active_target
                # を直接参照するため、これ自体は必須ではない)。
                self._real_game.target_plant_pos = target
                if chosen_site in self._waypoint_dist_maps:
                    self._reached_waypoint = False
                else:
                    self._reached_waypoint = True
                self._log(
                    f"[SITE OVERRIDE] chosen_site={chosen_site} target={target}"
                )

    @staticmethod
    def _choose_weighted_site():
        left_w = max(0.0, float(SITE_SELECTION_WEIGHTS.get("left", 0.0)))
        right_w = max(0.0, float(SITE_SELECTION_WEIGHTS.get("right", 0.0)))
        total = left_w + right_w
        if total <= 0.0:
            return random.choice(["left", "right"])
        return "left" if random.random() < (left_w / total) else "right"

    # -- ログ ---------------------------------------------------------
    def _log(self, message):
        if not self.debug:
            return
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except OSError:
            pass

    # -- ユーティリティ -------------------------------------------------
    def _smoke_cells(self):
        cells = set()
        for smoke in getattr(self.game, "smokes", []):
            cells.update(smoke.get("cells", ()))
        return cells

    def _get_target_dist_map(self, target_pos):
        target_pos = (int(target_pos[0]), int(target_pos[1]))
        if target_pos != self._cached_target_pos:
            self._target_dist_map = _bfs_distance_map(self.game.grid, target_pos)
            self._cached_target_pos = target_pos
        return self._target_dist_map

    def _time_critical(self, pos):
        """優先設置場所への到達+設置完了(PLANT_REQUIRED_TICKS)が、残りTickでは
        間に合わない場合True。間に合わない場合のみ通常マスでの妥協設置を許可する。"""
        dist_map = self._get_target_dist_map(self._active_target)
        r, c = int(pos[0]), int(pos[1])
        dist_to_priority = dist_map[r, c]
        if dist_to_priority < 0:
            return True  # 経路が無い(通常起こらないが保険)
        elapsed_ticks = getattr(self.game, "battle_tick", 0)
        remaining_ticks = ROUND_DURATION_TICKS - elapsed_ticks
        ticks_needed = dist_to_priority + PLANT_REQUIRED_TICKS
        return remaining_ticks < ticks_needed

    def _plantable_cell(self, pos):
        """このマスでPLANTを許可するか。優先設置場所は常に許可。通常のプラント
        可能マス(2)は、優先設置場所への到達が時間的に間に合わない場合のみ
        妥協的に許可する。"""
        pos_t = (int(pos[0]), int(pos[1]))
        if pos_t == self._active_target:
            return True
        if pos_t in self._plant_cells_by_site.get(self._active_site, ()):
            return self._time_critical(pos_t)
        return False

    def _update_sighting(self, char, chars, smoke_cells):
        grid = self.game.grid
        visible = [
            other for other in chars
            if getattr(other, "is_alive", True)
            and getattr(other, "team", None) != char.team
            and _has_los(grid, char.pos, other.pos, smoke_cells)
        ]
        if visible:
            nearest = min(
                visible,
                key=lambda o: max(abs(o.pos[0] - char.pos[0]), abs(o.pos[1] - char.pos[1])),
            )
            self._sighting = {"pos": tuple(nearest.pos), "name": nearest.name, "tick_ago": 0}
        elif self._sighting is not None:
            self._sighting["tick_ago"] += 1
            if self._sighting["tick_ago"] > SIGHTING_STALENESS_CAP:
                self._sighting = None

    # -- 観測構築(train_attacker_carry.py の build_observation と同一構造) --
    def _build_observation(self, char, chars, smoke_cells, dist_map, elapsed_ticks, reached_waypoint, on_target, next_step=None):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        grid = self.game.grid
        height, width = grid.shape
        r0, c0 = int(char.pos[0]), int(char.pos[1])

        obs[0] = char.pos[0] / height
        obs[1] = char.pos[1] / width
        obs[2] = char.hp / char.max_hp if char.max_hp else 0.0
        obs[3] = 1.0 if getattr(char, "moved_this_tick", False) else 0.0

        ability_index = {"SMOKE": 4, "FLASH": 5, "RECON": 6, "HUNT": 7}.get(char.ability_name)
        if ability_index is not None:
            obs[ability_index] = 1.0
        charges = {
            "SMOKE": char.smoke_charges, "FLASH": char.flash_charges, "RECON": char.recon_charges,
        }.get(char.ability_name, 0)
        obs[8] = 1.0 if charges > 0 else 0.0

        bfs_dist = dist_map[r0, c0] if dist_map is not None else -1
        if bfs_dist < 0:
            bfs_dist = height + width
        obs[9] = min(bfs_dist, height + width) / (height + width)
        obs[10] = 1.0 if reached_waypoint else 0.0
        obs[11] = 1.0 if on_target else 0.0

        visible_enemies = [
            o for o in chars
            if getattr(o, "is_alive", True) and o.team != char.team
            and _has_los(grid, char.pos, o.pos, smoke_cells)
        ]
        obs[12] = 1.0 if visible_enemies else 0.0
        obs[13] = len(visible_enemies) / 5.0
        if visible_enemies:
            nearest = min(
                visible_enemies,
                key=lambda o: max(abs(o.pos[0] - r0), abs(o.pos[1] - c0)),
            )
            obs[14] = (nearest.pos[0] - r0) / height
            obs[15] = (nearest.pos[1] - c0) / width
            dist = max(abs(nearest.pos[0] - r0), abs(nearest.pos[1] - c0))
            obs[16] = min(dist, height) / height

        obs[17] = 1.0 if any(
            getattr(o, "is_alive", True) and o.team != char.team
            and (getattr(o, "blind_remaining", 0) > 0 or getattr(o, "reveal_remaining", 0) > 0)
            for o in chars
        ) else 0.0

        own_smoke_active = any(
            s.get("owner") is not None
            and any(c.name == s.get("owner") and c.team == char.team for c in chars)
            for s in getattr(self.game, "smokes", [])
        )
        obs[18] = 1.0 if own_smoke_active else 0.0
        obs[19] = min(elapsed_ticks, 100) / 100.0
        obs[20] = sum(
            1 for o in chars if getattr(o, "is_alive", True) and o.team != char.team
        ) / 5.0

        char_facing = getattr(char, "facing", None)
        if char_facing in FACING_DIRS:
            obs[21 + FACING_DIRS.index(char_facing)] = 1.0

        # 次に進む予定のマスへの方向と、そのマスからさらに奥がどれだけ開けて
        # いるか(壁情報のみ。敵の実位置は使わない。train側と同一ロジック)。
        if next_step is not None:
            dr, dc = int(next_step[0]) - r0, int(next_step[1]) - c0
            if (dr, dc) in CARDINAL:
                dir_idx = CARDINAL.index((dr, dc))
                obs[29 + dir_idx] = 1.0
                openness = self._cell_openness.get((int(next_step[0]), int(next_step[1])))
                if openness is not None:
                    obs[33] = openness[dir_idx]

        return obs

    def _build_mask(self, char, on_target, ability_available=True):
        """向き(facing)は移動・アビリティとは無関係に常に自由選択できるため、
        base(0-2)側のマスクをfacing方向数だけ展開する
        (tv2_train_attacker_carry.pyと同一規約)。
        ability_available=Falseの場合は、chargesが残っていてもABILITYを選べない
        (train側のゲートと同一方針。SMOKEの無駄撃ちによるチャージ浪費を防ぐ)。"""
        base_mask = np.ones(BASE_ACTION_DIM, dtype=bool)
        charges = {
            "SMOKE": char.smoke_charges, "FLASH": char.flash_charges, "RECON": char.recon_charges,
        }.get(char.ability_name, 0)
        if charges <= 0 or char.ability_name == "HUNT" or not ability_available:
            base_mask[ACTION_ABILITY] = False
        base_mask[ACTION_PLANT] = bool(on_target)
        return np.repeat(base_mask, len(FACING_DIRS))

    def _select_action(self, obs, mask):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs_t).squeeze(0).cpu().numpy()
            q_values = np.where(mask, q_values, -np.inf)
            if not self.greedy and random.random() < self.epsilon:
                valid = np.flatnonzero(mask)
                return int(np.random.choice(valid)) if len(valid) else 0
            return int(np.argmax(q_values))

    @staticmethod
    def _decode_action(action_idx):
        """action_idx = base_idx(0-2) * 8 + facing_idx(0-7)。
        tv2_train_attacker_carry.pyのdecode_actionと同一規約。
        戻り値は (decoded_base_action, facing)。"""
        idx = int(action_idx)
        base_idx, facing_idx = divmod(idx, len(FACING_DIRS))
        if base_idx == ACTION_PLANT:
            decoded = "PLANT"
        elif base_idx == ACTION_ABILITY:
            decoded = "ABILITY"
        else:
            decoded = "NONE"
        return decoded, FACING_DIRS[facing_idx]

    # -- エスコート・ヒューリスティック -----------------------------------
    def _escort_ability_action(self, char, visible_enemies):
        charges = {
            "SMOKE": char.smoke_charges, "FLASH": char.flash_charges, "RECON": char.recon_charges,
        }.get(char.ability_name, 0)
        if charges <= 0 or char.ability_name == "HUNT":
            return None

        if char.ability_name == "SMOKE" and len(visible_enemies) >= 2:
            return ("SMOKE", tuple(map(int, visible_enemies[0].pos)))

        if char.ability_name == "FLASH" and visible_enemies:
            closest = min(
                visible_enemies,
                key=lambda e: max(abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])),
            )
            dist = max(abs(closest.pos[0] - char.pos[0]), abs(closest.pos[1] - char.pos[1]))
            if dist <= 5:
                return ("FLASH", tuple(map(int, closest.pos)))

        if char.ability_name == "RECON" and not visible_enemies and self._plant_cells:
            return ("RECON", random.choice(self._plant_cells))

        return None

    def _decide_escort_move(self, char, carrier, occupied):
        grid = self.game.grid
        if carrier is None or not getattr(carrier, "is_alive", True):
            return list(char.pos)
        dist = max(abs(carrier.pos[0] - char.pos[0]), abs(carrier.pos[1] - char.pos[1]))
        if random.random() < 0.3:
            nxt = _random_step(grid, tuple(char.pos), occupied)
        elif dist > 5:
            nxt = _bfs_next_step(grid, tuple(char.pos), tuple(carrier.pos), occupied, allow_adjacent_goal=True)
        else:
            nxt = _random_step(grid, tuple(char.pos), occupied)
        return [int(nxt[0]), int(nxt[1])]

    # -- メインエントリポイント -----------------------------------------
    def decide_move(self, char, game_state):
        chars = game_state.get("chars", [])
        grid = self.game.grid
        r, c = int(char.pos[0]), int(char.pos[1])
        smoke_cells = self._smoke_cells()

        occupied = {
            tuple(o.pos) for o in chars
            if o is not char and getattr(o, "is_alive", True)
        }

        # --- キャリア以外(エスコート)はヒューリスティック ---
        if not getattr(char, "has_spike", False):
            carrier = next(
                (o for o in chars if getattr(o, "is_alive", True) and o.team == char.team and getattr(o, "has_spike", False)),
                None,
            )
            visible_enemies = [
                o for o in chars
                if getattr(o, "is_alive", True) and o.team != char.team
                and _has_los(grid, char.pos, o.pos, smoke_cells)
            ]
            ability = self._escort_ability_action(char, visible_enemies)
            if ability is not None:
                return list(char.pos), {"ability": ability[0], "target": ability[1]}
            next_pos = self._decide_escort_move(char, carrier, occupied)
            return next_pos

        # --- ここからキャリア ---
        if self._active_target is None:
            # AI_CONTROLLED_SITE_SELECTION=Falseの場合などのフォールバック。
            fallback = game_state.get("target_plant_pos")
            self._active_target = (
                (int(fallback[0]), int(fallback[1])) if fallback is not None
                else (self._plant_cells[0] if self._plant_cells else (r, c))
            )
            width = grid.shape[1]
            self._active_site = "left" if self._active_target[1] < width // 2 else "right"
            self._reached_waypoint = self._active_site not in self._waypoint_dist_maps

        # 中継地点通過判定(未通過の場合のみ)。
        if not self._reached_waypoint:
            waypoint_cell = self.waypoint_cells[self._active_site]
            waypoint_dist = max(abs(r - waypoint_cell[0]), abs(c - waypoint_cell[1]))
            if waypoint_dist <= 1:
                self._reached_waypoint = True

        goal = self.waypoint_cells[self._active_site] if not self._reached_waypoint else self._active_target
        on_target = self._plantable_cell((r, c))

        if self._reached_waypoint:
            dist_map = self._get_target_dist_map(self._active_target)
        else:
            dist_map = self._waypoint_dist_maps[self._active_site]

        self._update_sighting(char, chars, smoke_cells)
        elapsed_ticks = getattr(self.game, "battle_tick", 0)

        # 移動は決定的BFS経路探索(占有マスは自分以外の全キャラ)。観測の
        # 先読み特徴量にも同じ移動先を使うため、行動決定より先に計算する。
        # 中継地点は隣接到達でOK(waypoint_dist<=1で別途判定済み)だが、最終
        # プラント目標は正確な到達が必須(隣接停止のままだとon_targetが
        # 永久にFalseになりPLANTを一切選べなくなる)。
        own_occupied = occupied
        allow_adjacent = not self._reached_waypoint
        nxt = _bfs_next_step(grid, (r, c), goal, own_occupied, allow_adjacent_goal=allow_adjacent)
        next_pos = [int(nxt[0]), int(nxt[1])]

        # マスク構築前に、ABILITYを選ぶ意味があるか(射程内に敵、またはSMOKEなら
        # 定点が射程内)を判定しておく。train側のability_availableゲートと同一方針
        # (無駄撃ちでチャージを浪費させないため、意味が無ければ選択肢から外す)。
        visible_enemies = [
            o for o in chars
            if getattr(o, "is_alive", True) and o.team != char.team
            and _has_los(grid, char.pos, o.pos, smoke_cells)
        ]
        nearby_enemies = [
            o for o in visible_enemies
            if max(abs(o.pos[0] - r), abs(o.pos[1] - c)) <= ABILITY_RANGE
        ]
        lineup_candidates = []
        if char.ability_name == "SMOKE":
            lineup_candidates = [
                cell for cell in self.smoke_lineup_cells_by_site.get(self._active_site, [])
                if max(abs(cell[0] - r), abs(cell[1] - c)) <= ABILITY_RANGE
            ]
        ability_available = (
            char.ability_name != "SMOKE" or bool(nearby_enemies) or bool(lineup_candidates)
        )

        obs = self._build_observation(
            char, chars, smoke_cells, dist_map, elapsed_ticks, self._reached_waypoint, on_target,
            next_step=next_pos,
        )
        mask = self._build_mask(char, on_target, ability_available)
        action_idx = self._select_action(obs, mask)
        decoded, facing = self._decode_action(action_idx)

        if decoded == "PLANT":
            return list(char.pos), "PLANT"

        if decoded == "ABILITY":
            target_pos = None
            if nearby_enemies:
                nearest = min(
                    nearby_enemies,
                    key=lambda o: max(abs(o.pos[0] - r), abs(o.pos[1] - c)),
                )
                target_pos = tuple(map(int, nearest.pos))
            elif lineup_candidates:
                target_pos = min(
                    lineup_candidates,
                    key=lambda cell: max(abs(cell[0] - r), abs(cell[1] - c)),
                )
            if target_pos is None and self._sighting is not None:
                target_pos = self._sighting["pos"]
            if target_pos is None and self._plant_cells:
                target_pos = random.choice(self._plant_cells)

            if target_pos is not None:
                return next_pos, {"ability": char.ability_name, "target": target_pos}

        # 向き(facing)は移動方向とは無関係にDQNが選択する。battle_logic.pyが
        # 移動と同時にこのfacing指定を適用する(被弾直後の強制向きが最優先)。
        return next_pos, {"facing": facing}