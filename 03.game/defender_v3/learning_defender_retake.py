"""learning_defender_retake.py (修正版)

train_defender_retake.py の観測変更(BFS距離マップベースのdist_to_plant /
good_dir)に合わせて更新。以前のバージョンとは観測ベクトルの意味が異なるため、
再学習済みのモデル(dqn_defender_retake_*.pt)と組み合わせて使うこと。

完全に自己完結。controllers.py / battle_logic.py / abilities_los.py への
依存はなく、必要なLOS・BFS・行動マスク・観測構築ロジックはこのファイル内に
複製する。game_core からは定数のみ参照する。

重要: 5人の Defender 全員に対して、このクラスの「同一インスタンス」を
defender_controller として割り当てること。味方が使用中のアビリティ効果
（特にSMOKE）を追跡するため、チーム内で状態を共有する設計になっている。
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
    DEFUSE_REQUIRED_TICKS,
    SPIKE_DETONATION_TICKS,
    SMOKE_DURATION_TICKS,
    RECON_REVEAL_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right (行動ID 0-3と対応)
N_ACTIONS = 7
MOVE_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
ACTION_DEFUSE = 5
ACTION_ABILITY = 6

SITE_ZONE_RADIUS = 6
DEFUSE_SAFETY_MARGIN_TICKS = 4
ENTRY_READY_RADIUS = 3
ENTRY_SAFETY_MARGIN_TICKS = DEFUSE_SAFETY_MARGIN_TICKS + ENTRY_READY_RADIUS
ROLE_INDEX = {"フラッシュ": 0, "スモーカー": 1, "シーカー": 2}

DEFAULT_MODEL_PATH = "data/defender_retake_data/dqn_defender_retake_best.pt"


# ---------------------------------------------------------------------------
# Dueling DQN (train_defender_retake.py と同一構造)
# ---------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions))

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# 補助関数(LOS / BFS)。abilities_los.py / controllers.py とは独立した複製実装。
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


def _smoke_allows_line(cells, smoke_cells):
    if not cells or len(cells) <= 2:
        return True
    return not any(cell in smoke_cells for cell in cells)


def _has_los(grid, p1, p2, smoke_cells):
    cells = _line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    return _smoke_allows_line(cells, smoke_cells)


def _bfs_distance_map(grid, goal):
    """goal(プラント地点)から各床マスへの最短距離マップ(壁越え不可)。
    train_defender_retake.py の bfs_distance_map() と同一方式。"""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _good_directions(grid, dist_map, r, c):
    """現在地から上下左右のうち、BFS距離マップ上で実際にプラントへの距離を
    縮められる方向を1.0とする4次元フラグ(train_defender_retake.py と同一)。"""
    height, width = grid.shape
    good = [0.0, 0.0, 0.0, 0.0]
    raw = dist_map[r, c]
    if raw < 0:
        return good
    for i, (dr, dc) in enumerate(CARDINAL_MOVES):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
            nd = dist_map[nr, nc]
            if nd != -1 and nd < raw:
                good[i] = 1.0
    return good


def _alive_occupied_positions(chars, moving_char=None):
    occupied = set()
    for other in chars:
        if other is moving_char or not getattr(other, "is_alive", True):
            continue
        pos = getattr(other, "pos", None)
        if pos is None:
            continue
        occupied.add((int(pos[0]), int(pos[1])))
    return occupied


def _is_walkable(grid, pos, blocked):
    height, width = grid.shape
    r, c = pos
    return 0 <= r < height and 0 <= c < width and grid[r, c] != 1 and (r, c) not in blocked


def _get_next_pos_random(grid, pos, chars, moving_char=None):
    r, c = int(pos[0]), int(pos[1])
    blocked = _alive_occupied_positions(chars, moving_char)
    valid = []
    for dr, dc in CARDINAL_MOVES:
        cand = (r + dr, c + dc)
        if _is_walkable(grid, cand, blocked):
            valid.append(cand)
    if not valid:
        return [r, c]
    return list(random.choice(valid))


def _candidate_goals(grid, goal, blocked, allow_adjacent_goal):
    goal = (int(goal[0]), int(goal[1]))
    candidates = []
    if _is_walkable(grid, goal, blocked):
        candidates.append(goal)
    if allow_adjacent_goal or goal in blocked:
        for dr, dc in CARDINAL_MOVES:
            adj = (goal[0] + dr, goal[1] + dc)
            if _is_walkable(grid, adj, blocked):
                candidates.append(adj)
    return list(dict.fromkeys(candidates))


def _move_towards_target(grid, pos, target, chars, moving_char=None, allow_adjacent_goal=False):
    """BFSで壁・生存キャラクターを避けながらtargetへ1マス進む(controllers.py非依存の複製)。
    プラント前(is_planted=False)の待避移動にのみ使用する。"""
    start = (int(pos[0]), int(pos[1]))
    goal = (int(target[0]), int(target[1]))
    blocked = _alive_occupied_positions(chars, moving_char)
    blocked.discard(start)

    if start == goal:
        return [start[0], start[1]]

    candidate_goals = _candidate_goals(grid, goal, blocked, allow_adjacent_goal)
    if not candidate_goals:
        return _get_next_pos_random(grid, start, chars, moving_char)

    candidate_goal_set = set(candidate_goals)
    queue = deque([start])
    parent = {start: None}
    reached = None

    while queue:
        cur = queue.popleft()
        if cur in candidate_goal_set:
            reached = cur
            break
        r, c = cur
        for dr, dc in CARDINAL_MOVES:
            nxt = (r + dr, c + dc)
            if nxt in parent or not _is_walkable(grid, nxt, blocked):
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if reached is None:
        return _get_next_pos_random(grid, start, chars, moving_char)

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return [start[0], start[1]]
    return [int(step[0]), int(step[1])]


def _own_ability_charge(char):
    ability = char.ability_name
    if ability == "SMOKE":
        return char.smoke_charges
    if ability == "FLASH":
        return char.flash_charges
    if ability == "RECON":
        return char.recon_charges
    return 0

# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningDefenderRetakeController:
    """Retake(サイト再突入・スパイク解除)フェーズ専用のAI。

    decide_move(char, game_state) は battle_logic.py の既存インタフェースに
    合わせて以下のいずれかを返す:
        - next_pos (list[int,int])                        : 通常移動
        - (next_pos, "DEFUSE")                              : 解除
        - (next_pos, {"ability": name, "target": (r, c)})   : アビリティ使用

    5人のDefender全員に対して同一インスタンスを割り当てることを前提とする。
    味方が撃ったSMOKEの残存を内部トラッキングするため。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, obs_dim=None, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self._obs_dim = obs_dim
        self._model_path = model_path
        self.model = None
        self.reset_round()

    def _lazy_init_model(self, obs_dim):
        self.model = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
        try:
            state_dict = torch.load(self._model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            if self.verbose:
                print(f"[LearningDefenderRetakeController] loaded: {self._model_path}")
        except Exception as exc:
            print(f"[LOAD ERROR] defender retake model '{self._model_path}' の読込に失敗: {exc}")
        self.model.eval()
        self._obs_dim = obs_dim

    # -- ラウンド開始時のリセット ------------------------------------------
    def reset_round(self):
        self._dist_map = None
        self._dist_map_key = None  # プラント地点座標。変化時のみ再計算する
        self._site_zone = None
        self._processed_this_tick = set()
        self._smoke_remaining = {}   # name -> 残りTick数
        self._smoke_cells_cache = set()

    # -- 味方アビリティ使用状況の追跡 ---------------------------------------
    def _advance_tick_if_needed(self, name):
        if name in self._processed_this_tick:
            self._processed_this_tick = set()
            expired = []
            for n, remaining in self._smoke_remaining.items():
                remaining -= 1
                if remaining <= 0:
                    expired.append(n)
                else:
                    self._smoke_remaining[n] = remaining
            for n in expired:
                del self._smoke_remaining[n]
            if not self._smoke_remaining:
                self._smoke_cells_cache = set()
        self._processed_this_tick.add(name)

    def _register_own_smoke_cast(self, planted_pos, grid, caster_name):
        pr, pc = int(planted_pos[0]), int(planted_pos[1])
        height, width = grid.shape
        cells = {
            (rr, cc)
            for rr in range(pr - 1, pr + 2)
            for cc in range(pc - 1, pc + 2)
            if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] != 1
        }
        self._smoke_remaining[caster_name] = SMOKE_DURATION_TICKS
        self._smoke_cells_cache = self._smoke_cells_cache | cells

    def _ally_ability_active(self, char, chars):
        for enemy in chars:
            if getattr(enemy, "team", None) != char.team and getattr(enemy, "is_alive", True):
                if enemy.blind_remaining > 0 or enemy.reveal_remaining > 0:
                    return True
        return bool(self._smoke_remaining)

    # -- BFS距離マップ・サイトゾーン(遅延計算・キャッシュ) --------------------
    def _get_dist_map(self, grid, plant_pos):
        key = (int(plant_pos[0]), int(plant_pos[1]))
        if self._dist_map is None or self._dist_map_key != key:
            self._dist_map = _bfs_distance_map(grid, plant_pos)
            self._dist_map_key = key
            self._site_zone = {
                (r, c)
                for r in range(grid.shape[0])
                for c in range(grid.shape[1])
                if 0 <= self._dist_map[r, c] <= SITE_ZONE_RADIUS
            }
        return self._dist_map, self._site_zone

    # -- 観測構築(train_defender_retake.py の RetakeEnv.build_observation と対応) --
    def _build_observation(self, char, game_state):
        grid = game_state["grid"]
        height, width = grid.shape
        chars = game_state.get("chars", [])
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos") or game_state.get("target_plant_pos")
        detonate_timer = game_state.get("detonate_timer", SPIKE_DETONATION_TICKS)

        if planted_pos is None:
            planted_pos = tuple(char.pos)

        dist_map, site_zone = self._get_dist_map(grid, planted_pos)

        r, c = char.pos
        pr, pc = planted_pos

        good_dir = _good_directions(grid, dist_map, r, c)
        raw_dist = dist_map[r, c]
        dist_to_plant = min(1.0, raw_dist / (height + width)) if raw_dist >= 0 else 1.0

        in_site_zone = 1.0 if (r, c) in site_zone else 0.0
        adjacent_to_plant = 1.0 if max(abs(pr - r), abs(pc - c)) <= 1 else 0.0

        own_charge = 1.0 if _own_ability_charge(char) > 0 else 0.0
        blind_norm = char.blind_remaining / BLIND_DURATION_TICKS if BLIND_DURATION_TICKS else 0.0
        defuse_norm = char.defuse_timer / DEFUSE_REQUIRED_TICKS
        detonate_norm = detonate_timer / SPIKE_DETONATION_TICKS

        allies = [a for a in chars if a.team == char.team and getattr(a, "is_alive", True)]
        enemies = [e for e in chars if e.team != char.team and getattr(e, "is_alive", True)]

        allies_alive_norm = len(allies) / 5.0
        allies_in_zone = sum(1 for a in allies if tuple(a.pos) in site_zone) / 5.0
        allies_near_entry = sum(
            1 for a in allies if max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
        ) / 5.0

        others = [a for a in allies if a is not char]
        if others:
            nearest_ally_dist = min(
                max(abs(a.pos[0] - r), abs(a.pos[1] - c)) for a in others
            ) / max(height, width)
        else:
            nearest_ally_dist = 1.0

        ally_ability_active = 1.0 if self._ally_ability_active(char, chars) else 0.0

        smoke_cells = self._smoke_cells_cache
        visible_enemies = [
            e for e in enemies if _has_los(grid, tuple(char.pos), tuple(e.pos), smoke_cells)
        ]
        visible_enemies.sort(key=lambda e: max(abs(e.pos[0] - r), abs(e.pos[1] - c)))

        enemy_feats = []
        for e in visible_enemies[:2]:
            edx = (e.pos[1] - c) / width
            edy = (e.pos[0] - r) / height
            edist = max(abs(e.pos[0] - r), abs(e.pos[1] - c)) / max(height, width)
            ehp = e.hp / e.max_hp
            eblind = 1.0 if e.blind_remaining > 0 else 0.0
            erevealed = 1.0 if (e.reveal_remaining > 0 or getattr(e, "los_revealed", False)) else 0.0
            enemy_feats.extend([edx, edy, edist, ehp, eblind, erevealed])
        while len(enemy_feats) < 12:
            enemy_feats.append(0.0)

        role_onehot = [0.0, 0.0, 0.0]
        role_onehot[ROLE_INDEX.get(char.role, 0)] = 1.0

        obs = [
            r / height, c / width,
            char.hp / char.max_hp,
            *good_dir,
            dist_to_plant,
            in_site_zone, adjacent_to_plant,
            own_charge, blind_norm, defuse_norm, detonate_norm,
            allies_alive_norm, allies_in_zone, allies_near_entry, nearest_ally_dist,
            ally_ability_active,
            len(visible_enemies) / 5.0,
            len(enemies) / 5.0,
        ] + enemy_feats + role_onehot

        return np.array(obs, dtype=np.float32), planted_pos, is_planted

    # -- 行動マスク(train_defender_retake.py の RetakeEnv.action_mask と対応) ----
    def _action_mask(self, char, game_state, planted_pos, is_planted):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        detonate_timer = game_state.get("detonate_timer", SPIKE_DETONATION_TICKS)
        height, width = grid.shape

        mask = np.zeros(N_ACTIONS, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = _alive_occupied_positions(chars, char)

        for a in range(4):
            dr, dc = MOVE_DELTAS[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and (nr, nc) not in occupied:
                mask[a] = True
        mask[4] = True

        pr, pc = int(planted_pos[0]), int(planted_pos[1])
        dist_to_plant = max(abs(pr - r), abs(pc - c))
        mask[ACTION_DEFUSE] = bool(is_planted and dist_to_plant <= 1)

        has_charge = _own_ability_charge(char) > 0
        mask[ACTION_ABILITY] = bool(has_charge and not self._ally_ability_active(char, chars))

        # 💡追加: train_defender_retake.py の action_mask と同一条件。
        # 時間に余裕があり、敵が視認できている場合は移動をマスクして足を止めさせる。
        time_critical = detonate_timer <= ENTRY_SAFETY_MARGIN_TICKS
        if not time_critical:
            enemy_visible = any(
                getattr(e, "is_alive", True)
                and _has_los(grid, tuple(char.pos), tuple(e.pos), self._smoke_cells_cache)
                for e in chars if e.team != char.team
            )
            if enemy_visible:
                mask[0] = mask[1] = mask[2] = mask[3] = False

        return mask

    # -- メイン ----------------------------------------------------------
    def decide_move(self, char, game_state):
        self._advance_tick_if_needed(char.name)

        grid = game_state["grid"]
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")
        target_plant_pos = game_state.get("target_plant_pos")
        chars = game_state.get("chars", [])

        if not is_planted:
            fallback_target = planted_pos or target_plant_pos
            if fallback_target is None:
                return list(char.pos)
            return _move_towards_target(grid, char.pos, fallback_target, chars, char, allow_adjacent_goal=True)

        obs, resolved_plant_pos, _ = self._build_observation(char, game_state)

        if self.model is None:
            self._lazy_init_model(obs.shape[0])

        mask = self._action_mask(char, game_state, resolved_plant_pos, is_planted)

        state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action = int(torch.argmax(q_values).item())

        if self.verbose:
            print(f"[RETAKE] {char.name} pos={tuple(char.pos)} action={action}")

        if action in (0, 1, 2, 3, 4):
            dr, dc = MOVE_DELTAS[action]
            return [char.pos[0] + dr, char.pos[1] + dc]

        if action == ACTION_DEFUSE:
            return list(char.pos), "DEFUSE"

        if char.ability_name == "SMOKE":
            self._register_own_smoke_cast(resolved_plant_pos, grid, char.name)

        target = (int(resolved_plant_pos[0]), int(resolved_plant_pos[1]))
        return list(char.pos), {"ability": char.ability_name, "target": target}

    def reset_round_public(self):
        self.reset_round()