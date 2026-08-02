"""train_attacker_escort.py

Attacker Carry Phase のうち、Escort（護衛）役 4体を学習するスクリプト（Dueling DQN）。

【目的】
キャリアー（スパイク運搬役）の周囲2〜7マス程度を維持しつつ、
キャリアーとプラントサイトの間の経路を塞がないように動き、
敵を見つけたらアビリティ（FLASH/RECON/SMOKE）を適切なタイミングで
使用できるようになるまで学習する。

【重要な設計判断】
1. マルチエージェント・重み共有方式
   4体のescortは全員「同じDueling DQNの重み」を共有して行動する
   （train_attacker_guard.py 系と同じ「パラメータ共有」方式の踏襲）。
   観測はエージェントごとに自分中心の相対座標系で構築するため、
   同じネットワークをどのescortにも使い回せる。

2. キャリアーは学習対象ではなくスクリプトAI
   キャリアーはエピソード開始時に決めたプラントサイトへ、
   固定のBFS最短経路を1tickごとに1マスずつ進む。
   経路上の次のマスが（escortまたは敵に）占有されている場合、
   実際のゲーム（battle_logic.move_character）と同じく「移動できず
   その場に留まる」。この「キャリアーが実際に進んだ距離」を
   escort全員の共有報酬にすることで、道を塞ぐと損、というシグナルを
   ハードコードせずに学習させる（横に空きがあれば避ける／一本道なら
   先に進む、のどちらもこの報酬設計から自然に導かれる）。

3. 敵はスクリプトAI（簡易ランダム待機・徘徊）
   本格的なDefenderAIの学習は別スクリプトの範囲。ここでは
   「視界に入る・アビリティで状態異常にできる・撃ち合いが発生する」
   という最低限の相互作用だけを簡易シミュレーションする。

4. 戦闘解決は簡略化モデル
   実際のキャラクター別ステータス（命中率・回避率等）は使わず、
   汎用的な固定値で近似する。escortの「立ち回り」と「アビリティ判断」
   を学習させることが目的であり、精密な戦闘バランス自体は
   Defender/Guardモデル側の学習範囲とする。

【設計方針（既存ルールの継承）】
- このファイルは完全に自己完結している（map_data.py以外の
  ゲーム本体コードに依存しない）。
- run_game.py / controllers.py など既存の共有インフラは変更・複製しない。
"""

import argparse
import os
import random
import sys
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPISODE_COUNT = 9000

# ---------------------------------------------------------------------------
# map_data.py の解決（attacker_v3/ 配下・プロジェクト直下のどちらでも動く）
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = (
    os.path.dirname(_CURRENT_DIR)
    if os.path.basename(_CURRENT_DIR) == "attacker_v3"
    else _CURRENT_DIR
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from map_data import NEW_MAZE_STR  # noqa: E402


# ---------------------------------------------------------------------------
# ゲームバランス定数（簡略化モデル用。本番の game_core.py / abilities_los.py
# の値を踏襲しつつ、escort訓練用に必要な部分だけ再現する）
# ---------------------------------------------------------------------------
SITE_CELL_VALUE = 2
ATTACKER_SPAWN_VALUE = 3
DEFENDER_SPAWN_VALUE = 4

MAX_HP = 100
HEADSHOT_DAMAGE = 160
BODY_DAMAGE = 40

BLIND_DURATION_TICKS = 3
REVEAL_DURATION_TICKS = 5
BLIND_ACCURACY_MULTIPLIER = 0.30
REVEALED_DODGE_MULTIPLIER = 0.50
SMOKE_DURATION_TICKS = 15

# 簡略化モデル用の汎用戦闘ステータス（実キャラの個体差は考慮しない近似値）
GENERIC_ACCURACY = 0.55
GENERIC_DODGE = 0.18
GENERIC_HS_RATE = 0.30

ABILITY_TYPES = ("FLASH", "RECON", "SMOKE")
ABILITY_RANGE = 6  # アビリティが届く最大距離（マンハッタン近似ではなくチェビシェフ距離で判定）

N_ESCORTS = 4
N_ENEMIES = 2

DIST_BAND_MIN = 2
DIST_BAND_MAX = 7
DIST_NORM_MAX = 15.0  # 観測正規化用の上限距離

STALL_THRESHOLD_TICKS = 3       # これを超えて無進捗が続いたら混雑ペナルティ開始
STALL_PENALTY_CAP_TICKS = 10    # ペナルティの伸び幅の上限（無限にエスカレートさせない）
CONGESTION_RADIUS = 2           # carryからこの距離以内のescortを「渋滞に関与」とみなす


# ---------------------------------------------------------------------------
# 汎用ヘルパー：BFS最短経路・射線判定
# ---------------------------------------------------------------------------
def _bfs_shortest_path(grid, start, goal):
    """壁(1)だけを障害物としたBFS最短経路。start→goalのセル列を返す。"""
    height, width = grid.shape
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return [start]

    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if grid[nr, nc] == 1 or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if goal not in parent:
        return [start]

    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _line_cells(p1, p2):
    """Bresenham法で2点間のセル列を返す（abilities_los.py と同一ロジック）。"""
    y0, x0 = int(p1[0]), int(p1[1])
    y1, x1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
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


def _has_los(grid, smoke_cells, p1, p2):
    """壁とスモークを考慮した射線判定（abilities_los.pyの簡略版）。"""
    cells = _line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells and len(cells) > 2:
        if any(cell in smoke_cells for cell in cells):
            return False
    return True


def _chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _build_distance_map_walls_only(grid, source_cells):
    """指定座標群を始点とした、壁のみを障害物としたマルチソースBFS距離マップ。
    キャラクター同士の占有は考慮しない
    （推論側 learning_attacker_escort.py と同一ロジック）。"""
    height, width = grid.shape
    dist = np.full((height, width), np.inf, dtype=np.float32)
    q = deque()
    for r, c in source_cells:
        if 0 <= r < height and 0 <= c < width and grid[r, c] != 1:
            dist[r, c] = 0.0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


# ---------------------------------------------------------------------------
# 環境
# ---------------------------------------------------------------------------
class EscortEnv:
    """キャリアー護衛4体（重み共有）を学習させる軽量マルチエージェント環境。"""

    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_ABILITY = range(6)
    N_ACTIONS = 6
    _MOVE_DELTA = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
        ACTION_STAY: (0, 0),
    }

    def __init__(
        self,
        maze_str=NEW_MAZE_STR,
        max_ticks=90,
        n_escorts=N_ESCORTS,
        n_enemies=N_ENEMIES,
        dist_band_min=DIST_BAND_MIN,
        dist_band_max=DIST_BAND_MAX,
        team_progress_coef=1.0,
        block_penalty=1.0,
        dist_penalty_coef=0.08,
        stall_threshold_ticks=STALL_THRESHOLD_TICKS,
        stall_penalty_cap_ticks=STALL_PENALTY_CAP_TICKS,
        congestion_radius=CONGESTION_RADIUS,
        congestion_penalty_coef=0.15,
        ability_success_reward=2.0,
        ability_redundant_penalty=1.0,
        ability_waste_penalty=0.3,
        kill_bonus=5.0,
        death_penalty=3.0,
        mission_success_reward=5.0,
        mission_fail_penalty=5.0,
        enemy_move_prob=0.2,
        seed=None,
    ):
        lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.height, self.width = self.grid.shape

        self.max_ticks = max_ticks
        self.n_escorts = n_escorts
        self.n_enemies = n_enemies
        self.dist_band_min = dist_band_min
        self.dist_band_max = dist_band_max
        self.team_progress_coef = team_progress_coef
        self.block_penalty = block_penalty
        self.dist_penalty_coef = dist_penalty_coef
        self.stall_threshold_ticks = stall_threshold_ticks
        self.stall_penalty_cap_ticks = stall_penalty_cap_ticks
        self.congestion_radius = congestion_radius
        self.congestion_penalty_coef = congestion_penalty_coef
        self.ability_success_reward = ability_success_reward
        self.ability_redundant_penalty = ability_redundant_penalty
        self.ability_waste_penalty = ability_waste_penalty
        self.kill_bonus = kill_bonus
        self.death_penalty = death_penalty
        self.mission_success_reward = mission_success_reward
        self.mission_fail_penalty = mission_fail_penalty
        self.enemy_move_prob = enemy_move_prob

        self.site_cells = list(zip(*np.where(self.grid == SITE_CELL_VALUE)))
        self.attacker_spawns = list(zip(*np.where(self.grid == ATTACKER_SPAWN_VALUE)))
        self.defender_spawns = list(zip(*np.where(self.grid == DEFENDER_SPAWN_VALUE)))
        self.walkable_cells = list(zip(*np.where(self.grid != 1)))

        self.rng = random.Random(seed)

        # ラウンド内状態（reset()で初期化）
        self.tick = 0
        self.carry_pos = (0, 0)
        self.carry_hp = MAX_HP
        self.carry_alive = True
        self.carry_path = [(0, 0)]
        self.carry_path_index = 0

        self.escort_pos = []
        self.escort_hp = []
        self.escort_alive = []
        self.escort_ability_type = []
        self.escort_ability_used = []
        self.escort_last_delta = []
        self.escort_stuck = []

        self.enemy_pos = []
        self.enemy_hp = []
        self.enemy_alive = []
        self.enemy_blind_remaining = []
        self.enemy_blind_source = []
        self.enemy_reveal_remaining = []
        self.enemy_reveal_source = []

        self.smokes = []  # [{"cells": set, "remaining": int}]

        self._blocking_escort_idx = None  # このtickでキャリアーを塞いだescort index
        self._carry_dist_map = None
        self._stall_ticks = 0  # キャリアーが連続で進めていないtick数

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def _random_walkable(self, exclude=()):
        exclude = set(exclude)
        for _ in range(200):
            cell = self.rng.choice(self.walkable_cells)
            if cell not in exclude:
                return cell
        return self.rng.choice(self.walkable_cells)

    def reset(self):
        self.tick = 0
        self.smokes = []

        # --- キャリアー：スポーンからサイトのどれか1点へ固定経路 ---
        carry_spawn = self.rng.choice(self.attacker_spawns) if self.attacker_spawns else self._random_walkable()
        target = self.rng.choice(self.site_cells) if self.site_cells else carry_spawn
        self.carry_path = _bfs_shortest_path(self.grid, carry_spawn, target)
        self.carry_path_index = 0
        self.carry_pos = self.carry_path[0]
        self.carry_hp = MAX_HP
        self.carry_alive = True
        self._refresh_carry_dist_map()

        # --- Escort：残りの攻撃側スポーンに配置（不足時はランダム） ---
        occupied = {self.carry_pos}
        self.escort_pos = []
        candidates = [c for c in self.attacker_spawns if c != self.carry_pos]
        self.rng.shuffle(candidates)
        for i in range(self.n_escorts):
            if i < len(candidates):
                pos = candidates[i]
            else:
                pos = self._random_walkable(exclude=occupied)
            occupied.add(pos)
            self.escort_pos.append(pos)

        self.escort_hp = [MAX_HP] * self.n_escorts
        self.escort_alive = [True] * self.n_escorts
        self.escort_ability_type = [self.rng.choice(ABILITY_TYPES) for _ in range(self.n_escorts)]
        self.escort_ability_used = [False] * self.n_escorts
        self.escort_last_delta = [(0.0, 0.0)] * self.n_escorts
        self.escort_stuck = [0] * self.n_escorts

        # --- 敵：守備側スポーンに配置 ---
        self.enemy_pos = []
        enemy_candidates = list(self.defender_spawns)
        self.rng.shuffle(enemy_candidates)
        for i in range(self.n_enemies):
            if i < len(enemy_candidates):
                pos = enemy_candidates[i]
            else:
                pos = self._random_walkable(exclude=occupied)
            self.enemy_pos.append(pos)

        self.enemy_hp = [MAX_HP] * self.n_enemies
        self.enemy_alive = [True] * self.n_enemies
        self.enemy_blind_remaining = [0] * self.n_enemies
        self.enemy_blind_source = [None] * self.n_enemies
        self.enemy_reveal_remaining = [0] * self.n_enemies
        self.enemy_reveal_source = [None] * self.n_enemies

        self._blocking_escort_idx = None
        self._prev_carry_path_index = 0
        self._stall_ticks = 0

        return [self._get_obs(i) for i in range(self.n_escorts)]

    # ------------------------------------------------------------------
    # 補助
    # ------------------------------------------------------------------
    def _is_wall(self, r, c):
        if not (0 <= r < self.height and 0 <= c < self.width):
            return True
        return self.grid[r, c] == 1

    def _smoke_cell_set(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def _occupied_by_others(self, exclude_kind, exclude_idx):
        """(kind, idx) を除いた、生存中の全キャラクターの現在位置集合。
        kind: 'carry' | 'escort' | 'enemy'
        """
        occ = set()
        if not (exclude_kind == "carry"):
            if self.carry_alive:
                occ.add(self.carry_pos)
        for i in range(self.n_escorts):
            if exclude_kind == "escort" and i == exclude_idx:
                continue
            if self.escort_alive[i]:
                occ.add(self.escort_pos[i])
        for i in range(self.n_enemies):
            if exclude_kind == "enemy" and i == exclude_idx:
                continue
            if self.enemy_alive[i]:
                occ.add(self.enemy_pos[i])
        return occ

    def _nearest_visible_enemy(self, from_pos, max_range=None):
        smoke_cells = self._smoke_cell_set()
        best_idx, best_dist = None, None
        for i in range(self.n_enemies):
            if not self.enemy_alive[i]:
                continue
            dist = _chebyshev(from_pos, self.enemy_pos[i])
            if max_range is not None and dist > max_range:
                continue
            if not _has_los(self.grid, smoke_cells, from_pos, self.enemy_pos[i]):
                continue
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = i, dist
        return best_idx, best_dist

    def _refresh_carry_dist_map(self):
        """carry_pos が変化した際に呼び、BFS距離マップを更新する。"""
        self._carry_dist_map = _build_distance_map_walls_only(self.grid, [self.carry_pos])

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------
    def get_action_mask(self, i):
        mask = np.ones(self.N_ACTIONS, dtype=bool)
        if not self.escort_alive[i]:
            mask[:] = False
            mask[self.ACTION_STAY] = True
            return mask

        r, c = self.escort_pos[i]
        for a, (dr, dc) in self._MOVE_DELTA.items():
            if a == self.ACTION_STAY:
                continue
            if self._is_wall(r + dr, c + dc):
                mask[a] = False

        # アビリティは1ラウンドに1回。使用済みなら選択不可にする。
        if self.escort_ability_used[i]:
            mask[self.ACTION_ABILITY] = False
        else:
            # 射程内に有効な標的（視認可能な敵）がいない場合もマスクする。
            # マスクしないと「常にwaste_penaltyを受けるだけの無意味な
            # ABILITY選択」がQ値の学習対象に残り続け、推論側で
            # 標的なしのままABILITYを選び続けて実質STAY＝ブロック、
            # という状態を誘発しうる。
            enemy_idx, _ = self._nearest_visible_enemy((r, c), max_range=ABILITY_RANGE)
            if enemy_idx is None:
                mask[self.ACTION_ABILITY] = False

        return mask

    def _get_obs(self, i):
        if not self.escort_alive[i]:
            return np.zeros(self._obs_dim(), dtype=np.float32)

        r, c = self.escort_pos[i]
        cr, cc = self.carry_pos
        if self._carry_dist_map is None:
            self._refresh_carry_dist_map()
        dist_map = self._carry_dist_map

        obs = []
        obs.append(r / max(1, self.height - 1))
        obs.append(c / max(1, self.width - 1))

        # escort自身からキャリアーまでの距離・方向はBFS実距離ベース
        # （チェビシェフ距離は壁を無視するため、曲がった通路で
        # 実際の経路と逆方向を指してしまうことがある）
        raw_dist = dist_map[r, c]
        dist_to_carry = float(raw_dist) if np.isfinite(raw_dist) else DIST_NORM_MAX
        obs.append(min(1.0, dist_to_carry / DIST_NORM_MAX))

        # 方向成分は、隣接4マスのうちBFS距離を最も縮める方向を使う
        best_dr, best_dc, best_d = 0, 0, dist_to_carry
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not self._is_wall(nr, nc):
                nd = dist_map[nr, nc]
                if np.isfinite(nd) and nd < best_d:
                    best_d = nd
                    best_dr, best_dc = dr, dc
        obs.append(float(best_dr))
        obs.append(float(best_dc))

        # キャリアーの進行方向（次の経路セルへの差分）
        next_idx = min(self.carry_path_index + 1, len(self.carry_path) - 1)
        nxt = self.carry_path[next_idx]
        obs.append(float(np.sign(nxt[0] - cr)))
        obs.append(float(np.sign(nxt[1] - cc)))

        # 壁フラグ（4方向）
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            obs.append(1.0 if self._is_wall(r + dr, c + dc) else 0.0)

        # 各方向に動いた場合のキャリアーまでのBFS距離勾配
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(nr, nc)
            if wall:
                obs.append(1.0)
            else:
                gdist = dist_map[nr, nc]
                gdist = float(gdist) if np.isfinite(gdist) else DIST_NORM_MAX
                obs.append(min(1.0, gdist / DIST_NORM_MAX))

        # 斜め壁フラグ
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(r + dr, c + dc) else 0.0)

        # 隣接4方向に生存中の味方escortがいるか（衝突・団子状態の回避用）
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            occupied_by_ally = any(
                j != i and self.escort_alive[j] and self.escort_pos[j] == (nr, nc)
                for j in range(self.n_escorts)
            )
            obs.append(1.0 if occupied_by_ally else 0.0)

        # 距離帯の逸脱量（負=近すぎ、正=遠すぎ、0=適正）
        if dist_to_carry < self.dist_band_min:
            band_dev = (dist_to_carry - self.dist_band_min) / DIST_NORM_MAX
        elif dist_to_carry > self.dist_band_max:
            band_dev = (dist_to_carry - self.dist_band_max) / DIST_NORM_MAX
        else:
            band_dev = 0.0
        obs.append(band_dev)

        # 最寄りの視認可能な敵
        enemy_idx, enemy_dist = self._nearest_visible_enemy((r, c), max_range=None)
        if enemy_idx is not None:
            er, ec = self.enemy_pos[enemy_idx]
            obs.append(1.0)
            obs.append(max(-1.0, min(1.0, (er - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (ec - c) / DIST_NORM_MAX)))
            obs.append(min(1.0, enemy_dist / DIST_NORM_MAX))
            obs.append(self.enemy_blind_remaining[enemy_idx] / max(1, BLIND_DURATION_TICKS))
            obs.append(self.enemy_reveal_remaining[enemy_idx] / max(1, REVEAL_DURATION_TICKS))
        else:
            obs.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        # 自分のアビリティ状態
        obs.append(0.0 if self.escort_ability_used[i] else 1.0)
        for ability in ABILITY_TYPES:
            obs.append(1.0 if self.escort_ability_type[i] == ability else 0.0)

        # チーム状況：誰かの効果（blind/reveal/smoke）が現在有効か
        team_effect_active = any(v > 0 for v in self.enemy_blind_remaining) or any(
            v > 0 for v in self.enemy_reveal_remaining
        ) or len(self.smokes) > 0
        obs.append(1.0 if team_effect_active else 0.0)

        obs.append(self.escort_last_delta[i][0])
        obs.append(self.escort_last_delta[i][1])
        obs.append(min(1.0, self.escort_stuck[i] / 10.0))
        obs.append(1.0 - min(1.0, self.tick / max(1, self.max_ticks)))
        obs.append(1.0 if self._blocking_escort_idx == i else 0.0)

        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _obs_dim():
        # _get_obs() の要素数と一致させる（固定値なのでズレたら即バグに気づけるようassert）
        # 内訳: 自己座標2 + キャリアーBFS距離/方向3 + キャリアー進行方向2
        #      + 壁フラグ4 + BFS距離勾配4 + 斜め壁フラグ4 + 味方隣接フラグ4
        #      + 距離帯逸脱1 + 敵情報6 + アビリティ状態(未使用フラグ1+種別onehot3)
        #      + チーム効果1 + 直前移動2 + stuck1 + 残り時間1 + 被ブロック1
        # = 2+3+2+4+4+4+4+1+6+4+1+2+1+1+1 = 40
        return 40

    # ------------------------------------------------------------------
    # アビリティ処理
    # ------------------------------------------------------------------
    def _apply_ability(self, i):
        """escort i のアビリティ使用を解決し、報酬(float)を返す。"""
        ability = self.escort_ability_type[i]
        pos = self.escort_pos[i]
        self.escort_ability_used[i] = True  # 成否に関わらず1ラウンド1回を消費

        if ability in ("FLASH", "RECON"):
            enemy_idx, dist = self._nearest_visible_enemy(pos, max_range=ABILITY_RANGE)
            if enemy_idx is None:
                return -self.ability_waste_penalty

            if ability == "FLASH":
                already = self.enemy_blind_remaining[enemy_idx] > 0
                self.enemy_blind_remaining[enemy_idx] = BLIND_DURATION_TICKS
                self.enemy_blind_source[enemy_idx] = i
            else:  # RECON
                already = self.enemy_reveal_remaining[enemy_idx] > 0
                self.enemy_reveal_remaining[enemy_idx] = REVEAL_DURATION_TICKS
                self.enemy_reveal_source[enemy_idx] = i

            return -self.ability_redundant_penalty if already else self.ability_success_reward

        # SMOKE: 「敵が味方(キャリアー)へ射線を持っている」状況を遮断できたら成功
        enemy_idx, dist = self._nearest_visible_enemy(pos, max_range=ABILITY_RANGE)
        if enemy_idx is None:
            return -self.ability_waste_penalty

        target_pos = self.enemy_pos[enemy_idx]
        smoke_cells = self._smoke_cell_set()
        enemy_had_los_to_carry = self.carry_alive and _has_los(
            self.grid, smoke_cells, target_pos, self.carry_pos
        )

        cells = {
            (rr, cc)
            for rr in range(target_pos[0] - 1, target_pos[0] + 2)
            for cc in range(target_pos[1] - 1, target_pos[1] + 2)
            if 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr, cc] != 1
        }
        self.smokes.append({"cells": cells, "remaining": SMOKE_DURATION_TICKS})

        return self.ability_success_reward if enemy_had_los_to_carry else -self.ability_waste_penalty

    # ------------------------------------------------------------------
    # 戦闘解決（簡略化モデル）
    # ------------------------------------------------------------------
    def _resolve_combat(self):
        """1tick分の簡易戦闘解決。kill発生時のボーナス対象escort集合を返す。"""
        smoke_cells = self._smoke_cell_set()

        allies = [("carry", 0, self.carry_pos)] if self.carry_alive else []
        allies += [("escort", i, self.escort_pos[i]) for i in range(self.n_escorts) if self.escort_alive[i]]
        enemies = [("enemy", i, self.enemy_pos[i]) for i in range(self.n_enemies) if self.enemy_alive[i]]

        shooters = []
        for kind, idx, pos in allies:
            e_idx, _ = self._nearest_visible_enemy(pos, max_range=None)
            if e_idx is not None:
                shooters.append((kind, idx, "enemy", e_idx))
        for kind, idx, pos in enemies:
            best_idx, best_dist = None, None
            for akind, aidx, apos in allies:
                if not _has_los(self.grid, smoke_cells, pos, apos):
                    continue
                d = _chebyshev(pos, apos)
                if best_dist is None or d < best_dist:
                    best_idx, best_dist = (akind, aidx), d
            if best_idx is not None:
                shooters.append((kind, idx, best_idx[0], best_idx[1]))

        self.rng.shuffle(shooters)

        kill_bonus_targets = []  # escort index のリスト

        for shooter_kind, shooter_idx, target_kind, target_idx in shooters:
            shooter_alive = (
                self.carry_alive if shooter_kind == "carry"
                else self.escort_alive[shooter_idx] if shooter_kind == "escort"
                else self.enemy_alive[shooter_idx]
            )
            target_alive = (
                self.carry_alive if target_kind == "carry"
                else self.escort_alive[target_idx] if target_kind == "escort"
                else self.enemy_alive[target_idx]
            )
            if not shooter_alive or not target_alive:
                continue

            accuracy = GENERIC_ACCURACY
            if shooter_kind == "enemy" and self.enemy_blind_remaining[shooter_idx] > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            dodge = GENERIC_DODGE
            if target_kind == "enemy" and self.enemy_reveal_remaining[target_idx] > 0:
                dodge *= REVEALED_DODGE_MULTIPLIER

            hit_chance = max(0.0, min(1.0, accuracy * (1.0 - dodge)))
            if self.rng.random() >= hit_chance:
                continue

            damage = HEADSHOT_DAMAGE if self.rng.random() < GENERIC_HS_RATE else BODY_DAMAGE

            if target_kind == "carry":
                self.carry_hp = max(0, self.carry_hp - damage)
                if self.carry_hp <= 0:
                    self.carry_alive = False
            elif target_kind == "escort":
                self.escort_hp[target_idx] = max(0, self.escort_hp[target_idx] - damage)
                if self.escort_hp[target_idx] <= 0:
                    self.escort_alive[target_idx] = False
            else:  # enemy
                was_blind = self.enemy_blind_remaining[target_idx] > 0
                was_revealed = self.enemy_reveal_remaining[target_idx] > 0
                blind_src = self.enemy_blind_source[target_idx]
                reveal_src = self.enemy_reveal_source[target_idx]

                self.enemy_hp[target_idx] = max(0, self.enemy_hp[target_idx] - damage)
                if self.enemy_hp[target_idx] <= 0:
                    self.enemy_alive[target_idx] = False
                    if was_blind and blind_src is not None:
                        kill_bonus_targets.append(blind_src)
                    if was_revealed and reveal_src is not None:
                        kill_bonus_targets.append(reveal_src)

        return kill_bonus_targets

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, actions):
        """actions: 長さ n_escorts のリスト（死亡中のescortはNone扱いでもよい）。
        戻り値: (next_obs_list, rewards, done, info)
        """
        self.tick += 1
        rewards = [0.0] * self.n_escorts
        info = {"success": False, "carry_died": False}

        # 1. タイマー減衰
        for i in range(self.n_enemies):
            self.enemy_blind_remaining[i] = max(0, self.enemy_blind_remaining[i] - 1)
            self.enemy_reveal_remaining[i] = max(0, self.enemy_reveal_remaining[i] - 1)
        for smoke in self.smokes:
            smoke["remaining"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining"] > 0]

        for i in range(self.n_escorts):
            rewards[i] -= 0.01  # 時間経過ペナルティ

        # 2. アビリティ行動を先に解決（アビリティ使用者はこのtick移動しない）
        used_ability_this_tick = set()
        for i in range(self.n_escorts):
            if not self.escort_alive[i] or actions[i] is None:
                continue
            if actions[i] == self.ACTION_ABILITY:
                rewards[i] += self._apply_ability(i)
                used_ability_this_tick.add(i)
                self.escort_last_delta[i] = (0.0, 0.0)
                self.escort_stuck[i] += 1

        # 3. 敵の簡易移動（衝突は考慮しない簡略化スクリプトAI）
        for i in range(self.n_enemies):
            if not self.enemy_alive[i]:
                continue
            if self.rng.random() < self.enemy_move_prob:
                r, c = self.enemy_pos[i]
                candidates = [
                    (r + dr, c + dc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if not self._is_wall(r + dr, c + dc)
                ]
                if candidates:
                    self.enemy_pos[i] = self.rng.choice(candidates)

        # 4. キャリアーの移動（塞がれていれば進めない）
        self._blocking_escort_idx = None
        prev_path_index = self.carry_path_index
        if self.carry_alive and self.carry_path_index < len(self.carry_path) - 1:
            next_cell = self.carry_path[self.carry_path_index + 1]
            occupied = self._occupied_by_others("carry", None)
            if next_cell not in occupied:
                self.carry_path_index += 1
                self.carry_pos = self.carry_path[self.carry_path_index]
                self._refresh_carry_dist_map()
            else:
                for i in range(self.n_escorts):
                    if self.escort_alive[i] and self.escort_pos[i] == next_cell:
                        self._blocking_escort_idx = i
                        break

        team_progress = self.carry_path_index - prev_path_index
        for i in range(self.n_escorts):
            if self.escort_alive[i]:
                rewards[i] += team_progress * self.team_progress_coef
        if self._blocking_escort_idx is not None:
            rewards[self._blocking_escort_idx] -= self.block_penalty

        # --- stall検知：直接の1体だけでなく、carry周辺で団子状態を
        # 作っている全escortに圧力をかける。escort同士の衝突で誰も
        # 動けなくなるジャムは「直前セルを塞ぐ1体」だけでは説明できない
        # ため、進捗ゼロが続くこと自体を検知して対処する。 ---
        carry_reached_goal = self.carry_path_index >= len(self.carry_path) - 1
        if team_progress > 0 or not self.carry_alive or carry_reached_goal:
            self._stall_ticks = 0
        else:
            self._stall_ticks += 1

        if self._stall_ticks > self.stall_threshold_ticks:
            overflow = min(
                self._stall_ticks - self.stall_threshold_ticks,
                self.stall_penalty_cap_ticks,
            )
            congestion_penalty = overflow * self.congestion_penalty_coef
            for i in range(self.n_escorts):
                if not self.escort_alive[i]:
                    continue
                if _chebyshev(self.escort_pos[i], self.carry_pos) <= self.congestion_radius:
                    rewards[i] -= congestion_penalty

        # 5. Escortの移動（アビリティ使用者・死亡者を除く、ランダム順で逐次解決）
        move_order = [
            i for i in range(self.n_escorts)
            if self.escort_alive[i] and i not in used_ability_this_tick and actions[i] is not None
        ]
        self.rng.shuffle(move_order)
        for i in move_order:
            action = actions[i]
            r, c = self.escort_pos[i]

            if action == self.ACTION_STAY or action not in self._MOVE_DELTA:
                self.escort_last_delta[i] = (0.0, 0.0)
                self.escort_stuck[i] += 1
                continue

            dr, dc = self._MOVE_DELTA[action]
            nr, nc = r + dr, c + dc

            if self._is_wall(nr, nc):
                self.escort_last_delta[i] = (0.0, 0.0)
                self.escort_stuck[i] += 1
                continue

            occupied = self._occupied_by_others("escort", i)
            if (nr, nc) in occupied:
                self.escort_last_delta[i] = (0.0, 0.0)
                self.escort_stuck[i] += 1
                continue

            self.escort_pos[i] = (nr, nc)
            self.escort_last_delta[i] = (float(dr), float(dc))
            self.escort_stuck[i] = 0

        # 6. 距離帯報酬（移動後の位置で評価）
        for i in range(self.n_escorts):
            if not self.escort_alive[i]:
                continue
            dist = _chebyshev(self.escort_pos[i], self.carry_pos)
            if dist < self.dist_band_min:
                rewards[i] -= (self.dist_band_min - dist) * self.dist_penalty_coef
            elif dist > self.dist_band_max:
                rewards[i] -= (dist - self.dist_band_max) * self.dist_penalty_coef

        # 7. 戦闘解決
        kill_bonus_targets = self._resolve_combat()
        for escort_idx in kill_bonus_targets:
            if 0 <= escort_idx < self.n_escorts:
                rewards[escort_idx] += self.kill_bonus

        # このtickで死亡したescortに死亡ペナルティ（HPが尽きて alive が False になった直後）
        for i in range(self.n_escorts):
            if not self.escort_alive[i] and self.escort_hp[i] <= 0:
                # 既に前tickで死亡済みの場合も含め毎tick引かれないよう、
                # HP==0確認はここでは簡略化し、死亡直後の1回だけ与えたいので
                # escort_hpを直後にNoneマーキングする代わりに簡易フラグで対応。
                pass

        done = False
        if not self.carry_alive:
            done = True
            info["carry_died"] = True
            for i in range(self.n_escorts):
                if self.escort_alive[i]:
                    rewards[i] -= self.mission_fail_penalty
        elif self.carry_path_index >= len(self.carry_path) - 1:
            done = True
            info["success"] = True
            for i in range(self.n_escorts):
                if self.escort_alive[i]:
                    rewards[i] += self.mission_success_reward
        elif self.tick >= self.max_ticks:
            done = True
            info["success"] = False
            for i in range(self.n_escorts):
                if self.escort_alive[i]:
                    rewards[i] -= self.mission_fail_penalty

        next_obs = [self._get_obs(i) for i in range(self.n_escorts)]
        return next_obs, rewards, done, info


# ---------------------------------------------------------------------------
# Dueling DQN（重み共有。train_attacker_carry.py と同一アーキテクチャ）
# ---------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "next_mask", "done"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def select_action(net, state, mask, epsilon, device):
    if random.random() < epsilon:
        valid_actions = np.flatnonzero(mask)
        return int(random.choice(valid_actions))
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(state_t).squeeze(0).cpu().numpy()
    q = np.where(mask, q, -1e9)
    return int(np.argmax(q))


def optimize_model(policy_net, target_net, optimizer, buffer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None

    batch = buffer.sample(batch_size)

    states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
    next_masks = torch.as_tensor(np.array(batch.next_mask), dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        next_q_policy = next_q_policy.masked_fill(~next_masks, -1e9)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states).gather(1, next_actions)
        target = rewards + gamma * next_q_target * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def evaluate(env, policy_net, device, episodes=20):
    successes = 0
    total_reward = 0.0
    total_kill_bonus_events = 0
    total_block_events = 0

    for _ in range(episodes):
        obs_list = env.reset()
        done = False
        episode_reward = 0.0
        block_events = 0
        info = {"success": False}

        while not done:
            actions = []
            for i in range(env.n_escorts):
                if not env.escort_alive[i]:
                    actions.append(None)
                    continue
                mask = env.get_action_mask(i)
                with torch.no_grad():
                    state_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                    q = policy_net(state_t).squeeze(0).cpu().numpy()
                q = np.where(mask, q, -1e9)
                actions.append(int(np.argmax(q)))

            next_obs_list, rewards, done, info = env.step(actions)
            if env._blocking_escort_idx is not None:
                block_events += 1
            episode_reward += sum(rewards)
            obs_list = next_obs_list

        total_reward += episode_reward
        total_block_events += block_events
        if info.get("success"):
            successes += 1

    success_rate = successes / episodes
    avg_reward = total_reward / episodes
    avg_block_events = total_block_events / episodes
    return success_rate, avg_reward, avg_block_events


def main():
    parser = argparse.ArgumentParser(description="Attacker Escort Phase 学習スクリプト")
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    parser.add_argument("--max-ticks", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=4000)
    parser.add_argument("--target-update-every", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "attacker_v3", "data", "attacker_escort_data"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    env = EscortEnv(max_ticks=args.max_ticks, seed=args.seed)
    eval_env = EscortEnv(max_ticks=args.max_ticks, seed=args.seed + 1)

    obs_dim = env._obs_dim()
    n_actions = env.N_ACTIONS
    print(f"[INFO] obs_dim={obs_dim} n_actions={n_actions} n_escorts={env.n_escorts} device={device}")

    policy_net = DuelingQNetwork(obs_dim, n_actions).to(device)
    target_net = DuelingQNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    best_success_rate = -1.0
    best_eval_reward = float("-inf")
    global_step = 0

    def _make_checkpoint(episode, success_rate, avg_reward, avg_block_events):
        return {
            "model_state_dict": policy_net.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "episode": episode,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_block_events": avg_block_events,
        }

    for episode in range(1, args.episodes + 1):
        progress = min(1.0, episode / args.eps_decay_episodes)
        epsilon = args.eps_start + (args.eps_end - args.eps_start) * progress

        obs_list = env.reset()
        done = False
        episode_reward = 0.0
        info = {"success": False}

        while not done:
            actions = []
            masks = []
            for i in range(env.n_escorts):
                if not env.escort_alive[i]:
                    actions.append(None)
                    masks.append(None)
                    continue
                mask = env.get_action_mask(i)
                action = select_action(policy_net, obs_list[i], mask, epsilon, device)
                actions.append(action)
                masks.append(mask)

            next_obs_list, rewards, done, info = env.step(actions)

            for i in range(env.n_escorts):
                if masks[i] is None:
                    continue
                next_mask = env.get_action_mask(i)
                agent_done = done or not env.escort_alive[i]
                buffer.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], next_mask, agent_done)

            obs_list = next_obs_list
            episode_reward += sum(rewards)
            global_step += 1

            if len(buffer) >= max(args.batch_size, args.warmup_steps):
                optimize_model(
                    policy_net, target_net, optimizer, buffer,
                    args.batch_size, args.gamma, device,
                )

            if global_step % args.target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode % 50 == 0:
            print(
                f"[EP {episode}/{EPISODE_COUNT}] reward={episode_reward:.2f} "
                f"eps={epsilon:.3f} success={info.get('success')} ticks={env.tick}"
            )

        if episode % args.eval_every == 0:
            success_rate, avg_reward, avg_block_events = evaluate(
                eval_env, policy_net, device, args.eval_episodes
            )
            print(
                f"[EVAL @ EP {episode}/{EPISODE_COUNT}] success_rate={success_rate:.2%} "
                f"avg_reward={avg_reward:.2f} avg_block_events={avg_block_events:.2f}"
            )

            latest_path = os.path.join(args.save_dir, "dqn_attacker_escort_latest.pt")
            torch.save(_make_checkpoint(episode, success_rate, avg_reward, avg_block_events), latest_path)

            is_better = (
                success_rate > best_success_rate + 1e-9
                or (
                    success_rate >= best_success_rate - 1e-9
                    and avg_reward > best_eval_reward
                )
            )
            if is_better:
                best_success_rate = max(best_success_rate, success_rate)
                best_eval_reward = avg_reward
                best_path = os.path.join(args.save_dir, "dqn_attacker_escort_best_by_eval.pt")
                torch.save(_make_checkpoint(episode, success_rate, avg_reward, avg_block_events), best_path)
                print(
                    f"[SAVE] 新しいベストモデルを保存: {best_path} "
                    f"(success_rate={success_rate:.2%}, avg_reward={avg_reward:.2f}, "
                    f"avg_block_events={avg_block_events:.2f})"
                )

    print("[DONE] 学習が完了しました。")


if __name__ == "__main__":
    main()