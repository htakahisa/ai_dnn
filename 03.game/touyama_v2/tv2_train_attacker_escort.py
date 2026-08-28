"""touyama_v2/train_attacker_escort.py

固定チーム(Tortlilyan/いぐるん/ろびぃな/夢の街/えんぺん)専用の
Attacker Carry Phase「escort(護衛)」学習スクリプト(Dueling DQN)。

【目的】
キャリアー(スパイク運搬役)の周囲2〜7マス程度を維持しつつ、キャリアーと
プラントサイトの間の経路を塞がないように動き、敵を見つけたらアビリティ
(FLASH/RECON/SMOKE)を適切なタイミングで使用できるようになるまで学習する。

【汎用版(train_attacker_escort.py)からの主な変更点】
1. キャラクター別の固定ステータス・固定アビリティ
   汎用版はGENERIC_ACCURACY等の汎用値とABILITY_TYPESのランダム割当を
   使っていたが、本版はcharacter_stats_touyama.py + コンボ(ふわんだりぃず)
   + タイガーパッシブで確定した実効ステータスを、キャリアー・エスコート
   それぞれに適用する(train_attacker_carry.py / train_attacker_guard.py と
   同一の_compute_touyama_effective_stats()を使用)。

2. HUNT(タイガー/Tortlilyan)対応
   アビリティonehotにHUNTを追加(3種→4種、OBS_DIM=40→41)。
   HUNTはアビリティチャージ0として初期化し、常にアビリティ行動を
   マスクする(タイガーはパッシブのみで使用アビリティを持たないため)。

3. キャリアー役の可変化(ハンドオフ想定)
   train_attacker_carry.pyと同じHANDOFF_AUGMENT_PROBパターンを導入。
   通常は「ろびぃな」がキャリアーだが、一定確率で他のロースター
   メンバーがキャリアーとして開始し、残り4人がエスコートを担当する
   (retrieveフェーズからの引き継ぎ・キャリア交代に一般化するため)。

4. 戦闘解決の精度向上
   battle_logic.pyと同じ「反応速度降順での逐次解決」「移動中射撃精度
   低下(MOVING_ACCURACY)」「移動中被弾しやすさ(MOVING_TARGET_HIT_MULTIPLIER)」
   を追加。汎用版にはこれらがなく、キャラクター間の反応速度差が
   全く反映されない設計だった。

5. game_core.pyの定数を正式にimport
   汎用版はゲーム定数(MAX_HP等)をこのファイル内に再定義していたが、
   他のtouyama_v2ファイルの慣習(定数専用ファイルとして参照)に合わせた。

【design方針(既存ルールの継承)】
- 完全に自己完結: run_game.py / controllers.py / battle_logic.py /
  abilities_los.py は一切importしない。必要なロジックはすべてこのファイル
  内に複製する。map_data.py / character_stats_touyama.py / game_core.py は
  定数専用ファイルとして参照する(import制限の対象外)。
- run_game.py / controllers.py は変更しない。

【マルチエージェント方式(汎用版から継承)】
4体のescortは全員「同じDueling DQNの重み」を共有して行動する
(train_attacker_guard.py系と同じ「パラメータ共有」方式)。観測は
エージェントごとに自分中心の相対座標系で構築するため、同じネットワークを
どのescortにも使い回せる。

【キャリアーはスクリプトAI】
キャリアーは学習対象ではない。エピソード開始時に決めたプラントサイトへ、
固定のBFS最短経路を1tickごとに1マスずつ進む。経路上の次のマスが
(escortまたは敵に)占有されている場合、実際のゲーム(battle_logic.py
move_character)と同じく「移動できずその場に留まる」。この「キャリアーが
実際に進んだ距離」をescort全員の共有報酬にすることで、道を塞ぐと損、
というシグナルをハードコードせずに学習させる。

【敵はスクリプトAI】
本格的なDefenderAIの学習は別スクリプトの範囲。ここでは「視界に入る・
アビリティで状態異常にできる・撃ち合いが発生する」という最低限の
相互作用だけを簡易シミュレーションする(N_ENEMIES=2、簡易ランダム徘徊)。

保存先: touyama_v2/data/attacker_escort_touyama_data/
チェックポイントは{"model_state_dict","obs_dim","n_actions","episode",
"success_rate","avg_reward","avg_block_events","roster_order",
"spike_holder_default"} を含むdict形式で保存する。
"""

import argparse
import math
import os
import random
import sys
from collections import deque, namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR

from game_core import (
    MAX_HP,
    HEADSHOT_DAMAGE,
    BODY_DAMAGE,
    MOVING_ACCURACY,
    MOVING_TARGET_HIT_MULTIPLIER,
    BLIND_ACCURACY_MULTIPLIER,
    REVEALED_DODGE_MULTIPLIER,
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
    SMOKE_DURATION_TICKS,
    FACING_VECTORS,
    SHOOTING_SITE_DIGREE,
)

from tv2_character_stats_touyama import CHARACTER_TABLE as TOUYAMA_STATS_TABLE
import tv2_common_rl
from tv2_common_rl import DEVICE, DuelingQNet, ReplayBuffer, select_action, optimize_double_dqn_step
from tv2_common_attacker import (
    TOUYAMA_ROSTER_ORDER,
    TOUYAMA_SPIKE_HOLDER,
    DEFAULT_ACCURACY,
    DEFAULT_DODGE,
    DEFAULT_HS_RATE,
    DEFAULT_REACTION,
    compute_touyama_effective_stats,
    print_effective_stats,
)

EPISODE_COUNT = 5000
EVAL_MIN_EPISODE = EPISODE_COUNT * 0.7

# ---------------------------------------------------------------------------
# マップ上の意味付け(map_data.py準拠)
# ---------------------------------------------------------------------------
SITE_CELL_VALUE = 2
ATTACKER_SPAWN_VALUE = 3
DEFENDER_SPAWN_VALUE = 4

ABILITY_TYPES = ("FLASH", "RECON", "SMOKE", "HUNT")  # HUNTはアビリティ行動を持たない(常にマスク)
ABILITY_RANGE = 6  # アビリティが届く最大距離(チェビシェフ距離で判定)

FACING_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

N_ESCORTS = 4  # touyama固定チーム5人からキャリアー1人を除いた人数
N_ENEMIES = 2  # 敵はこのフェーズでは簡易スクリプトAI(本格学習は別スクリプトの範囲)

DIST_BAND_MIN = 2
DIST_BAND_MAX = 7
DIST_NORM_MAX = 15.0  # 観測正規化用の上限距離

STALL_THRESHOLD_TICKS = 3       # これを超えて無進捗が続いたら混雑ペナルティ開始
STALL_PENALTY_CAP_TICKS = 10    # ペナルティの伸び幅の上限(無限にエスカレートさせない)
CONGESTION_RADIUS = 2           # carryからこの距離以内のescortを「渋滞に関与」とみなす

AHEAD_BLOCK_PENALTY_COEF = 0.2  # 「サイト-エスコート-キャリアー」順で、退避可能なのに
                                 # キャリアーの最短経路を塞いでいる場合の追加ペナルティ係数
PATH_EQUALITY_TOL = 0.5         # BFS距離の等価判定の許容誤差(float32のBFS距離用)

ESCORT_HOLD_TICKS = 3           # 開始直後この tick 数だけ移動アクションをマスクし、
                                 # キャリアーが先に動き出すまで待たせる(初動ブロック対策)

HANDOFF_AUGMENT_PROB = 0.25  # 一定確率でキャリアー役をろびぃな以外から選ぶ(train_attacker_carry.pyと同一方針)

FACING_ALIGN_WEIGHT_ESCORT = 0.01  # 敵不可視時のみ有効。進行方向を向くほど+、背を向けるほど-(弱いshaping)

SIGHTING_STALENESS_CAP = 20        # チーム共有の目撃情報を保持する最大tick数(carry/guardと同一方針)
TEAM_SIGHTING_ALIGN_WEIGHT = 0.02  # 自分が直接視認していない時のみ有効。チーム共有の目撃位置を
                                    # 向くほど+、逆を向くほど-(弱いshaping。直接視認時は交戦報酬に委ねる)

BEST_MODEL_EPS_THRESHOLD = 0.15   # epsilonがこの値以下に下がるまではbest候補として扱わない
                                   # (探索が多い段階のたまたま良いevalで固定されるのを防ぐ)
BEST_MODEL_SMOOTH_WINDOW = 5      # 直近何回分のeval結果を平均してbest判定に使うか

# (common_attackerからimport済みのため削除)

TOUYAMA_EFFECTIVE_STATS = compute_touyama_effective_stats(TOUYAMA_STATS_TABLE)
print_effective_stats(TOUYAMA_EFFECTIVE_STATS, "Attacker/escort")


# ---------------------------------------------------------------------------
# 汎用ヘルパー: BFS最短経路・射線判定(abilities_los.py / controllers.py
# と同等のロジックをこのファイル内に複製)
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


def _has_los(grid, smoke_cells, p1, p2):
    """壁とスモークを考慮した射線判定(tv2_common_rl.has_los利用。引数順序が
    異なる点に注意: このファイルはgrid, smoke_cells, p1, p2の順)。"""
    return tv2_common_rl.has_los(grid, p1, p2, smoke_cells)

def _chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _facing_from_delta(dr, dc, fallback):
    """移動delta(dr, dc)から4方向facingを判定する(battle_logic._facing_from_deltaと同一ロジック)。"""
    dr, dc = int(dr), int(dc)
    if dr == 0 and dc == 0:
        return fallback
    if dr != 0:
        return "N" if dr < 0 else "S"
    return "W" if dc < 0 else "E"


def _facing_towards(from_pos, to_pos):
    """from_posからto_posへ最も近い8方向のfacingを返す(battle_logic._facing_towardsと同一ロジック)。"""
    dc = float(to_pos[1] - from_pos[1])
    dr = float(to_pos[0] - from_pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0:
        return None
    nx, ny = dc / dist, dr / dist
    best_dir, best_dot = None, -2.0
    for direction, (fx, fy) in FACING_VECTORS.items():
        dot = fx * nx + fy * ny
        if dot > best_dot:
            best_dot = dot
            best_dir = direction
    return best_dir


def _facing_angle_diff(facing, from_pos, to_pos):
    """facingと、from_pos→to_pos方向との角度差(度)を返す(battle_logic._facing_angle_diffと同一ロジック)。"""
    dc = float(to_pos[1] - from_pos[1])
    dr = float(to_pos[0] - from_pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0:
        return 0.0
    fx, fy = FACING_VECTORS[facing]
    dot = max(-1.0, min(1.0, (fx * dc + fy * dr) / dist))
    return math.degrees(math.acos(dot))


def _facing_accuracy_multiplier(facing, from_pos, to_pos):
    """正面100%～真横50%まで、角度差に応じて線形に精度を落とす(battle_logicと同一ロジック)。"""
    angle = _facing_angle_diff(facing, from_pos, to_pos)
    return 1.0 - min(SHOOTING_SITE_DIGREE, angle) / SHOOTING_SITE_DIGREE * 0.5


def _direction_alignment(facing, from_pos, to_pos):
    """facingが、from_pos→to_pos方向とどれだけ一致しているか(-1〜1、cosθ相当)。
    tv2_train_attacker_guard.pyの警戒ポイントshapingと同一の考え方。"""
    dc = float(to_pos[1] - from_pos[1])
    dr = float(to_pos[0] - from_pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0 or facing not in FACING_VECTORS:
        return 0.0
    fx, fy = FACING_VECTORS[facing]
    return (fx * dc + fy * dr) / dist


def _build_distance_map_walls_only(grid, source_cells):
    """指定座標群を始点とした、壁のみを障害物としたマルチソースBFS距離マップ。
    キャラクター同士の占有は考慮しない
    (推論側 learning_attacker_escort.py と同一ロジックにする想定)。"""
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
    """touyama_v2固定チームのキャリアー護衛4体(重み共有)を学習させる
    軽量マルチエージェント環境。"""

    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_ABILITY = range(6)
    BASE_N_ACTIONS = 6
    N_ACTIONS = BASE_N_ACTIONS * len(FACING_DIRS)
    _MOVE_DELTA = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
        ACTION_STAY: (0, 0),
    }

    @staticmethod
    def decode_action(action_idx):
        """action_idx = base_idx(0-5) * 8 + facing_idx(0-7)。
        base(上下左右/STAY/ABILITY)とfacing(N/NE/E/SE/S/SW/W/NW)は完全に独立
        (tv2_train_attacker_carry.pyのdecode_actionと同一規約)。"""
        idx = int(action_idx)
        base_idx, facing_idx = divmod(idx, len(FACING_DIRS))
        return base_idx, FACING_DIRS[facing_idx]

    def __init__(
        self,
        maze_str=NEW_MAZE_STR,
        max_ticks=100,
        n_escorts=N_ESCORTS,
        n_enemies=N_ENEMIES,
        dist_band_min=DIST_BAND_MIN,
        dist_band_max=DIST_BAND_MAX,
        team_progress_coef=1.0,
        block_penalty=1.0,
        dist_penalty_coef=0.08,
        potential_coef=0.15,  # 常時距離短縮shaping: バンド内でも近づけば得、離れれば損にする
        stall_threshold_ticks=STALL_THRESHOLD_TICKS,
        stall_penalty_cap_ticks=STALL_PENALTY_CAP_TICKS,
        congestion_radius=CONGESTION_RADIUS,
        congestion_penalty_coef=0.15,
        ahead_block_penalty_coef=AHEAD_BLOCK_PENALTY_COEF,
        ability_success_reward=2.0,
        ability_redundant_penalty=1.0,
        ability_waste_penalty=0.3,
        kill_bonus=5.0,
        death_penalty=3.0,
        mission_success_reward=5.0,
        mission_fail_penalty=5.0,
        enemy_move_prob=0.2,
        handoff_augment_prob=HANDOFF_AUGMENT_PROB,
        hold_ticks=ESCORT_HOLD_TICKS,
        seed=None,
    ):
        self.grid = tv2_common_rl.parse_grid(maze_str)
        self.height, self.width = self.grid.shape

        self.max_ticks = max_ticks
        self.n_escorts = n_escorts
        self.n_enemies = n_enemies
        self.dist_band_min = dist_band_min
        self.dist_band_max = dist_band_max
        self.team_progress_coef = team_progress_coef
        self.block_penalty = block_penalty
        self.dist_penalty_coef = dist_penalty_coef
        self.potential_coef = potential_coef
        self.stall_threshold_ticks = stall_threshold_ticks
        self.stall_penalty_cap_ticks = stall_penalty_cap_ticks
        self.congestion_radius = congestion_radius
        self.congestion_penalty_coef = congestion_penalty_coef
        self.ahead_block_penalty_coef = ahead_block_penalty_coef
        self.ability_success_reward = ability_success_reward
        self.ability_redundant_penalty = ability_redundant_penalty
        self.ability_waste_penalty = ability_waste_penalty
        self.kill_bonus = kill_bonus
        self.death_penalty = death_penalty
        self.mission_success_reward = mission_success_reward
        self.mission_fail_penalty = mission_fail_penalty
        self.enemy_move_prob = enemy_move_prob
        self.handoff_augment_prob = handoff_augment_prob
        self.hold_ticks = hold_ticks

        self.site_cells = list(zip(*np.where(self.grid == SITE_CELL_VALUE)))
        self.attacker_spawns = list(zip(*np.where(self.grid == ATTACKER_SPAWN_VALUE)))
        self.defender_spawns = list(zip(*np.where(self.grid == DEFENDER_SPAWN_VALUE)))
        self.walkable_cells = list(zip(*np.where(self.grid != 1)))

        self.rng = random.Random(seed)

        # ラウンド内状態(reset()で初期化)
        self.tick = 0
        self.carry_pos = (0, 0)
        self.carry_hp = MAX_HP
        self.carry_alive = True
        self.carry_name = TOUYAMA_SPIKE_HOLDER
        self.carry_accuracy = 0.0
        self.carry_dodge = 0.0
        self.carry_hs_rate = 0.0
        self.carry_reaction = 0.0
        self.carry_moved = False
        self.carry_facing = "N"
        self.carry_forced_facing_next_tick = None
        self.carry_facing_forced_this_tick = False
        self.carry_path = [(0, 0)]
        self.carry_path_index = 0

        self.escort_pos = []
        self.escort_hp = []
        self.escort_alive = []
        self.escort_name = []
        self.escort_ability_type = []
        self.escort_ability_used = []
        self.escort_accuracy = []
        self.escort_dodge = []
        self.escort_hs_rate = []
        self.escort_reaction = []
        self.escort_moved = []
        self.escort_facing = []
        self.escort_forced_facing_next_tick = []
        self.escort_facing_forced_this_tick = []
        self.escort_last_delta = []
        self.escort_stuck = []

        self.enemy_pos = []
        self.enemy_hp = []
        self.enemy_alive = []
        self.enemy_accuracy = []
        self.enemy_dodge = []
        self.enemy_hs_rate = []
        self.enemy_reaction = []
        self.enemy_moved = []
        self.enemy_facing = []
        self.enemy_forced_facing_next_tick = []
        self.enemy_facing_forced_this_tick = []
        self.enemy_blind_remaining = []
        self.enemy_blind_source = []
        self.enemy_reveal_remaining = []
        self.enemy_reveal_source = []

        self.smokes = []  # [{"cells": set, "remaining": int}]

        self._blocking_escort_idx = None  # このtickでキャリアーを塞いだescort index
        self.team_sighting = None  # {"pos": (r, c), "idx": int, "tick_ago": int} or None。
                                    # carrier+escort全体で共有する最新の敵目撃情報。
        self._carry_dist_map = None
        self._goal_dist_map = None  # 設置目標地点からの壁のみBFS距離マップ(reset()で1回だけ計算)
        self.goal_pos = None
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

    def _resolve_spawn_collision(self, pos, occupied):
        """スポーン候補が重複/壁だった場合に、BFSで最寄りの空きマスへ逃がす
        (tv2_common_rl.resolve_spawn_collision使用)。"""
        return tv2_common_rl.resolve_spawn_collision(self.grid, pos, occupied)

    def reset(self):
        self.tick = 0
        self.smokes = []

        # --- キャリアー役を決定(train_attacker_carry.pyと同一のハンドオフ方針) ---
        handoff = self.rng.random() < self.handoff_augment_prob
        if handoff:
            carrier_name = self.rng.choice(TOUYAMA_ROSTER_ORDER)
        else:
            carrier_name = TOUYAMA_SPIKE_HOLDER
        self.carry_name = carrier_name

        carrier_stats = TOUYAMA_EFFECTIVE_STATS[carrier_name]
        self.carry_accuracy = carrier_stats["accuracy"]
        self.carry_dodge = carrier_stats["dodge_rate"]
        self.carry_hs_rate = carrier_stats["hs_rate"]
        self.carry_reaction = carrier_stats["reaction"] + self.rng.uniform(-10, 10)

        # --- キャリアーとescort4人のスポーン位置を決定 ---
        occupied = set()
        if handoff:
            spawn_positions = self.rng.sample(self.walkable_cells, min(5, len(self.walkable_cells)))
        else:
            spawn_positions = list(self.attacker_spawns[:5]) if len(self.attacker_spawns) >= 5 else list(self.attacker_spawns)
            while len(spawn_positions) < 5:
                spawn_positions.append(self._random_walkable(exclude=occupied))

        carry_base_pos = spawn_positions[TOUYAMA_ROSTER_ORDER.index(carrier_name) % len(spawn_positions)]
        self.carry_pos = self._resolve_spawn_collision(carry_base_pos, occupied)
        occupied.add(self.carry_pos)

        # --- escort名(ロースター順、キャリアーを除いた4人) ---
        escort_names = [name for name in TOUYAMA_ROSTER_ORDER if name != carrier_name]

        self.escort_pos = []
        self.escort_name = []
        self.escort_ability_type = []
        self.escort_ability_used = []
        self.escort_accuracy = []
        self.escort_dodge = []
        self.escort_hs_rate = []
        self.escort_reaction = []
        for i, name in enumerate(escort_names):
            stats = TOUYAMA_EFFECTIVE_STATS[name]
            base_pos = spawn_positions[TOUYAMA_ROSTER_ORDER.index(name) % len(spawn_positions)]
            pos = self._resolve_spawn_collision(base_pos, occupied)
            occupied.add(pos)

            self.escort_pos.append(pos)
            self.escort_name.append(name)
            self.escort_ability_type.append(stats["ability"])
            self.escort_accuracy.append(stats["accuracy"])
            self.escort_dodge.append(stats["dodge_rate"])
            self.escort_hs_rate.append(stats["hs_rate"])
            self.escort_reaction.append(stats["reaction"] + self.rng.uniform(-10, 10))
            # HUNT(タイガー)はアビリティを持たないため、最初から「使用済み」扱いにして
            # get_action_mask()が常にACTION_ABILITYを弾くようにする。
            self.escort_ability_used.append(stats["ability"] == "HUNT")

        self.n_escorts = len(escort_names)
        self.escort_hp = [MAX_HP] * self.n_escorts
        self.escort_alive = [True] * self.n_escorts
        self.escort_moved = [False] * self.n_escorts
        self.escort_facing = ["N"] * self.n_escorts
        self.escort_forced_facing_next_tick = [None] * self.n_escorts
        self.escort_facing_forced_this_tick = [False] * self.n_escorts
        self.escort_last_delta = [(0.0, 0.0)] * self.n_escorts
        self.escort_stuck = [0] * self.n_escorts

        # --- キャリアー：スポーンからサイトのどれか1点へ固定経路 ---
        target = self.rng.choice(self.site_cells) if self.site_cells else self.carry_pos
        self.carry_path = _bfs_shortest_path(self.grid, self.carry_pos, target)
        self.carry_path_index = 0
        self.carry_hp = MAX_HP
        self.carry_alive = True
        self.carry_moved = False
        self.carry_facing = "N"
        self.carry_forced_facing_next_tick = None
        self.carry_facing_forced_this_tick = False
        self.goal_pos = target
        # goal(設置目標地点)は1ラウンド中固定なので、reset時に1回だけ計算する。
        self._goal_dist_map = _build_distance_map_walls_only(self.grid, [target])
        self._refresh_carry_dist_map()
        # 常時距離短縮shaping用: 各escortの直前tickでのcarryまでのBFS距離を記録しておく
        self.escort_prev_dist = [self._carry_bfs_dist(pos) for pos in self.escort_pos]

        # --- 敵：守備側スポーンに配置(当面ヒューリスティック) ---
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
        self.enemy_accuracy = [DEFAULT_ACCURACY] * self.n_enemies
        self.enemy_dodge = [DEFAULT_DODGE] * self.n_enemies
        self.enemy_hs_rate = [DEFAULT_HS_RATE] * self.n_enemies
        self.enemy_reaction = [DEFAULT_REACTION + self.rng.uniform(-10, 10) for _ in range(self.n_enemies)]
        self.enemy_moved = [False] * self.n_enemies
        self.enemy_facing = ["S"] * self.n_enemies
        self.enemy_forced_facing_next_tick = [None] * self.n_enemies
        self.enemy_facing_forced_this_tick = [False] * self.n_enemies
        self.enemy_blind_remaining = [0] * self.n_enemies
        self.enemy_blind_source = [None] * self.n_enemies
        self.enemy_reveal_remaining = [0] * self.n_enemies
        self.enemy_reveal_source = [None] * self.n_enemies

        self._blocking_escort_idx = None
        self._prev_carry_path_index = 0
        self._stall_ticks = 0

        self.team_sighting = None
        self._update_team_sighting()

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

    def _pos_of(self, kind, idx):
        if kind == "carry":
            return self.carry_pos
        if kind == "escort":
            return self.escort_pos[idx]
        return self.enemy_pos[idx]

    def _facing_of(self, kind, idx):
        if kind == "carry":
            return self.carry_facing
        if kind == "escort":
            return self.escort_facing[idx]
        return self.enemy_facing[idx]

    def _set_forced_facing(self, kind, idx, facing):
        """撃たれた側へ次tickの強制facingを設定する(battle_logicと同一方針)。"""
        if facing is None:
            return
        if kind == "carry":
            self.carry_forced_facing_next_tick = facing
        elif kind == "escort":
            self.escort_forced_facing_next_tick[idx] = facing
        else:
            self.enemy_forced_facing_next_tick[idx] = facing

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

    def _all_ally_positions(self):
        """carrier(生存時)+生存中escort全員の現在位置。チーム共有目撃判定に使う。"""
        positions = []
        if self.carry_alive:
            positions.append(self.carry_pos)
        for i in range(self.n_escorts):
            if self.escort_alive[i]:
                positions.append(self.escort_pos[i])
        return positions

    def _update_team_sighting(self):
        """carrier+escort全体の視認情報を統合し、チーム共有の目撃情報として1つ
        保持する(carry.pyのSightingMemory / guard.pyのGuardMemoryと同一方針)。
        誰か一人でも視認できていれば共有され、以後SIGHTING_STALENESS_CAP tickの
        間は視認が途切れても保持される。"""
        smoke_cells = self._smoke_cell_set()
        ally_positions = self._all_ally_positions()

        visible_indices = [
            i for i in range(self.n_enemies)
            if self.enemy_alive[i]
            and any(_has_los(self.grid, smoke_cells, apos, self.enemy_pos[i]) for apos in ally_positions)
        ]

        if visible_indices:
            tracked_idx = None
            if self.team_sighting is not None and self.team_sighting["idx"] in visible_indices:
                tracked_idx = self.team_sighting["idx"]
            if tracked_idx is None:
                tracked_idx = min(
                    visible_indices,
                    key=lambda i: min(
                        _chebyshev(apos, self.enemy_pos[i]) for apos in ally_positions
                    ) if ally_positions else 0,
                )
            self.team_sighting = {
                "pos": tuple(self.enemy_pos[tracked_idx]), "idx": tracked_idx, "tick_ago": 0,
            }
        elif self.team_sighting is not None:
            self.team_sighting["tick_ago"] += 1
            if self.team_sighting["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.team_sighting = None

    def _refresh_carry_dist_map(self):
        """carry_pos が変化した際に呼び、BFS距離マップを更新する。"""
        self._carry_dist_map = _build_distance_map_walls_only(self.grid, [self.carry_pos])

    def _carry_bfs_dist(self, pos):
        """posからcarry_posまでの壁のみBFS実距離。_get_obs()と同じ計算元
        (self._carry_dist_map)を使うことで、報酬とobsの距離定義を一致させる。
        到達不能な場合はグリッドサイズ相当の大きな値にフォールバックする。"""
        if self._carry_dist_map is None:
            return float(self.height + self.width)
        r, c = pos
        d = self._carry_dist_map[r, c]
        return float(d) if np.isfinite(d) else float(self.height + self.width)

    def _ahead_of_carry(self, pos):
        """posがキャリアーよりgoalに近いか(=キャリアーより前に出ているか)。"""
        if self._goal_dist_map is None:
            return False
        pr, pc = pos
        cr, cc = self.carry_pos
        pd, cd = self._goal_dist_map[pr, pc], self._goal_dist_map[cr, cc]
        if not (np.isfinite(pd) and np.isfinite(cd)):
            return False
        return pd < cd

    def _path_status(self, pos):
        """posが「キャリアーの最短経路上(のいずれか)」かどうか、および
        経路上でない隣接マス(退避先)が存在するかを判定する。
        carry_dist_map(始点=carry_pos)とgoal_dist_map(始点=goal)の和が
        carry→goalの最短距離と一致するセル = 最短経路上、という判定を使う。
        (特定の1本の経路だけでなく、同じ長さの全最短経路をカバーできる)
        """
        if self._carry_dist_map is None or self._goal_dist_map is None:
            return False, False
        r, c = pos
        cr, cc = self.carry_pos
        total = self._goal_dist_map[cr, cc]
        if not np.isfinite(total):
            return False, False

        def _on_path(rr, cc_):
            d1 = self._carry_dist_map[rr, cc_]
            d2 = self._goal_dist_map[rr, cc_]
            if not (np.isfinite(d1) and np.isfinite(d2)):
                return False
            return abs(d1 + d2 - total) < PATH_EQUALITY_TOL

        on_path = _on_path(r, c)
        escape_available = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self._is_wall(nr, nc):
                continue
            if not _on_path(nr, nc):
                escape_available = True
                break
        return on_path, escape_available

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------
    def get_action_mask(self, i):
        base_mask = np.ones(self.BASE_N_ACTIONS, dtype=bool)
        if not self.escort_alive[i]:
            base_mask[:] = False
            base_mask[self.ACTION_STAY] = True
            return np.repeat(base_mask, len(FACING_DIRS))

        r, c = self.escort_pos[i]
        for a, (dr, dc) in self._MOVE_DELTA.items():
            if a == self.ACTION_STAY:
                continue
            if self._is_wall(r + dr, c + dc):
                base_mask[a] = False

        # 開始直後hold_ticks分は移動を禁止し、キャリアーが先に動き出すまで待つ。
        # (STAYは常に許可のまま。アビリティは対象がいれば通常通り使用可能。
        #  向き(facing)は移動・アビリティとは無関係に常に自由選択できる)
        if self.tick < self.hold_ticks:
            for a in (self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT):
                base_mask[a] = False

        # アビリティは1ラウンドに1回。使用済み(HUNTは常時この扱い)なら選択不可にする。
        if self.escort_ability_used[i]:
            base_mask[self.ACTION_ABILITY] = False
        else:
            # 射程内に有効な標的(視認可能な敵)がいない場合もマスクする。
            enemy_idx, _ = self._nearest_visible_enemy((r, c), max_range=ABILITY_RANGE)
            if enemy_idx is None:
                base_mask[self.ACTION_ABILITY] = False

        return np.repeat(base_mask, len(FACING_DIRS))

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

        # キャリアーの進行方向(次の経路セルへの差分)
        next_idx = min(self.carry_path_index + 1, len(self.carry_path) - 1)
        nxt = self.carry_path[next_idx]
        obs.append(float(np.sign(nxt[0] - cr)))
        obs.append(float(np.sign(nxt[1] - cc)))

        # 壁フラグ(4方向)
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

        # 隣接4方向に生存中の味方escortがいるか(衝突・団子状態の回避用)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            occupied_by_ally = any(
                j != i and self.escort_alive[j] and self.escort_pos[j] == (nr, nc)
                for j in range(self.n_escorts)
            )
            obs.append(1.0 if occupied_by_ally else 0.0)

        # 距離帯の逸脱量(負=近すぎ、正=遠すぎ、0=適正)
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

        # 自分のアビリティ状態(未使用フラグ + 種別onehot。HUNTを含め4種)
        obs.append(0.0 if self.escort_ability_used[i] else 1.0)
        for ability in ABILITY_TYPES:
            obs.append(1.0 if self.escort_ability_type[i] == ability else 0.0)

        # チーム状況：誰かの効果(blind/reveal/smoke)が現在有効か
        team_effect_active = any(v > 0 for v in self.enemy_blind_remaining) or any(
            v > 0 for v in self.enemy_reveal_remaining
        ) or len(self.smokes) > 0
        obs.append(1.0 if team_effect_active else 0.0)

        obs.append(self.escort_last_delta[i][0])
        obs.append(self.escort_last_delta[i][1])
        obs.append(min(1.0, self.escort_stuck[i] / 10.0))
        obs.append(1.0 - min(1.0, self.tick / max(1, self.max_ticks)))
        obs.append(1.0 if self._blocking_escort_idx == i else 0.0)

        # 自分の現在facing(N/S/E/W)のonehot
        for d in FACING_DIRS:
            obs.append(1.0 if self.escort_facing[i] == d else 0.0)

        # 「サイト-エスコート-キャリアー」順(自分がキャリアーより前に出ているか)判定用。
        ahead = self._ahead_of_carry((r, c))
        on_path, escape_available = self._path_status((r, c))
        obs.append(1.0 if ahead else 0.0)
        obs.append(1.0 if on_path else 0.0)
        obs.append(1.0 if escape_available else 0.0)

        # チーム共有の目撃情報(carrier+escort全体。自分が直接視認していなくても
        # 誰かが見ていれば共有される)。visibleフラグが立っている時のみ、facingを
        # そちらへ向ける学習をTEAM_SIGHTING_ALIGN_WEIGHTで後押しする。
        if self.team_sighting is not None:
            tr, tc = self.team_sighting["pos"]
            obs.append(1.0)
            obs.append(max(-1.0, min(1.0, (tr - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (tc - c) / DIST_NORM_MAX)))
            obs.append(min(self.team_sighting["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _obs_dim():
        # _get_obs() の要素数と一致させる(固定値なのでズレたら即バグに気づけるようassert)
        # 内訳: 自己座標2 + キャリアーBFS距離/方向3 + キャリアー進行方向2
        #      + 壁フラグ4 + BFS距離勾配4 + 斜め壁フラグ4 + 味方隣接フラグ4
        #      + 距離帯逸脱1 + 敵情報6 + アビリティ状態(未使用フラグ1+種別onehot4)
        #      + チーム効果1 + 直前移動2 + stuck1 + 残り時間1 + 被ブロック1
        #      + 前方判定(ahead/on_path/escape_available)3
        # = 2+3+2+4+4+4+4+1+6+5+1+2+1+1+1+3+8+4 = 56
        # (41→44: 「サイト-エスコート-キャリアー」順での道譲り判定用に3次元追加)
        # (44→52: 自身の現在facing onehotを8次元追加。当初4方向限定は行動選択側の
        #  バグであり、facingは本来8方向自由選択なので合わせて修正)
        # (52→56: チーム共有の目撃情報(visibleフラグ+方向2+経過tick)を4次元追加。
        #  自分だけでなくcarrier/他escortが見ている敵の情報も反映するため)
        return 56

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

        if ability == "SMOKE":
            # 「敵が味方(キャリアー)へ射線を持っている」状況を遮断できたら成功
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

        # HUNT(タイガー)はここに到達しない(常にマスクされているため)。保険としてwasteを返す。
        return -self.ability_waste_penalty

    # ------------------------------------------------------------------
    # 戦闘解決
    # ------------------------------------------------------------------
    def _resolve_combat(self):
        """1tick分の戦闘解決。kill発生時のボーナス対象escort集合を返す。

        battle_logic.py _resolve_all_shots と同じく、反応速度の高い順に
        逐次解決する(同値はシャッフルでランダム順)。移動中の射撃精度低下
        (MOVING_ACCURACY)・移動中の被弾しやすさ(MOVING_TARGET_HIT_MULTIPLIER)・
        向き(facing)による命中率補正・正面±SHOOTING_SITE_DIGREE度を超える
        相手を撃てない制限・被弾時の次tick強制向き変更も反映する
        (battle_logic._resolve_all_shotsと同一方針)。
        """
        smoke_cells = self._smoke_cell_set()

        def _stats_for(kind, idx):
            if kind == "carry":
                return {
                    "accuracy": self.carry_accuracy, "dodge": self.carry_dodge,
                    "hs_rate": self.carry_hs_rate, "reaction": self.carry_reaction,
                    "moved": self.carry_moved,
                }
            if kind == "escort":
                return {
                    "accuracy": self.escort_accuracy[idx], "dodge": self.escort_dodge[idx],
                    "hs_rate": self.escort_hs_rate[idx], "reaction": self.escort_reaction[idx],
                    "moved": self.escort_moved[idx],
                }
            return {
                "accuracy": self.enemy_accuracy[idx], "dodge": self.enemy_dodge[idx],
                "hs_rate": self.enemy_hs_rate[idx], "reaction": self.enemy_reaction[idx],
                "moved": self.enemy_moved[idx],
            }

        allies = [("carry", 0, self.carry_pos)] if self.carry_alive else []
        allies += [("escort", i, self.escort_pos[i]) for i in range(self.n_escorts) if self.escort_alive[i]]
        enemies = [("enemy", i, self.enemy_pos[i]) for i in range(self.n_enemies) if self.enemy_alive[i]]

        shooters = []
        for kind, idx, pos in allies:
            facing = self._facing_of(kind, idx)
            e_idx, _ = self._nearest_visible_enemy(pos, max_range=None)
            if e_idx is not None and _facing_angle_diff(facing, pos, self.enemy_pos[e_idx]) <= SHOOTING_SITE_DIGREE:
                shooters.append((kind, idx, "enemy", e_idx))
        for kind, idx, pos in enemies:
            facing = self._facing_of(kind, idx)
            best_idx, best_dist = None, None
            for akind, aidx, apos in allies:
                if not _has_los(self.grid, smoke_cells, pos, apos):
                    continue
                if _facing_angle_diff(facing, pos, apos) > SHOOTING_SITE_DIGREE:
                    continue
                d = _chebyshev(pos, apos)
                if best_dist is None or d < best_dist:
                    best_idx, best_dist = (akind, aidx), d
            if best_idx is not None:
                shooters.append((kind, idx, best_idx[0], best_idx[1]))

        # シャッフル後に反応速度降順で安定ソート(同値だけランダム順になる。
        # battle_logic.py _resolve_all_shots と同一方針)
        self.rng.shuffle(shooters)
        shooters.sort(key=lambda s: _stats_for(s[0], s[1])["reaction"], reverse=True)

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

            shooter_stats = _stats_for(shooter_kind, shooter_idx)
            target_stats = _stats_for(target_kind, target_idx)
            shooter_pos = self._pos_of(shooter_kind, shooter_idx)
            target_pos = self._pos_of(target_kind, target_idx)

            accuracy = MOVING_ACCURACY if shooter_stats["moved"] else shooter_stats["accuracy"]
            # 正面からの角度差による補正(正面100%～真横50%、battle_logicと同一ロジック)
            accuracy *= _facing_accuracy_multiplier(
                self._facing_of(shooter_kind, shooter_idx), shooter_pos, target_pos
            )
            if shooter_kind == "enemy" and self.enemy_blind_remaining[shooter_idx] > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            dodge = target_stats["dodge"]
            if target_kind == "enemy" and self.enemy_reveal_remaining[target_idx] > 0:
                dodge *= REVEALED_DODGE_MULTIPLIER

            hit_chance = accuracy * (1.0 - dodge)
            if target_stats["moved"]:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = self.rng.random() < hit_chance

            # 命中・被弾を問わず、撃たれたら次tickだけ相手の方向を強制的に
            # 向く(battle_logic._resolve_all_shotsと同一ロジック)。
            self._set_forced_facing(target_kind, target_idx, _facing_towards(target_pos, shooter_pos))

            if not hit:
                continue

            damage = HEADSHOT_DAMAGE if self.rng.random() < shooter_stats["hs_rate"] else BODY_DAMAGE

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
        """actions: 長さ n_escorts のリスト(死亡中のescortはNone扱いでもよい)。
        各要素は decode_action() で (base_action, facing) に分解される
        組み合わせ済みindex(0-23)。
        戻り値: (next_obs_list, rewards, done, info)
        """
        self.tick += 1
        rewards = [0.0] * self.n_escorts
        info = {"success": False, "carry_died": False}

        # 0. moved_this_tickフラグをリセット(このtickの実移動でのみTrueにする)。
        # 前tickで被弾していれば、このtickだけ強制的に相手の方向を向かせる
        # (battle_logic.move_characterと同一ロジック)。
        self.carry_moved = False
        self.carry_facing_forced_this_tick = False
        if self.carry_forced_facing_next_tick:
            self.carry_facing = self.carry_forced_facing_next_tick
            self.carry_facing_forced_this_tick = True
        self.carry_forced_facing_next_tick = None

        self.escort_moved = [False] * self.n_escorts
        for i in range(self.n_escorts):
            self.escort_facing_forced_this_tick[i] = False
            if self.escort_forced_facing_next_tick[i]:
                self.escort_facing[i] = self.escort_forced_facing_next_tick[i]
                self.escort_facing_forced_this_tick[i] = True
            self.escort_forced_facing_next_tick[i] = None

        self.enemy_moved = [False] * self.n_enemies
        for i in range(self.n_enemies):
            self.enemy_facing_forced_this_tick[i] = False
            if self.enemy_forced_facing_next_tick[i]:
                self.enemy_facing[i] = self.enemy_forced_facing_next_tick[i]
                self.enemy_facing_forced_this_tick[i] = True
            self.enemy_forced_facing_next_tick[i] = None

        # 0.5 escortのactionをbase_action/facingへ分解しておく
        # (アビリティ・移動どちらの分岐でも同じfacingを使うため先に解いておく)。
        decoded_base_actions = [None] * self.n_escorts
        chosen_facings = [None] * self.n_escorts
        for i in range(self.n_escorts):
            if actions[i] is None:
                continue
            base_action, facing = self.decode_action(actions[i])
            decoded_base_actions[i] = base_action
            chosen_facings[i] = facing

        # 0.6 facing整合shaping用に、移動で位置が変わる前(行動決定時点)の
        # 視認状態を保持しておく。敵が見えている時はこのshadingを無効化する。
        pre_action_visible_enemy = [
            bool(
                self.escort_alive[i]
                and self._nearest_visible_enemy(self.escort_pos[i], max_range=None)[0] is not None
            )
            for i in range(self.n_escorts)
        ]

        # 0.7 このtickの行動決定に使われたのは、前tick終了時点のteam_sightingの
        # 状態(=このstep()冒頭時点の self.team_sighting)。_update_team_sighting()
        # はin-placeでtick_agoをインクリメントするため、コピーを取って退避する。
        team_sighting_for_reward = dict(self.team_sighting) if self.team_sighting is not None else None

        # 1. タイマー減衰
        for i in range(self.n_enemies):
            self.enemy_blind_remaining[i] = max(0, self.enemy_blind_remaining[i] - 1)
            self.enemy_reveal_remaining[i] = max(0, self.enemy_reveal_remaining[i] - 1)
        for smoke in self.smokes:
            smoke["remaining"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining"] > 0]

        for i in range(self.n_escorts):
            rewards[i] -= 0.01  # 時間経過ペナルティ

        # 2. アビリティ行動を先に解決(アビリティ使用者はこのtick移動しない)
        used_ability_this_tick = set()
        for i in range(self.n_escorts):
            if not self.escort_alive[i] or decoded_base_actions[i] is None:
                continue
            if decoded_base_actions[i] == self.ACTION_ABILITY:
                rewards[i] += self._apply_ability(i)
                used_ability_this_tick.add(i)
                self.escort_last_delta[i] = (0.0, 0.0)
                self.escort_stuck[i] += 1

        # 3. 敵の簡易移動(衝突は考慮しない簡略化スクリプトAI)
        for i in range(self.n_enemies):
            if not self.enemy_alive[i]:
                continue
            prev_pos = self.enemy_pos[i]
            if self.rng.random() < self.enemy_move_prob:
                r, c = self.enemy_pos[i]
                candidates = [
                    (r + dr, c + dc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if not self._is_wall(r + dr, c + dc)
                ]
                if candidates:
                    new_pos = self.rng.choice(candidates)
                    if new_pos != self.enemy_pos[i]:
                        self.enemy_moved[i] = True
                    self.enemy_pos[i] = new_pos
            if not self.enemy_facing_forced_this_tick[i]:
                self.enemy_facing[i] = _facing_from_delta(
                    self.enemy_pos[i][0] - prev_pos[0],
                    self.enemy_pos[i][1] - prev_pos[1],
                    self.enemy_facing[i],
                )

        # 4. キャリアーの移動(塞がれていれば進めない)
        self._blocking_escort_idx = None
        prev_path_index = self.carry_path_index
        prev_carry_pos = self.carry_pos
        if self.carry_alive and self.carry_path_index < len(self.carry_path) - 1:
            next_cell = self.carry_path[self.carry_path_index + 1]
            occupied = self._occupied_by_others("carry", None)
            if next_cell not in occupied:
                self.carry_path_index += 1
                self.carry_pos = self.carry_path[self.carry_path_index]
                self.carry_moved = True
                self._refresh_carry_dist_map()
            else:
                for i in range(self.n_escorts):
                    if self.escort_alive[i] and self.escort_pos[i] == next_cell:
                        self._blocking_escort_idx = i
                        break
        if not self.carry_facing_forced_this_tick:
            self.carry_facing = _facing_from_delta(
                self.carry_pos[0] - prev_carry_pos[0],
                self.carry_pos[1] - prev_carry_pos[1],
                self.carry_facing,
            )

        team_progress = self.carry_path_index - prev_path_index
        for i in range(self.n_escorts):
            if self.escort_alive[i]:
                rewards[i] += team_progress * self.team_progress_coef
        if self._blocking_escort_idx is not None:
            rewards[self._blocking_escort_idx] -= self.block_penalty

        # --- stall検知：直接の1体だけでなく、carry周辺で団子状態を
        # 作っている全escortに圧力をかける。---
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
                if self._carry_bfs_dist(self.escort_pos[i]) <= self.congestion_radius:
                    # 前に出ていて、かつ退避不可能(狭い通路など)な場合は、
                    # 進むしかない状況なのでcongestion_penaltyの対象から除外する。
                    if self._ahead_of_carry(self.escort_pos[i]):
                        _, escape_available = self._path_status(self.escort_pos[i])
                        if not escape_available:
                            continue
                    rewards[i] -= congestion_penalty

        # 5. Escortの移動(アビリティ使用者・死亡者を除く、ランダム順で逐次解決)
        move_order = [
            i for i in range(self.n_escorts)
            if self.escort_alive[i] and i not in used_ability_this_tick and decoded_base_actions[i] is not None
        ]
        self.rng.shuffle(move_order)
        for i in move_order:
            action = decoded_base_actions[i]
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
            self.escort_moved[i] = True
            self.escort_last_delta[i] = (float(dr), float(dc))
            self.escort_stuck[i] = 0

        # 5.5 escortの向き(facing)を確定する。
        # 移動・STAY・アビリティいずれでもfacingは移動方向と無関係にDQNが
        # 選んだ方向を優先する(tv2_train_attacker_carry.pyと同一方針)。
        # ただし前tickで被弾していれば、それが常に最優先。
        for i in range(self.n_escorts):
            if not self.escort_alive[i] or chosen_facings[i] is None:
                continue
            if self.escort_facing_forced_this_tick[i]:
                continue
            self.escort_facing[i] = chosen_facings[i]

        # 5.6 facing整合の弱いshaping報酬(敵不可視時のみ)。
        # tv2_train_attacker_carry.pyと同一方針: 進行方向を向くほど+、背を
        # 向けるほど-。敵が見えている時(pre_action_visible_enemy=True)は
        # 通常の交戦報酬(命中率経由)に完全に委ね、この項は加算しない。
        # 被弾による強制facing(escort_facing_forced_this_tick)がかかった
        # tickも、本人の意思ではないため対象から除外する。
        for i in range(self.n_escorts):
            if not self.escort_alive[i]:
                continue
            if pre_action_visible_enemy[i] or self.escort_facing_forced_this_tick[i]:
                continue
            facing = self.escort_facing[i]
            dr, dc = self.escort_last_delta[i]
            move_len = math.hypot(dr, dc)
            if move_len == 0.0 or facing not in FACING_VECTORS:
                continue
            fx, fy = FACING_VECTORS[facing]
            alignment = (fx * dc + fy * dr) / move_len
            rewards[i] += FACING_ALIGN_WEIGHT_ESCORT * alignment

        # 5.7 チーム共有の目撃情報に対するfacing整合shaping(自分が直接視認して
        # いない時のみ有効)。carrier/他escortが見ている敵の方向を向くほど+、
        # 逆を向くほど-。直接視認している場合は通常の交戦報酬(命中率経由)に
        # 完全に委ね、このshapingは加えない。被弾による強制facingがかかった
        # tickも本人の意思ではないため対象から除外する。
        if team_sighting_for_reward is not None:
            for i in range(self.n_escorts):
                if not self.escort_alive[i]:
                    continue
                if pre_action_visible_enemy[i] or self.escort_facing_forced_this_tick[i]:
                    continue
                facing = self.escort_facing[i]
                alignment = _direction_alignment(facing, self.escort_pos[i], team_sighting_for_reward["pos"])
                rewards[i] += TEAM_SIGHTING_ALIGN_WEIGHT * alignment

        # 6. 距離帯報酬(移動後の位置で評価)。obsのescort→carry距離と同じく
        # BFS実距離を使う(Chebyshevだと曲がった通路で「壁越しに近い」を
        # 「実際に近い」と誤判定し、中継点付近で追従圧力が消えてしまうため)。
        for i in range(self.n_escorts):
            if not self.escort_alive[i]:
                continue
            dist = self._carry_bfs_dist(self.escort_pos[i])
            if dist < self.dist_band_min:
                rewards[i] -= (self.dist_band_min - dist) * self.dist_penalty_coef
            elif dist > self.dist_band_max:
                rewards[i] -= (dist - self.dist_band_max) * self.dist_penalty_coef

        # 6.1 常時距離短縮のポテンシャルベース報酬。
        #     バンド内(dist_band_min〜dist_band_max)ではdist_penaltyが0になり
        #     「これ以上近づく理由がない」状態になるため、バンド内外を問わず
        #     毎tick「前tickよりcarryに近づいたか」だけで加減点する。
        #     これによりcarryが既に静止していても、遠いescortは接近を続ける。
        for i in range(self.n_escorts):
            if not self.escort_alive[i]:
                continue
            cur_dist = self._carry_bfs_dist(self.escort_pos[i])
            rewards[i] += (self.escort_prev_dist[i] - cur_dist) * self.potential_coef
            self.escort_prev_dist[i] = cur_dist

        # 6.5 「サイト-エスコート-キャリアー」順での道譲りペナルティ。
        #     退避可能(=道を空けられる)なのに、キャリアーの最短経路上に留まって
        #     前に出ている場合のみ罰する。退避不可能な狭い通路では罰さず、
        #     そのまま進ませる(先に進む方の挙動を許容する)。
        for i in range(self.n_escorts):
            if not self.escort_alive[i]:
                continue
            pos = self.escort_pos[i]
            if not self._ahead_of_carry(pos):
                continue
            on_path, escape_available = self._path_status(pos)
            if on_path and escape_available and self._carry_bfs_dist(pos) <= self.congestion_radius:
                rewards[i] -= self.ahead_block_penalty_coef

        # 7. 戦闘解決
        kill_bonus_targets = self._resolve_combat()
        for escort_idx in kill_bonus_targets:
            if 0 <= escort_idx < self.n_escorts:
                rewards[escort_idx] += self.kill_bonus

        # 移動・戦闘が確定した後の位置関係でチーム共有目撃情報を更新する。
        # 次tickのobs(_get_obs)と行動決定はこの更新後の状態を参照する。
        self._update_team_sighting()

        done = False
        if not self.carry_alive:
            done = True
            info["carry_died"] = True
            for i in range(self.n_escorts):
                if self.escort_alive[i]:
                    rewards[i] -= self.mission_fail_penalty
        elif carry_reached_goal:
            # carryが到着済みでも、escort全員がdist_band_max以内に収まるまでは
            # doneにしない(=接近行動そのものを学習対象に含める)。
            # carry_reached_goalは5.の停滞検知セクションで既に計算済みの値を再利用する。
            all_escorts_settled = all(
                (not self.escort_alive[i])
                or self._carry_bfs_dist(self.escort_pos[i]) <= self.dist_band_max
                for i in range(self.n_escorts)
            )
            if all_escorts_settled:
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
            # else: 継続。ポテンシャルshaping(6.1)がこの間の接近を評価する。
        elif self.tick >= self.max_ticks:
            done = True
            info["success"] = False
            for i in range(self.n_escorts):
                if self.escort_alive[i]:
                    rewards[i] -= self.mission_fail_penalty

        next_obs = [self._get_obs(i) for i in range(self.n_escorts)]
        return next_obs, rewards, done, info


# ---------------------------------------------------------------------------
# Dueling DQN(重み共有。他のtouyama_v2学習ファイルと同一アーキテクチャ)
# ---------------------------------------------------------------------------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "next_mask", "done"))
# DuelingQNetwork(hidden=128) は tv2_common_rl.DuelingQNet と層構成が完全一致。
# ReplayBuffer / select_action / optimize_model は tv2_common_rl に統合。
# 注意: 元のselect_actionはマスク全滅を想定していなかった(ACTION_STAYが
# 常にTrueのため)。tv2_common_rl.select_actionはfallback_action引数を持つが、
# デフォルト0でも実質未使用となる想定。

def evaluate(env, policy_net, device, episodes=20):
    successes = 0
    total_reward = 0.0
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
    parser = argparse.ArgumentParser(description="touyama_v2 Attacker Escort Phase 学習スクリプト")
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    parser.add_argument("--max-ticks", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=int(EPISODE_COUNT * 0.8))
    parser.add_argument("--target-update-every", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=50)  # 20だと分散が大きいため引き上げ
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--hold-ticks", type=int, default=ESCORT_HOLD_TICKS)
    parser.add_argument("--best-model-eps-threshold", type=float, default=BEST_MODEL_EPS_THRESHOLD)
    parser.add_argument("--best-model-smooth-window", type=int, default=BEST_MODEL_SMOOTH_WINDOW)
    parser.add_argument("--ahead-block-penalty-coef", type=float, default=AHEAD_BLOCK_PENALTY_COEF)
    parser.add_argument("--potential-coef", type=float, default=0.15)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/attacker_escort_touyama_data/",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    env = EscortEnv(
        max_ticks=args.max_ticks, hold_ticks=args.hold_ticks,
        ahead_block_penalty_coef=args.ahead_block_penalty_coef,
        potential_coef=args.potential_coef, seed=args.seed,
    )
    eval_env = EscortEnv(
        max_ticks=args.max_ticks, hold_ticks=args.hold_ticks,
        ahead_block_penalty_coef=args.ahead_block_penalty_coef,
        potential_coef=args.potential_coef, seed=args.seed + 1,
    )

    obs_dim = env._obs_dim()
    n_actions = env.N_ACTIONS
    print(f"[INFO] obs_dim={obs_dim} n_actions={n_actions} n_escorts={env.n_escorts} device={device}")

    policy_net = DuelingQNet(obs_dim, n_actions).to(device)
    target_net = DuelingQNet(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(Transition, args.buffer_size)

    best_success_rate = -1.0
    best_eval_reward = float("-inf")
    global_step = 0
    eval_history = deque(maxlen=args.best_model_smooth_window)  # 直近evalのsuccess_rate/avg_reward

    def _make_checkpoint(episode, success_rate, avg_reward, avg_block_events):
        return {
            "model_state_dict": policy_net.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "episode": episode,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_block_events": avg_block_events,
            "roster_order": list(TOUYAMA_ROSTER_ORDER),
            "spike_holder_default": TOUYAMA_SPIKE_HOLDER,
            "hold_ticks": args.hold_ticks,
        }

    start_time = time.perf_counter()
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
                action = select_action(
                    policy_net, obs_list[i], mask, epsilon,
                    device=device, fallback_action=EscortEnv.ACTION_STAY,
                )
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
                batch = buffer.sample(args.batch_size)
                optimize_double_dqn_step(
                    policy_net, target_net, optimizer,
                    batch.state, batch.action, batch.reward,
                    batch.next_state, batch.done, batch.next_mask,
                    args.gamma, device=device,
                )

            if global_step % args.target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode % 50 == 0:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter();
            print(
                f"[EP {episode}/{args.episodes}] reward={episode_reward:.2f} elapse={elapsed_time:.1f} "
                f"eps={epsilon:.3f} success={info.get('success')} ticks={env.tick} "
                f"carrier={env.carry_name}"
            )
        
        if episode % args.eval_every == 0:
            success_rate, avg_reward, avg_block_events = evaluate(
                eval_env, policy_net, device, args.eval_episodes
            )
            eval_history.append((success_rate, avg_reward))
            smoothed_success_rate = sum(s for s, _ in eval_history) / len(eval_history)
            smoothed_avg_reward = sum(r for _, r in eval_history) / len(eval_history)

            print(
                f"[EVAL @ EP {episode}/{args.episodes}] success_rate={success_rate:.2%} "
                f"avg_reward={avg_reward:.2f} avg_block_events={avg_block_events:.2f} "
                f"eps={epsilon:.3f} smoothed_success_rate={smoothed_success_rate:.2%} "
                f"smoothed_avg_reward={smoothed_avg_reward:.2f}"
            )

            latest_path = os.path.join(args.save_dir, "dqn_attacker_escort_touyama_latest.pt")
            torch.save(_make_checkpoint(episode, success_rate, avg_reward, avg_block_events), latest_path)

            # epsilonがまだ閾値より高い(=探索が多く方策が未収束)間はbest候補から除外し、
            # 直近best_model_smooth_window回のeval平均(smoothed_*)で比較することで、
            # 1回だけたまたま良かったevalがbestとして固定されるのを防ぐ。
            eligible_for_best = epsilon <= args.best_model_eps_threshold
            is_better = (
                eligible_for_best
                and (
                    smoothed_success_rate > best_success_rate + 1e-9
                    or (
                        smoothed_success_rate >= best_success_rate - 1e-9
                        and smoothed_avg_reward > best_eval_reward
                    )
                )
            )
            if episode < EVAL_MIN_EPISODE:
                print(f"[SAVE skip] episode={episode} < EVAL_MIN_EPISODE={EVAL_MIN_EPISODE}")
            elif is_better:
                best_success_rate = max(best_success_rate, smoothed_success_rate)
                best_eval_reward = smoothed_avg_reward
                best_path = os.path.join(args.save_dir, "dqn_attacker_escort_touyama_best_by_eval.pt")
                torch.save(_make_checkpoint(episode, success_rate, avg_reward, avg_block_events), best_path)
                print(
                    f"[SAVE] 新しいベストモデルを保存: {best_path} "
                    f"(success_rate={success_rate:.2%}, avg_reward={avg_reward:.2f}, "
                    f"smoothed_success_rate={smoothed_success_rate:.2%}, "
                    f"avg_block_events={avg_block_events:.2f})"
                )

    print("[DONE] 学習が完了しました。")


if __name__ == "__main__":
    main()