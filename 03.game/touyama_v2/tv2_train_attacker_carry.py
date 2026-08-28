"""touyama_v2/train_attacker_carry.py

固定チーム(いぐるん/夢の街/ろびぃな/Tortlilyan/えんぺん)専用の
Attacker「carry phase」学習スクリプト(簡略版)。

【設計方針(簡略版)】
- 移動はDQNの行動空間に含めない。中継地点(6/7)未到達ならそこへ、到達後は
  優先設置場所(5)へ、BFS経路探索(bfs_next_step、他キャラは障害物として回避)で
  決定的に移動する。エスコートとの衝突回避は移動適用順(キャリア最優先)と
  経路探索の占有マス回避で自然に解消される。
- DQNが学習するのはアビリティ使用判断(NONE/ABILITY)と明示PLANTの3値のみ。
- 優先設置場所(5)にちょうど到達した場合のみプラント成功として扱う。
  妥協設置(通常の2マス)・時間切れによるフォールバックは行わない
  (確実に優先地点へ到達してプラントすることを最優先するため)。
- エスコート4人・敵5人はヒューリスティックで動かす(引き続き将来の専用モデル
  差し替えに対応できる設計)。

train_attacker_guard.py / train_defender_search.py と同一規約:
  - run_game.py / controllers.py / battle_logic.py / abilities_los.py は
    一切importしない。必要なロジックはすべてこのファイル内に複製する。
  - map_data.py / character_stats_touyama.py / game_core.py は定数専用
    ファイルとして参照する(import制限の対象外)。
  - run_game.py / controllers.py は変更しない。

マップは map_data_carry.py を使用する(grid==5: 優先設置場所, grid==6/7: サイト別
中継地点)。5/6/7は本番map_data.pyには存在しないため、チェックポイントに座標として
保存し、本番実行時はgridの値に依存せずその座標を使う(learning側で対応)。

保存先: touyama_v2/data/attacker_carry_touyama_data/
チェックポイントは {"model_state_dict","obs_dim","n_actions","episode",
"success_rate","priority_cells","has_priority_cells","waypoint_cells"} を
含むdict形式で保存する。
"""

import os
import sys
import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- after ---
from tv2_map_data_carry import NEW_MAZE_STR
from tv2_character_stats_touyama import CHARACTER_TABLE as TOUYAMA_STATS_TABLE

from game_core import (
    MAX_HP,
    BODY_DAMAGE,
    HEADSHOT_DAMAGE,
    MOVING_ACCURACY,
    MOVING_TARGET_HIT_MULTIPLIER,
    BLIND_ACCURACY_MULTIPLIER,
    REVEALED_DODGE_MULTIPLIER,
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
    SMOKE_DURATION_TICKS,
    ROUND_DURATION_TICKS,
    PLANT_REQUIRED_TICKS,
    FACING_VECTORS,
    SHOOTING_SITE_DIGREE,
)

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

EPISODE_COUNT = 3000
EVAL_MIN_EPISODE = EPISODE_COUNT * 0.7   # これ未満のepisodeでは評価・保存を行わない(初期の不安定な重みを保存しないため)

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = "data/attacker_carry_touyama_data/"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_attacker_carry_touyama_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_attacker_carry_touyama_latest.pt")

CARDINAL = tv2_common_rl.CARDINAL_MOVES

# 観測29次元。行動空間は「アビリティ不使用/使用」「明示PLANT」の3値のみ
# (移動はDQNの行動空間に含めない。BFS経路探索で決定的に処理する)。
# 21-28: 自分(carrier)の現在の向き(N/NE/E/SE/S/SW/W/NW)のone-hot。向き選択
# (facing_idx)を状態から独立に学習できないバグの修正のため追加。
OBS_DIM = 34
# 移動はBFS決定的なため行動空間に含めない。ここに含めるのはアビリティ使用判断
# (NONE/ABILITY)・明示PLANTの3値と、向き(facing)選択の直積のみ。
# tv2_train_defender_search.pyと同一規約: action_idx = base_idx(0-2)*8 + facing_idx(0-7)。
# 移動先(BFS)と向きは無関係に選べる(例: 前進しながら横を警戒する)。
# 向きには直接報酬を与えず、通常の交戦結果(命中率補正経由)を通じて間接的に学習させる。
BASE_ACTION_DIM = 3
ACTION_NONE = 0
ACTION_ABILITY = 1
ACTION_PLANT = 2
FACING_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_DIM = BASE_ACTION_DIM * len(FACING_DIRS)  # 24

# サイト別中継地点: 右サイト=6, 左サイト=7。
WAYPOINT_VALUE_BY_SITE = {"right": 6, "left": 7}

# スモーク事前設置(lineup)用の定点。中継地点やtarget_plant_posとは無関係の、
# 実際に射線を切りたい座標をサイトごとに手動で指定する
# (例: 「このマスに焚けば、敵の定番ポジションからサイトへの射線が切れる」)。
# 1サイトに複数点あってもよい(carrierに近い方から優先的に使う)。
# 未設定(空リスト)の間はこの機能は発火しない。マップを見ながら座標を埋めること。
SMOKE_LINEUP_CELLS_BY_SITE = {
    "left": [(7, 3)],   # 例: [(12, 5), (10, 3)]
    "right": [(9,39)],  # 例: [(12, 30)]
}

# target_plant_pos抽選時のサイト選択確率。学習は必ず50/50にする(片方のサイトの
# 学習内容が不足するのを防ぐため)。実プレイでの左右比率は推論側
# (learning_attacker_carry_touyama.py)のAI_CONTROLLED_SITE_SELECTIONで別途決定する。
SITE_SELECTION_WEIGHTS = {"left": 0.5, "right": 0.5}

# プラント可能マスは 2(通常) と 5(優先設置場所)の両方。ただしプラント成功として
# 認めるのは常に優先設置場所(5)のみ(target_plant_posは必ずPRIORITY_CELLSから選ぶ)。
SITE_VALUES = frozenset({2, 5})

N_ATTACKERS = 5  # touyama固定チーム(このフェーズではAttacker)
N_DEFENDERS = 5  # 敵(ヒューリスティック)
MAX_TICKS = ROUND_DURATION_TICKS  # 100: ラウンド制限時間と一致させる

ABILITY_RANGE = 7
SIGHTING_STALENESS_CAP = 20
HANDOFF_AUGMENT_PROB = 0.25  # 一定確率でスポーン以外(拾得後の合流)からスタート


# ---------------------------------------------------------------------------
# touyama_v2 固定チーム定義
# ---------------------------------------------------------------------------
TOUYAMA_EFFECTIVE_STATS = compute_touyama_effective_stats(TOUYAMA_STATS_TABLE)
print_effective_stats(TOUYAMA_EFFECTIVE_STATS, "Attacker/carry・簡略版")

# 報酬パラメータ(簡略版: 移動はDQNの決定事項ではないためPROGRESS_REWARDは廃止)
STEP_PENALTY = -0.001
DAMAGE_TAKEN_PENALTY_SCALE = -0.002  # 被弾ダメージ1あたり
KILL_REWARD = 0.4
DEATH_PENALTY = -1.0            # キャリア死亡=スパイクドロップ(retrieveフェーズへ引き継ぎ)
ABILITY_WHIFF_PENALTY = -0.05
ABILITY_OVERLAP_PENALTY = -0.05
PLANT_WHIFF_PENALTY = -0.05     # 目標地点外でPLANTを選んだ場合(マスクが機能していれば理論上到達しない保険)
ON_TARGET_NON_PLANT_PENALTY = -0.05  # 目標地点ちょうどにいるのにPLANT以外を選んだ場合の毎tickペナルティ
PLANT_TICK_BONUS = 0.05         # 明示PLANT行動が1Tick成功して進むごとのボーナス
PLANT_SUCCESS_REWARD = 1.5              # 優先設置場所でのプラント完了(最優先ゴール)
PLANT_SUCCESS_REWARD_COMPROMISE = 0.8   # 妥協設置(通常マス2)成功時。優先設置場所より低くし優先順位を維持する
TIME_EXPIRE_PENALTY = -2.0      # ラウンド時間切れ(未設置)。全ペナルティ中で最も重くする
TEAM_WIPE_PENALTY = -0.2        # 味方(エスコート込み)全滅だがキャリアは生存継続中
WAYPOINT_REACHED_REWARD = 0.10  # サイト別中継地点(6=右/7=左)を初通過した時の一度きりのボーナス
FACING_ALIGN_WEIGHT = 0.01      # 敵不可視時のみ有効。進行方向を向くほど+、背を向けるほど-(弱いshaping)
SMOKE_LINEUP_REWARD = 0.35      # 定点スモーク使用のボーナス。ABILITY_WHIFF_PENALTY(0.05)より
                                 # 十分大きくし、「見えてから焚く」より「定点で先に焚く」方を有利にする
SMOKE_LINEUP_WAIT_PENALTY = -0.03  # 定点スモークが射程内で使用可能なのに未使用のまま経過する毎tickのペナルティ。
                                    # 「近づいてから焚く」を不利にし、射程に入ったらなるべく早く焚くよう誘導する。


# ============================================================================
# マップ読み込み
# ============================================================================

GRID = tv2_common_rl.parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
ATTACKER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 3]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) in SITE_VALUES]
# 優先設置場所(5)。target_plant_posは必ずここから選ぶ(構造的にサイト内に1つも
# 無い場合のみ、後述のPLANT_CELLS_BY_SITEへ構造的フォールバックする)。
PRIORITY_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) == 5]

# --- チョークポイント(狭い通路)の自動検出 + 手動追加 ------------------------
# 「開放マスがちょうど2方向しかない」壁のみの判定なので、味方・敵の位置には
# 依存しない。既知の場所を追加したい場合はここへ (r, c) を足す。
MANUAL_CHOKEPOINT_CELLS = []


def _detect_corridor_chokepoints(grid):
    height, width = grid.shape
    cells = []
    for r in range(height):
        for c in range(width):
            if grid[r, c] == 1:
                continue
            open_dirs = [
                (dr, dc) for dr, dc in tv2_common_rl.CARDINAL_MOVES
                if 0 <= r + dr < height and 0 <= c + dc < width and grid[r + dr, c + dc] != 1
            ]
            if len(open_dirs) == 2:
                cells.append((r, c))
    cells.extend(tuple(cell) for cell in MANUAL_CHOKEPOINT_CELLS)
    return list(dict.fromkeys(cells))


def _multi_source_bfs_distance_map(grid, source_cells):
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    q = deque()
    for r, c in source_cells:
        if 0 <= r < height and 0 <= c < width and grid[r, c] != 1 and dist[r, c] == -1:
            dist[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in tv2_common_rl.CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


CHOKEPOINT_CELLS = _detect_corridor_chokepoints(GRID)
CHOKEPOINT_DIST_MAP = (
    _multi_source_bfs_distance_map(GRID, CHOKEPOINT_CELLS) if CHOKEPOINT_CELLS else None
)

WAYPOINT_CELLS = {}
for _site, _value in WAYPOINT_VALUE_BY_SITE.items():
    _cells = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) == _value]
    if len(_cells) == 1:
        WAYPOINT_CELLS[_site] = _cells[0]
    elif len(_cells) > 1:
        raise RuntimeError(
            f"map_data_carry.py にサイト別中継地点(値={_value}, site={_site})が"
            f"{len(_cells)}個あります。1個だけにしてください。"
        )
    else:
        print(
            f"[WARN] map_data_carry.py にサイト別中継地点(値={_value}, site={_site})が"
            f"見つかりません。このサイトは経路強制なしの従来挙動になります。"
        )

if len(ATTACKER_SPAWNS) < N_ATTACKERS:
    raise RuntimeError(
        f"ATTACKER_SPAWNSが{len(ATTACKER_SPAWNS)}マスしかなく、固定チーム{N_ATTACKERS}人分を配置できません。"
    )
if len(DEFENDER_SPAWNS) < N_DEFENDERS:
    raise RuntimeError(
        f"DEFENDER_SPAWNSが{len(DEFENDER_SPAWNS)}マスしかなく、敵{N_DEFENDERS}人分を配置できません。"
    )
if not PLANT_CELLS:
    raise RuntimeError("map_data_carry.py にプラント可能マス(2 または 5)が見つかりません。")
if not PRIORITY_CELLS:
    print("[WARN] map_data_carry.py に優先設置場所(5)が見つかりません。target_plant_posは通常マス(2)から選ばれます。")

# サイト別分割(判定基準: 列がWIDTH//2未満なら左)。
PLANT_CELLS_BY_SITE = {"left": [], "right": []}
for _cell in PLANT_CELLS:
    _site_key = "left" if _cell[1] < WIDTH // 2 else "right"
    PLANT_CELLS_BY_SITE[_site_key].append(_cell)

PRIORITY_CELLS_BY_SITE = {"left": [], "right": []}
for _cell in PRIORITY_CELLS:
    _site_key = "left" if _cell[1] < WIDTH // 2 else "right"
    PRIORITY_CELLS_BY_SITE[_site_key].append(_cell)

for _site_key in ("left", "right"):
    if not PLANT_CELLS_BY_SITE[_site_key]:
        print(f"[WARN] map_data_carry.py に{_site_key}サイトのプラント可能マスが見つかりません。")
    if not PRIORITY_CELLS_BY_SITE[_site_key]:
        print(
            f"[WARN] map_data_carry.py に{_site_key}サイトの優先設置場所(5)が見つかりません。"
            f"このサイトが選ばれた場合、通常のプラント可能マス(2)からtargetを選びます(構造的フォールバック)。"
        )

_site_weight_sum = sum(SITE_SELECTION_WEIGHTS.get(s, 0.0) for s in ("left", "right"))
if abs(_site_weight_sum - 1.0) > 1e-6:
    print(
        f"[WARN] SITE_SELECTION_WEIGHTSの合計が1.0ではありません(現在: {_site_weight_sum})。"
        f"そのままの比率で正規化して使用します。"
    )


# ============================================================================
# LOS・BFS
# ============================================================================

def has_los(p1, p2, smoke_cells=None):
    return tv2_common_rl.has_los(GRID, p1, p2, smoke_cells)


def bfs_distance_map(goal):
    return tv2_common_rl.bfs_distance_map(GRID, goal)


def bfs_next_step(start, goal, occupied, allow_adjacent_goal=True):
    return tv2_common_rl.bfs_next_step(GRID, start, goal, occupied, allow_adjacent_goal=allow_adjacent_goal)


def _random_step(pos, occupied):
    return tv2_common_rl.random_step(GRID, pos, occupied)


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


def _facing_angle_diff(shooter, target):
    """射手のfacingと、射手→標的方向との角度差(度)を返す(battle_logic._facing_angle_diffと同一ロジック)。"""
    dc = float(target.pos[1] - shooter.pos[1])
    dr = float(target.pos[0] - shooter.pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0:
        return 0.0
    fx, fy = FACING_VECTORS[shooter.facing]
    dot = max(-1.0, min(1.0, (fx * dc + fy * dr) / dist))
    return math.degrees(math.acos(dot))


def _facing_accuracy_multiplier(shooter, target):
    """正面100%～真横50%まで、角度差に応じて線形に精度を落とす(battle_logicと同一ロジック)。"""
    angle = _facing_angle_diff(shooter, target)
    return 1.0 - min(SHOOTING_SITE_DIGREE, angle) / SHOOTING_SITE_DIGREE * 0.5


def _choose_weighted_site():
    left_w = max(0.0, float(SITE_SELECTION_WEIGHTS.get("left", 0.0)))
    right_w = max(0.0, float(SITE_SELECTION_WEIGHTS.get("right", 0.0)))
    total = left_w + right_w
    if total <= 0.0:
        return random.choice(["left", "right"])
    return "left" if random.random() < (left_w / total) else "right"


# 各PLANT_CELLへの個別BFS距離マップ(target_plant_pos固定用。観測の距離特徴量に使う)。
PLANT_DIST_MAPS = {cell: bfs_distance_map(cell) for cell in PLANT_CELLS}
WAYPOINT_DIST_MAPS = {site: bfs_distance_map(cell) for site, cell in WAYPOINT_CELLS.items()}


def _ray_openness(pos, direction, max_range=12):
    """壁のみに基づく直線方向の見通し距離(0〜1に正規化)。
    次のマスへ進んだ時にどれだけ視界が開けるか=奇襲されやすいかの
    事前情報として使う。敵の実位置は一切参照しないフェアな情報。"""
    r, c = int(pos[0]), int(pos[1])
    dr, dc = direction
    dist = 0
    while dist < max_range:
        nr, nc = r + dr * (dist + 1), c + dc * (dist + 1)
        if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH) or GRID[nr, nc] == 1:
            break
        dist += 1
    return dist / max_range


CELL_OPENNESS = {
    cell: [_ray_openness(cell, d) for d in CARDINAL]
    for cell in WALKABLE
}


# ============================================================================
# ユニットスタブ
# ============================================================================

class UnitStub:
    def __init__(self, name, team, pos, role, ability, accuracy, dodge_rate, hs_rate, reaction, has_spike=False):
        self.name = name
        self.team = team  # "A"(touyama/carry+escort) or "D"(敵)
        self.pos = list(pos)
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.is_alive = True
        self.role = role
        self.ability_name = ability
        self.charges = 0 if ability in ("HUNT", "NONE") else 1
        self.blind_remaining = 0
        self.reveal_remaining = 0
        self.moved_this_tick = False
        self.has_spike = has_spike
        self.kills = 0
        self.accuracy = accuracy
        self.dodge_rate = dodge_rate
        self.hs_rate = hs_rate
        self.reaction = reaction + random.uniform(-10, 10)
        # 向き。Attacker(carry/escort)は北向き、Defenderは南向きでスタート(game_core.Characterと同一)。
        self.facing = "S" if team == "D" else "N"
        self.forced_facing_next_tick = None
        self.facing_forced_this_tick = False


def _resolve_spawn_collision(pos, occupied):
    return tv2_common_rl.resolve_spawn_collision(GRID, pos, occupied)


def _build_fixed_attackers(carrier_name, handoff=False):
    attackers = []
    occupied = set()

    if handoff:
        base_positions = random.sample(WALKABLE, N_ATTACKERS)
    else:
        base_positions = list(ATTACKER_SPAWNS[:N_ATTACKERS])

    for i, name in enumerate(TOUYAMA_ROSTER_ORDER):
        stats = TOUYAMA_EFFECTIVE_STATS[name]
        base_pos = base_positions[i % len(base_positions)]
        spawn_pos = _resolve_spawn_collision(base_pos, occupied)
        occupied.add(spawn_pos)

        unit = UnitStub(
            name, "A", spawn_pos, stats["ability"], stats["ability"],
            stats["accuracy"], stats["dodge_rate"], stats["hs_rate"], stats["reaction"],
            has_spike=(name == carrier_name),
        )
        attackers.append(unit)
    return attackers


DEFENDER_SITE_BIAS_STRENGTH = 0.5        # 0=完全ランダム、1=常にサイト近傍優先。過学習防止のため1未満推奨
DEFENDER_CHOKEPOINT_BIAS_STRENGTH = 0.3  # 0=チョークポイントを考慮しない。過信させすぎないよう控えめに


def _build_defenders(bias_cell=None, bias_strength=0.0, chokepoint_bias_strength=0.0):
    """bias_cell付近・チョークポイント付近ほどDefenderが選ばれやすくする。
    2つのバイアスは独立した強さで重ね合わせ、残りは一様分布のまま残す
    (どちらも0なら従来通り完全ランダム)。"""
    pool = list(DEFENDER_SPAWNS)
    n = min(N_DEFENDERS, len(pool))

    weight_components = []
    total_bias = 0.0

    if bias_cell is not None and bias_strength > 0.0:
        dist_map = bfs_distance_map(bias_cell)
        raw = np.array([
            1.0 / (1.0 + (dist_map[c[0], c[1]] if dist_map[c[0], c[1]] >= 0 else HEIGHT + WIDTH))
            for c in pool
        ])
        weight_components.append((bias_strength, raw / raw.sum()))
        total_bias += bias_strength

    if chokepoint_bias_strength > 0.0 and CHOKEPOINT_DIST_MAP is not None:
        raw = np.array([
            1.0 / (1.0 + (
                CHOKEPOINT_DIST_MAP[c[0], c[1]] if CHOKEPOINT_DIST_MAP[c[0], c[1]] >= 0 else HEIGHT + WIDTH
            ))
            for c in pool
        ])
        weight_components.append((chokepoint_bias_strength, raw / raw.sum()))
        total_bias += chokepoint_bias_strength

    if weight_components and total_bias > 0.0:
        total_bias = min(1.0, total_bias)
        p = (1.0 - total_bias) * np.full(len(pool), 1.0 / len(pool))
        for strength, dist in weight_components:
            p = p + strength * dist
        p = p / p.sum()
        idx = np.random.choice(len(pool), size=n, replace=False, p=p)
        d_spawns = [pool[i] for i in idx]
    else:
        d_spawns = random.sample(pool, n)

    return [
        UnitStub(
            f"D{i+1}", "D", pos, "NONE", random.choice(["FLASH", "SMOKE", "RECON", "NONE"]),
            DEFAULT_ACCURACY, DEFAULT_DODGE, DEFAULT_HS_RATE, DEFAULT_REACTION,
        )
        for i, pos in enumerate(d_spawns)
    ]


# ============================================================================
# ヒューリスティック(エスコート4人 / 敵5人)
# ============================================================================

def _heuristic_ability_action(unit, visible_enemies):
    if unit.charges <= 0 or unit.ability_name in ("HUNT", "NONE"):
        return None

    if unit.ability_name == "SMOKE" and len(visible_enemies) >= 2:
        target = visible_enemies[0]
        return ("SMOKE", tuple(map(int, target.pos)))

    if unit.ability_name == "FLASH" and visible_enemies:
        closest = min(
            visible_enemies,
            key=lambda e: max(abs(e.pos[0] - unit.pos[0]), abs(e.pos[1] - unit.pos[1])),
        )
        dist = max(abs(closest.pos[0] - unit.pos[0]), abs(closest.pos[1] - unit.pos[1]))
        if dist <= 5:
            return ("FLASH", tuple(map(int, closest.pos)))

    if unit.ability_name == "RECON" and not visible_enemies:
        target = random.choice(PLANT_CELLS)
        return ("RECON", target)

    return None


def _apply_ability(unit, ability_name, target_pos, smokes, all_units, smoke_cells):
    unit.charges -= 1
    if ability_name == "SMOKE":
        tr, tc = int(target_pos[0]), int(target_pos[1])
        cells = {
            (rr, cc)
            for rr in range(tr - 1, tr + 2)
            for cc in range(tc - 1, tc + 2)
            if 0 <= rr < HEIGHT and 0 <= cc < WIDTH and GRID[rr, cc] != 1
        }
        smokes.append({"cells": cells, "remaining_ticks": SMOKE_DURATION_TICKS, "team": unit.team})
    elif ability_name == "FLASH":
        for other in all_units:
            if other.is_alive and other.team != unit.team and has_los(target_pos, other.pos, smoke_cells):
                other.blind_remaining = max(other.blind_remaining, BLIND_DURATION_TICKS)
    elif ability_name == "RECON":
        for other in all_units:
            if other.is_alive and other.team != unit.team and has_los(target_pos, other.pos, smoke_cells):
                other.reveal_remaining = max(other.reveal_remaining, REVEAL_DURATION_TICKS)


def _escort_move(unit, carrier, occupied):
    if carrier is None or not carrier.is_alive:
        return _random_step(tuple(unit.pos), occupied)
    dist = max(abs(carrier.pos[0] - unit.pos[0]), abs(carrier.pos[1] - unit.pos[1]))
    if random.random() < 0.3:
        return _random_step(tuple(unit.pos), occupied)
    if dist > 5:
        return bfs_next_step(tuple(unit.pos), tuple(carrier.pos), occupied, allow_adjacent_goal=True)
    return _random_step(tuple(unit.pos), occupied)


def _defender_move(unit, occupied):
    return _random_step(tuple(unit.pos), occupied)


# ============================================================================
# 索敵メモリ(キャリア視点)
# ============================================================================

class SightingMemory:
    def __init__(self):
        self.last_seen_enemy = None

    def reset(self):
        self.last_seen_enemy = None

    def update(self, carrier, defenders, smoke_cells):
        if carrier is None or not carrier.is_alive:
            return
        visible = [d for d in defenders if d.is_alive and has_los(carrier.pos, d.pos, smoke_cells)]
        if visible:
            nearest = min(
                visible,
                key=lambda d: max(abs(d.pos[0] - carrier.pos[0]), abs(d.pos[1] - carrier.pos[1])),
            )
            self.last_seen_enemy = {"pos": tuple(nearest.pos), "name": nearest.name, "tick_ago": 0}
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ============================================================================
# ネットワーク
# ============================================================================

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "next_mask", "done"))
# AttackerCarryDuelingDQN(hidden=64) は tv2_common_rl.DuelingQNet(OBS_DIM, ACTION_DIM, hidden=64)
# で層構成が完全一致(value/advantage headはhidden//2=32で従来と同じ)。
# ReplayBufferは共通版を使用。

# ============================================================================
# 観測構築(簡略版: エスコート関連を削除。移動方向ガイダンスも不要)
# ============================================================================

def build_observation(
    carrier, defenders, smoke_cells, own_smoke_active, elapsed_ticks, dist_map,
    reached_waypoint, on_target, next_step=None,
):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    r0, c0 = int(carrier.pos[0]), int(carrier.pos[1])

    obs[0] = carrier.pos[0] / HEIGHT
    obs[1] = carrier.pos[1] / WIDTH
    obs[2] = carrier.hp / carrier.max_hp if carrier.max_hp else 0.0
    obs[3] = 1.0 if carrier.moved_this_tick else 0.0

    ability_index = {"SMOKE": 4, "FLASH": 5, "RECON": 6, "HUNT": 7}.get(carrier.ability_name)
    if ability_index is not None:
        obs[ability_index] = 1.0
    obs[8] = 1.0 if carrier.charges > 0 else 0.0

    bfs_dist = dist_map[r0, c0] if dist_map is not None else -1
    if bfs_dist < 0:
        bfs_dist = HEIGHT + WIDTH
    obs[9] = min(bfs_dist, HEIGHT + WIDTH) / (HEIGHT + WIDTH)
    obs[10] = 1.0 if reached_waypoint else 0.0
    obs[11] = 1.0 if on_target else 0.0

    visible_enemies = [d for d in defenders if d.is_alive and has_los(carrier.pos, d.pos, smoke_cells)]
    obs[12] = 1.0 if visible_enemies else 0.0
    obs[13] = len(visible_enemies) / 5.0
    if visible_enemies:
        nearest = min(
            visible_enemies,
            key=lambda d: max(abs(d.pos[0] - r0), abs(d.pos[1] - c0)),
        )
        obs[14] = (nearest.pos[0] - r0) / HEIGHT
        obs[15] = (nearest.pos[1] - c0) / WIDTH
        dist = max(abs(nearest.pos[0] - r0), abs(nearest.pos[1] - c0))
        obs[16] = min(dist, HEIGHT) / HEIGHT
    obs[17] = 1.0 if any(d.is_alive and (d.blind_remaining > 0 or d.reveal_remaining > 0) for d in defenders) else 0.0
    obs[18] = 1.0 if own_smoke_active else 0.0
    obs[19] = min(elapsed_ticks, MAX_TICKS) / MAX_TICKS
    obs[20] = sum(1 for d in defenders if d.is_alive) / 5.0

    if carrier.facing in FACING_DIRS:
        obs[21 + FACING_DIRS.index(carrier.facing)] = 1.0

    # 次に進む予定のマス(BFSで決定済みなので事前計算可能)への方向と、
    # そのマスからさらに奥がどれだけ開けているか。敵の実位置は使わない。
    if next_step is not None:
        r0, c0 = int(carrier.pos[0]), int(carrier.pos[1])
        dr, dc = int(next_step[0]) - r0, int(next_step[1]) - c0
        if (dr, dc) in CARDINAL:
            dir_idx = CARDINAL.index((dr, dc))
            obs[29 + dir_idx] = 1.0
            obs[33] = CELL_OPENNESS.get(tuple(next_step), [0.0, 0.0, 0.0, 0.0])[dir_idx]

    return obs


def decode_action(action_idx):
    """action_idx = base_idx(0-2) * 8 + facing_idx(0-7)。
    base_idxはNONE/ABILITY/PLANT、facing_idxは向き(N/NE/E/SE/S/SW/W/NW)で、
    両者は完全に独立(tv2_train_defender_search.pyのdecode_actionと同一規約)。
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


def build_action_mask(unit, on_target, ability_available=True):
    """向き(facing)は移動・アビリティとは無関係に常に自由選択できるため、
    base(0-2)側のマスクをfacing方向数だけ展開する
    (tv2_train_defender_search.pyと同一規約)。
    ability_available=Falseの場合は、chargesが残っていてもABILITYを選べない
    (SMOKEが定点圏外なのに無駄撃ちしてチャージを失うのを防ぐゲート)。"""
    base_mask = np.ones(BASE_ACTION_DIM, dtype=bool)
    if unit.charges <= 0 or unit.ability_name in ("HUNT", "NONE") or not ability_available:
        base_mask[ACTION_ABILITY] = False
    base_mask[ACTION_PLANT] = bool(on_target)
    return np.repeat(base_mask, len(FACING_DIRS))


# ============================================================================
# 環境本体
# ============================================================================

class CarryEnv:
    """carryフェーズを模した簡易環境(簡略版)。
    移動は決定的BFS経路探索。DQNが学習するのはアビリティ使用判断とPLANTのみ。"""

    def __init__(self):
        self.sighting = SightingMemory()
        self.attackers = []
        self.defenders = []
        self.carrier = None
        self.smokes = []
        self.elapsed_ticks = 0
        self.plant_progress = 0
        self.dist_map = None
        self.match_over_reason = None
        self._prev_kills = {}
        self._prev_alive = {}
        self._prev_hp = {}
        self.active_site = None
        self.target_plant_pos = None
        self.reached_waypoint = True
        self.smoke_lineup_used = False

    def reset(self):
        self.sighting.reset()
        self.smokes = []
        self.elapsed_ticks = 0
        self.plant_progress = 0
        self.match_over_reason = None
        self.smoke_lineup_used = False

        handoff = random.random() < HANDOFF_AUGMENT_PROB
        if handoff:
            carrier_name = random.choice(TOUYAMA_ROSTER_ORDER)
        else:
            carrier_name = TOUYAMA_SPIKE_HOLDER

        self.attackers = _build_fixed_attackers(carrier_name, handoff=handoff)
        self.carrier = next(a for a in self.attackers if a.name == carrier_name)

        self.active_site = _choose_weighted_site()
        # 優先設置場所(5)を最優先でtargetにする。構造的に1つも無いサイトの場合のみ
        # 通常のプラント可能マス(2)から選ぶ。
        candidates = PRIORITY_CELLS_BY_SITE.get(self.active_site) or []
        if not candidates:
            candidates = PLANT_CELLS_BY_SITE.get(self.active_site) or []
        if not candidates:
            fallback_site = "right" if self.active_site == "left" else "left"
            candidates = (
                PRIORITY_CELLS_BY_SITE.get(fallback_site)
                or PLANT_CELLS_BY_SITE.get(fallback_site)
                or PLANT_CELLS
            )
            self.active_site = fallback_site
        self.target_plant_pos = random.choice(candidates)
        self.defenders = _build_defenders(
            bias_cell=self.target_plant_pos, bias_strength=DEFENDER_SITE_BIAS_STRENGTH,
            chokepoint_bias_strength=DEFENDER_CHOKEPOINT_BIAS_STRENGTH,
        )

        if self.active_site in WAYPOINT_DIST_MAPS:
            self.reached_waypoint = False
            self.dist_map = WAYPOINT_DIST_MAPS[self.active_site]
        else:
            self.reached_waypoint = True
            self.dist_map = PLANT_DIST_MAPS[self.target_plant_pos]

        self.sighting.update(self.carrier, self.defenders, self._smoke_cells())

        all_units = self.attackers + self.defenders
        self._prev_kills = {u.name: u.kills for u in all_units}
        self._prev_alive = {u.name: u.is_alive for u in all_units}
        self._prev_hp = {u.name: u.hp for u in all_units}

        return self._collect_observation()

    def _smoke_cells(self):
        cells = set()
        for s in self.smokes:
            if s["remaining_ticks"] > 0:
                cells.update(s["cells"])
        return cells

    def _own_smoke_active(self):
        return any(s["team"] == "A" and s["remaining_ticks"] > 0 for s in self.smokes)

    def _current_nav_goal(self):
        if not self.reached_waypoint:
            return WAYPOINT_CELLS[self.active_site]
        return self.target_plant_pos

    def _time_critical(self, pos):
        """優先設置場所への到達+設置完了(PLANT_REQUIRED_TICKS)が、残りTickでは
        間に合わない場合True。間に合わない場合のみ通常マスでの妥協設置を許可する。"""
        r, c = int(pos[0]), int(pos[1])
        dist_to_priority = PLANT_DIST_MAPS[self.target_plant_pos][r, c]
        if dist_to_priority < 0:
            return True  # 経路が無い(通常起こらないが保険)
        remaining_ticks = MAX_TICKS - self.elapsed_ticks
        ticks_needed = dist_to_priority + PLANT_REQUIRED_TICKS
        return remaining_ticks < ticks_needed

    def _plantable_cell(self, pos):
        """このマスでPLANTを許可するか。優先設置場所(5)は常に許可。通常のプラント
        可能マス(2)は、優先設置場所への到達が時間的に間に合わない場合のみ妥協的に許可する。"""
        pos_t = tuple(map(int, pos))
        if pos_t == self.target_plant_pos:
            return True
        if pos_t in PLANT_CELLS_BY_SITE.get(self.active_site, ()):
            return self._time_critical(pos_t)
        return False

    def _carrier_ability_target_available(self):
        """ABILITYを選ぶ「意味」があるかどうか。charges>0だけで許可すると、
        SMOKEが定点(射程内)に到達する前にランダムなプラントマスへ無駄撃ち
        してチャージを浪費してしまう(1ラウンド1回しか使えないため致命的)。
        PLANTのon_targetゲートと同様に、ABILITY自体をマスクして温存させる。
        FLASH/RECONは従来通りchargesがあれば常に選択可とする(sighting経由の
        フォールバックにも一定の価値があるため)。"""
        if self.carrier.ability_name != "SMOKE":
            return True
        smoke_cells = self._smoke_cells()
        visible_enemies = [
            d for d in self.defenders if d.is_alive and has_los(self.carrier.pos, d.pos, smoke_cells)
        ]
        nearby_enemies = [
            d for d in visible_enemies
            if max(abs(d.pos[0] - self.carrier.pos[0]), abs(d.pos[1] - self.carrier.pos[1])) <= ABILITY_RANGE
        ]
        if nearby_enemies:
            return True
        lineup_candidates = [
            cell for cell in SMOKE_LINEUP_CELLS_BY_SITE.get(self.active_site, [])
            if max(abs(int(cell[0]) - int(self.carrier.pos[0])), abs(int(cell[1]) - int(self.carrier.pos[1]))) <= ABILITY_RANGE
        ]
        return bool(lineup_candidates)

    def _smoke_lineup_opportunity(self):
        """定点スモークを「今すぐ焚ける」状態かどうか(近接脅威なし・射程内に定点あり)。
        wait penalty算出専用。"""
        if self.carrier.ability_name != "SMOKE" or self.carrier.charges <= 0:
            return False
        smoke_cells = self._smoke_cells()
        nearby_enemies = [
            d for d in self.defenders
            if d.is_alive and has_los(self.carrier.pos, d.pos, smoke_cells)
            and max(abs(d.pos[0] - self.carrier.pos[0]), abs(d.pos[1] - self.carrier.pos[1])) <= ABILITY_RANGE
        ]
        if nearby_enemies:
            return False
        lineup_candidates = [
            cell for cell in SMOKE_LINEUP_CELLS_BY_SITE.get(self.active_site, [])
            if max(abs(int(cell[0]) - int(self.carrier.pos[0])), abs(int(cell[1]) - int(self.carrier.pos[1]))) <= ABILITY_RANGE
        ]
        return bool(lineup_candidates)

    def _carrier_lookahead_step(self):
        """Carrierが次に進む予定のマスを事前計算する(実際の移動判定と同一の
        BFSロジック・同一occupied集合を使うため、後でstep()が計算する実際の
        移動先と一致する)。"""
        if not self.carrier.is_alive:
            return None
        occupied = {tuple(u.pos) for u in self.attackers + self.defenders if u.is_alive}
        own_occupied = occupied - {tuple(self.carrier.pos)}
        goal = self._current_nav_goal()
        allow_adjacent = not self.reached_waypoint
        return bfs_next_step(
            tuple(self.carrier.pos), goal, own_occupied, allow_adjacent_goal=allow_adjacent
        )

    def _collect_observation(self):
        smoke_cells = self._smoke_cells()
        on_target = self.carrier.is_alive and self._plantable_cell(self.carrier.pos)
        next_step = self._carrier_lookahead_step()
        ability_available = self.carrier.is_alive and self._carrier_ability_target_available()
        obs = build_observation(
            self.carrier, self.defenders, smoke_cells, self._own_smoke_active(),
            self.elapsed_ticks, self.dist_map, self.reached_waypoint, on_target,
            next_step=next_step,
        )
        mask = build_action_mask(self.carrier, on_target, ability_available)
        return obs, mask

    def step(self, action_idx):
        self.elapsed_ticks += 1
        for u in self.attackers + self.defenders:
            u.moved_this_tick = False
            u.facing_forced_this_tick = False
            if u.forced_facing_next_tick:
                u.facing = u.forced_facing_next_tick
                u.facing_forced_this_tick = True
            u.forced_facing_next_tick = None

        pre_flash_recon_active = any(
            d.is_alive and (d.blind_remaining > 0 or d.reveal_remaining > 0) for d in self.defenders
        )

        smoke_cells = self._smoke_cells()
        occupied = {tuple(u.pos) for u in self.attackers + self.defenders if u.is_alive}

        move_plans = []
        ability_whiff = False
        ability_overlap = False

        carrier_alive = self.carrier.is_alive
        on_target_before_action = carrier_alive and self._plantable_cell(self.carrier.pos)
        on_priority_before_action = carrier_alive and tuple(self.carrier.pos) == self.target_plant_pos
        smoke_lineup_opportunity_before_action = carrier_alive and self._smoke_lineup_opportunity()

        # --- 敵(Defender)側: ランダム移動+近接ヒューリスティックアビリティ ---
        for d in self.defenders:
            if not d.is_alive:
                continue
            own_occupied = occupied - {tuple(d.pos)}
            visible = [a for a in self.attackers if a.is_alive and has_los(d.pos, a.pos, smoke_cells)]
            ability = _heuristic_ability_action(d, visible)
            if ability is not None:
                _apply_ability(d, ability[0], ability[1], self.smokes, self.attackers + self.defenders, smoke_cells)
                move_plans.append((d, tuple(d.pos)))
                continue
            nxt = _defender_move(d, own_occupied)
            move_plans.append((d, nxt))

        # --- エスコート4人: ヒューリスティック(carrierの意思決定からは独立) ---
        for a in self.attackers:
            if a is self.carrier or not a.is_alive:
                continue
            own_occupied = occupied - {tuple(a.pos)}
            visible = [d for d in self.defenders if d.is_alive and has_los(a.pos, d.pos, smoke_cells)]
            ability = _heuristic_ability_action(a, visible)
            if ability is not None:
                _apply_ability(a, ability[0], ability[1], self.smokes, self.attackers + self.defenders, smoke_cells)
                move_plans.append((a, tuple(a.pos)))
                continue
            nxt = _escort_move(a, self.carrier, own_occupied)
            move_plans.append((a, nxt))

        # --- キャリア: 移動はBFS決定的、DQNはアビリティ/PLANTのみ判断 ---
        plant_action_chosen = False
        carrier_explicit_facing = None
        carrier_had_visible_enemy = False
        carrier_move_dir = None
        smoke_lineup_used = False
        if carrier_alive:
            decoded, carrier_explicit_facing = decode_action(action_idx)
            visible_enemies = [
                d for d in self.defenders if d.is_alive and has_los(self.carrier.pos, d.pos, smoke_cells)
            ]
            carrier_had_visible_enemy = bool(visible_enemies)

            if decoded == "PLANT":
                move_plans.append((self.carrier, tuple(self.carrier.pos)))
                plant_action_chosen = True
            else:
                own_occupied = occupied - {tuple(self.carrier.pos)}
                goal = self._current_nav_goal()
                # 中継地点は隣接到達でOK(別途waypoint_dist<=1で判定済み)だが、
                # 最終プラント目標は正確な到達が必須。隣接停止のままだとon_targetが
                # 永久にFalseになりPLANTを一切選べなくなる(今回のバグの本体)。
                allow_adjacent = not self.reached_waypoint
                nxt = bfs_next_step(tuple(self.carrier.pos), goal, own_occupied, allow_adjacent_goal=allow_adjacent)
                move_plans.append((self.carrier, nxt))
                carrier_move_dir = (nxt[0] - self.carrier.pos[0], nxt[1] - self.carrier.pos[1])

                if decoded == "ABILITY":
                    is_smoke = self.carrier.ability_name == "SMOKE"
                    # 「見えているか」ではなく「射程内に脅威がいるか」で判定する。
                    # 遠方に敵が1体見えているだけで定点スモークを諦めるのは不自然なため。
                    nearby_enemies = [
                        d for d in visible_enemies
                        if max(abs(d.pos[0] - self.carrier.pos[0]), abs(d.pos[1] - self.carrier.pos[1])) <= ABILITY_RANGE
                    ]
                    # 定点(SMOKE_LINEUP_CELLS_BY_SITE)のうち、射程内にあるものだけを候補にする。
                    # goal(中継地点/設置目標)とは無関係の、事前に指定した「射線を切る座標」。
                    lineup_candidates = [
                        cell for cell in SMOKE_LINEUP_CELLS_BY_SITE.get(self.active_site, [])
                        if max(abs(int(cell[0]) - int(self.carrier.pos[0])), abs(int(cell[1]) - int(self.carrier.pos[1]))) <= ABILITY_RANGE
                    ]
                    smoke_lineup = is_smoke and not nearby_enemies and bool(lineup_candidates)
                    ability_whiff = (not nearby_enemies) and not smoke_lineup
                    ability_overlap = pre_flash_recon_active and self.carrier.ability_name in ("FLASH", "RECON")
                    if self.carrier.charges > 0:
                        if nearby_enemies:
                            nearest = min(
                                nearby_enemies,
                                key=lambda d: max(abs(d.pos[0] - self.carrier.pos[0]), abs(d.pos[1] - self.carrier.pos[1])),
                            )
                            _apply_ability(
                                self.carrier, self.carrier.ability_name, tuple(nearest.pos),
                                self.smokes, self.attackers + self.defenders, smoke_cells,
                            )
                        elif smoke_lineup:
                            # 射程内に脅威がいない状態で、指定済みの定点へスモークを焚く(事前設置)。
                            lineup_target = min(
                                lineup_candidates,
                                key=lambda cell: max(
                                    abs(int(cell[0]) - int(self.carrier.pos[0])),
                                    abs(int(cell[1]) - int(self.carrier.pos[1])),
                                ),
                            )
                            smoke_lineup_used = True
                            self.smoke_lineup_used = True
                            _apply_ability(
                                self.carrier, self.carrier.ability_name, tuple(map(int, lineup_target)),
                                self.smokes, self.attackers + self.defenders, smoke_cells,
                            )
                        elif self.sighting.last_seen_enemy is not None:
                            _apply_ability(
                                self.carrier, self.carrier.ability_name, self.sighting.last_seen_enemy["pos"],
                                self.smokes, self.attackers + self.defenders, smoke_cells,
                            )
                        else:
                            _apply_ability(
                                self.carrier, self.carrier.ability_name, random.choice(PLANT_CELLS),
                                self.smokes, self.attackers + self.defenders, smoke_cells,
                            )   

        # --- 移動の適用: キャリア最優先 ---
        move_plans.sort(key=lambda pair: 0 if pair[0] is self.carrier else 1)

        for unit, target_pos in move_plans:
            if not unit.is_alive:
                continue
            old_pos = tuple(unit.pos)
            nr, nc = int(target_pos[0]), int(target_pos[1])
            in_bounds = 0 <= nr < HEIGHT and 0 <= nc < WIDTH
            is_wall = in_bounds and GRID[nr, nc] == 1
            occ = any(
                other is not unit and other.is_alive and tuple(other.pos) == (nr, nc)
                for other in self.attackers + self.defenders
            )
            if in_bounds and not is_wall and not occ:
                unit.pos = [nr, nc]
            unit.moved_this_tick = tuple(unit.pos) != old_pos

            # キャリアはDQNが選んだ向き(移動方向とは無関係)を優先する。
            # ただし前Tickで被弾していれば、それが常に最優先(battle_logicと同一)。
            explicit_facing = carrier_explicit_facing if unit is self.carrier else None
            if not unit.facing_forced_this_tick:
                if explicit_facing in FACING_VECTORS:
                    unit.facing = explicit_facing
                else:
                    unit.facing = _facing_from_delta(
                        unit.pos[0] - old_pos[0], unit.pos[1] - old_pos[1], unit.facing
                    )

        self._resolve_shots()

        for u in self.attackers + self.defenders:
            u.blind_remaining = max(0, u.blind_remaining - 1)
            u.reveal_remaining = max(0, u.reveal_remaining - 1)
        for s in self.smokes:
            s["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

        plant_tick_progress = False
        plant_completed = False
        plant_tick_progress = False
        plant_completed = False
        plant_completed_on_priority = False
        if plant_action_chosen and on_target_before_action and self.carrier.is_alive:
            self.plant_progress += 1
            plant_tick_progress = True
            if self.plant_progress >= PLANT_REQUIRED_TICKS:
                plant_completed = True
                plant_completed_on_priority = on_priority_before_action
        else:
            self.plant_progress = 0

        self.sighting.update(self.carrier, self.defenders, self._smoke_cells())

        # 中継地点通過判定(未通過の場合のみ)。
        waypoint_bonus = 0.0
        if self.carrier.is_alive and not self.reached_waypoint:
            wr, wc = int(self.carrier.pos[0]), int(self.carrier.pos[1])
            waypoint_cell = WAYPOINT_CELLS[self.active_site]
            waypoint_dist = max(abs(wr - waypoint_cell[0]), abs(wc - waypoint_cell[1]))
            if waypoint_dist <= 1:
                self.reached_waypoint = True
                waypoint_bonus = WAYPOINT_REACHED_REWARD
                self.dist_map = PLANT_DIST_MAPS[self.target_plant_pos]

        reward, done = self._compute_reward(
            ability_whiff, ability_overlap, plant_tick_progress, plant_completed,
            plant_action_chosen, on_target_before_action, waypoint_bonus,
            plant_completed_on_priority, carrier_explicit_facing,
            carrier_had_visible_enemy, carrier_move_dir, smoke_lineup_used,
            smoke_lineup_opportunity_before_action,
        )

        all_units = self.attackers + self.defenders
        self._prev_kills = {u.name: u.kills for u in all_units}
        self._prev_alive = {u.name: u.is_alive for u in all_units}
        self._prev_hp = {u.name: u.hp for u in all_units}

        obs, mask = self._collect_observation() if self.carrier.is_alive else (
            np.zeros(OBS_DIM, dtype=np.float32),
            np.repeat(np.array([True, False, False], dtype=bool), len(FACING_DIRS)),
        )
        return obs, mask, reward, done

    def _resolve_shots(self):
        alive = [u for u in self.attackers + self.defenders if u.is_alive]
        smoke_cells = self._smoke_cells()
        shot_intents = []

        for shooter in alive:
            targets = [t for t in alive if t.team != shooter.team and has_los(shooter.pos, t.pos, smoke_cells)]
            # 正面から左右x度を超える(背後含む)相手は視界外のため撃てない(battle_logicと同一ロジック)。
            targets = [t for t in targets if _facing_angle_diff(shooter, t) <= SHOOTING_SITE_DIGREE]
            if not targets:
                continue
            target = min(
                targets,
                key=lambda t: (
                    max(abs(t.pos[0] - shooter.pos[0]), abs(t.pos[1] - shooter.pos[1])),
                    t.hp, t.name,
                ),
            )
            shot_intents.append((shooter, target))

        random.shuffle(shot_intents)
        shot_intents.sort(key=lambda pair: pair[0].reaction, reverse=True)

        for shooter, target in shot_intents:
            if not shooter.is_alive or not target.is_alive:
                continue

            accuracy = MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            # 正面からの角度差による補正(正面100%～真横50%、battle_logicと同一ロジック)
            accuracy *= _facing_accuracy_multiplier(shooter, target)
            if shooter.blind_remaining > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            debuffed = target.blind_remaining > 0 or target.reveal_remaining > 0
            effective_dodge = target.dodge_rate * (REVEALED_DODGE_MULTIPLIER if debuffed else 1.0)
            hit_chance = accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            # 命中・被弾を問わず、撃たれたら次のTickだけ相手の方向を強制的に向く(battle_logicと同一ロジック)。
            target.forced_facing_next_tick = _facing_towards(target.pos, shooter.pos)

            if random.random() < hit_chance:
                headshot = random.random() < shooter.hs_rate
                damage = HEADSHOT_DAMAGE if headshot else BODY_DAMAGE
                target.hp = max(0, target.hp - damage)
                if target.hp <= 0:
                    target.is_alive = False
                    shooter.kills += 1
                    if target.has_spike:
                        target.has_spike = False

    def _compute_reward(
        self, ability_whiff, ability_overlap, plant_tick_progress, plant_completed,
        plant_action_chosen, on_target_before_action, waypoint_bonus=0.0,
        plant_completed_on_priority=False, chosen_facing=None,
        had_visible_enemy=False, move_dir=None, smoke_lineup_used=False,
        smoke_lineup_opportunity_before_action=False,
    ):
        reward = STEP_PENALTY + waypoint_bonus + (SMOKE_LINEUP_REWARD if smoke_lineup_used else 0.0)
        if smoke_lineup_opportunity_before_action and not smoke_lineup_used:
            reward += SMOKE_LINEUP_WAIT_PENALTY

        # 視認中の敵がいない時だけ、進行方向を向く弱いshaping報酬を与える。
        # 敵が見えている時は通常の交戦報酬(命中率経由)に完全に委ねる。
        if not had_visible_enemy and chosen_facing in FACING_VECTORS and move_dir is not None:
            dr, dc = move_dir
            move_len = math.hypot(dr, dc)
            if move_len > 0:
                fx, fy = FACING_VECTORS[chosen_facing]
                alignment = (fx * dc + fy * dr) / move_len
                reward += FACING_ALIGN_WEIGHT * alignment

        was_alive = self._prev_alive.get(self.carrier.name, True)

        if was_alive and not self.carrier.is_alive:
            reward += DEATH_PENALTY
            self.match_over_reason = "carrier_died"
            return reward, True

        if not self.carrier.is_alive:
            return reward, True

        hp_lost = max(0, self._prev_hp.get(self.carrier.name, self.carrier.hp) - self.carrier.hp)
        reward += DAMAGE_TAKEN_PENALTY_SCALE * hp_lost

        if ability_whiff:
            reward += ABILITY_WHIFF_PENALTY
        if ability_overlap:
            reward += ABILITY_OVERLAP_PENALTY

        new_kills = self.carrier.kills - self._prev_kills.get(self.carrier.name, self.carrier.kills)
        if new_kills > 0:
            reward += KILL_REWARD * new_kills

        if plant_action_chosen and not on_target_before_action:
            reward += PLANT_WHIFF_PENALTY

        if on_target_before_action and not plant_action_chosen:
            reward += ON_TARGET_NON_PLANT_PENALTY

        if plant_tick_progress:
            reward += PLANT_TICK_BONUS

        if plant_completed:
            if plant_completed_on_priority:
                reward += PLANT_SUCCESS_REWARD
                self.match_over_reason = "planted"
            else:
                reward += PLANT_SUCCESS_REWARD_COMPROMISE
                self.match_over_reason = "planted_compromise"
            return reward, True

        if self.elapsed_ticks >= MAX_TICKS:
            reward += TIME_EXPIRE_PENALTY
            self.match_over_reason = "time_expired"
            return reward, True

        if not any(a.is_alive for a in self.attackers if a is not self.carrier):
            if self.match_over_reason != "escort_wiped_noted":
                reward += TEAM_WIPE_PENALTY
                self.match_over_reason = "escort_wiped_noted"

        return reward, False


# ============================================================================
# 学習ループ
# ============================================================================

def epsilon_by_episode(episode, total_episodes=EPISODE_COUNT, eps_start=1.0, eps_end=0.05, decay_ratio=0.8):
    decay_episodes = total_episodes * decay_ratio
    return max(eps_end, eps_start - (eps_start - eps_end) * episode / decay_episodes)


# select_action / optimize は tv2_common_rl.select_action /
# tv2_common_rl.optimize_double_dqn_step を使用(呼び出し側train()を参照)。

EVAL_EVERY = 20          # 何episode毎にgreedy評価を行うか
EVAL_EPISODES = 50       # 1回の評価で回すgreedy episode数(20は分散が大きすぎるため増量)
EVAL_SEED = 12345        # 評価専用の固定シード。チェックポイント間で同じ「問題セット」を
                         # 使うことで、evalごとの当たり外れ(乱数ガチャ)を除いて公平比較する
EVAL_CONFIRM_WINDOW = 3  # best更新判定に使う直近eval結果の個数(移動平均で判定しブレを抑える)


def evaluate(policy_net, episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """epsilon=0の完全greedyでロールアウトし、探索ノイズを含まない実力を測る。
    学習は行わない(バッファへのpush・optimizeなし)。
    固定シードでロールアウトすることで、チェックポイント間で同じ配置パターンを
    使い、eval結果の比較を公平にする(学習用RNGの状態には影響しない)。"""
    env = CarryEnv()
    policy_net.eval()
    reward_total = 0.0
    success_count = 0
    smoke_lineup_count = 0

    # 学習用のグローバル乱数状態を汚さないよう退避してから固定シードに切り替える
    rng_state = random.getstate()
    random.seed(seed)
    try:
        with torch.no_grad():
            for _ in range(episodes):
                obs, mask = env.reset()
                episode_reward = 0.0
                for _tick in range(MAX_TICKS):
                    action = select_action(policy_net, obs, mask, epsilon=0.0, fallback_action=0)
                    obs, mask, reward, done = env.step(action)
                    episode_reward += reward
                    if done:
                        break
                reward_total += episode_reward
                if env.match_over_reason in ("planted", "planted_compromise"):
                    success_count += 1
                if env.smoke_lineup_used:
                    smoke_lineup_count += 1
    finally:
        random.setstate(rng_state)

    policy_net.train()
    avg_reward = reward_total / episodes
    success_rate = success_count / episodes
    smoke_lineup_rate = smoke_lineup_count / episodes
    return avg_reward, success_rate, smoke_lineup_rate


def train(
    episodes=EPISODE_COUNT,
    batch_size=128,
    gamma=0.99,
    lr=1e-4,
    buffer_size=200_000,
    target_update_every=1000,
):
    policy_net = DuelingQNet(OBS_DIM, ACTION_DIM, hidden=64).to(DEVICE)
    target_net = DuelingQNet(OBS_DIM, ACTION_DIM, hidden=64).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(Transition, capacity=buffer_size)
    env = CarryEnv()

    global_step = 0
    best_eval_reward = -float("inf")
    episode_reward_history = deque(maxlen=100)
    episode_success_history = deque(maxlen=100)
    eval_reward_recent = deque(maxlen=EVAL_CONFIRM_WINDOW)

    def _save_checkpoint(path, episode_no, success_rate_value):
        torch.save(
            {
                "model_state_dict": policy_net.state_dict(),
                "obs_dim": OBS_DIM,
                "n_actions": ACTION_DIM,
                "episode": episode_no,
                "success_rate": success_rate_value,
                "priority_cells": list(PRIORITY_CELLS),
                "has_priority_cells": bool(PRIORITY_CELLS),
                "waypoint_cells": {site: tuple(cell) for site, cell in WAYPOINT_CELLS.items()},
                "smoke_lineup_cells_by_site": {
                    site: [tuple(map(int, cell)) for cell in cells]
                    for site, cells in SMOKE_LINEUP_CELLS_BY_SITE.items()
                },
            },
            path,
        )

    start_time = time.perf_counter()

    for episode in range(1, episodes + 1):
        obs, mask = env.reset()
        episode_reward_total = 0.0
        epsilon = epsilon_by_episode(episode)

        for tick in range(MAX_TICKS):
            action = select_action(policy_net, obs, mask, epsilon, fallback_action=0)
            next_obs, next_mask, reward, done = env.step(action)
            episode_reward_total += reward

            buffer.push(obs, action, reward, next_obs, next_mask, float(done))

            obs, mask = next_obs, next_mask
            global_step += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                optimize_double_dqn_step(
                    policy_net, target_net, optimizer,
                    batch.obs, batch.action, batch.reward,
                    batch.next_obs, batch.done, batch.next_mask,
                    gamma,
                )

            if global_step % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_reward_history.append(episode_reward_total)
        episode_success_history.append(
            1.0 if env.match_over_reason in ("planted", "planted_compromise") else 0.0
        )
        avg_reward = sum(episode_reward_history) / len(episode_reward_history)
        success_rate = sum(episode_success_history) / len(episode_success_history)

        if episode % 20 == 0:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter()
            print(
                f"[EP {episode}/{episodes}] reward={episode_reward_total:.3f} elapse={elapsed_time:.1f} "
                f"avg100={avg_reward:.3f} success100={success_rate:.3f} "
                f"epsilon={epsilon_by_episode(episode):.3f} "
                f"buffer={len(buffer)} reason={env.match_over_reason}"
            )

        # --- best model判定はexploration込みのavg100ではなく、epsilon=0の
        # greedy評価(evaluate())で行う。学習中のepsilonが高いうちは
        # ランダム行動の当たり外れで保存されてしまうのを防ぐため。
        if episode >= EVAL_MIN_EPISODE and episode % EVAL_EVERY == 0:
            eval_reward, eval_success_rate, eval_smoke_lineup_rate = evaluate(policy_net)
            eval_reward_recent.append(eval_reward)
            eval_reward_smoothed = sum(eval_reward_recent) / len(eval_reward_recent)
            print(
                f"[EVAL @ EP {episode}] eval_reward={eval_reward:.3f} "
                f"eval_success={eval_success_rate:.3f} smoke_lineup={eval_smoke_lineup_rate:.3f} "
                f"smoothed={eval_reward_smoothed:.3f} "
                f"(greedy, {EVAL_EPISODES}episodes, fixed_seed={EVAL_SEED})"
            )
            # 直近EVAL_CONFIRM_WINDOW回分が揃ってから、その移動平均でbest判定する
            # (1回だけたまたま良い結果を引いて即保存する事故を防ぐ)。
            if len(eval_reward_recent) >= EVAL_CONFIRM_WINDOW and eval_reward_smoothed > best_eval_reward:
                best_eval_reward = eval_reward_smoothed
                _save_checkpoint(MODEL_SAVE_PATH, episode, eval_success_rate)
                print(
                    f"[SAVE] best model updated: smoothed_eval_reward={eval_reward_smoothed:.3f} "
                    f"(latest eval_success={eval_success_rate:.3f}) -> {MODEL_SAVE_PATH}"
                )

        if episode % 100 == 0:
            _save_checkpoint(MODEL_LATEST_PATH, episode, success_rate)

    print("[DONE] training finished.")


if __name__ == "__main__":
    train()