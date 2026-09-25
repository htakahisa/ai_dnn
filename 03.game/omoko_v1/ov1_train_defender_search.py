"""omoko_v1/ov1_train_defender_search.py

固定チーム(いぐるん/夢の街/ひつじさん/Tortlilyan/えんぺん)専用の
Defender「search phase」学習スクリプト（プラント前限定）。

train_defender_search.py(root/defender_v3)をベースに、以下を固定化した:
  - ステータス: character_stats.py の生値 + 常時発動コンボ
    「ふわんだりぃず」(ひつじさん/えんぺん/いぐるん) + タイガーパッシブ(Tortlilyan)
  - ロール: 上記5人のロールに完全固定(ランダム選択なし)
  - スポーン: ロースター順 = DEFENDER_SPAWNS(行優先走査順)の対応で固定
    (run_game.py の実際のスポーン割り当てロジックと一致)

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py などのfeatureモジュールは一切importしない。
character_stats.py は game_core.py / map_data.py と同様、
ロジックを含まない定数専用ファイルとして参照する(import制限の対象外)。

学習データ・チェックポイントは omoko_v1/data/defender_search_data/
以下に保存する。

--------------------------------------------------------------------------
優先順位ツリー(_compute_rewards / SearchEnv._priority_mode_and_distmap):
    1. スパイク確定情報(spike_pos)     -- 最優先。SPIKE_PULL_REWARD
    2. 敵目撃情報(last_seen_enemy)      -- 次点。retake準備として全員が寄る。
       SIGHTING_PULL_REWARD
    3. どちらも無い場合                -- 担当する有利ポジション(7)へ向かい、
       到着後は静止する。DEFENSE_POSITION_PULL_REWARD / HOLD_POSITION_BONUS
--------------------------------------------------------------------------
"""

import os
import sys
import random
import math
from collections import deque, namedtuple
import time

# 標準出力のバッファリングによってログ表示が遅延・停止して見える問題を防ぐため、
# 実行時のコマンド(-uの有無)に依存せず、スクリプト側で明示的に行バッファリングへ切り替える。
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR

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
    RECON_REVEAL_SIZE,
    ROUND_DURATION_TICKS,
    PLANT_REQUIRED_TICKS,
    FACING_VECTORS,
    SHOOTING_SITE_DIGREE,
)
from map_data_defender_setup import DEFENDER_SETUP_MASK_STR, DEFENDER_SETUP_TICKS


from ov1_map_data_search import SEARCH_MAZE_STR
from character_stats import CHARACTER_TABLE as STATS_TABLE
import ov1_common_rl
from ov1_common_rl import DEVICE, DuelingQNet, ReplayBuffer, select_action, optimize_double_dqn_step
from ov1_ultimate_training import (
    collect_orb_tick,
    initialize_ultimate,
    orb_context,
    orb_priority,
    spend_ultimate,
    ultimate_context,
    ultimate_ready,
    ultimate_use_reward,
    valid_orb_cells,
)
from ov1_common_defender import (
    ROSTER_ORDER,
    compute_effective_stats,
    print_effective_stats,
)
from ov1_search_site_context import (
    SITE_CONTEXT_DIM, SITE_SUPPORT_RADIUS, site_context, site_index,
    walking_distance, regroup_pressure,
)
from ov1_defender_search_support import find_support_ability_plan

EPISODE_COUNT = 3000
EVAL_EVERY = 200          # 何エピソードごとにepsilon=0評価を行うか
EVAL_EPISODES = 100       # 1回の評価で何エピソード分プレイして平均するか
EVAL_MIN_EPISODE = int(EPISODE_COUNT * 0.7)  # epsilonが十分下がるまでbest更新の対象外にする
#MIN_EVAL_ARRIVAL_RATE = 0.70  # 配置到着率がこれ未満のモデルはbest候補から除外

# 各キャラの監視座標(複数可、配列形式)。マップではなくコード上で直接指定する。
# 壁越しなど視認不可能な座標は登録しないこと(視認可否のチェックはここでは行わない)。
# 味方が直線上に立ち、一時的に視線を塞ぐことはあり得るが許容する。
DEFENSE_WATCH_POINTS = {
    "ねこさん": [(16, 21)],
    "とりさん": [(16, 21)],
    "おもこ": [(16, 21)],
    "いぬさん": [(16, 21)],
    "ひつじさん": [(13, 23)],
}


# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = str(Path(__file__).resolve().parent / "data" / "defender_search_data")
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_defender_search_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_defender_search_latest.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------

CARDINAL = ov1_common_rl.CARDINAL_MOVES
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
AGENT_ID_DIM = len(ROSTER_ORDER)
AGENT_ID_OFFSET = 46
ULTIMATE_CONTEXT_DIM = 6
ORB_CONTEXT_DIM = 7
TACTICAL_CONTEXT_DIM = 8
OBS_DIM = (
    AGENT_ID_OFFSET + AGENT_ID_DIM
    + ULTIMATE_CONTEXT_DIM + ORB_CONTEXT_DIM + TACTICAL_CONTEXT_DIM
    + SITE_CONTEXT_DIM
)
              # + 8(自身のfacing one-hot。従来欠落していたため追加。POMDP化を防ぐ)
# 移動(5方向)*アビリティ有無(10通り) と 向き(N/NE/E/SE/S/SW/W/NW、8通り)を
# 完全に独立した直積として扱う: action_idx = base_idx(0-9) * 8 + facing_idx(0-7)。
# 移動先と向きは無関係に指定できる(例: 前進しながら後ろを向く)。
# 向きには直接報酬を与えず、battle_logic.pyと同じ命中率補正を経由した通常の
# 交戦結果(KILL_REWARD/DEATH_PENALTY)を通じて間接的に学習させる。
BASE_ACTION_DIM = 12
ACTION_ULTIMATE_BASE = 10
ACTION_ORB_BASE = 11
FACING_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_DIM = BASE_ACTION_DIM * len(FACING_DIRS)  # 96

N_DEFENDERS = 5
N_ATTACKERS = 5
MAX_TICKS = ROUND_DURATION_TICKS  # 90
SCHEDULED_SMOKE_DELAY_TICKS = 20  # Setup終了後、Sマークへ自動スモークするまで

ABILITY_RANGE = 8       # FLASH/RECONの照準可能な最大距離
SIGHTING_STALENESS_CAP = 30  # 観測に入れる目撃情報の経過tickをこの値で正規化する
REINFORCE_FROM_TICK = 30
SIGHTING_MEMORY_TICKS = MAX_TICKS - REINFORCE_FROM_TICK
SPIKE_CARRIER_MEMORY_TICKS = 12
REACH_RADIUS = 0        # 担当ポジションへ「到着した」とみなすBFS距離

# 敵(Attacker)側の既定ステータス(当面ヒューリスティックのため簡易値のまま)
DEFAULT_ACCURACY = 0.70
DEFAULT_DODGE = 0.42
DEFAULT_HS_RATE = 0.30
DEFAULT_REACTION = 150.0

ROLES = ["FLASH", "SMOKE", "RECON", "HUNT"]

# ---------------------------------------------------------------------------
# omoko_v1 固定チーム定義
# ---------------------------------------------------------------------------
SPIKE_HOLDER = "ひつじさん"  # このsearch phaseでは未使用。carry/guard学習用に保持。

# 💡注意: accuracy/hs_rate/dodge_rateのスケールが0-100→0-1に変わる
# (common_defender.compute_effective_stats に統一。合意事項)。
EFFECTIVE_STATS = compute_effective_stats(STATS_TABLE)
print_effective_stats(EFFECTIVE_STATS, "Defender/search")


# 報酬パラメータ
# 優先度: SPIKE > SIGHTING > DEFENSE_POSITION > HOLD_POSITION
# の順で明確に重みを引き離し、「待機の方が得」という学習結果を防ぐ。
STEP_PENALTY = -0.001
SPIKE_PULL_REWARD = 0.025        # 旧チェックポイントとの診断互換用
SPIKE_GROUND_PULL_REWARD = 0.01
SPIKE_GROUND_APPROACH_RADIUS = 4 # 担当ポジションからこの距離以内でのみSPIKE_GROUND_PULL_REWARDを付与
SIGHTING_PULL_REWARD = 0.12      # 30tick以降、敵目撃方向への増援を優先する
SIGHTING_IDLE_PENALTY = -0.04    # 遠い目撃地点へ移動できるのに待機しない
SITE_ROTATE_REWARD = 0.10        # 反対サイトから判明したサイトへ増援する
SITE_HOLD_REWARD = 0.025        # 担当サイトの有効な射線を維持する
SITE_ABANDON_PENALTY = -0.04    # 担当サイトから理由なく離れる
REGROUP_REWARD = 0.09          # 孤立・人数不利時に味方へ合流する
COMBAT_HOLD_BONUS = 0.01         # 射線中の静止はごく弱く評価し、再配置を妨げない
COMBAT_MOVE_PENALTY = 0.0        # 移動可否は射線参加と戦闘結果から学習させる
CROSSFIRE_JOIN_REWARD = 0.12     # 味方が見ている対象へ新たに射線参加
CROSSFIRE_MAINTAIN_REWARD = 0.015  # 2人以上の射線を維持（毎tickは弱く）
DEFENSE_POSITION_PULL_REWARD = 0.03   # 平常時、担当7地点へ寄る(ポテンシャル差分)
HOLD_POSITION_BONUS = 0.02            # 担当地点到着後、静止
HOLD_POSITION_PENALTY = -0.01         # 担当地点到着後、無駄にうろつく
ABILITY_WHIFF_PENALTY = -0.05
ABILITY_OVERLAP_PENALTY = -0.05
ABILITY_AIMED_REWARD = 0.03      # 視認あり・射程内で使用(外れても付与、whiffとは排他)
ABILITY_HIT_BONUS = 0.10         # 上記に加え、実際に命中/デバフ成立した場合に追加
SMOKE_SUPPORT_TARGET_BONUS = 0.08  # SMOKE: 味方の射線を保って敵の背後を遮断
DEBUFF_KILL_BONUS = 0.3
HOLD_ANGLE_BONUS = 0.005
HOLD_ANGLE_PENALTY = 0.0
# 💡追加: facing整合の弱いshaping報酬。自分が敵を直接視認していない時のみ有効。
# 優先順位はTeamMemoryの優先度ツリーと揃える: spike > sighting > 担当地点。
# 直接視認時は既存の交戦報酬(命中率経由)に完全に委ね、このshapingは加えない。
FACING_POSITION_REWARD = 0.25  # 配置後に指定監視位置を向く
FACING_ROUTE_REWARD = 0.08     # 移動経路の次方向(曲がり角の先読みを含む)を向く
FACING_SETUP_REWARD_TICKS = 5  # 到着後、監視方向を評価する tick 数
# 旧 combat/priority facing shaping は簡略化方針により無効化する。
FACING_ALIGN_SPIKE_WEIGHT = 0.0
FACING_ALIGN_SIGHTING_WEIGHT = 0.0
FACING_ALIGN_VISIBLE_WEIGHT = 0.0
FACING_ALIGN_POSITION_WEIGHT = 0.0
FACING_COMBAT_CORRECT_REWARD = 0.0
FACING_COMBAT_INCORRECT_PENALTY = 0.0
# 旧 combat shaping の互換用。旧ブロックは報酬係数が0のため無効だが、
# 内部の relevance 計算が参照するため定義だけ残す。
FACING_MEMORY_MAX_DISTANCE = 10
FACING_MEMORY_MAX_STALENESS = 12
SPIKE_WATCH_HOLD_BONUS = 0.02       # 落下中スパイクにLOSが通っている間、静止(待ち伏せ)
SPIKE_WATCH_MOVE_PENALTY = -0.01    # 同上、無駄にうろつく
KILL_REWARD = 0.5
DEATH_PENALTY = -0.5
ROUND_WIN_REWARD = 1.0          # 時間切れ・全滅によるDefender勝利
PLANT_PENALTY = -0.5            # このフェーズの範囲外(プラント成立)に至った場合


def crossfire_coordination_reward(unit_name, pre_los, post_los):
    """共有対象への射線参加を評価する純粋関数。"""
    joined = (
        not pre_los.get(unit_name, False)
        and post_los.get(unit_name, False)
        and any(
            has_los_before
            for name, has_los_before in pre_los.items()
            if name != unit_name
        )
    )
    reward = CROSSFIRE_JOIN_REWARD if joined else 0.0
    if sum(post_los.values()) >= 2 and post_los.get(unit_name, False):
        reward += CROSSFIRE_MAINTAIN_REWARD
    return reward


# ============================================================================
# マップ読み込み(map_data.NEW_MAZE_STR / map_data_search.SEARCH_MAZE_STR
# のみ参照。パース処理は自前で複製)
# ============================================================================

GRID = ov1_common_rl.parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
ATTACKER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 3]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 2]

if len(DEFENDER_SPAWNS) < len(ROSTER_ORDER):
    raise RuntimeError(
        f"DEFENDER_SPAWNSが{len(DEFENDER_SPAWNS)}マスしかなく、"
        f"固定ロースター{len(ROSTER_ORDER)}人分を配置できません。"
    )

# ロースター順(ROSTER_ORDER)に、マップ上の文字 a,b,c,d,e (Setup Phase
# 到達地点) / A,B,C,D,E (Search Phase最終担当地点) をそのまま1:1で対応させる。
# a/A=roster[0], b/B=roster[1], c/C=roster[2], d/D=roster[3], e/E=roster[4]。
# 5人×5地点の総当たり最適化は不要で、マップ側で明示的に指定された担当地点へ
# ロースター順のままそのまま割り当てるだけでよい。
SETUP_POSITION_CHARS = {
    name: chr(ord("a") + i) for i, name in enumerate(ROSTER_ORDER)
}
DEFENSE_POSITION_CHARS = {
    name: chr(ord("A") + i) for i, name in enumerate(ROSTER_ORDER)
}


def _find_marker_position(maze_str, char):
    """maze_str中で指定char(1文字)が出現するマスを1つ返す。SEARCH_MAZE_STRは
    a-e/A-E/z等の非数字マーカーを含むため、ov1_common_rl.parse_grid(int変換)
    は使わず文字列を直接走査する。"""
    lines = [l for l in maze_str.strip("\n").split("\n") if l.strip()]
    hits = [
        (r, c) for r, line in enumerate(lines) for c, ch in enumerate(line)
        if ch == char
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"map_data_search.py の文字'{char}'は1マスのみである必要がありますが、"
            f"{len(hits)}マス見つかりました: {hits}"
        )
    return hits[0]


def _find_marker_positions(maze_str, char):
    """maze_str中の同一マーカーを全て返す(順序は行優先)。"""
    lines = [l for l in maze_str.strip("\n").split("\n") if l.strip()]
    return [
        (r, c) for r, line in enumerate(lines) for c, marker in enumerate(line)
        if marker == char
    ]


SETUP_ASSIGNMENT = {
    name: _find_marker_position(SEARCH_MAZE_STR, ch)
    for name, ch in SETUP_POSITION_CHARS.items()
}
DEFENSE_ASSIGNMENT = {
    name: _find_marker_position(SEARCH_MAZE_STR, ch)
    for name, ch in DEFENSE_POSITION_CHARS.items()
}

# SETUP_POSITIONS/DEFENSE_POSITIONS はロースター順の担当地点リスト(既存コードとの互換用)。
SETUP_POSITIONS = [SETUP_ASSIGNMENT[name] for name in ROSTER_ORDER]
DEFENSE_POSITIONS = [DEFENSE_ASSIGNMENT[name] for name in ROSTER_ORDER]
SMOKE_SITE_POSITIONS = _find_marker_positions(SEARCH_MAZE_STR, "S")




def _nearest_watch_point(name, pos):
    """登録された監視座標のうち、posから最も近い1点を返す(Chebyshev距離)。
    同距離の場合は座標(row, col)が小さい方を採用する。未登録ならNone。"""
    points = DEFENSE_WATCH_POINTS.get(name)
    if not points:
        return None
    r0, c0 = pos
    best = None
    best_dist = None
    for wp in points:
        dist = max(abs(wp[0] - r0), abs(wp[1] - c0))
        if best_dist is None or dist < best_dist or (dist == best_dist and wp < best):
            best_dist = dist
            best = wp
    return best


def _extract_site_positions(cells, max_sites=2):
    """プラント可能セル群を、単純な距離クラスタリングでサイト代表座標にまとめる。"""
    if not cells:
        return []
    clusters = []
    for cell in cells:
        placed = False
        for cluster in clusters:
            cr, cc = cluster["centroid"]
            if max(abs(cell[0] - cr), abs(cell[1] - cc)) <= 6:
                cluster["cells"].append(cell)
                rs = [c[0] for c in cluster["cells"]]
                cs = [c[1] for c in cluster["cells"]]
                cluster["centroid"] = (sum(rs) / len(rs), sum(cs) / len(cs))
                placed = True
                break
        if not placed:
            clusters.append({"cells": [cell], "centroid": (float(cell[0]), float(cell[1]))})
    clusters.sort(key=lambda c: -len(c["cells"]))
    return [min(
        c["cells"],
        key=lambda cell: (cell[0] - c["centroid"][0]) ** 2
        + (cell[1] - c["centroid"][1]) ** 2,
    ) for c in clusters[:max_sites]]


SITE_POSITIONS = _extract_site_positions(PLANT_CELLS, max_sites=2)
if not SITE_POSITIONS:
    SITE_POSITIONS = [(HEIGHT / 2.0, WIDTH / 2.0)]


# ============================================================================
# LOS・BFS(abilities_los.py / controllers.py と同等のロジックを複製)
# ============================================================================

def has_los(p1, p2, smoke_cells=None):
    return ov1_common_rl.has_los(GRID, p1, p2, smoke_cells)


def _facing_from_delta(dr, dc, fallback):
    """battle_logic.py の _facing_from_delta と同一ロジック(自己完結ルールにより複製)。"""
    if dr == 0 and dc == 0:
        return fallback
    if dr != 0:
        return "N" if dr < 0 else "S"
    return "W" if dc < 0 else "E"


def _facing_accuracy_multiplier(shooter_facing, shooter_pos, target_pos):
    """battle_logic.py の _facing_accuracy_multiplier と同一ロジック(自己完結ルールにより複製)。
    正面100%～真横/背後50%まで、角度差に応じて線形に精度を落とす。"""
    dc = float(target_pos[1] - shooter_pos[1])
    dr = float(target_pos[0] - shooter_pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0:
        return 1.0
    fx, fy = FACING_VECTORS[shooter_facing]
    dot = max(-1.0, min(1.0, (fx * dc + fy * dr) / dist))
    angle = math.degrees(math.acos(dot))
    return 1.0 - min(SHOOTING_SITE_DIGREE, angle) / SHOOTING_SITE_DIGREE * 0.5


def _facing_angle_diff(shooter_facing, shooter_pos, target_pos):
    dc = float(target_pos[1] - shooter_pos[1])
    dr = float(target_pos[0] - shooter_pos[0])
    distance = math.hypot(dc, dr)
    if distance == 0:
        return 0.0
    fx, fy = FACING_VECTORS[shooter_facing]
    dot = max(-1.0, min(1.0, (fx * dc + fy * dr) / distance))
    return math.degrees(math.acos(dot))


# 観測・報酬計算用のfacingエンコード順。FACING_DIRS(action decode用)と
# 同一の並びだが、混同を避けるため別名で持つ(guard/carryのALL_FACINGSと同一方針)。
ALL_FACINGS = FACING_DIRS


def _facing_alignment(facing, from_pos, to_pos):
    """facingが、from_pos→to_pos方向とどれだけ一致しているか(-1〜1、cosθ相当)。
    ov1_train_attacker_carry.py / ov1_train_attacker_guard.pyのfacing整合
    shapingと同一の考え方。"""
    dc = float(to_pos[1] - from_pos[1])
    dr = float(to_pos[0] - from_pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0 or facing not in FACING_VECTORS:
        return 0.0
    fx, fy = FACING_VECTORS[facing]
    return (fx * dc + fy * dr) / dist


def _expected_facing(from_pos, to_pos):
    """from_posからto_posへ最も近い8方向のfacingを返す。"""
    dc = float(to_pos[1] - from_pos[1])
    dr = float(to_pos[0] - from_pos[0])
    dist = math.hypot(dc, dr)
    if dist == 0:
        return None
    nx, ny = dc / dist, dr / dist
    return max(
        FACING_VECTORS,
        key=lambda direction: (
            FACING_VECTORS[direction][0] * nx
            + FACING_VECTORS[direction][1] * ny
        ),
    )


# 各キャラの監視方向(固定ラベル)。担当地点(DEFENSE_ASSIGNMENT)と
# 監視座標(DEFENSE_WATCH_POINTS)から起動時に1度だけ計算し、以後は座標を
# 毎tick動的計算しない。Setup Phase中の移動〜担当地点到着後の静止まで、
# 交戦情報が無い間は常にこのラベルだけをforced facingとして使うことで、
# 経路方向依存のN/Sふらつきを無くす。
DEFENSE_WATCH_FACING = {}
for _name in ROSTER_ORDER:
    _watch_pos = _nearest_watch_point(_name, DEFENSE_ASSIGNMENT[_name])
    if _watch_pos is not None:
        DEFENSE_WATCH_FACING[_name] = _expected_facing(
            DEFENSE_ASSIGNMENT[_name], _watch_pos
        )
print("[omoko_v1] 固定監視方向:", DEFENSE_WATCH_FACING)


def _facing_memory_relevance(from_pos, memory):
    """遠すぎる・古すぎるlast_seen_enemyをfacing対象から外す。"""
    if memory is None:
        return 0.0
    target = memory.get("pos")
    if target is None:
        return 0.0
    distance = max(
        abs(int(target[0]) - int(from_pos[0])),
        abs(int(target[1]) - int(from_pos[1])),
    )
    staleness = max(0, int(memory.get("tick_ago", 0)))
    if distance > FACING_MEMORY_MAX_DISTANCE:
        return 0.0
    if staleness > FACING_MEMORY_MAX_STALENESS:
        return 0.0
    distance_factor = 1.0 - distance / FACING_MEMORY_MAX_DISTANCE
    staleness_factor = 1.0 - staleness / FACING_MEMORY_MAX_STALENESS
    return max(0.0, distance_factor * staleness_factor)


def bfs_distance_map(goal):
    return ov1_common_rl.bfs_distance_map(GRID, goal)


bfs_best_direction = ov1_common_rl.bfs_best_direction

def bfs_best_direction(dist_map, r0, c0):
    """dist_map上で、(r0,c0)から見て最も距離が縮む隣接方向(dr,dc)を返す。
    到達不能・移動不要なら(0,0)を返す。"""
    if dist_map is None:
        return 0, 0
    cur = dist_map[r0, c0]
    if cur < 0:
        return 0, 0
    best_dr, best_dc, best_d = 0, 0, cur
    for dr, dc in CARDINAL:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and dist_map[nr, nc] >= 0:
            if dist_map[nr, nc] < best_d:
                best_d = dist_map[nr, nc]
                best_dr, best_dc = dr, dc
    return best_dr, best_dc


def bfs_best_direction_unoccupied(dist_map, r0, c0, occupied):
    """bfs_best_directionと同じだが、occupied(他ユニットが現在いるマス)は
    移動先候補から除外する。1マス幅の通路で味方が静止していると、通常版は
    毎tick同じ塞がったマスを指し続けて詰まって見えるため、その回避用。
    候補が全て塞がっている場合は(0,0)を返す(無駄な足踏みを防ぐ)。"""
    if dist_map is None:
        return 0, 0
    cur = dist_map[r0, c0]
    if cur < 0:
        return 0, 0
    best_dr, best_dc, best_d = 0, 0, cur
    for dr, dc in CARDINAL:
        nr, nc = r0 + dr, c0 + dc
        if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
            continue
        if dist_map[nr, nc] < 0 or (nr, nc) in occupied:
            continue
        if dist_map[nr, nc] < best_d:
            best_d = dist_map[nr, nc]
            best_dr, best_dc = dr, dc
    return best_dr, best_dc


def planned_route_facing(dist_map, pos):
    """経路上の次方向を基本に、次 tick で曲がる場合は曲がる方向を返す。

    例: 上、上、右、右 の経路では、facing は 上、右、右、右 になる。
    dist_map は目的地からの BFS 距離なので、現在位置と1マス先から
    それぞれ最短経路の方向を調べれば、次の曲がり角を先読みできる。
    """
    if dist_map is None:
        return None
    r0, c0 = int(pos[0]), int(pos[1])
    first = bfs_best_direction(dist_map, r0, c0)
    if first == (0, 0):
        return None

    nr, nc = r0 + first[0], c0 + first[1]
    if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
        return _facing_from_delta(first[0], first[1], None)

    second = bfs_best_direction(dist_map, nr, nc)
    desired = second if second != (0, 0) and second != first else first
    return _facing_from_delta(desired[0], desired[1], None)


def bfs_distance_map_avoiding(goal, avoid_cells):
    """GRID(壁)に加えて、avoid_cells(現在tick時点で他ユニットが占有中の
    マス)も歩行不可として扱う一時的な距離マップを計算する。
    goal自身がavoid_cellsに含まれていても、goalは常に到達可能として扱う
    (味方が自分の担当地点で待機している場合はそのまま素通りできる)。"""
    avoid = set(avoid_cells) - {tuple(goal)}
    if not avoid:
        return bfs_distance_map(goal)
    grid = GRID.copy()
    for (r, c) in avoid:
        if 0 <= r < HEIGHT and 0 <= c < WIDTH:
            grid[r, c] = 1
    return ov1_common_rl.bfs_distance_map(grid, goal)


def bfs_best_direction_detour(dist_map_static, goal, r0, c0, avoid_cells):
    """position phase専用: まず静的な距離マップ(dist_map_static)上で
    占有マスを避けた隣接方向を試す(bfs_best_direction_unoccupied、安価)。
    それが(0,0)を返し、かつ未到着の場合(=隣接候補が全て塞がっている等で
    詰まっている)のみ、avoid_cellsを一時的に壁として扱った動的BFSを
    再計算し、実際に迂回可能な経路の方向を返す(必要な時だけ呼ぶため
    毎tickフルBFSするより許容コスト)。"""
    cur = dist_map_static[r0, c0]
    dr, dc = bfs_best_direction_unoccupied(dist_map_static, r0, c0, avoid_cells)
    if (dr, dc) != (0, 0) or cur <= REACH_RADIUS:
        return dr, dc
    dynamic_map = bfs_distance_map_avoiding(goal, avoid_cells)
    return bfs_best_direction(dynamic_map, r0, c0)


SITE_DIST_MAPS = [bfs_distance_map(tuple(map(int, s))) for s in SITE_POSITIONS]
DEFENSE_POS_DIST_MAPS = [bfs_distance_map(pos) for pos in DEFENSE_POSITIONS]

# ---------------------------------------------------------------------------
# Defender Setup Phase(配置フェーズ)用: 通常の壁に加えて進入禁止マスも
# 歩行不可として扱う専用グリッド・距離マップ。
# defender_setup_phase.py / map_data_defender_setup.py の判定ロジックは
# importせず、必要な制約だけをここに複製する。
# ---------------------------------------------------------------------------
SETUP_MASK_GRID = ov1_common_rl.parse_grid(DEFENDER_SETUP_MASK_STR)
if SETUP_MASK_GRID.shape != GRID.shape:
    raise RuntimeError(
        f"map_data_defender_setup.py のサイズがmap_data.pyと不一致です: "
        f"setup={SETUP_MASK_GRID.shape} base={GRID.shape}"
    )
SETUP_WALK_GRID = np.where(
    (GRID == 1) | (SETUP_MASK_GRID == 1), 1, 0
).astype(np.int32)


def bfs_distance_map_setup(goal):
    return ov1_common_rl.bfs_distance_map(SETUP_WALK_GRID, goal)


SETUP_DEFENSE_POS_DIST_MAPS = [
    bfs_distance_map_setup(pos) for pos in SETUP_POSITIONS
]

# --- 整合性チェック: 各キャラのスポーンから担当ポジションへ実際に到達可能か検証する ---
# map_data_search.py(担当地点)と map_data.py / map_data_defender_setup.py
# (実際の移動判定グリッド)がズレていると、BFS距離が-1(到達不能)になり、
# 学習側はそれを検知できず「方向情報なし」のまま扱ってしまう(これが詰まって見える原因)。
# 起動時に必ず検出できるよう、ここで到達可能性を明示的に検証する。
for _i, _name in enumerate(ROSTER_ORDER):
    _setup_pos = SETUP_POSITIONS[_i]
    _pos = DEFENSE_POSITIONS[_i]
    _spawn = DEFENDER_SPAWNS[_i]
    _setup_dist = SETUP_DEFENSE_POS_DIST_MAPS[_i][_spawn[0], _spawn[1]]
    _normal_dist = DEFENSE_POS_DIST_MAPS[_i][_setup_pos[0], _setup_pos[1]]
    if _setup_dist < 0:
        raise RuntimeError(
            f"[整合性エラー] {_name}: setup担当ポジション{_setup_pos}へSetup Phase中のグリッドで到達不能です。"
            f"spawn={_spawn}。map_data_defender_setup.pyの進入禁止マスク(1)が"
            f"担当地点や経路を塞いでいないか確認してください。"
        )
    if _normal_dist < 0:
        raise RuntimeError(
            f"[整合性エラー] {_name}: 担当ポジション{_pos}へ通常グリッドで到達不能です。"
            f"setup後地点={_setup_pos}。map_data_search.pyの壁配置がmap_data.pyとズレていないか確認してください。"
        )


# DEFENSE_ASSIGNMENT はマップ読み込み時点(上のブロック)で
# 5/6/7/8/9 とロースター順の対応から直接確定済みのため、ここでの
# 総当たり最適化は不要。確認用のログのみ出力する。
# print("[omoko_v1] 固定 担当ポジション割り当て(マップ上の5/6/7/8/9をロースター順に直接対応):")
# for i, _name in enumerate(ROSTER_ORDER):
#     _pos = DEFENSE_ASSIGNMENT[_name]
#     _spawn = DEFENDER_SPAWNS[i]
#     _dist = DEFENSE_POS_DIST_MAPS[i][_spawn[0], _spawn[1]]
#     print(
#         f"  {_name}: spawn={_spawn} -> "
#         f"pos={_pos}(value={DEFENSE_POSITION_VALUES[_name]}) dist={_dist}"
#     )

# print("[omoko_v1][DIAG] ひつじさん担当地点からのLOS確認:")
# _robina_pos = DEFENSE_ASSIGNMENT["ひつじさん"]
# for _cell in PLANT_CELLS:
#     _los = has_los(_robina_pos, _cell)
#     print(f"  plant_cell {_cell}: los={_los}")

# --- 診断用: マップ構造そのものに起因する偏りが無いか確認する ---
# 各キャラのスポーン地点から「最も近いdefense position」までの純粋なBFS距離
# (貪欲割当・シャッフル順のバイアスを除いた理論上の最短値)を出力する。
# もしこれ自体が上段/下段で大きく偏っていれば、マップ側(map_data_search.py の
# 7の配置)がそもそも不公平であることが確定する。
print(f"[omoko_v1][DIAG] 配置 {ROSTER_ORDER[0]}, {ROSTER_ORDER[1]}, {ROSTER_ORDER[2]}, {ROSTER_ORDER[3]}, {ROSTER_ORDER[4]}")
for i, name in enumerate(ROSTER_ORDER):
    spawn = DEFENDER_SPAWNS[i]
    nearest_dist = min(
        bfs_distance_map(spawn)[pos[0], pos[1]]
        for pos in DEFENSE_POSITIONS
    )
    all_dists = sorted(
        bfs_distance_map(spawn)[pos[0], pos[1]] for pos in DEFENSE_POSITIONS
    )
    print(
        f"[omoko_v1][DIAG] {name} spawn={spawn} "
        f"nearest_defense_dist={nearest_dist} "
    )


# ============================================================================
# ユニットスタブ(game_core.Characterの必要最小限の複製。継承・importはしない)
# ============================================================================

class UnitStub:
    def __init__(self, name, team, pos, role, has_spike=False):
        self.name = name
        self.team = team  # "A" or "D"
        self.pos = list(pos)
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.is_alive = True
        self.role = role
        self.ability_name = role
        self.charges = 0 if role == "HUNT" else 1
        initialize_ultimate(self, role)
        self.blind_remaining = 0
        self.reveal_remaining = 0
        self.moved_this_tick = False
        self.moved_last_tick = False
        self.has_spike = has_spike
        self.kills = 0
        self.accuracy = DEFAULT_ACCURACY
        self.dodge_rate = DEFAULT_DODGE
        self.hs_rate = DEFAULT_HS_RATE
        self.reaction = DEFAULT_REACTION + random.uniform(-10, 10)

        # Defender専用: 割り当てられた待機ポジション(7)とそのBFS距離マップ。
        self.assigned_defense_pos = None
        self.assigned_defense_dist_map = None
        # Defender専用: Setup Phase用(進入禁止マス考慮)の距離マップ。
        self.assigned_setup_dist_map = None

        # Defender専用: 現在アクティブな優先モード("spike"/"sighting"/"position")
        # と、そのモードで前tickに観測したBFS距離。モード切替直後は基準値を
        # 揃えるためだけに使い、報酬は発生させない(_compute_rewards参照)。
        self.prev_priority_mode = None
        self.prev_priority_dist = None
        self.prev_priority_target_key = None

        # 向き(N/S/E/W)。battle_logic.pyの実ゲームと同様、移動delta or
        # TURN行動によって更新される。射撃精度補正(_facing_accuracy_multiplier)に使う。
        self.facing = "S"


def _build_fixed_defenders():
    """omoko_v1固定チーム(5人)をDefenderとして生成する。

    ロースター順 = DEFENDER_SPAWNS(行優先走査順)の対応で固定し、
    run_game.py の実際のスポーン割り当てロジック(ロースターi番目 =
    area_4[i])と一致させる。ステータス・ロールは_compute_effective_stats
    で確定済みの値をそのまま使う(ランダム選択・ランダムjitterなし)。
    """
    defenders = []
    for i, name in enumerate(ROSTER_ORDER):
        stats = EFFECTIVE_STATS[name]
        unit = UnitStub(name, "D", DEFENDER_SPAWNS[i], stats["ability"])
        unit.accuracy = stats["accuracy"]
        unit.dodge_rate = stats["dodge_rate"]
        unit.hs_rate = stats["hs_rate"]
        unit.reaction = stats["reaction"]
        defenders.append(unit)
    return defenders


def _build_attackers():
    """敵(Attacker)側は当面ヒューリスティック対応のため、従来通り
    ランダムスポーン・ランダムロール・既定値ステータスで生成する。

    将来、敵専用の固定ステータスファイルを用意する場合は、この関数だけを
    差し替えれば済むよう分離してある(引き継ぎ資料の設計方針に準拠)。
    """
    a_spawns = random.sample(ATTACKER_SPAWNS, min(N_ATTACKERS, len(ATTACKER_SPAWNS)))
    attackers = [
        UnitStub(f"A{i+1}", "A", pos, random.choice(ROLES))
        for i, pos in enumerate(a_spawns)
    ]
    carrier = random.choice(attackers)
    carrier.has_spike = True
    return attackers


# ============================================================================
# チーム共有メモリ(Defender視点。スパイク確定情報 / 敵目撃情報)
# ============================================================================

class TeamMemory:
    def __init__(self):
        self.spike_pos = None
        self.spike_held = False  # True: 保持者が移動中(緊急) / False: 地面に落下(待ち伏せ可)
        self.spike_tick_ago = 0
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}
        # 撃破・視界切れの後も、次の接敵に備えて最後の敵方向を維持する。
        self.last_enemy_threat = None  # {"pos": (r, c), "name": str, "source": str}

    def reset(self):
        self.spike_pos = None
        self.spike_held = False
        self.spike_tick_ago = 0
        self.last_seen_enemy = None
        self.last_enemy_threat = None

    def update(self, defenders, attackers, smoke_cells, spike_ground_pos=None,
               round_tick=0, shots=()):
        alive_defenders = [d for d in defenders if d.is_alive]
        visible_enemies = []
        for d in alive_defenders:
            for a in attackers:
                if not a.is_alive:
                    continue
                if has_los(d.pos, a.pos, smoke_cells):
                    visible_enemies.append(a)

        spike_holder = next((a for a in visible_enemies if a.has_spike), None)
        if spike_holder is not None:
            self.spike_pos = tuple(spike_holder.pos)
            self.spike_held = True
            self.spike_tick_ago = 0
        elif spike_ground_pos is not None and any(
            has_los(d.pos, spike_ground_pos, smoke_cells) for d in alive_defenders
        ):
            self.spike_pos = tuple(spike_ground_pos)
            self.spike_held = False
            self.spike_tick_ago = 0
        elif round_tick >= REINFORCE_FROM_TICK and visible_enemies and self.spike_held:
            # 保持者の古い座標より、現在確認できた敵を増援先として優先する。
            self.spike_pos = None
            self.spike_held = False
            self.spike_tick_ago = 0
        elif self.spike_pos is not None and self.spike_held:
            self.spike_tick_ago += 1
            if self.spike_tick_ago > SPIKE_CARRIER_MEMORY_TICKS:
                self.spike_pos = None
                self.spike_held = False
                self.spike_tick_ago = 0
        elif (self.spike_pos is not None and spike_ground_pos is None
              and any(has_los(d.pos, self.spike_pos, smoke_cells)
                      for d in alive_defenders)):
            # 以前の落下地点が見えており、もうスパイクが無ければ記憶を消す。
            self.spike_pos = None

        if round_tick < REINFORCE_FROM_TICK:
            self.last_seen_enemy = None
            self.last_enemy_threat = None
        elif visible_enemies:
            previous_name = (self.last_seen_enemy or {}).get("name")
            target = next((a for a in visible_enemies if a.name == previous_name), visible_enemies[0])
            self.last_seen_enemy = {"pos": tuple(target.pos), "name": target.name, "tick_ago": 0}
            self.last_enemy_threat = {
                "pos": tuple(target.pos), "name": target.name, "source": "sighting"
            }
        elif self.last_seen_enemy is not None:
            memory = self.last_seen_enemy
            memory["tick_ago"] += 1
            target_dead = any(
                a.name == memory["name"] and not a.is_alive for a in attackers
            )
            if target_dead or memory["tick_ago"] > SIGHTING_MEMORY_TICKS:
                self.last_seen_enemy = None

        # 視認できなくても撃たれた場合は射手の方向を記憶する。
        # 同tickに直接視認できていれば、視認情報の方を優先する。
        if round_tick >= REINFORCE_FROM_TICK and not visible_enemies:
            incoming = [
                shot for shot in shots
                if shot.get("shooter") in attackers
                and shot.get("target") in defenders
                and getattr(shot.get("shooter"), "is_alive", False)
            ]
            if incoming:
                shooter = incoming[-1]["shooter"]
                self.last_enemy_threat = {
                    "pos": tuple(shooter.pos), "name": shooter.name, "source": "shot"
                }


# ============================================================================
# ネットワーク
# ============================================================================

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "next_mask", "done"))
# ネットワーク(旧DefenderSearchDuelingDQN)・ReplayBufferは ov1_common_rl に統合。

# ============================================================================
# 観測構築
# ============================================================================

def build_observation(
    unit, defenders, attackers, team_memory, smoke_cells, own_smoke_active, round_timer,
    spike_dist_map, sighting_dist_map, unit_has_spike_los, in_setup_phase=False,
    available_orbs=(), allies=(),
):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    r0, c0 = int(unit.pos[0]), int(unit.pos[1])

    obs[0] = unit.pos[0] / HEIGHT
    obs[1] = unit.pos[1] / WIDTH
    obs[2] = unit.hp / unit.max_hp if unit.max_hp else 0.0
    obs[3] = 1.0 if unit.moved_this_tick else 0.0

    ability_index = {"SMOKE": 4, "FLASH": 5, "RECON": 6, "HUNT": 7}[unit.role]
    obs[ability_index] = 1.0
    obs[8] = 1.0 if unit.charges > 0 else 0.0

    visible_enemies = [
        a for a in attackers if a.is_alive and has_los(unit.pos, a.pos, smoke_cells)
    ]
    obs[9] = 1.0 if visible_enemies else 0.0

    teammates = [d for d in defenders if d is not unit and d.is_alive]
    obs[10] = len(teammates) / 4.0
    if teammates:
        nearest_d = min(
            max(abs(t.pos[0] - unit.pos[0]), abs(t.pos[1] - unit.pos[1])) for t in teammates
        )
        obs[11] = min(nearest_d, HEIGHT) / HEIGHT

    obs[12] = 1.0 if any(
        a.is_alive and (a.blind_remaining > 0 or a.reveal_remaining > 0) for a in attackers
    ) else 0.0
    obs[13] = 1.0 if own_smoke_active else 0.0

    if team_memory.spike_pos is not None:
        obs[14] = 1.0
        best_dr, best_dc = bfs_best_direction(spike_dist_map, r0, c0)
        obs[15] = float(best_dr)
        obs[16] = float(best_dc)

    if team_memory.last_seen_enemy is not None:
        ls = team_memory.last_seen_enemy
        obs[17] = 1.0
        best_dr, best_dc = bfs_best_direction(sighting_dist_map, r0, c0)
        obs[18] = float(best_dr)
        obs[19] = float(best_dc)
        obs[20] = min(ls["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP

        obs[21] = len(visible_enemies) / 5.0
        if visible_enemies:
            nearest_enemy = min(
                visible_enemies,
            key=lambda a: max(abs(a.pos[0] - unit.pos[0]), abs(a.pos[1] - unit.pos[1])),
        )
            obs[22] = (nearest_enemy.pos[0] - unit.pos[0]) / HEIGHT
            obs[23] = (nearest_enemy.pos[1] - unit.pos[1]) / WIDTH
            dist = max(abs(nearest_enemy.pos[0] - unit.pos[0]), abs(nearest_enemy.pos[1] - unit.pos[1]))
            obs[24] = min(dist, HEIGHT) / HEIGHT

    if len(SITE_POSITIONS) >= 1:
        obs[25] = (SITE_POSITIONS[0][0] - unit.pos[0]) / HEIGHT
        obs[26] = (SITE_POSITIONS[0][1] - unit.pos[1]) / WIDTH
    if len(SITE_POSITIONS) >= 2:
        obs[27] = (SITE_POSITIONS[1][0] - unit.pos[0]) / HEIGHT
        obs[28] = (SITE_POSITIONS[1][1] - unit.pos[1]) / WIDTH

    obs[29] = min(round_timer, MAX_TICKS) / MAX_TICKS
    obs[30] = 1.0 if in_setup_phase else 0.0  # Setup Phase中かどうか(旧: 予備次元)

    team_visible_enemies = [
        attacker
        for attacker in attackers
        if attacker.is_alive
        and any(
            defender.is_alive and has_los(defender.pos, attacker.pos, smoke_cells)
            for defender in defenders
        )
    ]
    # 味方の視認または落下スパイク情報がある間は、担当地点への強制帰還を止める。
    in_position_mode = (
        team_memory.spike_pos is None
        and team_memory.last_seen_enemy is None
        and not team_visible_enemies
    )
    if in_setup_phase and unit.assigned_setup_dist_map is not None:
        dist_map = unit.assigned_setup_dist_map
        bfs_dist = dist_map[r0, c0]
        if bfs_dist < 0:
            bfs_dist = HEIGHT + WIDTH
        obs[31] = min(max(bfs_dist, 0), HEIGHT + WIDTH) / (HEIGHT + WIDTH)
        best_dr, best_dc = bfs_best_direction(dist_map, r0, c0)
        obs[32] = float(best_dr)
        obs[33] = float(best_dc)
        obs[34] = 1.0 if bfs_dist <= REACH_RADIUS else 0.0
    elif in_position_mode and unit.assigned_defense_pos is not None:
        dist_map = unit.assigned_defense_dist_map
        bfs_dist = dist_map[r0, c0]
        if bfs_dist < 0:
            bfs_dist = HEIGHT + WIDTH
        obs[31] = min(max(bfs_dist, 0), HEIGHT + WIDTH) / (HEIGHT + WIDTH)
        best_dr, best_dc = bfs_best_direction(dist_map, r0, c0)
        obs[32] = float(best_dr)
        obs[33] = float(best_dc)
        obs[34] = 1.0 if bfs_dist <= REACH_RADIUS else 0.0

    obs[35] = 1.0 if (
        team_memory.spike_pos is not None
        and not team_memory.spike_held
        and unit_has_spike_los
    ) else 0.0

    # [36-43] 自身の向き(8方向)one-hot。従来欠落しており、facingが命中率へ
    # 直接影響するにもかかわらず観測できない状態だった(POMDP化)。
    # carry/escort/guardと同一方針で追加する。
    if unit.facing in ALL_FACINGS:
        obs[36 + ALL_FACINGS.index(unit.facing)] = 1.0

    watch_pos = _nearest_watch_point(unit.name, (r0, c0))
    if watch_pos is not None:
        obs[44] = (watch_pos[0] - r0) / HEIGHT
        obs[45] = (watch_pos[1] - c0) / WIDTH

    # 共有ネットワークがキャラごとの担当・facing方針を区別できるようにする。
    agent_id = ROSTER_ORDER.index(unit.name)
    obs[AGENT_ID_OFFSET + agent_id] = 1.0

    team_sees_enemy = bool(team_visible_enemies)
    context_offset = AGENT_ID_OFFSET + AGENT_ID_DIM
    obs[context_offset:context_offset + 4] = ultimate_context(
        unit,
        self_sighting=bool(visible_enemies),
        team_sighting=team_sees_enemy,
        tactical=(not in_setup_phase and team_sees_enemy),
    )
    if team_visible_enemies:
        shared_target = min(
            team_visible_enemies,
            key=lambda enemy: max(
                abs(enemy.pos[0] - r0), abs(enemy.pos[1] - c0)
            ),
        )
        obs[context_offset + 4] = (shared_target.pos[0] - r0) / HEIGHT
        obs[context_offset + 5] = (shared_target.pos[1] - c0) / WIDTH
    obs[context_offset + 6:context_offset + 13] = orb_context(
        unit, available_orbs, GRID, allies
    )

    # 共有された戦術対象と、各1歩候補から対象へ射線が通るかを観測する。
    # これにより、壁形状を推論側の移動ルールにせず、射線参加を学習できる。
    tactical_offset = context_offset + ULTIMATE_CONTEXT_DIM + ORB_CONTEXT_DIM
    visible_holders = [enemy for enemy in team_visible_enemies if enemy.has_spike]
    if visible_holders:
        tactical_target = min(
            visible_holders,
            key=lambda enemy: max(abs(enemy.pos[0] - r0), abs(enemy.pos[1] - c0)),
        ).pos
        ground_spike_target = False
    elif team_visible_enemies:
        tactical_target = min(
            team_visible_enemies,
            key=lambda enemy: max(abs(enemy.pos[0] - r0), abs(enemy.pos[1] - c0)),
        ).pos
        ground_spike_target = False
    elif team_memory.spike_pos is not None:
        tactical_target = team_memory.spike_pos
        ground_spike_target = not team_memory.spike_held
    else:
        tactical_target = (
            team_memory.last_seen_enemy["pos"]
            if team_memory.last_seen_enemy is not None else None
        )
        ground_spike_target = False

    if tactical_target is not None:
        target_pos = tuple(map(int, tactical_target))
        obs[tactical_offset] = 1.0
        obs[tactical_offset + 1] = 1.0 if ground_spike_target else 0.0
        ally_coverage = sum(
            defender is not unit
            and defender.is_alive
            and has_los(defender.pos, target_pos, smoke_cells)
            for defender in defenders
        )
        obs[tactical_offset + 2] = ally_coverage / 4.0
        occupied = {
            tuple(defender.pos)
            for defender in defenders
            if defender.is_alive and defender is not unit
        }
        for move_idx, (dr, dc) in enumerate(MOVES):
            nr, nc = r0 + dr, c0 + dc
            valid = (
                0 <= nr < HEIGHT and 0 <= nc < WIDTH
                and GRID[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            obs[tactical_offset + 3 + move_idx] = (
                1.0 if valid and has_los((nr, nc), target_pos, smoke_cells) else 0.0
            )

    site_offset = tactical_offset + TACTICAL_CONTEXT_DIM
    site_target = (
        team_memory.spike_pos if team_memory.spike_pos is not None
        else team_memory.last_seen_enemy["pos"]
        if team_memory.last_seen_enemy is not None
        else team_visible_enemies[0].pos if team_visible_enemies else None
    )
    obs[site_offset:site_offset + SITE_CONTEXT_DIM] = site_context(
        unit, defenders, team_visible_enemies, SITE_DIST_MAPS,
        unit.assigned_defense_pos, site_target, team_memory.spike_held,
        HEIGHT, WIDTH,
    )

    return obs


def decode_action(action_idx):
    """action_idx = base_idx(0-9) * 8 + facing_idx(0-7)。
    base_idxは移動(5方向)*アビリティ有無、facing_idxは向き
    (N/NE/E/SE/S/SW/W/NW)で、両者は完全に独立
    (移動先と向きは無関係に組み合わせられる)。
    戻り値は (move, use_ability, facing)。"""
    action_idx = int(action_idx)
    base_idx, facing_idx = divmod(action_idx, len(FACING_DIRS))
    if base_idx == ACTION_ULTIMATE_BASE:
        return (0, 0), False, True, False, FACING_DIRS[facing_idx]
    if base_idx == ACTION_ORB_BASE:
        return (0, 0), False, False, True, FACING_DIRS[facing_idx]
    move_idx, use_ability = divmod(base_idx, 2)
    return MOVES[move_idx], bool(use_ability), False, False, FACING_DIRS[facing_idx]


def encode_action(move, use_ability, facing, use_ultimate=False, use_orb=False):
    if use_ultimate:
        return ACTION_ULTIMATE_BASE * len(FACING_DIRS) + FACING_DIRS.index(facing)
    if use_orb:
        return ACTION_ORB_BASE * len(FACING_DIRS) + FACING_DIRS.index(facing)
    move_idx = MOVES.index(move)
    base_idx = move_idx * 2 + (1 if use_ability else 0)
    facing_idx = FACING_DIRS.index(facing)
    return base_idx * len(FACING_DIRS) + facing_idx


def build_action_mask(
    unit, occupied, lock_movement=False, has_target_info=True,
    in_setup_phase=False, forced_facing=None,
):
    """lock_movement=True の場合、stay(move_idx=0)以外の移動を禁止する。
    交戦中(敵が視認できている間)は静止させ、射撃の当たりやすさを優先する。
    has_target_info=False(有効な味方支援目標が無い、または本人が交戦中)の場合、
    use_abilityを除外する。空撃ちによるチャージ消費と射撃中断を防ぐ。
    向き(facing)は移動・アビリティとは無関係に常に自由選択できるため、
    base(0-9)側のマスクを4倍に展開するだけでよい。"""
    base_mask = np.ones(BASE_ACTION_DIM, dtype=bool)
    r, c = int(unit.pos[0]), int(unit.pos[1])
    for move_idx, (dr, dc) in enumerate(MOVES):
        if lock_movement and move_idx != 0:
            base_mask[move_idx * 2] = False
            base_mask[move_idx * 2 + 1] = False
            continue
        nr, nc = r + dr, c + dc
        walkable = (
            0 <= nr < HEIGHT and 0 <= nc < WIDTH
            and GRID[nr, nc] != 1
            and (nr, nc) not in occupied
        )
        if walkable and in_setup_phase:
            walkable = SETUP_MASK_GRID[nr, nc] == 0
        if not walkable:
            base_mask[move_idx * 2] = False
            base_mask[move_idx * 2 + 1] = False

    if in_setup_phase or unit.charges <= 0 or unit.role == "HUNT" or not has_target_info:
        for move_idx in range(5):
            base_mask[move_idx * 2 + 1] = False

    base_mask[ACTION_ULTIMATE_BASE] = bool(not in_setup_phase and ultimate_ready(unit))
    base_mask[ACTION_ORB_BASE] = False

    # base_idxごとに向き4通りをまとめて許可/禁止する(encode_actionのbase_idx*4+facing_idxと対応)。
    action_mask = np.repeat(base_mask, len(FACING_DIRS))
    if forced_facing in FACING_DIRS:
        facing_idx = FACING_DIRS.index(forced_facing)
        for base_idx in range(BASE_ACTION_DIM):
            if base_mask[base_idx]:
                start = base_idx * len(FACING_DIRS)
                action_mask[start:start + len(FACING_DIRS)] = False
                action_mask[start + facing_idx] = True
    return action_mask


def _forced_watch_facing(unit, visible_enemies, team_memory):
    """交戦情報(視認中の敵・記憶中の目撃情報)が無い間、監視方向を固定
    ラベル(DEFENSE_WATCH_FACING)で強制する。Setup Phase中の移動や
    担当地点への移動中も含め常に適用する(座標からの動的計算はしない)。"""
    if visible_enemies or team_memory.last_seen_enemy is not None:
        return None
    return DEFENSE_WATCH_FACING.get(unit.name)


def _forced_combat_facing(unit, visible_enemies, team_memory):
    """視認中の敵、または最後に確認・被弾した敵の方向を優先する。"""
    if visible_enemies:
        target = min(
            visible_enemies,
            key=lambda a: max(
                abs(a.pos[0] - unit.pos[0]), abs(a.pos[1] - unit.pos[1])
            ),
        )
        facing = _expected_facing(tuple(unit.pos), tuple(target.pos))
        if facing is not None:
            unit._combat_facing = facing
        return getattr(unit, "_combat_facing", None)
    threat = getattr(team_memory, "last_enemy_threat", None)
    if threat is not None:
        facing = _expected_facing(tuple(unit.pos), tuple(threat["pos"]))
        if facing is not None:
            unit._combat_facing = facing
        return getattr(unit, "_combat_facing", None)
    return None


# ============================================================================
# 環境本体
# ============================================================================

class SearchEnv:
    """プラント前フェーズを模した簡易マルチエージェント環境。

    Defender = omoko_v1固定チーム(5人、固定ステータス・固定ロール・
    固定スポーン)。Attacker側は本物のcontrollers.pyロジックではなく、
    このファイル内に複製した簡易ヒューリスティックで動かす(将来、敵専用
    ステータスファイルに差し替え可能な _build_attackers() に分離済み)。
    """

    def __init__(self):
        self.team_memory = TeamMemory()
        self.defenders = []
        self.attackers = []
        self.smokes = []  # [{"cells": set, "remaining_ticks": int, "team": str}]
        self.round_timer = MAX_TICKS
        self.carrier_target_site_idx = 0
        self.attacker_follow_probability = 0.6
        self.planted = False
        self.match_over_reason = None
        self._prev_kills = {}
        self._prev_alive = {}
        self.spike_dist_map = None
        self.sighting_dist_map = None
        self.spike_ground_pos = None
        self.available_orbs = set()
        self.in_setup_phase = False
        self.setup_ticks_remaining = 0
        self.scheduled_smoke_fired = False

        # --- 一時デバッグ用: 特定キャラの毎tick実況トレース。
        # train()側で特定エピソードだけTrueにする。
        self.debug_trace = False
        self.debug_trace_name = "ひつじさん"

        # --- 診断用: positionモード中(spike/sighting情報が無いとき)の
        # キャラ別「平均BFS距離・到着率・移動率」を1エピソード分蓄積する。
        # train()側がエピソード終了ごとにこれを読み取り、履歴に積算する。
        
        self.position_mode_stats = {
            name: {"dist_sum": 0.0, "dist_count": 0, "moved_count": 0, "arrived_count": 0}
            for name in ROSTER_ORDER
        }
        # エピソード中に指定配置位置へ一度でも到達したか。
        # position mode から索敵/戦闘 mode へ移行した後の到達も記録する。
        self.episode_arrived = {name: False for name in ROSTER_ORDER}
        self.position_facing_ticks = {name: 0 for name in ROSTER_ORDER}

        # --- 診断用: アビリティ使用の内訳(視認あり使用/命中/外れ/whiff/overlap/debuffキル)を
        # 1エピソード分蓄積する。train()側がエピソード終了ごとにこれを読み取り、履歴に積算する。
        self.ability_diag_stats = {
            name: {
                "aimed": 0, "hit": 0, "miss": 0, "whiff": 0, "overlap": 0,
                "debuff_kill": 0, "opportunity": 0, "own_any_enemy_seen": 0,
            }
            for name in ROSTER_ORDER
        }

    # -- 初期化 --------------------------------------------------------
    def reset(self):
        self.team_memory.reset()
        self.smokes = []
        self.available_orbs = valid_orb_cells(GRID)
        self.round_timer = MAX_TICKS
        self.planted = False
        self.match_over_reason = None
        self.spike_ground_pos = None
        self.in_setup_phase = DEFENDER_SETUP_TICKS > 0
        self.setup_ticks_remaining = DEFENDER_SETUP_TICKS
        self.scheduled_smoke_fired = False

        # --- 診断用: 新しいエピソードの開始時に集計をリセット ---
        
        self.position_mode_stats = {
            name: {"dist_sum": 0.0, "dist_count": 0, "moved_count": 0, "arrived_count": 0}
            for name in ROSTER_ORDER
        }
        self.episode_arrived = {name: False for name in ROSTER_ORDER}
        self.position_facing_ticks = {name: 0 for name in ROSTER_ORDER}
        self.ability_diag_stats = {
            name: {
                "aimed": 0, "hit": 0, "miss": 0, "whiff": 0, "overlap": 0,
                "debuff_kill": 0, "opportunity": 0, "own_any_enemy_seen": 0,
            }
            for name in ROSTER_ORDER
        }

        self.defenders = _build_fixed_defenders()
        self.attackers = _build_attackers()

        self.carrier_target_site_idx = random.randrange(len(SITE_POSITIONS))
        # 密集ラッシュと分散進行を両方経験させる。
        self.attacker_follow_probability = random.choice((0.35, 0.65, 0.9))

        self._assign_defense_positions()
        self._update_episode_arrival_flags()

        self.team_memory.update(self.defenders, self.attackers, self._smoke_cells(), self.spike_ground_pos)
        self._update_priority_dist_maps()
        self._prev_kills = {u.name: u.kills for u in self.defenders + self.attackers}
        self._prev_alive = {u.name: u.is_alive for u in self.defenders + self.attackers}

        return self._collect_observations()

    def _scheduled_smoke_targets(self):
        """指定tickにSマーカーへ投げるスモーク役を距離順に割り当てる。"""
        search_tick = MAX_TICKS - self.round_timer + 1
        if (
            self.scheduled_smoke_fired
            or self.in_setup_phase
            or search_tick != SCHEDULED_SMOKE_DELAY_TICKS
            or not SMOKE_SITE_POSITIONS
            or any(
                attacker.is_alive and defender.is_alive
                and has_los(defender.pos, attacker.pos, self._smoke_cells())
                for defender in self.defenders for attacker in self.attackers
            )
        ):
            return {}

        eligible = [
            defender for defender in self.defenders
            if defender.is_alive and defender.role == "SMOKE" and defender.charges > 0
        ]
        assignments = {}
        unused = set(defender.name for defender in eligible)
        site_dist_maps = {site: bfs_distance_map(site) for site in SMOKE_SITE_POSITIONS}
        for site in SMOKE_SITE_POSITIONS:
            candidates = [defender for defender in eligible if defender.name in unused]
            if not candidates:
                break
            dist_map = site_dist_maps[site]

            def smoke_distance(defender):
                distance = dist_map[int(defender.pos[0]), int(defender.pos[1])]
                return distance if distance >= 0 else HEIGHT + WIDTH

            nearest = min(
                candidates,
                key=lambda defender: (
                    smoke_distance(defender),
                    defender.name,
                ),
            )
            assignments[nearest.name] = tuple(site)
            unused.remove(nearest.name)
        self.scheduled_smoke_fired = True
        return assignments

    def _update_episode_arrival_flags(self):
        """エピソード全体の配置位置到達状況を記録する。

        position_mode_stats の arrived_count は position mode 中の診断用であり、
        mode 切り替え後に到着したケースを数えられない。そのため、こちらは
        毎 tick、指定座標との一致を確認して一度 True になったら保持する。
        """
        for d in self.defenders:
            target = getattr(d, "assigned_defense_pos", None)
            if target is not None and tuple(map(int, d.pos)) == tuple(map(int, target)):
                self.episode_arrived[d.name] = True

    def _assign_defense_positions(self):
        """omoko_v1固定チームはスポーン位置が毎エピソード同一のため、
        ランダムシャッフル+早い者勝ち貪欲法ではなく、事前に一意計算した
        全組合せ最適解(DEFENSE_ASSIGNMENT)を毎回そのまま使う。
        これにより、複数キャラが同じ近場ポジションを取り合い、その結果を
        エピソードごとの運で分け合うという構造的な偏りを解消する。"""
        for d in self.defenders:
            pos = DEFENSE_ASSIGNMENT[d.name]
            idx = DEFENSE_POSITIONS.index(pos)
            d.assigned_defense_pos = pos
            d.assigned_defense_dist_map = DEFENSE_POS_DIST_MAPS[idx]
            d.assigned_setup_dist_map = SETUP_DEFENSE_POS_DIST_MAPS[idx]
            d.prev_priority_mode = None
            d.prev_priority_dist = None
            d.prev_priority_target_key = None

        # --- 一時デバッグ用: 担当ポジションが本当に5人とも別々かを1度だけ確認する ---
        if not getattr(self, "_debug_printed_assignment", False):
            self._debug_printed_assignment = True
            print("[omoko_v1][DIAG] 担当ポジション割り当て確認(全員別々であるべき):")
            for d in self.defenders:
                print(f"  {d.name}: spawn={tuple(d.pos)} -> assigned_pos={d.assigned_defense_pos}")

            # --- 一時デバッグ用: 実際にBFSが計算した最短経路そのものを座標列で出力する。
            # 「右へ迂回」が本当にマップ構造上の最短経路なのか、それとも別の不具合
            # (Setup用マップ取り違え等)なのかを目視で確定するため。
            def _trace_path(dist_map, start, label):
                path = [tuple(map(int, start))]
                r0, c0 = int(start[0]), int(start[1])
                for _ in range(200):
                    if dist_map[r0, c0] <= 0:
                        break
                    dr, dc = bfs_best_direction(dist_map, r0, c0)
                    if dr == 0 and dc == 0:
                        break
                    r0, c0 = r0 + dr, c0 + dc
                    path.append((r0, c0))
                print(f"    [{label}] path(len={len(path)})={path}")

            # print("[omoko_v1][DIAG] 各キャラのBFS最短経路(占有無視・壁のみ考慮):")
            # for d in self.defenders:
            #     print(f"  {d.name}: target={d.assigned_defense_pos}")
            #     _trace_path(d.assigned_setup_dist_map, d.pos, "SETUP grid")
            #     _trace_path(d.assigned_defense_dist_map, d.pos, "NORMAL grid")

    def _update_priority_dist_maps(self):
        self.spike_dist_map = (
            bfs_distance_map(self.team_memory.spike_pos)
            if self.team_memory.spike_pos is not None else None
        )
        self.sighting_dist_map = (
            bfs_distance_map(self.team_memory.last_seen_enemy["pos"])
            if self.team_memory.last_seen_enemy is not None else None
        )

    def _smoke_cells(self):
        cells = set()
        for s in self.smokes:
            if s["remaining_ticks"] > 0:
                cells.update(s["cells"])
        return cells

    def _own_smoke_active(self, team):
        return any(s["team"] == team and s["remaining_ticks"] > 0 for s in self.smokes)

    def _collect_observations(self):
        smoke_cells = self._smoke_cells()
        occupied = {
            tuple(u.pos) for u in self.defenders + self.attackers if u.is_alive
        }
        obs_dict, mask_dict = {}, {}
        for d in self.defenders:
            if not d.is_alive:
                continue
            unit_has_spike_los = (
                self.team_memory.spike_pos is not None
                and not self.team_memory.spike_held
                and has_los(d.pos, self.team_memory.spike_pos, smoke_cells)
            )
            obs_dict[d.name] = build_observation(
                d, self.defenders, self.attackers, self.team_memory,
                smoke_cells, self._own_smoke_active("D"), self.round_timer,
                self.spike_dist_map, self.sighting_dist_map, unit_has_spike_los,
                in_setup_phase=self.in_setup_phase,
                available_orbs=self.available_orbs, allies=self.defenders,
            )
            own_occupied = occupied - {tuple(d.pos)}
            visible_enemies_for_mask = [
                a for a in self.attackers if a.is_alive and has_los(d.pos, a.pos, smoke_cells)
            ]
            has_enemy_los = bool(visible_enemies_for_mask) and not self.in_setup_phase
            # 静止して撃つか、人数不利で退避するかをモデルに選ばせる。
            lock_movement = False
            # 直接交戦中は撃ち続ける。射線のない味方のみ支援アビリティを選ぶ。
            # Setup Phase中はアビリティ自体が使用不可のためhas_target_info=False固定。
            if self.in_setup_phase or has_enemy_los or d.charges <= 0:
                has_target_info = False
            else:
                support_plan = find_support_ability_plan(
                    GRID, d, self.defenders, self.attackers, d.role, smoke_cells,
                    max_aim_range=ABILITY_RANGE,
                )
                has_target_info = support_plan is not None
            forced_facing = _forced_combat_facing(
                d, visible_enemies_for_mask, self.team_memory
            )
            if forced_facing is None:
                forced_facing = _forced_watch_facing(
                    d, visible_enemies_for_mask, self.team_memory
                )
            mask_dict[d.name] = build_action_mask(
                d, own_occupied, lock_movement=lock_movement, has_target_info=has_target_info,
                in_setup_phase=self.in_setup_phase, forced_facing=forced_facing,
            )
            if (
                not self.in_setup_phase
                and tuple(d.pos) in self.available_orbs
                and orb_priority(d, self.defenders)
            ):
                start = ACTION_ORB_BASE * len(FACING_DIRS)
                mask_dict[d.name][start:start + len(FACING_DIRS)] = True
        return obs_dict, mask_dict


    # -- Attacker側の簡易ヒューリスティック ------------------------------
    def _attacker_decide_move(self, unit):
        goal_dist_map = SITE_DIST_MAPS[self.carrier_target_site_idx]
        r, c = int(unit.pos[0]), int(unit.pos[1])

        if unit.has_spike:
            best_move = (0, 0)
            best_dist = goal_dist_map[r, c]
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1:
                    d = goal_dist_map[nr, nc]
                    if d >= 0 and (best_dist < 0 or d < best_dist):
                        best_dist = d
                        best_move = (dr, dc)
            if random.random() < 0.15:
                best_move = random.choice(CARDINAL)
            return best_move

        carrier = next((a for a in self.attackers if a.is_alive and a.has_spike), None)
        if carrier is not None and random.random() < self.attacker_follow_probability:
            dr = 1 if carrier.pos[0] > r else (-1 if carrier.pos[0] < r else 0)
            dc = 1 if carrier.pos[1] > c else (-1 if carrier.pos[1] < c else 0)
            candidates = [m for m in [(dr, 0), (0, dc)] if m != (0, 0)]
            random.shuffle(candidates)
            for mdr, mdc in candidates:
                nr, nc = r + mdr, c + mdc
                if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1:
                    return (mdr, mdc)

        valid = [
            (dr, dc) for dr, dc in CARDINAL
            if 0 <= r + dr < HEIGHT and 0 <= c + dc < WIDTH and GRID[r + dr, c + dc] != 1
        ]
        return random.choice(valid) if valid else (0, 0)

    def _step_setup_phase(self, action_dict):
        """Defender Setup Phase(配置フェーズ)の1tick処理。

        Defenderの移動のみ行い、Attackerの移動・戦闘・アビリティ・
        round_timerの進行は一切行わない
        (battle_logic._run_defender_setup_tick と同等の制約)。
        移動先はSETUP_MASK_GRID(map_data_defender_setup.py)で制限する。
        """
        for u in self.defenders:
            u.moved_last_tick = u.moved_this_tick
            u.moved_this_tick = False

        smoke_cells = self._smoke_cells()

        move_plans = []
        actual_action_dict = {}
        prev_dists = {
            d.name: float(d.assigned_setup_dist_map[int(d.pos[0]), int(d.pos[1])])
            for d in self.defenders
            if d.is_alive and d.assigned_setup_dist_map is not None
        }

        # position mode(スパイク情報も敵目撃情報も無い状態)かつ担当地点未到着の
        # 間は、スポーン・担当地点がどちらも毎エピソード固定である以上、移動方向を
        # RLに手探りさせる意味がない。既知のBFS最短方向をそのまま強制適用する。
        in_position_phase = (
            self.team_memory.last_seen_enemy is None
            and (
                self.team_memory.spike_pos is None
                or not self.team_memory.spike_held
            )
        )

        # search phase(in_position_phase)と同様、担当地点は毎エピソード固定のため
        # 移動方向はBFS最短(占有マス回避付き)で強制する。RLに委ねるのは今後の
        # アビリティ使用判断のみとし、行き止まりでの迷走を防ぐ。
        occupied_now = {tuple(u.pos) for u in self.defenders if u.is_alive}

        for d in self.defenders:
            if not d.is_alive or d.name not in action_dict:
                continue
            (dr, dc), use_ability, _use_ultimate, _use_orb, facing = decode_action(action_dict[d.name])

            visible_enemies = [
                a for a in self.attackers if a.is_alive and has_los(d.pos, a.pos, smoke_cells)
            ]
            self_occupied = occupied_now - {tuple(d.pos)}

            forced_facing = _forced_combat_facing(d, visible_enemies, self.team_memory)
            if forced_facing is None:
                forced_facing = _forced_watch_facing(d, visible_enemies, self.team_memory)
            if forced_facing is not None:
                facing = forced_facing

            if in_position_phase and d.assigned_defense_dist_map is not None:
                r0, c0 = int(d.pos[0]), int(d.pos[1])
                cur_dist = d.assigned_defense_dist_map[r0, c0]
                if cur_dist > REACH_RADIUS:
                    dr, dc = bfs_best_direction_detour(
                        d.assigned_defense_dist_map, d.assigned_defense_pos, r0, c0, self_occupied
                    )
                else:
                    dr, dc = 0, 0
            # 向き(facing)は移動先の決定方法(BFS強制/ネットワーク)と無関係に、
            # ネットワークが選んだ向きをそのまま毎tick適用する。
            d.facing = facing
            d._facing_eval_pos = tuple(d.pos)
            d._facing_route_map = d.assigned_setup_dist_map
            d._facing_move = (dr, dc)
            actual_action_dict[d.name] = encode_action((dr, dc), use_ability, facing)
            move_plans.append((d, (dr, dc)))

        for unit, (dr, dc) in move_plans:
            if dr == 0 and dc == 0:
                continue
            old_pos = tuple(unit.pos)
            nr, nc = unit.pos[0] + dr, unit.pos[1] + dc
            in_bounds = 0 <= nr < HEIGHT and 0 <= nc < WIDTH
            is_wall = in_bounds and GRID[nr, nc] == 1
            is_setup_blocked = in_bounds and SETUP_MASK_GRID[nr, nc] == 1
            occupied = any(
                other is not unit and other.is_alive and tuple(other.pos) == (nr, nc)
                for other in self.defenders
            )
            if in_bounds and not is_wall and not is_setup_blocked and not occupied:
                unit.pos = [nr, nc]
            unit.moved_this_tick = tuple(unit.pos) != old_pos

        self.setup_ticks_remaining -= 1
        if self.setup_ticks_remaining <= 0:
            self.in_setup_phase = False

        self._update_episode_arrival_flags()

        rewards = {}
        for d in self.defenders:
            r = STEP_PENALTY
            dist_map = d.assigned_setup_dist_map
            prev = prev_dists.get(d.name)
            if dist_map is not None and prev is not None and prev >= 0:
                r0, c0 = int(d.pos[0]), int(d.pos[1])
                bfs_dist = dist_map[r0, c0]
                if bfs_dist >= 0:
                    delta = prev - bfs_dist
                    if bfs_dist > REACH_RADIUS:
                        r += DEFENSE_POSITION_PULL_REWARD * delta
                    else:
                        r += HOLD_POSITION_BONUS if not d.moved_this_tick else HOLD_POSITION_PENALTY
                        if bfs_dist <= REACH_RADIUS:
                            watch_pos = _nearest_watch_point(d.name, tuple(d.pos))
                            if watch_pos is not None:
                                r += FACING_ALIGN_POSITION_WEIGHT * _facing_alignment(
                                    d.facing, tuple(d.pos), watch_pos
                                )
            rewards[d.name] = r

        obs_dict, mask_dict = self._collect_observations()
        done = False
        return obs_dict, mask_dict, rewards, done, actual_action_dict

    # -- メインステップ ---------------------------------------------------
    def step(self, action_dict):
        if self.in_setup_phase:
            return self._step_setup_phase(action_dict)

        for u in self.defenders + self.attackers:
            u.moved_last_tick = u.moved_this_tick
            u.moved_this_tick = False
            u._position_before_step = tuple(u.pos)

        pre_tick_enemy_debuffed = {
            a.name: (a.blind_remaining > 0 or a.reveal_remaining > 0)
            for a in self.attackers if a.is_alive
        }
        pre_tick_flash_recon_active = any(
            a.is_alive and (a.blind_remaining > 0 or a.reveal_remaining > 0)
            for a in self.attackers
        )

        smoke_cells = self._smoke_cells()
        scheduled_smoke_targets = self._scheduled_smoke_targets()

        move_plans = []
        ability_requests = []
        ability_whiff = {}
        ability_overlap = {}
        held_angle = {}
        ability_smoke_support = {}
        support_impacts = {}
        ultimate_tactical = {}
        ultimate_alignment = {}
        orb_rewards = {}

        for name, target in scheduled_smoke_targets.items():
            defender = next((d for d in self.defenders if d.name == name), None)
            if defender is not None and defender.is_alive and defender.charges > 0:
                defender.charges -= 1
                ability_requests.append((defender, target))
                ability_whiff[name] = False

        team_visible_enemies = [
            attacker
            for attacker in self.attackers
            if attacker.is_alive
            and any(
                defender.is_alive and has_los(defender.pos, attacker.pos, smoke_cells)
                for defender in self.defenders
            )
        ]
        team_sees_enemy = bool(team_visible_enemies)
        visible_holders = [enemy for enemy in team_visible_enemies if enemy.has_spike]
        if visible_holders:
            crossfire_enemy = visible_holders[0]
            crossfire_ground_pos = None
        elif team_visible_enemies:
            crossfire_enemy = team_visible_enemies[0]
            crossfire_ground_pos = None
        else:
            crossfire_enemy = None
            crossfire_ground_pos = (
                tuple(self.team_memory.spike_pos)
                if self.team_memory.spike_pos is not None and not self.team_memory.spike_held
                else None
            )
        pre_crossfire_target = (
            tuple(crossfire_enemy.pos) if crossfire_enemy is not None else crossfire_ground_pos
        )
        pre_crossfire_los = {
            defender.name: bool(
                defender.is_alive
                and pre_crossfire_target is not None
                and has_los(defender.pos, pre_crossfire_target, smoke_cells)
            )
            for defender in self.defenders
        }
        # 退避の判断は移動前に見えていた人数差で評価する。移動後に敵が
        # 視界から消えた場合も、味方へ合流できた行動を学習できる。
        self._pressure_before_step = {}
        for defender in self.defenders:
            if not defender.is_alive:
                continue
            other_allies = [ally for ally in self.defenders
                            if ally is not defender and ally.is_alive]
            nearest_ally = min(
                other_allies,
                key=lambda ally: max(abs(ally.pos[0] - defender.pos[0]),
                                      abs(ally.pos[1] - defender.pos[1])),
                default=None,
            )
            self._pressure_before_step[defender.name] = (
                regroup_pressure(defender, other_allies, team_visible_enemies),
                tuple(nearest_ally.pos) if nearest_ally is not None else None,
            )

        carriers = [a for a in self.attackers if a.is_alive and a.has_spike]
        others = [a for a in self.attackers if a.is_alive and not a.has_spike]
        for unit in carriers + others:
            dr, dc = self._attacker_decide_move(unit)
            move_plans.append((unit, (dr, dc)))

        # position mode(スパイク情報も敵目撃情報も無い状態)かつ担当地点未到着の
        # 間は、スポーン・担当地点がどちらも毎エピソード固定である以上、移動方向を
        # RLに手探りさせる意味がない。既知のBFS最短方向をそのまま強制適用する。
        in_position_phase = (
            self.team_memory.spike_pos is None
            and self.team_memory.last_seen_enemy is None
            and not team_sees_enemy
        )

        actual_action_dict = {}
        defenders_occupied_now = {tuple(u.pos) for u in self.defenders if u.is_alive}

        for d in self.defenders:
            if not d.is_alive or d.name not in action_dict:
                continue
            (dr, dc), use_ability, use_ultimate, use_orb, facing = decode_action(action_dict[d.name])

            visible_enemies = [
                a for a in self.attackers if a.is_alive and has_los(d.pos, a.pos, smoke_cells)
            ]
            has_enemy_los = bool(visible_enemies)
            support_plan = None if has_enemy_los or d.charges <= 0 else find_support_ability_plan(
                GRID, d, self.defenders, self.attackers, d.role, smoke_cells,
                max_aim_range=ABILITY_RANGE,
            )
            if has_enemy_los:
                use_ability = False
            if use_ultimate and spend_ultimate(d):
                tactical = has_enemy_los or team_sees_enemy
                ultimate_tactical[d.name] = tactical
                targets = visible_enemies or team_visible_enemies
                if targets:
                    target = min(
                        targets,
                        key=lambda enemy: max(
                            abs(enemy.pos[0] - d.pos[0]),
                            abs(enemy.pos[1] - d.pos[1]),
                        ),
                    )
                    ultimate_alignment[d.name] = _facing_alignment(
                        facing, tuple(d.pos), tuple(target.pos)
                    )
                dr, dc = 0, 0
                if d.ultimate_name == "MONITOR":
                    for enemy in self.attackers:
                        if enemy.is_alive:
                            enemy.reveal_remaining = max(enemy.reveal_remaining, REVEAL_DURATION_TICKS)
                elif d.ultimate_name == "TUNNEL":
                    targets = visible_enemies
                    if not targets and team_sees_enemy:
                        targets = [enemy for enemy in self.attackers if enemy.is_alive]
                    for enemy in targets:
                        enemy.blind_remaining = max(enemy.blind_remaining, BLIND_DURATION_TICKS)
            if use_orb:
                dr, dc = 0, 0
                _completed, orb_rewards[d.name] = collect_orb_tick(d, self.available_orbs)
            self_occupied = defenders_occupied_now - {tuple(d.pos)}
            forced_facing = _forced_combat_facing(
                d, visible_enemies, self.team_memory
            )
            if forced_facing is None:
                forced_facing = _forced_watch_facing(
                    d, visible_enemies, self.team_memory
                )
            if forced_facing is not None:
                # マスクだけでなく実行側でも保証し、replayに実際のfacingを記録する。
                facing = forced_facing

            if d.name in scheduled_smoke_targets:
                dr, dc = 0, 0
                use_ability = False

            if in_position_phase and d.assigned_defense_dist_map is not None:
                r0, c0 = int(d.pos[0]), int(d.pos[1])
                cur_dist = d.assigned_defense_dist_map[r0, c0]
                if cur_dist > REACH_RADIUS:
                    self_occupied = defenders_occupied_now - {tuple(d.pos)}
                    dr, dc = bfs_best_direction_detour(
                        d.assigned_defense_dist_map, d.assigned_defense_pos, r0, c0, self_occupied
                    )
                else:
                    dr, dc = 0, 0
            # spike情報/sighting情報がある場合、以前はBFSで強制的に接近させていたが、
            # 「入口(担当ポジション)で待ち構える方が有利」な状況を学習できるよう
            # 強制上書きをやめ、ネットワークが選んだ移動(dr, dc)をそのまま使う。
            # 接近への誘導自体はSPIKE_PULL_REWARD/SIGHTING_PULL_REWARD
            # (ポテンシャル差分報酬)側に残っているため、接近の学習は引き続き可能。

            # 向き(facing)は移動先の決定方法(BFS強制/ネットワーク)と無関係に、
            # ネットワークが選んだ向きをそのまま毎tick適用する。
            if use_ultimate or use_orb or (use_ability and support_plan is not None):
                dr, dc = 0, 0
            d.facing = facing
            d._facing_eval_pos = tuple(d.pos)
            d._facing_route_map = d.assigned_defense_dist_map if in_position_phase else None
            d._facing_move = (dr, dc)
            actual_action_dict[d.name] = encode_action(
                (dr, dc), use_ability, facing, use_ultimate=use_ultimate, use_orb=use_orb
            )
            move_plans.append((d, (dr, dc)))
            # マスクと実行で同じ現在の味方目撃情報と投射経路を使う。
            # 過去の目撃位置だけではアビリティを使わない。
            has_target_info = support_plan is not None

            if has_target_info:
                stats = self.ability_diag_stats.get(d.name)
                if stats is not None:
                    stats["opportunity"] += 1
            if d.role == "SMOKE" and visible_enemies:
                stats = self.ability_diag_stats.get(d.name)
                if stats is not None:
                    stats["own_any_enemy_seen"] += 1

            if has_enemy_los and (dr, dc) == (0, 0):
                held_angle[d.name] = "held_with_los"
            elif has_enemy_los:
                held_angle[d.name] = "moved_with_los"
            else:
                held_angle[d.name] = "no_los"

            if use_ability:
                ability_whiff[d.name] = not has_target_info
                ability_overlap[d.name] = (
                    pre_tick_flash_recon_active and d.role in ("FLASH", "RECON")
                )
                if d.charges > 0:
                    # 狙う相手の有無にかかわらず、use_ability選択時点でチャージを消費する。
                    # 以前はここで消費しておらず、狙う対象が無い場合にチャージが無限に温存され、
                    # 同じユニットが毎tickwhiffを繰り返し選べてしまっていた。
                    d.charges -= 1
                    if support_plan is not None:
                        if d.role == "SMOKE":
                            ability_smoke_support[d.name] = True
                        ability_requests.append((d, support_plan.aim))
                        if support_plan.impact is not None:
                            support_impacts[d.name] = support_plan.impact
        for unit, (dr, dc) in move_plans:
            if not unit.is_alive:
                continue
            old_pos = tuple(unit.pos)
            nr, nc = unit.pos[0] + dr, unit.pos[1] + dc
            if dr == 0 and dc == 0:
                continue
            in_bounds = 0 <= nr < HEIGHT and 0 <= nc < WIDTH
            is_wall = in_bounds and GRID[nr, nc] == 1
            occupied = any(
                other is not unit and other.is_alive and tuple(other.pos) == (nr, nc)
                for other in self.defenders + self.attackers
            )
            if in_bounds and not is_wall and not occupied:
                # facingは既にstep()内でネットワークの選択値がunit.facingへ
                # 直接反映済み(移動方向とは無関係)のため、ここでは上書きしない。
                unit.pos = [nr, nc]
            unit.moved_this_tick = tuple(unit.pos) != old_pos

        post_crossfire_target = (
            tuple(crossfire_enemy.pos) if crossfire_enemy is not None else crossfire_ground_pos
        )
        post_crossfire_los = {
            defender.name: bool(
                defender.is_alive
                and post_crossfire_target is not None
                and has_los(defender.pos, post_crossfire_target, smoke_cells)
            )
            for defender in self.defenders
        }

        if self.spike_ground_pos is not None:
            picker = next(
                (a for a in self.attackers if a.is_alive and tuple(a.pos) == self.spike_ground_pos),
                None,
            )
            if picker is not None:
                picker.has_spike = True
                self.spike_ground_pos = None

        ability_hit = {}
        for unit, target_pos in ability_requests:
            # チャージは選択時点(上のforループ内)で既に消費済みのためここでは減算しない。
            hit_any = False
            if unit.role == "SMOKE":
                tr, tc = int(target_pos[0]), int(target_pos[1])
                cells = {
                    (rr, cc)
                    for rr in range(tr - 1, tr + 2)
                    for cc in range(tc - 1, tc + 2)
                    if 0 <= rr < HEIGHT and 0 <= cc < WIDTH and GRID[rr, cc] != 1
                }
                self.smokes.append({
                    "cells": cells,
                    "remaining_ticks": SMOKE_DURATION_TICKS,
                    "team": unit.team,
                })
                # 診断用: SMOKEの「命中」= 展開セルが敵の現在地を実際に覆っているか
                hit_any = any(a.is_alive and tuple(a.pos) in cells for a in self.attackers)
            elif unit.role == "FLASH":
                impact = support_impacts.get(unit.name)
                for a in self.attackers:
                    if a.is_alive and has_los(
                        impact if impact is not None else target_pos,
                        a.pos, smoke_cells,
                    ):
                        a.blind_remaining = max(a.blind_remaining, BLIND_DURATION_TICKS)
                        hit_any = True
            elif unit.role == "RECON":
                impact = support_impacts.get(unit.name)
                for a in self.attackers:
                    if a.is_alive and (
                        max(abs(impact[0] - a.pos[0]),
                            abs(impact[1] - a.pos[1])) <= RECON_REVEAL_SIZE // 2
                        if impact is not None
                        else has_los(target_pos, a.pos, smoke_cells)
                    ):
                        a.reveal_remaining = max(a.reveal_remaining, REVEAL_DURATION_TICKS)
                        hit_any = True
            ability_hit[unit.name] = hit_any

        # 診断用: aimed(視認ありで使用)/hit/miss/whiff/overlap を集計
        for name, whiffed in ability_whiff.items():
            stats = self.ability_diag_stats.get(name)
            if stats is None:
                continue
            if whiffed:
                stats["whiff"] += 1
            else:
                stats["aimed"] += 1
                if ability_hit.get(name):
                    stats["hit"] += 1
                else:
                    stats["miss"] += 1
            if ability_overlap.get(name):
                stats["overlap"] += 1

        if not any(a.is_alive and a.has_spike for a in self.attackers):
            dropped_holder = next((a for a in self.attackers if a.has_spike), None)
            if dropped_holder is not None:
                self.spike_ground_pos = tuple(dropped_holder.pos)
                dropped_holder.has_spike = False

        self._resolve_shots()

        for u in self.defenders + self.attackers:
            u.blind_remaining = max(0, u.blind_remaining - 1)
            u.reveal_remaining = max(0, u.reveal_remaining - 1)
        for s in self.smokes:
            s["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

        self.team_memory.update(
            self.defenders, self.attackers, self._smoke_cells(), self.spike_ground_pos,
            round_tick=MAX_TICKS - self.round_timer + 1,
            shots=self.last_shots,
        )
        self._update_priority_dist_maps()
        self.round_timer -= 1

        carrier = next((a for a in self.attackers if a.is_alive and a.has_spike), None)
        if carrier is not None:
            site = SITE_POSITIONS[self.carrier_target_site_idx]
            dist_to_site = max(abs(carrier.pos[0]-site[0]), abs(carrier.pos[1]-site[1]))
            if dist_to_site <= 1:
                if not hasattr(self, "_plant_progress"):
                    self._plant_progress = 0
                self._plant_progress += 1
                if self._plant_progress >= PLANT_REQUIRED_TICKS:
                    self.planted = True
            else:
                self._plant_progress = 0
        else:
            self._plant_progress = 0

        rewards = self._compute_rewards(
            pre_tick_enemy_debuffed, ability_whiff, ability_overlap, held_angle, ability_hit,
            pre_crossfire_los=pre_crossfire_los, post_crossfire_los=post_crossfire_los,
            ability_smoke_support=ability_smoke_support,
        )
        for name, tactical in ultimate_tactical.items():
            rewards[name] = rewards.get(name, 0.0) + ultimate_use_reward(tactical)
            rewards[name] += 0.25 * ultimate_alignment.get(name, 0.0)
        for name, orb_reward in orb_rewards.items():
            rewards[name] = rewards.get(name, 0.0) + orb_reward

        self._prev_kills = {u.name: u.kills for u in self.defenders + self.attackers}
        self._prev_alive = {u.name: u.is_alive for u in self.defenders + self.attackers}

        attackers_alive = any(a.is_alive for a in self.attackers)
        defenders_alive = any(d.is_alive for d in self.defenders)
        done = (
            self.planted
            or self.round_timer <= 0
            or not attackers_alive
            or not defenders_alive
        )

        if done:
            if self.planted:
                self.match_over_reason = "planted"
                for d in self.defenders:
                    rewards[d.name] = rewards.get(d.name, 0.0) + PLANT_PENALTY
            elif not attackers_alive or self.round_timer <= 0:
                self.match_over_reason = "defender_win"
                for d in self.defenders:
                    rewards[d.name] = rewards.get(d.name, 0.0) + ROUND_WIN_REWARD
            elif not defenders_alive:
                self.match_over_reason = "defender_wipe"

        self._update_episode_arrival_flags()
        obs_dict, mask_dict = self._collect_observations()
        return obs_dict, mask_dict, rewards, done, actual_action_dict

    def _resolve_shots(self):
        alive = [u for u in self.defenders + self.attackers if u.is_alive]
        smoke_cells = self._smoke_cells()
        shot_intents = []

        for shooter in alive:
            targets = [
                t for t in alive
                if t.team != shooter.team
                and has_los(shooter.pos, t.pos, smoke_cells)
                and (shooter.team != "D" or _facing_angle_diff(
                    shooter.facing, shooter.pos, t.pos
                ) <= SHOOTING_SITE_DIGREE)
            ]
            if not targets:
                continue
            target = min(
                targets,
                key=lambda t: (
                    max(abs(t.pos[0]-shooter.pos[0]), abs(t.pos[1]-shooter.pos[1])),
                    t.hp,
                    t.name,
                ),
            )
            shot_intents.append((shooter, target))

        random.shuffle(shot_intents)
        shot_intents.sort(key=lambda pair: pair[0].reaction, reverse=True)

        self.last_shots = []
        for shooter, target in shot_intents:
            if not shooter.is_alive or not target.is_alive:
                continue

            accuracy = MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            hs_rate = shooter.hs_rate * (0.1 if shooter.moved_this_tick else 1.0)
            # Defender側のみ向き(facing)を学習対象とするため命中率補正を適用する
            # (battle_logic.py._facing_accuracy_multiplierと同一ロジック)。
            # Attackerはヒューリスティックで向きを持たないため対象外(倍率1.0)。
            if shooter.team == "D":
                facing_mult = _facing_accuracy_multiplier(shooter.facing, shooter.pos, target.pos)
                accuracy *= facing_mult
                hs_rate *= facing_mult
            if shooter.moved_last_tick and not shooter.moved_this_tick:
                accuracy *= 0.75
                hs_rate *= 0.75
            if shooter.blind_remaining > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            debuffed = target.blind_remaining > 0 or target.reveal_remaining > 0
            effective_dodge = target.dodge_rate * (REVEALED_DODGE_MULTIPLIER if debuffed else 1.0)
            hit_chance = accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            distance = math.hypot(target.pos[0] - shooter.pos[0],
                                  target.pos[1] - shooter.pos[1])
            if distance <= 1.0:
                distance_multiplier = 2.0
            elif distance <= 15.0:
                distance_multiplier = 2.0 - (distance - 1.0) / 14.0
            elif distance <= 40.0:
                distance_multiplier = 1.0 - 0.25 * (distance - 15.0) / 25.0
            else:
                distance_multiplier = 0.75
            hit_chance *= distance_multiplier
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            if hit:
                headshot = random.random() < hs_rate
                damage = HEADSHOT_DAMAGE if headshot else BODY_DAMAGE
                target.hp = max(0, target.hp - damage)
                self.last_shots.append({"shooter": shooter, "target": target, "hit": True})
                if target.hp <= 0:
                    target.is_alive = False
                    shooter.kills += 1
                    if shooter.role == "HUNT":
                        shooter.hp = min(shooter.max_hp, shooter.hp + 50)
                    if target.has_spike:
                        target.has_spike = False
            else:
                self.last_shots.append({"shooter": shooter, "target": target, "hit": False})

    def _priority_mode_and_distmap(self, defender):
        # スパイク接近は弱い差分報酬に留め、射線参加報酬と戦闘結果が
        # 待機・横移動・接近のどれを選ぶかを決められるようにする。
        visible_enemies = [
            a for a in self.attackers
            if a.is_alive and has_los(defender.pos, a.pos, self._smoke_cells())
        ]
        visible_spike_holder = any(a.has_spike for a in visible_enemies)
        active_spike_target = (
            self.team_memory.spike_pos is not None
            and self.team_memory.spike_held
        )
        if self.team_memory.spike_pos is not None and self.spike_dist_map is not None:
            if self.team_memory.spike_held:
                return "spike", self.spike_dist_map, "spike"
            return "spike_ground", self.spike_dist_map, "spike_ground"
        if self.team_memory.last_seen_enemy is not None and self.sighting_dist_map is not None:
            return "sighting", self.sighting_dist_map, self.team_memory.last_seen_enemy["name"]
        if visible_enemies and not visible_spike_holder and not active_spike_target:
            return "combat_hold", None, "combat_hold"
        return "position", defender.assigned_defense_dist_map, "position"

    def _compute_rewards(
        self, pre_tick_enemy_debuffed, ability_whiff, ability_overlap, held_angle,
        ability_hit=None,
        pre_crossfire_los=None, post_crossfire_los=None,
        ability_smoke_support=None,
    ):
        ability_hit = ability_hit or {}
        ability_smoke_support = ability_smoke_support or {}
        pre_crossfire_los = pre_crossfire_los or {}
        post_crossfire_los = post_crossfire_los or {}
        rewards = {}
        smoke_cells = self._smoke_cells()
        team_visible_enemies = [
            enemy for enemy in self.attackers if enemy.is_alive
            and any(ally.is_alive and has_los(
                ally.pos, enemy.pos, smoke_cells
            ) for ally in self.defenders)
        ]
        for d in self.defenders:
            r = STEP_PENALTY

            mode, dist_map, target_key = self._priority_mode_and_distmap(d)
            r0, c0 = int(d.pos[0]), int(d.pos[1])
            bfs_dist = dist_map[r0, c0] if dist_map is not None else None
            if bfs_dist is not None and bfs_dist < 0:
                bfs_dist = None

            if mode == "combat_hold":
                # 直接視認中の本人は追撃しない。敵を見ていない味方は
                # sightingモードのまま増援として接近できる。
                r += (
                    COMBAT_HOLD_BONUS
                    if not d.moved_this_tick
                    else COMBAT_MOVE_PENALTY
                )

            r += crossfire_coordination_reward(
                d.name, pre_crossfire_los, post_crossfire_los
            )

            # 配置地点到着後の向きを明確に学習させる。
            # 移動中は向きの自由度を維持し、配置地点にいるときだけ
            # 現在の優先対象(spike / last seen enemy / watch point)に対する
            # 8方向の正解・不正解を直接報酬化する。
            # facing shaping は、移動経路と配置直後の監視方向だけに限定する。
            eval_pos = getattr(d, "_facing_eval_pos", tuple(d.pos))
            route_facing = planned_route_facing(
                getattr(d, "_facing_route_map", None), eval_pos
            )
            if route_facing is None:
                move_dr, move_dc = getattr(d, "_facing_move", (0, 0))
                route_facing = _facing_from_delta(move_dr, move_dc, None)
            if route_facing is not None and d.facing == route_facing:
                r += FACING_ROUTE_REWARD

            at_defense_pos = (
                d.assigned_defense_pos is not None
                and tuple(map(int, d.pos)) == tuple(map(int, d.assigned_defense_pos))
            )
            if at_defense_pos and self.team_memory.last_seen_enemy is None:
                facing_ticks = self.position_facing_ticks.get(d.name, 0)
                if facing_ticks < FACING_SETUP_REWARD_TICKS:
                    watch_pos = _nearest_watch_point(d.name, tuple(d.pos))
                    if watch_pos is not None:
                        expected_facing = _expected_facing(tuple(d.pos), watch_pos)
                        if expected_facing is not None and d.facing == expected_facing:
                            r += FACING_POSITION_REWARD
                    self.position_facing_ticks[d.name] = facing_ticks + 1

            if bfs_dist is None:
                d.prev_priority_mode = mode
                d.prev_priority_target_key = target_key
                d.prev_priority_dist = None
            elif (
                mode != d.prev_priority_mode
                or target_key != d.prev_priority_target_key
                or d.prev_priority_dist is None
            ):
                d.prev_priority_mode = mode
                d.prev_priority_target_key = target_key
                d.prev_priority_dist = bfs_dist
            else:
                delta = d.prev_priority_dist - bfs_dist
                d.prev_priority_dist = bfs_dist

                if mode == "sighting_hold":
                    r += HOLD_POSITION_BONUS if not d.moved_this_tick else HOLD_POSITION_PENALTY
                elif mode == "position":
                    if bfs_dist > REACH_RADIUS:
                        r += DEFENSE_POSITION_PULL_REWARD * delta
                    elif d.moved_this_tick:
                        r += HOLD_POSITION_PENALTY
                    else:
                        watch_pos = _nearest_watch_point(d.name, tuple(d.pos))
                        if watch_pos is not None:
                            facing_ok = _facing_alignment(d.facing, tuple(d.pos), watch_pos) > 0.5
                            r += HOLD_POSITION_BONUS if facing_ok else HOLD_POSITION_BONUS * 0.2
                        else:
                            r += HOLD_POSITION_BONUS

                    # --- 診断用: positionモード時のみ、BFS距離・到着・移動を記録 ---
                    stats = self.position_mode_stats.get(d.name)
                    if stats is not None:
                        stats["dist_sum"] += float(bfs_dist)
                        stats["dist_count"] += 1
                        if d.moved_this_tick:
                            stats["moved_count"] += 1
                        if bfs_dist <= REACH_RADIUS:
                            stats["arrived_count"] += 1

            site_target = (
                self.team_memory.spike_pos if self.team_memory.spike_pos is not None
                else self.team_memory.last_seen_enemy["pos"]
                if self.team_memory.last_seen_enemy is not None
                else team_visible_enemies[0].pos if team_visible_enemies else None
            )
            target_site = site_index(site_target, SITE_DIST_MAPS)
            if target_site is not None:
                target_map = SITE_DIST_MAPS[target_site]
                before_pos = getattr(d, "_position_before_step", tuple(d.pos))
                before_site_dist = walking_distance(before_pos, target_map)
                after_site_dist = walking_distance(d.pos, target_map)
                assigned_site = site_index(d.assigned_defense_pos, SITE_DIST_MAPS)
                sighting_weight = (
                    max(0.0, 1.0 - self.team_memory.last_seen_enemy["tick_ago"]
                        / SIGHTING_MEMORY_TICKS)
                    if self.team_memory.spike_pos is None
                    and self.team_memory.last_seen_enemy is not None else 1.0
                )
                if assigned_site != target_site:
                    if before_site_dist > SITE_SUPPORT_RADIUS and after_site_dist >= 0:
                        r += SITE_ROTATE_REWARD * sighting_weight * (
                            before_site_dist - after_site_dist
                        )
                elif not d.moved_this_tick and (
                    has_los(d.pos, site_target, smoke_cells)
                    or (watch_pos := _nearest_watch_point(d.name, tuple(d.pos))) is not None
                    and has_los(d.pos, watch_pos, smoke_cells)
                ):
                    r += SITE_HOLD_REWARD * sighting_weight
                elif (assigned_site == target_site and before_site_dist >= 0
                      and after_site_dist > before_site_dist):
                    r += SITE_ABANDON_PENALTY * sighting_weight

            if (mode == "sighting" and target_site is not None
                    and site_index(d.assigned_defense_pos, SITE_DIST_MAPS) != target_site
                    and walking_distance(d.pos, SITE_DIST_MAPS[target_site])
                    > SITE_SUPPORT_RADIUS):
                sees_enemy = any(
                    a.is_alive and has_los(d.pos, a.pos, smoke_cells)
                    for a in self.attackers
                )
                if not d.moved_this_tick and not sees_enemy:
                    r += SIGHTING_IDLE_PENALTY

            pressure, ally_pos = getattr(self, "_pressure_before_step", {}).get(
                d.name, (False, None)
            )
            if pressure and ally_pos is not None and d.is_alive:
                before_pos = getattr(d, "_position_before_step", tuple(d.pos))
                before_ally_dist = max(abs(before_pos[0] - ally_pos[0]),
                                       abs(before_pos[1] - ally_pos[1]))
                after_ally_dist = max(abs(d.pos[0] - ally_pos[0]),
                                      abs(d.pos[1] - ally_pos[1]))
                r += REGROUP_REWARD * pressure * (before_ally_dist - after_ally_dist)

            if d.name in ability_whiff:
                if ability_whiff[d.name]:
                    # 視認情報が無い状態での使用(既存のまま)
                    r += ABILITY_WHIFF_PENALTY
                else:
                    # 視認あり・射程内で使用(外れてもここまでは付与)
                    r += ABILITY_AIMED_REWARD
                    if ability_hit.get(d.name):
                        r += ABILITY_HIT_BONUS
                    if d.role == "SMOKE" and ability_smoke_support.get(d.name):
                        r += SMOKE_SUPPORT_TARGET_BONUS
            if ability_overlap.get(d.name):
                r += ABILITY_OVERLAP_PENALTY

            angle_state = held_angle.get(d.name)
            if angle_state == "held_with_los":
                r += HOLD_ANGLE_BONUS
            elif angle_state == "moved_with_los":
                r += HOLD_ANGLE_PENALTY

            # 敵を直接視認できているtickは、射撃結果が疎でも敵方向を
            # 向く行動に即時の学習信号を与える。
            visible_enemies = [
                a for a in self.attackers
                if a.is_alive and has_los(d.pos, a.pos, self._smoke_cells())
            ]
            if visible_enemies:
                nearest_enemy = min(
                    visible_enemies,
                    key=lambda a: max(
                        abs(a.pos[0] - d.pos[0]), abs(a.pos[1] - d.pos[1])
                    ),
                )
                r += FACING_ALIGN_VISIBLE_WEIGHT * _facing_alignment(
                    d.facing, tuple(d.pos), tuple(nearest_enemy.pos)
                )

            # 戦闘・ピーク中のfacingを明確に学習させる。
            # 直接視認できる場合は視認中の最寄り敵を優先し、視認できない場合は
            # TeamMemoryのlast_seen_enemyを使う。移動中もこの報酬を与えることで、
            # 敵へ寄りながら敵と反対方向を向く行動を抑制する。
            combat_target = None
            combat_facing_weight = 0.0
            if visible_enemies:
                combat_target = tuple(nearest_enemy.pos)
            elif self.team_memory.last_seen_enemy is not None:
                memory_relevance = _facing_memory_relevance(
                    tuple(d.pos), self.team_memory.last_seen_enemy
                )
                if memory_relevance > 0.0:
                    combat_target = tuple(self.team_memory.last_seen_enemy["pos"])
                    combat_facing_weight = memory_relevance

            if visible_enemies:
                combat_facing_weight = 1.0

            if combat_target is not None:
                combat_expected_facing = _expected_facing(
                    tuple(d.pos), combat_target
                )
                if combat_expected_facing is not None:
                    r += combat_facing_weight * (
                        FACING_COMBAT_CORRECT_REWARD
                        if d.facing == combat_expected_facing
                        else FACING_COMBAT_INCORRECT_PENALTY
                    )

            new_kills = d.kills - self._prev_kills.get(d.name, d.kills)
            if new_kills > 0:
                r += KILL_REWARD * new_kills
                
                for shot in getattr(self, "last_shots", []):
                    if (
                        shot["shooter"] is d
                        and shot["hit"]
                        and not shot["target"].is_alive
                        and pre_tick_enemy_debuffed.get(shot["target"].name, False)
                    ):
                        r += DEBUFF_KILL_BONUS
                        stats = self.ability_diag_stats.get(d.name)
                        if stats is not None:
                            stats["debuff_kill"] += 1

            # 💡追加: facing整合の弱いshaping報酬。自分が実際の敵を直接視認して
            # いない時のみ有効(直接視認時は通常の交戦報酬(命中率経由)に委ねる)。
            # 優先順位はmode(spike > sighting > position)にそのまま揃える。
            watch_pos, watch_weight = None, 0.0
            if team_memory := self.team_memory:
                if team_memory.spike_pos is not None:
                    watch_pos, watch_weight = team_memory.spike_pos, FACING_ALIGN_SPIKE_WEIGHT
                elif team_memory.last_seen_enemy is not None:
                    memory_relevance = _facing_memory_relevance(
                        tuple(d.pos), team_memory.last_seen_enemy
                    )
                    if memory_relevance > 0.0:
                        watch_pos = team_memory.last_seen_enemy["pos"]
                        watch_weight = FACING_ALIGN_SIGHTING_WEIGHT * memory_relevance
                    elif d.assigned_defense_pos is not None:
                        watch_pos = (
                            _nearest_watch_point(d.name, tuple(d.pos))
                            or d.assigned_defense_pos
                        )
                        watch_weight = FACING_ALIGN_POSITION_WEIGHT
                elif d.assigned_defense_pos is not None:
                    watch_pos = (
                        _nearest_watch_point(d.name, tuple(d.pos))
                        or d.assigned_defense_pos
                    )
                    watch_weight = FACING_ALIGN_POSITION_WEIGHT
            if watch_pos is not None:
                r += watch_weight * _facing_alignment(d.facing, tuple(d.pos), watch_pos)

            was_alive = self._prev_alive.get(d.name, True)
            if was_alive and not d.is_alive:
                r += DEATH_PENALTY

            rewards[d.name] = r
        return rewards


# ============================================================================
# 学習ループ
# ============================================================================

def epsilon_by_episode(episode, total_episodes=EPISODE_COUNT, eps_start=1.0, eps_end=0.05, decay_ratio=0.8):
    decay_episodes = total_episodes * decay_ratio
    return max(eps_end, eps_start - (eps_start - eps_end) * episode / decay_episodes)


# select_action / optimize は ov1_common_rl.select_action /
# ov1_common_rl.optimize_double_dqn_step を直接使用(呼び出し側train()を参照)。

def train(
    episodes=EPISODE_COUNT,
    batch_size=128,
    gamma=0.99,
    lr=1e-4,
    buffer_size=200_000,
    target_update_every=1000,
    warm_start=True,
):
    policy_net = DuelingQNet(OBS_DIM, ACTION_DIM).to(DEVICE)
    legacy_path = os.path.join(DATA_DIR, "dqn_defender_search_best_by_eval.pt")
    if warm_start and os.path.exists(legacy_path):
        state = torch.load(legacy_path, map_location=DEVICE)
        old_weight = state["feature.0.weight"]
        if old_weight.shape[1] == OBS_DIM - SITE_CONTEXT_DIM:
            expanded = torch.zeros(
                (old_weight.shape[0], OBS_DIM), dtype=old_weight.dtype,
                device=old_weight.device,
            )
            expanded[:, :old_weight.shape[1]] = old_weight
            state["feature.0.weight"] = expanded
        policy_net.load_state_dict(state)
        print(f"[WARM START] {legacy_path}")
    target_net = DuelingQNet(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(Transition, capacity=buffer_size)
    env = SearchEnv()

    global_step = 0
    best_avg_reward = -float("inf")
    episode_reward_history = deque(maxlen=100)
    eval_avg_reward_history = deque(maxlen=10)  # epsilon=0評価の平滑化用(5->10)

    # --- 診断用(1): キャラ別(ロール別)の直近100エピソード報酬履歴 ---
    per_name_reward_history = {name: deque(maxlen=100) for name in ROSTER_ORDER}
    per_name_episode_total = {name: 0.0 for name in ROSTER_ORDER}

    # --- 診断用(4): positionモード中の「平均BFS距離・到着率・移動率」履歴 ---
    per_name_avg_dist_history = {name: deque(maxlen=100) for name in ROSTER_ORDER}
    per_name_arrival_rate_history = {name: deque(maxlen=100) for name in ROSTER_ORDER}
    per_name_move_rate_history = {name: deque(maxlen=100) for name in ROSTER_ORDER}

    # --- 診断用(5): アビリティ使用内訳(aimed/hit/miss/whiff/overlap/debuff_kill)の直近100エピソード履歴 ---
    ABILITY_DIAG_KEYS = ("aimed", "hit", "miss", "whiff", "overlap", "debuff_kill", "opportunity", "own_any_enemy_seen")
    per_name_ability_history = {
        name: {key: deque(maxlen=100) for key in ABILITY_DIAG_KEYS}
        for name in ROSTER_ORDER
    }

    start_time = time.perf_counter()
    for episode in range(1, episodes + 1):
        # --- 一時デバッグ用: 500エピソードごとに1エピソードだけ実況トレースON ---
        env.debug_trace = (episode % 500 == 0)

        obs_dict, mask_dict = env.reset()
        episode_reward_total = 0.0
        epsilon = epsilon_by_episode(episode)

        # --- 診断用(1): このエピソードのキャラ別累計報酬をリセット ---
        for name in per_name_episode_total:
            per_name_episode_total[name] = 0.0

        for tick in range(MAX_TICKS + DEFENDER_SETUP_TICKS):

            action_dict = {
                name: select_action(policy_net, obs, mask_dict[name], epsilon)
                for name, obs in obs_dict.items()
            }

            next_obs_dict, next_mask_dict, rewards, done, actual_action_dict = env.step(action_dict)

            for name, obs in obs_dict.items():
                # position mode強制上書きにより、ネットワークが選んだ行動と
                # 実際に反映された行動がズレる場合があるため、学習用には
                # 実際に反映された方(actual_action_dict)を使う。
                action = actual_action_dict.get(name, action_dict[name])
                reward = rewards.get(name, 0.0)
                episode_reward_total += reward
                # --- 診断用(1): キャラ別に報酬を積算 ---
                per_name_episode_total[name] = per_name_episode_total.get(name, 0.0) + reward

                if name in next_obs_dict:
                    next_obs = next_obs_dict[name]
                    next_mask = next_mask_dict[name]
                    step_done = done
                else:
                    next_obs = obs
                    next_mask = mask_dict[name]
                    step_done = True

                buffer.push(obs, action, reward, next_obs, next_mask, float(step_done))

            obs_dict, mask_dict = next_obs_dict, next_mask_dict
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

            if done or not obs_dict:
                break

        # --- 診断用(4): このエピソードのpositionモード統計を履歴に積算 ---
        for name in ROSTER_ORDER:
            stats = env.position_mode_stats.get(name, {})
            dist_count = stats.get("dist_count", 0)
            # 到達判定は position mode の有無に依存させない。
            # position mode が短い/存在しないエピソードでも、実際に指定位置へ
            # 到達していれば best 判定用の履歴へ記録する。
            per_name_arrival_rate_history[name].append(
                1.0 if env.episode_arrived.get(name, False) else 0.0
            )
            if dist_count > 0:
                avg_dist = stats["dist_sum"] / dist_count
                move_rate = stats["moved_count"] / dist_count
                per_name_avg_dist_history[name].append(avg_dist)
                per_name_move_rate_history[name].append(move_rate)

        episode_reward_history.append(episode_reward_total)
        avg_reward = sum(episode_reward_history) / len(episode_reward_history)

        # --- 診断用(5): このエピソードのアビリティ使用内訳を履歴に積算 ---
        for name in ROSTER_ORDER:
            stats = env.ability_diag_stats.get(name, {})
            for key in ABILITY_DIAG_KEYS:
                per_name_ability_history[name][key].append(stats.get(key, 0))

        # --- 診断用(1): キャラ別報酬履歴に積算値を追加 ---
        for name in per_name_reward_history:
            per_name_reward_history[name].append(per_name_episode_total.get(name, 0.0))

        if episode % 200 == 0:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter();
            print(
                f"[EP {episode}/{episodes}] reward={episode_reward_total:.3f} "
                f"avg100={avg_reward:.3f} epsilon={epsilon_by_episode(episode):.3f} "
                f"buffer={len(buffer)} reason={env.match_over_reason} "
                f"elapse={elapsed_time:.1f} "
            )
            # --- 診断用(1): キャラ別(ロール別)の直近100エピソード平均報酬 ---
            per_name_str = " / ".join(
                f"{name}({EFFECTIVE_STATS[name]['ability']})="
                f"{(sum(per_name_reward_history[name]) / len(per_name_reward_history[name])):.3f}"
                for name in ROSTER_ORDER
                if len(per_name_reward_history[name]) > 0
            )
            #print(f"  [PER-CHAR avg100] {per_name_str}")

            # --- 診断用(4): positionモード中の平均BFS距離・到着率・移動率 ---
            position_diag_str = " \n ".join(
                f"{name}(dist={sum(per_name_avg_dist_history[name]) / len(per_name_avg_dist_history[name]):.2f},"
                f"arrive={sum(per_name_arrival_rate_history[name]) / len(per_name_arrival_rate_history[name]) * 100:.1f}%,"
                f"move={sum(per_name_move_rate_history[name]) / len(per_name_move_rate_history[name]) * 100:.1f}%)"
                for name in ROSTER_ORDER
                if len(per_name_avg_dist_history[name]) > 0
            )
            print(f"  [POSITION-MODE diag | arrive=episode-wide]\n {position_diag_str}")

            # --- 診断用(5): 直近<=100エピソード合計でのアビリティ使用内訳 ---
            # ability_diag_str = " \n ".join(
            #     f"{name}(aimed={sum(per_name_ability_history[name]['aimed'])},"
            #     f"hit={sum(per_name_ability_history[name]['hit'])},"
            #     f"miss={sum(per_name_ability_history[name]['miss'])},"
            #     f"whiff={sum(per_name_ability_history[name]['whiff'])},"
            #     f"overlap={sum(per_name_ability_history[name]['overlap'])},"
            #     f"dbuff_kill={sum(per_name_ability_history[name]['debuff_kill'])},"
            #     f"opp={sum(per_name_ability_history[name]['opportunity'])},"
            #     f"seen={sum(per_name_ability_history[name]['own_any_enemy_seen'])})"
            #     for name in ROSTER_ORDER
            # )
            # print(f"  [ABILITY diag, sum over last<=100 eps]\n {ability_diag_str}")

        if episode % EVAL_EVERY == 0:
            eval_metrics = evaluate_policy(policy_net, env, EVAL_EPISODES)
            eval_avg = eval_metrics["avg_reward"]
            eval_avg_reward_history.append(eval_avg)
            eval_avg_smoothed = sum(eval_avg_reward_history) / len(eval_avg_reward_history)
            eval_score = evaluation_score(eval_metrics)
            per_agent_arrival = " / ".join(
                f"{name}={rate:.3f}"
                for name, rate in eval_metrics["arrival_rate_by_name"].items()
            )
            print(
                f"  [EVAL per={EVAL_EPISODES}] "
                f"score={eval_score:.3f} arrival={eval_metrics['arrival_rate']:.3f} "
                f"avg={eval_avg:.3f} smoothed={eval_avg_smoothed:.3f} "
                f"all_arrived={eval_metrics['all_arrived_rate']:.3f} "
                f"facing={eval_metrics['facing_rate']:.3f} "
                f"plant_prevent={eval_metrics['plant_prevent_rate']:.3f} "
                f"win={eval_metrics['defender_win_rate']:.3f} "
                f"kills={eval_metrics['avg_kills']:.3f}"
            )
            print(f"  [EVAL arrival_by_agent] {per_agent_arrival}")

            if episode < EVAL_MIN_EPISODE:
                print(f"  [SAVE skip] episode={episode} < EVAL_MIN_EPISODE={EVAL_MIN_EPISODE} (epsilon依然高いため候補から除外)")
            # elif min(eval_metrics["arrival_rate_by_name"].values()) < MIN_EVAL_ARRIVAL_RATE:
            #     print(
            #         f"  [SAVE skip] per-agent arrival rate is below "
            #         f"MIN_EVAL_ARRIVAL_RATE={MIN_EVAL_ARRIVAL_RATE:.3f}"
            #     )
            elif eval_score > best_avg_reward:
                best_avg_reward = eval_score
                torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
                print(f"[SAVE] best model updated: score={eval_score:.3f} -> {MODEL_SAVE_PATH}")

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), MODEL_LATEST_PATH)

    if os.path.exists(MODEL_SAVE_PATH):
        best_state = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        policy_net.load_state_dict(best_state)
        final_best_eval = evaluate_policy(policy_net, env, EVAL_EPISODES)
        print(
            f"[FINAL BEST EVAL eps=0, n={EVAL_EPISODES}] "
            f"avg={final_best_eval['avg_reward']:.3f} "
            f"score={evaluation_score(final_best_eval):.3f} "
            f"arrival={final_best_eval['arrival_rate']:.3f} "
            f"all_arrived={final_best_eval['all_arrived_rate']:.3f} "
            f"facing={final_best_eval['facing_rate']:.3f} "
            f"plant_prevent={final_best_eval['plant_prevent_rate']:.3f} "
            f"win={final_best_eval['defender_win_rate']:.3f} "
            f"kills={final_best_eval['avg_kills']:.3f} "
            f"checkpoint={MODEL_SAVE_PATH}"
        )
        print(
            "[FINAL BEST arrival_by_agent] "
            + " / ".join(
                f"{name}={rate:.3f}"
                for name, rate in final_best_eval["arrival_rate_by_name"].items()
            )
        )
    else:
        print(f"[FINAL BEST EVAL skipped] checkpoint not found: {MODEL_SAVE_PATH}")

    print("[DONE] training finished.")

def evaluation_score(metrics):
    """目的に沿った評価スコア。到着率の低いモデルは別途候補から除外する。"""
    return (
        4.0 * metrics["defender_win_rate"]
        + 3.0 * metrics["plant_prevent_rate"]
        + 1.5 * metrics["facing_rate"]
        + 1.0 * metrics["arrival_rate"]
        + 0.5 * min(metrics["avg_kills"] / max(N_ATTACKERS, 1), 1.0)
    )


def evaluate_policy(policy_net, env, episodes=EVAL_EPISODES):
    """epsilon=0(greedy)でepisodes回プレイし、平均合計報酬を返す。
    学習(buffer/optimizer)には一切触れない評価専用ループ。
    policy_netのtrain/evalモードはBatchNorm等未使用のため実質影響ないが、
    将来的な拡張に備えてeval()/train()を明示的に切り替えておく。"""
    policy_net.eval()
    total = 0.0
    arrival_count = 0
    agent_count = 0
    all_arrived_count = 0
    arrival_by_name_count = {d.name: 0 for d in env.defenders}
    facing_checks = 0
    facing_correct = 0
    plant_prevented_count = 0
    defender_win_count = 0
    kill_total = 0
    with torch.no_grad():
        for _ in range(episodes):
            obs_dict, mask_dict = env.reset()
            ep_reward = 0.0
            arrived = {d.name: False for d in env.defenders}
            for _tick in range(MAX_TICKS + DEFENDER_SETUP_TICKS):
                action_dict = {
                    name: select_action(policy_net, obs, mask_dict[name], 0.0)
                    for name, obs in obs_dict.items()
                }
                obs_dict, mask_dict, rewards, done, _ = env.step(action_dict)
                ep_reward += sum(rewards.values())

                for d in env.defenders:
                    assigned_pos = getattr(d, "assigned_defense_pos", None)
                    if assigned_pos is None:
                        continue
                    # 実ゲームの配置判定と一致させるため、BFS距離ではなく
                    # assigned_defense_posとの実座標一致で判定する。
                    if tuple(d.pos) == tuple(assigned_pos):
                        arrived[d.name] = True
                    # facing評価は、スパイク・敵の既知情報がまだない初期配置中に限定する。
                    # 交戦開始後は敵方向を向くことが正しい場合があるため、
                    # watch point評価に混ぜない。
                    # facing評価は、到達後に移動した tick を含めず、
                    # このtickで実際に担当配置地点にいる場合だけ行う。
                    # 学習側のFACING_POSITION_REWARD条件と一致させる。
                    at_assigned_defense_pos = (
                        tuple(d.pos) == tuple(assigned_pos)
                    )
                    if (
                        at_assigned_defense_pos
                        and env.team_memory.spike_pos is None
                        and env.team_memory.last_seen_enemy is None
                    ):
                        facing_checks += 1
                        watch_pos = _nearest_watch_point(d.name, tuple(d.pos))
                        if (
                            watch_pos is not None
                            and _facing_alignment(d.facing, tuple(d.pos), watch_pos) > 0.5
                        ):
                            facing_correct += 1

                if done or not obs_dict:
                    break
            total += ep_reward
            arrival_count += sum(arrived.values())
            agent_count += len(arrived)
            for name, did_arrive in arrived.items():
                arrival_by_name_count[name] += int(did_arrive)
            all_arrived_count += int(bool(arrived) and all(arrived.values()))
            plant_prevented_count += int(not env.planted)
            defender_win_count += int(env.match_over_reason == "defender_win")
            kill_total += sum(d.kills for d in env.defenders)
    policy_net.train()
    return {
        "avg_reward": total / episodes,
        "arrival_rate": arrival_count / max(agent_count, 1),
        "arrival_rate_by_name": {
            name: count / episodes
            for name, count in arrival_by_name_count.items()
        },
        "all_arrived_rate": all_arrived_count / episodes,
        "facing_rate": facing_correct / max(facing_checks, 1),
        "plant_prevent_rate": plant_prevented_count / episodes,
        "defender_win_rate": defender_win_count / episodes,
        "avg_kills": kill_total / episodes,
    }

if __name__ == "__main__":
    train()
