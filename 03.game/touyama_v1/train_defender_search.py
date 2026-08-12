"""touyama_v1/train_defender_search.py

固定チーム(いぐるん/夢の街/ろびぃな/Tortlilyan/えんぺん)専用の
Defender「search phase」学習スクリプト（プラント前限定）。

train_defender_search.py(root/defender_v3)をベースに、以下を固定化した:
  - ステータス: character_stats_touyama.py の生値 + 常時発動コンボ
    「ふわんだりぃず」(ろびぃな/えんぺん/いぐるん) + タイガーパッシブ(Tortlilyan)
  - ロール: 上記5人のロールに完全固定(ランダム選択なし)
  - スポーン: ロースター順 = DEFENDER_SPAWNS(行優先走査順)の対応で固定
    (run_game.py の実際のスポーン割り当てロジックと一致)

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py などのfeatureモジュールは一切importしない。
character_stats_touyama.py は game_core.py / map_data.py と同様、
ロジックを含まない定数専用ファイルとして参照する(import制限の対象外)。

学習データ・チェックポイントは touyama_v1/data/defender_search_touyama_data/
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
from map_data_search_touyama import SEARCH_MAZE_STR
from character_stats_touyama import CHARACTER_TABLE as TOUYAMA_STATS_TABLE

import common_rl
from common_rl import DEVICE, DuelingQNet, ReplayBuffer, select_action, optimize_double_dqn_step
from common_defender import (
    TOUYAMA_ROSTER_ORDER,
    compute_touyama_effective_stats,
    print_effective_stats,
)

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
)

EPISODE_COUNT = 8000

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = "data/defender_search_touyama_data/"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_defender_search_touyama_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_defender_search_touyama_latest.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------

CARDINAL = common_rl.CARDINAL_MOVES
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 36  # 31(従来) + 4(BFS距離 + 推奨方向dr,dc + 到着フラグ) + 1(spike_watchフラグ)
ACTION_DIM = 10  # move_idx(0-4) * 2 + use_ability_flag(0/1)
ROLES = ["FLASH", "SMOKE", "RECON", "HUNT"]  # Attacker(敵)側の簡易ヒューリスティック用

N_DEFENDERS = 5
N_ATTACKERS = 5
MAX_TICKS = ROUND_DURATION_TICKS  # 90

ABILITY_RANGE = 8       # FLASH/RECONを即時適用してよい最大距離(簡易化)
SIGHTING_STALENESS_CAP = 30
REACH_RADIUS = 1        # 担当ポジションへ「到着した」とみなすBFS距離

# 敵(Attacker)側の既定ステータス(当面ヒューリスティックのため簡易値のまま)
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0

# ---------------------------------------------------------------------------
# touyama_v1 固定チーム定義
# ---------------------------------------------------------------------------
TOUYAMA_SPIKE_HOLDER = "ろびぃな"  # このsearch phaseでは未使用。carry/guard学習用に保持。

# 💡注意: accuracy/hs_rate/dodge_rateのスケールが0-100→0-1に変わる
# (common_defender.compute_touyama_effective_stats に統一。合意事項)。
TOUYAMA_EFFECTIVE_STATS = compute_touyama_effective_stats(TOUYAMA_STATS_TABLE)
print_effective_stats(TOUYAMA_EFFECTIVE_STATS, "Defender/search")


# 報酬パラメータ
# 優先度: SPIKE > SIGHTING > DEFENSE_POSITION > HOLD_POSITION
# の順で明確に重みを引き離し、「待機の方が得」という学習結果を防ぐ。
STEP_PENALTY = -0.001
SPIKE_PULL_REWARD = 0.08         # スパイク確定方向へ近づく(ポテンシャル差分)
SIGHTING_PULL_REWARD = 0.05      # 敵目撃方向へ近づく(ポテンシャル差分)
DEFENSE_POSITION_PULL_REWARD = 0.03   # 平常時、担当7地点へ寄る(ポテンシャル差分)
HOLD_POSITION_BONUS = 0.02            # 担当地点到着後、静止
HOLD_POSITION_PENALTY = -0.01         # 担当地点到着後、無駄にうろつく
ABILITY_WHIFF_PENALTY = -0.05
ABILITY_OVERLAP_PENALTY = -0.05
ABILITY_AIMED_REWARD = 0.03      # 視認あり・射程内で使用(外れても付与、whiffとは排他)
ABILITY_HIT_BONUS = 0.10         # 上記に加え、実際に命中/デバフ成立した場合に追加
DEBUFF_KILL_BONUS = 0.3
HOLD_ANGLE_BONUS = 0.02
HOLD_ANGLE_PENALTY = -0.01
SPIKE_WATCH_HOLD_BONUS = 0.02       # 落下中スパイクにLOSが通っている間、静止(待ち伏せ)
SPIKE_WATCH_MOVE_PENALTY = -0.01    # 同上、無駄にうろつく
KILL_REWARD = 0.5
DEATH_PENALTY = -0.5
ROUND_WIN_REWARD = 1.0          # 時間切れ・全滅によるDefender勝利
PLANT_PENALTY = -0.5            # このフェーズの範囲外(プラント成立)に至った場合


# ============================================================================
# マップ読み込み(map_data.NEW_MAZE_STR / map_data_search.SEARCH_MAZE_STR
# のみ参照。パース処理は自前で複製)
# ============================================================================

GRID = common_rl.parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
ATTACKER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 3]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 2]

if len(DEFENDER_SPAWNS) < len(TOUYAMA_ROSTER_ORDER):
    raise RuntimeError(
        f"DEFENDER_SPAWNSが{len(DEFENDER_SPAWNS)}マスしかなく、"
        f"固定ロースター{len(TOUYAMA_ROSTER_ORDER)}人分を配置できません。"
    )

_SEARCH_GRID = common_rl.parse_grid(SEARCH_MAZE_STR)

# ロースター順(TOUYAMA_ROSTER_ORDER)に、マップ上の値 5,6,7,8,9 をそのまま
# 1:1で対応させる。5=roster[0], 6=roster[1], 7=roster[2], 8=roster[3], 9=roster[4]。
# 5人×5地点の総当たり最適化は不要で、マップ側で明示的に指定された担当地点へ
# ロースター順のままそのまま割り当てるだけでよい。
TOUYAMA_DEFENSE_POSITION_VALUES = {
    name: 5 + i for i, name in enumerate(TOUYAMA_ROSTER_ORDER)
}


def _find_marker_position(grid, value):
    hits = [
        (r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])
        if grid[r, c] == value
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"map_data_search.py の値{value}は1マスのみである必要がありますが、"
            f"{len(hits)}マス見つかりました: {hits}"
        )
    return hits[0]


TOUYAMA_DEFENSE_ASSIGNMENT = {
    name: _find_marker_position(_SEARCH_GRID, value)
    for name, value in TOUYAMA_DEFENSE_POSITION_VALUES.items()
}

# DEFENSE_POSITIONS はロースター順の担当地点リスト(既存コードとの互換用)。
DEFENSE_POSITIONS = [TOUYAMA_DEFENSE_ASSIGNMENT[name] for name in TOUYAMA_ROSTER_ORDER]


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
    return [c["centroid"] for c in clusters[:max_sites]]


SITE_POSITIONS = _extract_site_positions(PLANT_CELLS, max_sites=2)
if not SITE_POSITIONS:
    SITE_POSITIONS = [(HEIGHT / 2.0, WIDTH / 2.0)]


# ============================================================================
# LOS・BFS(abilities_los.py / controllers.py と同等のロジックを複製)
# ============================================================================

def has_los(p1, p2, smoke_cells=None):
    return common_rl.has_los(GRID, p1, p2, smoke_cells)


def bfs_distance_map(goal):
    return common_rl.bfs_distance_map(GRID, goal)


bfs_best_direction = common_rl.bfs_best_direction

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


SITE_DIST_MAPS = [bfs_distance_map(tuple(map(int, s))) for s in SITE_POSITIONS]
DEFENSE_POS_DIST_MAPS = [bfs_distance_map(pos) for pos in DEFENSE_POSITIONS]


# TOUYAMA_DEFENSE_ASSIGNMENT はマップ読み込み時点(上のブロック)で
# 5/6/7/8/9 とロースター順の対応から直接確定済みのため、ここでの
# 総当たり最適化は不要。確認用のログのみ出力する。
print("[touyama_v1] 固定 担当ポジション割り当て(マップ上の5/6/7/8/9をロースター順に直接対応):")
for i, _name in enumerate(TOUYAMA_ROSTER_ORDER):
    _pos = TOUYAMA_DEFENSE_ASSIGNMENT[_name]
    _spawn = DEFENDER_SPAWNS[i]
    _dist = DEFENSE_POS_DIST_MAPS[i][_spawn[0], _spawn[1]]
    print(
        f"  {_name}: spawn={_spawn} -> "
        f"pos={_pos}(value={TOUYAMA_DEFENSE_POSITION_VALUES[_name]}) dist={_dist}"
    )

# --- 診断用: マップ構造そのものに起因する偏りが無いか確認する ---
# 各キャラのスポーン地点から「最も近いdefense position」までの純粋なBFS距離
# (貪欲割当・シャッフル順のバイアスを除いた理論上の最短値)を出力する。
# もしこれ自体が上段/下段で大きく偏っていれば、マップ側(map_data_search.py の
# 7の配置)がそもそも不公平であることが確定する。
print("[touyama_v1][DIAG] DEFENSE_POSITIONS(ロースター順、値5-9) 座標一覧:", DEFENSE_POSITIONS)
print("[touyama_v1][DIAG] DEFENDER_SPAWNS 座標一覧:", DEFENDER_SPAWNS)
for i, name in enumerate(TOUYAMA_ROSTER_ORDER):
    spawn = DEFENDER_SPAWNS[i]
    nearest_dist = min(
        bfs_distance_map(spawn)[pos[0], pos[1]]
        for pos in DEFENSE_POSITIONS
    )
    all_dists = sorted(
        bfs_distance_map(spawn)[pos[0], pos[1]] for pos in DEFENSE_POSITIONS
    )
    print(
        f"[touyama_v1][DIAG] {name} spawn={spawn} "
        f"nearest_defense_dist={nearest_dist} "
        f"all_defense_dists_sorted={all_dists}"
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
        self.blind_remaining = 0
        self.reveal_remaining = 0
        self.moved_this_tick = False
        self.has_spike = has_spike
        self.kills = 0
        self.accuracy = DEFAULT_ACCURACY
        self.dodge_rate = DEFAULT_DODGE
        self.hs_rate = DEFAULT_HS_RATE
        self.reaction = DEFAULT_REACTION + random.uniform(-10, 10)

        # Defender専用: 割り当てられた待機ポジション(7)とそのBFS距離マップ。
        self.assigned_defense_pos = None
        self.assigned_defense_dist_map = None

        # Defender専用: 現在アクティブな優先モード("spike"/"sighting"/"position")
        # と、そのモードで前tickに観測したBFS距離。モード切替直後は基準値を
        # 揃えるためだけに使い、報酬は発生させない(_compute_rewards参照)。
        self.prev_priority_mode = None
        self.prev_priority_dist = None
        self.prev_priority_target_key = None


def _build_fixed_defenders():
    """touyama_v1固定チーム(5人)をDefenderとして生成する。

    ロースター順 = DEFENDER_SPAWNS(行優先走査順)の対応で固定し、
    run_game.py の実際のスポーン割り当てロジック(ロースターi番目 =
    area_4[i])と一致させる。ステータス・ロールは_compute_touyama_effective_stats
    で確定済みの値をそのまま使う(ランダム選択・ランダムjitterなし)。
    """
    defenders = []
    for i, name in enumerate(TOUYAMA_ROSTER_ORDER):
        stats = TOUYAMA_EFFECTIVE_STATS[name]
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
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.spike_pos = None
        self.spike_held = False
        self.last_seen_enemy = None

    def update(self, defenders, attackers, smoke_cells, spike_ground_pos=None):
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
        elif spike_ground_pos is not None and any(
            has_los(d.pos, spike_ground_pos, smoke_cells) for d in alive_defenders
        ):
            self.spike_pos = tuple(spike_ground_pos)
            self.spike_held = False

        if visible_enemies:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next((a for a in visible_enemies if a.name == tracked_name), None)
            if tracked is None:
                tracked = min(
                    visible_enemies,
                    key=lambda a: min(
                        max(abs(a.pos[0] - d.pos[0]), abs(a.pos[1] - d.pos[1]))
                        for d in alive_defenders
                    ) if alive_defenders else 0,
                )
            self.last_seen_enemy = {
                "pos": tuple(tracked.pos), "name": tracked.name, "tick_ago": 0
            }
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ============================================================================
# ネットワーク
# ============================================================================

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "next_mask", "done"))
# ネットワーク(旧DefenderSearchDuelingDQN)・ReplayBufferは common_rl に統合。

# ============================================================================
# 観測構築
# ============================================================================

def build_observation(
    unit, defenders, attackers, team_memory, smoke_cells, own_smoke_active, round_timer,
    spike_dist_map, sighting_dist_map, unit_has_spike_los,
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
    obs[30] = 0.0  # 予備次元

    in_position_mode = team_memory.spike_pos is None and team_memory.last_seen_enemy is None
    if in_position_mode and unit.assigned_defense_pos is not None:
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

    return obs


def decode_action(action_idx):
    move_idx, use_ability = divmod(int(action_idx), 2)
    return MOVES[move_idx], bool(use_ability)


def encode_action(move, use_ability):
    move_idx = MOVES.index(move)
    return move_idx * 2 + (1 if use_ability else 0)


def build_action_mask(unit, occupied, lock_movement=False, has_target_info=True):
    """lock_movement=True の場合、stay(move_idx=0)以外の移動を禁止する。
    交戦中(敵が視認できている間)は静止させ、射撃の当たりやすさを優先する。
    has_target_info=False(敵の視認情報も直近の目撃情報も一切無い)の場合、
    use_ability action自体を選択肢から除外する。敵情報が無い状態での使用は
    ability_requestsに追加されず常に無意味(空撃ち)なため、探索でこの選択肢に
    ランダムにチャージを浪費させないための措置(2026-08-12 合意)。"""
    mask = np.ones(ACTION_DIM, dtype=bool)
    r, c = int(unit.pos[0]), int(unit.pos[1])
    for move_idx, (dr, dc) in enumerate(MOVES):
        if lock_movement and move_idx != 0:
            mask[move_idx * 2] = False
            mask[move_idx * 2 + 1] = False
            continue
        nr, nc = r + dr, c + dc
        walkable = (
            0 <= nr < HEIGHT and 0 <= nc < WIDTH
            and GRID[nr, nc] != 1
            and (nr, nc) not in occupied
        )
        if not walkable:
            mask[move_idx * 2] = False
            mask[move_idx * 2 + 1] = False

    if unit.charges <= 0 or unit.role == "HUNT" or not has_target_info:
        for move_idx in range(5):
            mask[move_idx * 2 + 1] = False

    return mask


# ============================================================================
# 環境本体
# ============================================================================

class SearchEnv:
    """プラント前フェーズを模した簡易マルチエージェント環境。

    Defender = touyama_v1固定チーム(5人、固定ステータス・固定ロール・
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
        self.planted = False
        self.match_over_reason = None
        self._prev_kills = {}
        self._prev_alive = {}
        self.spike_dist_map = None
        self.sighting_dist_map = None
        self.spike_ground_pos = None

        # --- 一時デバッグ用: 特定キャラの毎tick実況トレース。
        # train()側で特定エピソードだけTrueにする。
        self.debug_trace = False
        self.debug_trace_name = "ろびぃな"

        # --- 診断用: positionモード中(spike/sighting情報が無いとき)の
        # キャラ別「平均BFS距離・到着率・移動率」を1エピソード分蓄積する。
        # train()側がエピソード終了ごとにこれを読み取り、履歴に積算する。
        
        self.position_mode_stats = {
            name: {"dist_sum": 0.0, "dist_count": 0, "moved_count": 0, "arrived_count": 0}
            for name in TOUYAMA_ROSTER_ORDER
        }

        # --- 診断用: アビリティ使用の内訳(視認あり使用/命中/外れ/whiff/overlap/debuffキル)を
        # 1エピソード分蓄積する。train()側がエピソード終了ごとにこれを読み取り、履歴に積算する。
        self.ability_diag_stats = {
            name: {"aimed": 0, "hit": 0, "miss": 0, "whiff": 0, "overlap": 0, "debuff_kill": 0}
            for name in TOUYAMA_ROSTER_ORDER
        }

    # -- 初期化 --------------------------------------------------------
    def reset(self):
        self.team_memory.reset()
        self.smokes = []
        self.round_timer = MAX_TICKS
        self.planted = False
        self.match_over_reason = None
        self.spike_ground_pos = None

        # --- 診断用: 新しいエピソードの開始時に集計をリセット ---
        
        self.position_mode_stats = {
            name: {"dist_sum": 0.0, "dist_count": 0, "moved_count": 0, "arrived_count": 0}
            for name in TOUYAMA_ROSTER_ORDER
        }
        self.ability_diag_stats = {
            name: {"aimed": 0, "hit": 0, "miss": 0, "whiff": 0, "overlap": 0, "debuff_kill": 0}
            for name in TOUYAMA_ROSTER_ORDER
        }

        self.defenders = _build_fixed_defenders()
        self.attackers = _build_attackers()

        self.carrier_target_site_idx = random.randrange(len(SITE_POSITIONS))

        self._assign_defense_positions()

        self.team_memory.update(self.defenders, self.attackers, self._smoke_cells(), self.spike_ground_pos)
        self._update_priority_dist_maps()
        self._prev_kills = {u.name: u.kills for u in self.defenders + self.attackers}
        self._prev_alive = {u.name: u.is_alive for u in self.defenders + self.attackers}

        return self._collect_observations()

    def _assign_defense_positions(self):
        """touyama_v1固定チームはスポーン位置が毎エピソード同一のため、
        ランダムシャッフル+早い者勝ち貪欲法ではなく、事前に一意計算した
        全組合せ最適解(TOUYAMA_DEFENSE_ASSIGNMENT)を毎回そのまま使う。
        これにより、複数キャラが同じ近場ポジションを取り合い、その結果を
        エピソードごとの運で分け合うという構造的な偏りを解消する。"""
        for d in self.defenders:
            pos = TOUYAMA_DEFENSE_ASSIGNMENT[d.name]
            d.assigned_defense_pos = pos
            d.assigned_defense_dist_map = DEFENSE_POS_DIST_MAPS[DEFENSE_POSITIONS.index(pos)]
            d.prev_priority_mode = None
            d.prev_priority_dist = None
            d.prev_priority_target_key = None

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
            )
            own_occupied = occupied - {tuple(d.pos)}
            has_enemy_los = any(
                a.is_alive and has_los(d.pos, a.pos, smoke_cells) for a in self.attackers
            )
            # スパイク(地面)に射線が通った時点で、それ以上通路を進ませず
            # その場で待ち伏せさせる(敵が拾いに来るところを迎撃する方が有利なため)。
            lock_movement = has_enemy_los or unit_has_spike_los
            # 敵の視認情報も直近の目撃情報も一切無い場合、use_abilityは常に無意味
            # (ability_requestsに追加されない空撃ち)になるため、探索での浪費を防ぐ
            # ためマスクの時点で選択肢から除外する。
            has_target_info = has_enemy_los or (self.team_memory.last_seen_enemy is not None)
            mask_dict[d.name] = build_action_mask(
                d, own_occupied, lock_movement=lock_movement, has_target_info=has_target_info
            )
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
        if carrier is not None and random.random() < 0.6:
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

    # -- メインステップ ---------------------------------------------------
    def step(self, action_dict):
        for u in self.defenders + self.attackers:
            u.moved_this_tick = False

        pre_tick_enemy_debuffed = {
            a.name: (a.blind_remaining > 0 or a.reveal_remaining > 0)
            for a in self.attackers if a.is_alive
        }
        pre_tick_flash_recon_active = any(
            a.is_alive and (a.blind_remaining > 0 or a.reveal_remaining > 0)
            for a in self.attackers
        )

        smoke_cells = self._smoke_cells()

        move_plans = []
        ability_requests = []
        ability_whiff = {}
        ability_overlap = {}
        held_angle = {}

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
        )

        actual_action_dict = {}

        for d in self.defenders:
            if not d.is_alive or d.name not in action_dict:
                continue
            (dr, dc), use_ability = decode_action(action_dict[d.name])

            if in_position_phase and d.assigned_defense_dist_map is not None:
                r0, c0 = int(d.pos[0]), int(d.pos[1])
                cur_dist = d.assigned_defense_dist_map[r0, c0]
                if cur_dist > REACH_RADIUS:
                    dr, dc = bfs_best_direction(d.assigned_defense_dist_map, r0, c0)
            elif self.team_memory.spike_pos is not None and self.spike_dist_map is not None:
                # spike確定 or 接近中(spike_watchは除く): 目標座標は既知なので、
                # 移動方向はBFS最短方向を強制し、ネットワークにはアビリティ使用の
                # 判断のみ委ねる。spike_watch(既にLOSが通っている)は対象外。
                r0, c0 = int(d.pos[0]), int(d.pos[1])
                already_watching = (
                    not self.team_memory.spike_held
                    and has_los(d.pos, self.team_memory.spike_pos, smoke_cells)
                )
                if not already_watching:
                    dr, dc = bfs_best_direction(self.spike_dist_map, r0, c0)
            elif self.team_memory.last_seen_enemy is not None and self.sighting_dist_map is not None:
                # sighting(目撃情報ベース追跡): 同様にBFS最短方向を強制する。
                r0, c0 = int(d.pos[0]), int(d.pos[1])
                dr, dc = bfs_best_direction(self.sighting_dist_map, r0, c0)

            actual_action_dict[d.name] = encode_action((dr, dc), use_ability)
            move_plans.append((d, (dr, dc)))

            visible_enemies = [
                a for a in self.attackers if a.is_alive and has_los(d.pos, a.pos, smoke_cells)
            ]
            has_enemy_los = bool(visible_enemies)
            # use_abilityのマスク許可条件(has_target_info)と、実際にability_requestsへ
            # 追加されるかどうかの条件は一致している必要がある。以前はwhiff判定に
            # has_enemy_los(直接視認のみ)を使っており、sightingモード(記憶ベース)での
            # 正しい使用まで誤ってwhiff扱いしていた。
            has_target_info = has_enemy_los or (self.team_memory.last_seen_enemy is not None)

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
                    if visible_enemies:
                        nearest = min(
                            visible_enemies,
                            key=lambda a: max(abs(a.pos[0]-d.pos[0]), abs(a.pos[1]-d.pos[1])),
                        )
                        dist = max(abs(nearest.pos[0]-d.pos[0]), abs(nearest.pos[1]-d.pos[1]))
                        if dist <= ABILITY_RANGE:
                            ability_requests.append((d, tuple(nearest.pos)))
                    elif self.team_memory.last_seen_enemy is not None:
                        ability_requests.append((d, self.team_memory.last_seen_enemy["pos"]))

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
                unit.pos = [nr, nc]
            unit.moved_this_tick = tuple(unit.pos) != old_pos

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
                for a in self.attackers:
                    if a.is_alive and has_los(target_pos, a.pos, smoke_cells):
                        a.blind_remaining = max(a.blind_remaining, BLIND_DURATION_TICKS)
                        hit_any = True
            elif unit.role == "RECON":
                for a in self.attackers:
                    if a.is_alive and has_los(target_pos, a.pos, smoke_cells):
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

        self.team_memory.update(self.defenders, self.attackers, self._smoke_cells(), self.spike_ground_pos)
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
            pre_tick_enemy_debuffed, ability_whiff, ability_overlap, held_angle, ability_hit
        )

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

        obs_dict, mask_dict = self._collect_observations()
        return obs_dict, mask_dict, rewards, done, actual_action_dict

    def _resolve_shots(self):
        alive = [u for u in self.defenders + self.attackers if u.is_alive]
        smoke_cells = self._smoke_cells()
        shot_intents = []

        for shooter in alive:
            targets = [
                t for t in alive
                if t.team != shooter.team and has_los(shooter.pos, t.pos, smoke_cells)
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
            if shooter.blind_remaining > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            debuffed = target.blind_remaining > 0 or target.reveal_remaining > 0
            effective_dodge = target.dodge_rate * (REVEALED_DODGE_MULTIPLIER if debuffed else 1.0)
            hit_chance = accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            if hit:
                headshot = random.random() < shooter.hs_rate
                damage = HEADSHOT_DAMAGE if headshot else BODY_DAMAGE
                target.hp = max(0, target.hp - damage)
                self.last_shots.append({"shooter": shooter, "target": target, "hit": True})
                if target.hp <= 0:
                    target.is_alive = False
                    shooter.kills += 1
                    if target.has_spike:
                        target.has_spike = False
            else:
                self.last_shots.append({"shooter": shooter, "target": target, "hit": False})

    def _priority_mode_and_distmap(self, defender):
        if self.team_memory.spike_pos is not None and self.spike_dist_map is not None:
            if self.team_memory.spike_held:
                return "spike", self.spike_dist_map, "spike"
            if has_los(defender.pos, self.team_memory.spike_pos, self._smoke_cells()):
                return "spike_watch", self.spike_dist_map, "spike_watch"
            return "spike_approach", self.spike_dist_map, "spike_approach"
        if self.team_memory.last_seen_enemy is not None and self.sighting_dist_map is not None:
            target_key = f"sighting:{self.team_memory.last_seen_enemy.get('name')}"
            return "sighting", self.sighting_dist_map, target_key
        return "position", defender.assigned_defense_dist_map, "position"

    def _compute_rewards(self, pre_tick_enemy_debuffed, ability_whiff, ability_overlap, held_angle, ability_hit=None):
        ability_hit = ability_hit or {}
        rewards = {}
        for d in self.defenders:
            r = STEP_PENALTY

            mode, dist_map, target_key = self._priority_mode_and_distmap(d)
            r0, c0 = int(d.pos[0]), int(d.pos[1])
            bfs_dist = dist_map[r0, c0] if dist_map is not None else None
            if bfs_dist is not None and bfs_dist < 0:
                bfs_dist = None

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

                if mode == "spike":
                    r += SPIKE_PULL_REWARD * delta
                elif mode == "spike_approach":
                    r += SPIKE_PULL_REWARD * delta
                elif mode == "spike_watch":
                    r += SPIKE_WATCH_HOLD_BONUS if not d.moved_this_tick else SPIKE_WATCH_MOVE_PENALTY
                elif mode == "sighting":
                    r += SIGHTING_PULL_REWARD * delta
                else:
                    if bfs_dist > REACH_RADIUS:
                        r += DEFENSE_POSITION_PULL_REWARD * delta
                    else:
                        r += HOLD_POSITION_BONUS if not d.moved_this_tick else HOLD_POSITION_PENALTY

                    # --- 診断用: positionモード時のみ、BFS距離・到着・移動を記録 ---
                    stats = self.position_mode_stats.get(d.name)
                    if stats is not None:
                        stats["dist_sum"] += float(bfs_dist)
                        stats["dist_count"] += 1
                        if d.moved_this_tick:
                            stats["moved_count"] += 1
                        if bfs_dist <= REACH_RADIUS:
                            stats["arrived_count"] += 1

            if d.name in ability_whiff:
                if ability_whiff[d.name]:
                    # 視認情報が無い状態での使用(既存のまま)
                    r += ABILITY_WHIFF_PENALTY
                else:
                    # 視認あり・射程内で使用(外れてもここまでは付与)
                    r += ABILITY_AIMED_REWARD
                    if ability_hit.get(d.name):
                        r += ABILITY_HIT_BONUS
            if ability_overlap.get(d.name):
                r += ABILITY_OVERLAP_PENALTY

            angle_state = held_angle.get(d.name)
            if angle_state == "held_with_los":
                r += HOLD_ANGLE_BONUS
            elif angle_state == "moved_with_los":
                r += HOLD_ANGLE_PENALTY

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


# select_action / optimize は common_rl.select_action /
# common_rl.optimize_double_dqn_step を直接使用(呼び出し側train()を参照)。

def train(
    episodes=EPISODE_COUNT,
    batch_size=128,
    gamma=0.99,
    lr=1e-4,
    buffer_size=200_000,
    target_update_every=1000,
):
    policy_net = DuelingQNet(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net = DuelingQNet(OBS_DIM, ACTION_DIM).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(Transition, capacity=buffer_size)
    env = SearchEnv()

    global_step = 0
    best_avg_reward = -float("inf")
    episode_reward_history = deque(maxlen=100)

    # --- 診断用(1): キャラ別(ロール別)の直近100エピソード報酬履歴 ---
    per_name_reward_history = {name: deque(maxlen=100) for name in TOUYAMA_ROSTER_ORDER}
    per_name_episode_total = {name: 0.0 for name in TOUYAMA_ROSTER_ORDER}

    # --- 診断用(4): positionモード中の「平均BFS距離・到着率・移動率」履歴 ---
    per_name_avg_dist_history = {name: deque(maxlen=100) for name in TOUYAMA_ROSTER_ORDER}
    per_name_arrival_rate_history = {name: deque(maxlen=100) for name in TOUYAMA_ROSTER_ORDER}
    per_name_move_rate_history = {name: deque(maxlen=100) for name in TOUYAMA_ROSTER_ORDER}

    # --- 診断用(5): アビリティ使用内訳(aimed/hit/miss/whiff/overlap/debuff_kill)の直近100エピソード履歴 ---
    ABILITY_DIAG_KEYS = ("aimed", "hit", "miss", "whiff", "overlap", "debuff_kill")
    per_name_ability_history = {
        name: {key: deque(maxlen=100) for key in ABILITY_DIAG_KEYS}
        for name in TOUYAMA_ROSTER_ORDER
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

        for tick in range(MAX_TICKS):

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
        for name in TOUYAMA_ROSTER_ORDER:
            stats = env.position_mode_stats.get(name, {})
            dist_count = stats.get("dist_count", 0)
            if dist_count > 0:
                avg_dist = stats["dist_sum"] / dist_count
                arrival_rate = stats["arrived_count"] / dist_count
                move_rate = stats["moved_count"] / dist_count
                per_name_avg_dist_history[name].append(avg_dist)
                per_name_arrival_rate_history[name].append(arrival_rate)
                per_name_move_rate_history[name].append(move_rate)

        episode_reward_history.append(episode_reward_total)
        avg_reward = sum(episode_reward_history) / len(episode_reward_history)

        # --- 診断用(5): このエピソードのアビリティ使用内訳を履歴に積算 ---
        for name in TOUYAMA_ROSTER_ORDER:
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
                f"{name}({TOUYAMA_EFFECTIVE_STATS[name]['ability']})="
                f"{(sum(per_name_reward_history[name]) / len(per_name_reward_history[name])):.3f}"
                for name in TOUYAMA_ROSTER_ORDER
                if len(per_name_reward_history[name]) > 0
            )
            print(f"  [PER-CHAR avg100] {per_name_str}")

            # --- 診断用(4): positionモード中の平均BFS距離・到着率・移動率 ---
            position_diag_str = " / ".join(
                f"{name}(dist={sum(per_name_avg_dist_history[name]) / len(per_name_avg_dist_history[name]):.2f},"
                f"arrive={sum(per_name_arrival_rate_history[name]) / len(per_name_arrival_rate_history[name]) * 100:.1f}%,"
                f"move={sum(per_name_move_rate_history[name]) / len(per_name_move_rate_history[name]) * 100:.1f}%)"
                for name in TOUYAMA_ROSTER_ORDER
                if len(per_name_avg_dist_history[name]) > 0
            )
            
            print(f"  [POSITION-MODE diag] {position_diag_str}")

            # --- 診断用(5): 直近<=100エピソード合計でのアビリティ使用内訳 ---
            ability_diag_str = " / ".join(
                f"{name}(aimed={sum(per_name_ability_history[name]['aimed'])},"
                f"hit={sum(per_name_ability_history[name]['hit'])},"
                f"miss={sum(per_name_ability_history[name]['miss'])},"
                f"whiff={sum(per_name_ability_history[name]['whiff'])},"
                f"overlap={sum(per_name_ability_history[name]['overlap'])},"
                f"dbuff_kill={sum(per_name_ability_history[name]['debuff_kill'])})"
                for name in TOUYAMA_ROSTER_ORDER
            )
            print(f"  [ABILITY diag, sum over last<=100 eps] {ability_diag_str}")

        if avg_reward > best_avg_reward and len(episode_reward_history) >= 50:
            best_avg_reward = avg_reward
            torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
            print(f"[SAVE] best model updated: avg100={avg_reward:.3f} -> {MODEL_SAVE_PATH}")

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), MODEL_LATEST_PATH)

    print("[DONE] training finished.")


if __name__ == "__main__":
    train()