"""touyama_v1/train_attacker_carry.py

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

保存先: touyama_v1/data/attacker_carry_touyama_data/
チェックポイントは {"model_state_dict","obs_dim","n_actions","episode",
"success_rate","priority_cells","has_priority_cells","waypoint_cells"} を
含むdict形式で保存する。
"""

import os
import sys
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data_carry import NEW_MAZE_STR
from character_stats_touyama import CHARACTER_TABLE as TOUYAMA_STATS_TABLE

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
from character_stats_touyama import (
    CHARACTER_TABLE as TOUYAMA_STATS_TABLE,
    TOUYAMA_ROSTER_ORDER,
)

EPISODE_COUNT = 3000

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = "data/attacker_carry_touyama_data/"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_attacker_carry_touyama_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_attacker_carry_touyama_latest.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 観測21次元。行動空間は「アビリティ不使用/使用」「明示PLANT」の3値のみ
# (移動はDQNの行動空間に含めない。BFS経路探索で決定的に処理する)。
OBS_DIM = 21
ACTION_DIM = 3
ACTION_NONE = 0
ACTION_ABILITY = 1
ACTION_PLANT = 2

# サイト別中継地点: 右サイト=6, 左サイト=7。
WAYPOINT_VALUE_BY_SITE = {"right": 6, "left": 7}

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

ABILITY_RANGE = 8
SIGHTING_STALENESS_CAP = 20
HANDOFF_AUGMENT_PROB = 0.25  # 一定確率でスポーン以外(拾得後の合流)からスタート

# 敵(Defender)側の既定ステータス(当面ヒューリスティックのため簡易値のまま)
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0

# ---------------------------------------------------------------------------
# touyama_v1 固定チーム定義
# ---------------------------------------------------------------------------
TOUYAMA_ROSTER_ORDER = ["Tortlilyan", "いぐるん", "ろびぃな", "夢の街", "えんぺん"]
TOUYAMA_SPIKE_HOLDER = "ろびぃな"  # 通常ラウンド開始時の既定キャリア

TOUYAMA_COMBO_MEMBERS = {"ろびぃな", "えんぺん", "いぐるん"}
TOUYAMA_COMBO_BONUS = {
    "accuracy": 0.15,
    "hs_rate": 0.10,
    "dodge_rate": 0.20,
    "reaction": 30.0,
}

TOUYAMA_ROLE_TO_ABILITY = {
    "フラッシュ": "FLASH",
    "スモーカー": "SMOKE",
    "シーカー": "RECON",
    "タイガー": "HUNT",
}
TIGER_ACCURACY_BONUS = 0.10
TIGER_HS_BONUS = 0.05


def _compute_touyama_effective_stats():
    effective = {}
    for name in TOUYAMA_ROSTER_ORDER:
        raw = TOUYAMA_STATS_TABLE[name]
        accuracy = float(raw.hit_pct)
        hs_rate = float(raw.hs_pct)
        dodge_rate = float(raw.dodge_pct)
        reaction = float(raw.reaction)

        if raw.role == "タイガー":
            accuracy += TIGER_ACCURACY_BONUS
            hs_rate += TIGER_HS_BONUS

        if name in TOUYAMA_COMBO_MEMBERS:
            accuracy += TOUYAMA_COMBO_BONUS["accuracy"]
            hs_rate += TOUYAMA_COMBO_BONUS["hs_rate"]
            dodge_rate += TOUYAMA_COMBO_BONUS["dodge_rate"]
            reaction += TOUYAMA_COMBO_BONUS["reaction"]

        effective[name] = {
            "accuracy": max(0.0, accuracy),
            "hs_rate": max(0.0, min(1.0, hs_rate)),
            "dodge_rate": max(0.0, min(1.0, dodge_rate)),
            "reaction": max(0.0, reaction),
            "ability": TOUYAMA_ROLE_TO_ABILITY[raw.role],
        }
    return effective


TOUYAMA_EFFECTIVE_STATS = _compute_touyama_effective_stats()

print("[touyama_v1] 固定チーム(Attacker/carry・簡略版) 確定ステータス:")
for _name in TOUYAMA_ROSTER_ORDER:
    _s = TOUYAMA_EFFECTIVE_STATS[_name]
    print(
        f"  {_name}: acc={_s['accuracy']:.2f} hs={_s['hs_rate']:.2f} "
        f"dodge={_s['dodge_rate']:.2f} reaction={_s['reaction']:.0f} "
        f"ability={_s['ability']}"
    )


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


# ============================================================================
# マップ読み込み
# ============================================================================

def _parse_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = _parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
ATTACKER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 3]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) in SITE_VALUES]
# 優先設置場所(5)。target_plant_posは必ずここから選ぶ(構造的にサイト内に1つも
# 無い場合のみ、後述のPLANT_CELLS_BY_SITEへ構造的フォールバックする)。
PRIORITY_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) == 5]

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

def line_cells(p1, p2):
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


def has_los(p1, p2, smoke_cells=None):
    cells = line_cells(p1, p2)
    for r, c in cells:
        if GRID[r, c] == 1:
            return False
    if smoke_cells and len(cells) > 2:
        if any(cell in smoke_cells for cell in cells):
            return False
    return True


def bfs_distance_map(goal):
    """単一始点からの距離マップ(観測の距離特徴量用)。"""
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if GRID[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def bfs_next_step(start, goal, occupied, allow_adjacent_goal=True):
    """経路探索BFS(占有マス回避)。キャリアの移動・エスコートの追従の両方で使う。
    controllers.BaseController.move_towards_target と同等のロジックを複製。"""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return start

    candidate_goals = []
    if GRID[goal[0], goal[1]] != 1 and goal not in occupied:
        candidate_goals.append(goal)
    if allow_adjacent_goal or goal in occupied:
        for dr, dc in CARDINAL:
            adj = (goal[0] + dr, goal[1] + dc)
            if (
                0 <= adj[0] < HEIGHT and 0 <= adj[1] < WIDTH
                and GRID[adj[0], adj[1]] != 1 and adj not in occupied
            ):
                candidate_goals.append(adj)
    candidate_goals = list(dict.fromkeys(candidate_goals))
    if not candidate_goals:
        return _random_step(start, occupied)

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
            if not (0 <= nxt[0] < HEIGHT and 0 <= nxt[1] < WIDTH):
                continue
            if GRID[nxt[0], nxt[1]] == 1 or nxt in occupied:
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if reached is None:
        return _random_step(start, occupied)

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return start
    return step


def _random_step(pos, occupied):
    r, c = pos
    valid = [
        (r + dr, c + dc) for dr, dc in CARDINAL
        if 0 <= r + dr < HEIGHT and 0 <= c + dc < WIDTH
        and GRID[r + dr, c + dc] != 1 and (r + dr, c + dc) not in occupied
    ]
    return random.choice(valid) if valid else pos


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


def _resolve_spawn_collision(pos, occupied):
    if pos not in occupied and GRID[pos[0], pos[1]] != 1:
        return pos
    visited = {pos}
    queue = deque([pos])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
                continue
            if (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            if GRID[nr, nc] == 1:
                continue
            if (nr, nc) not in occupied:
                return (nr, nc)
            queue.append((nr, nc))
    return pos


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


def _build_defenders():
    d_spawns = random.sample(DEFENDER_SPAWNS, min(N_DEFENDERS, len(DEFENDER_SPAWNS)))
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


Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "next_mask", "done"))


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 観測構築(簡略版: エスコート関連を削除。移動方向ガイダンスも不要)
# ============================================================================

def build_observation(
    carrier, defenders, smoke_cells, own_smoke_active, elapsed_ticks, dist_map,
    reached_waypoint, on_target,
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

    return obs


def decode_action(action_idx):
    idx = int(action_idx)
    if idx == ACTION_PLANT:
        return "PLANT"
    if idx == ACTION_ABILITY:
        return "ABILITY"
    return "NONE"


def build_action_mask(unit, on_target):
    mask = np.ones(ACTION_DIM, dtype=bool)
    if unit.charges <= 0 or unit.ability_name in ("HUNT", "NONE"):
        mask[ACTION_ABILITY] = False
    mask[ACTION_PLANT] = bool(on_target)
    return mask


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

    def reset(self):
        self.sighting.reset()
        self.smokes = []
        self.elapsed_ticks = 0
        self.plant_progress = 0
        self.match_over_reason = None

        handoff = random.random() < HANDOFF_AUGMENT_PROB
        if handoff:
            carrier_name = random.choice(TOUYAMA_ROSTER_ORDER)
        else:
            carrier_name = TOUYAMA_SPIKE_HOLDER

        self.attackers = _build_fixed_attackers(carrier_name, handoff=handoff)
        self.defenders = _build_defenders()
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

    def _collect_observation(self):
        smoke_cells = self._smoke_cells()
        on_target = self.carrier.is_alive and self._plantable_cell(self.carrier.pos)
        obs = build_observation(
            self.carrier, self.defenders, smoke_cells, self._own_smoke_active(),
            self.elapsed_ticks, self.dist_map, self.reached_waypoint, on_target,
        )
        mask = build_action_mask(self.carrier, on_target)
        return obs, mask

    def step(self, action_idx):
        self.elapsed_ticks += 1
        for u in self.attackers + self.defenders:
            u.moved_this_tick = False

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
        if carrier_alive:
            decoded = decode_action(action_idx)
            visible_enemies = [
                d for d in self.defenders if d.is_alive and has_los(self.carrier.pos, d.pos, smoke_cells)
            ]

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

                if decoded == "ABILITY":
                    ability_whiff = not visible_enemies
                    ability_overlap = pre_flash_recon_active and self.carrier.ability_name in ("FLASH", "RECON")
                    if self.carrier.charges > 0:
                        if visible_enemies:
                            nearest = min(
                                visible_enemies,
                                key=lambda d: max(abs(d.pos[0] - self.carrier.pos[0]), abs(d.pos[1] - self.carrier.pos[1])),
                            )
                            dist = max(abs(nearest.pos[0] - self.carrier.pos[0]), abs(nearest.pos[1] - self.carrier.pos[1]))
                            if dist <= ABILITY_RANGE:
                                _apply_ability(
                                    self.carrier, self.carrier.ability_name, tuple(nearest.pos),
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
            plant_completed_on_priority,
        )

        all_units = self.attackers + self.defenders
        self._prev_kills = {u.name: u.kills for u in all_units}
        self._prev_alive = {u.name: u.is_alive for u in all_units}
        self._prev_hp = {u.name: u.hp for u in all_units}

        obs, mask = self._collect_observation() if self.carrier.is_alive else (
            np.zeros(OBS_DIM, dtype=np.float32), np.array([True, False, False], dtype=bool)
        )
        return obs, mask, reward, done

    def _resolve_shots(self):
        alive = [u for u in self.attackers + self.defenders if u.is_alive]
        smoke_cells = self._smoke_cells()
        shot_intents = []

        for shooter in alive:
            targets = [t for t in alive if t.team != shooter.team and has_los(shooter.pos, t.pos, smoke_cells)]
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
            if shooter.blind_remaining > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            debuffed = target.blind_remaining > 0 or target.reveal_remaining > 0
            effective_dodge = target.dodge_rate * (REVEALED_DODGE_MULTIPLIER if debuffed else 1.0)
            hit_chance = accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

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
        plant_completed_on_priority=False,
    ):
        reward = STEP_PENALTY + waypoint_bonus

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


def select_action(policy_net, obs, mask, epsilon):
    if random.random() < epsilon:
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return 0
        return int(np.random.choice(valid_indices))
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = policy_net(obs_t).squeeze(0).cpu().numpy()
        q_values = np.where(mask, q_values, -np.inf)
        return int(np.argmax(q_values))


def optimize(policy_net, target_net, optimizer, buffer, batch_size, gamma):
    if len(buffer) < batch_size:
        return None

    batch = buffer.sample(batch_size)
    obs_batch = torch.as_tensor(np.array(batch.obs), dtype=torch.float32, device=DEVICE)
    action_batch = torch.as_tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    next_obs_batch = torch.as_tensor(np.array(batch.next_obs), dtype=torch.float32, device=DEVICE)
    next_mask_batch = torch.as_tensor(np.array(batch.next_mask), dtype=torch.bool, device=DEVICE)
    done_batch = torch.as_tensor(batch.done, dtype=torch.float32, device=DEVICE)

    q_values = policy_net(obs_batch).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_q_policy = policy_net(next_obs_batch)
        next_q_policy = next_q_policy.masked_fill(~next_mask_batch, -float("inf"))
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_obs_batch).gather(1, next_actions).squeeze(1)
        target = reward_batch + gamma * next_q_target * (1.0 - done_batch)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


EVAL_EVERY = 20          # 何episode毎にgreedy評価を行うか
EVAL_EPISODES = 50       # 1回の評価で回すgreedy episode数(20は分散が大きすぎるため増量)
EVAL_MIN_EPISODE = 100   # これ未満のepisodeでは評価・保存を行わない(初期の不安定な重みを保存しないため)
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

    # 学習用のグローバル乱数状態を汚さないよう退避してから固定シードに切り替える
    rng_state = random.getstate()
    random.seed(seed)
    try:
        with torch.no_grad():
            for _ in range(episodes):
                obs, mask = env.reset()
                episode_reward = 0.0
                for _tick in range(MAX_TICKS):
                    action = select_action(policy_net, obs, mask, epsilon=0.0)
                    obs, mask, reward, done = env.step(action)
                    episode_reward += reward
                    if done:
                        break
                reward_total += episode_reward
                if env.match_over_reason in ("planted", "planted_compromise"):
                    success_count += 1
    finally:
        random.setstate(rng_state)

    policy_net.train()
    avg_reward = reward_total / episodes
    success_rate = success_count / episodes
    return avg_reward, success_rate


def train(
    episodes=EPISODE_COUNT,
    batch_size=128,
    gamma=0.99,
    lr=1e-4,
    buffer_size=200_000,
    target_update_every=1000,
):
    policy_net = AttackerCarryDuelingDQN().to(DEVICE)
    target_net = AttackerCarryDuelingDQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size)
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
            },
            path,
        )

    start_time = time.perf_counter()

    for episode in range(1, episodes + 1):
        obs, mask = env.reset()
        episode_reward_total = 0.0
        epsilon = epsilon_by_episode(episode)

        for tick in range(MAX_TICKS):
            action = select_action(policy_net, obs, mask, epsilon)
            next_obs, next_mask, reward, done = env.step(action)
            episode_reward_total += reward

            buffer.push(obs, action, reward, next_obs, next_mask, float(done))

            obs, mask = next_obs, next_mask
            global_step += 1

            optimize(policy_net, target_net, optimizer, buffer, batch_size, gamma)

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
            eval_reward, eval_success_rate = evaluate(policy_net)
            eval_reward_recent.append(eval_reward)
            eval_reward_smoothed = sum(eval_reward_recent) / len(eval_reward_recent)
            print(
                f"[EVAL @ EP {episode}] eval_reward={eval_reward:.3f} "
                f"eval_success={eval_success_rate:.3f} smoothed={eval_reward_smoothed:.3f} "
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