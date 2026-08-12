"""gc_v1/train_attacker_carry.py

固定チーム(Xdll/Syouta/Absol/eKo/SugarZ3ro)専用の
Attacker「carry phase」学習スクリプト。
スパイクをスポーンからプラント可能地点まで運び、自動発火するプラントを
完了させることを目的とする。学習対象はスパイク保持者(キャリア)のみ。

train_attacker_guard.py / train_defender_search.py と同一規約:
  - run_game.py / controllers.py / battle_logic.py / abilities_los.py は
    一切importしない。必要なロジックはすべてこのファイル内に複製する。
  - map_data.py / character_stats_gc.py / game_core.py は定数専用
    ファイルとして参照する(import制限の対象外)。
  - run_game.py / controllers.py は変更しない。

このフェーズで学習しないもの(=ヒューリスティックで代替):
  - エスコート4人 (DefaultAttackerControllerの護衛ロジックを複製)
  - 敵(Defender)5人 (DefaultDefenderControllerのロジックを複製)
  これらは将来、専用モデルに差し替え可能な設計(_build_escort_action /
  _build_defender_action だけを変更すればよい)。

プラントは明示的な行動(PLANT, action_idx=PLANT_ACTION_INDEX)として選択する仕様
(実ゲームのbattle_logic.py PLANT分岐と同一): プラント可能マス(2または5)に
立っている間だけPLANTが選択可能になり、選び続けている間だけ進捗が進む
(PLANT_REQUIRED_TICKS到達で成功終了)。PLANT以外の行動を選んだ瞬間、進捗は0に戻る。

ナビゲーション目標(target_plant_pos)はエピソード開始時に1点だけ選んで固定する
(実ゲームのinit_round()と同じ考え方)。毎tick最寄り地点を再計算しないため、
移動中に目標がすり替わってジグザグ移動を誘発することがない。

マップは map_data_carry_gc.py を使用する。同ファイルは map_data.py 上のプラント
可能マス(2)の一部に、学習専用の優先(代表)地点マーカー(5)を追加したもの。
5は本番map_data.pyには存在しないため、優先地点は学習済みチェックポイントに
座標(priority_cells)として保存し、本番実行時はgridの値に依存せずその座標を使う
(learning_attacker_carry.py側で対応予定)。

保存先: gc_v1/data/attacker_carry_gc_data/
チェックポイントは {"model_state_dict","obs_dim","n_actions","episode",
"success_rate","priority_cells","has_priority_cells"} を含むdict形式で保存する。
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

from map_data_carry_gc import NEW_MAZE_STR
from character_stats_gc import CHARACTER_TABLE as GC_STATS_TABLE

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
from character_stats_gc import (
    CHARACTER_TABLE as GC_STATS_TABLE,
    GC_ROSTER_ORDER,
)

EPISODE_COUNT = 8000

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = "data/attacker_carry_gc_data/"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_attacker_carry_gc_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_attacker_carry_gc_latest.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 29
ACTION_DIM = 11  # move_idx(0-4)*2 + use_ability_flag(0/1) の10種類 + 明示PLANT(index=10)
PLANT_ACTION_INDEX = 10

# サイト別ウェイポイント: 右サイト=6, 左サイト=7。
# 同じ値を複数マス配置してよい。対応サイトへ向かう際は、
# その候補群の「どれか1つ」を通ればウェイポイント通過とみなす。
WAYPOINT_VALUE_BY_SITE = {"right": 6, "left": 7}

# map_data_carry_gc.py上で、通常のプラント可能マス(2)と優先(代表)地点マーカー(5)の
# 両方をプラント可能マスとして扱う。5は本番map_data.py上では2として存在するマスに
# 学習用マーカーを上書きしたもの(map_data_carry_gc.py側の仕様)。
SITE_VALUES = frozenset({2, 5})

N_ATTACKERS = 5  # gc固定チーム(このフェーズではAttacker)
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
# gc_v1 固定チーム定義(train_defender_search.py / train_attacker_guard.py と同一)
# ---------------------------------------------------------------------------
# GC_ROSTER_ORDER は character_stats_gc.py から読み込む
GC_SPIKE_HOLDER = "Absol"  # GC既定キャリア


GC_ROLE_TO_ABILITY = {
    "フラッシュ": "FLASH",
    "スモーカー": "SMOKE",
    "シーカー": "RECON",
    "タイガー": "HUNT",
}
TIGER_ACCURACY_BONUS = 0.10
TIGER_HS_BONUS = 0.05


GC_COMBO_NAME = "幽霊部員de廃部待ったなし"
GC_COMBO_MEMBERS = set(GC_ROSTER_ORDER)
GC_PLAYER_BONUSES = {
    "Xdll": {"dodge_rate": 0.4, "mental": 3},
    "Syouta": {"reaction": 40, "mental": 3},
    "Absol": {"dodge_rate": 0.4, "mental": 3},
    "eKo": {"hs_rate": 0.4, "mental": 3},
    "SugarZ3ro": {"iq": 40, "mental": 3},
}

def _compute_gc_effective_stats():
    """character_stats_gc.py の生値に、常時発動するチームコンボ
    (ふわんだりぃず)とタイガーパッシブを適用した確定値を返す。
    他のgc_v1学習ファイルと同一ロジック。"""
    effective = {}
    for name in GC_ROSTER_ORDER:
        raw = GC_STATS_TABLE[name]
        accuracy = float(raw.hit_pct)
        hs_rate = float(raw.hs_pct)
        dodge_rate = float(raw.dodge_pct)
        reaction = float(raw.reaction)

        if raw.role == "タイガー":
            accuracy += TIGER_ACCURACY_BONUS
            hs_rate += TIGER_HS_BONUS

        if name in GC_COMBO_MEMBERS:
            accuracy += GC_PLAYER_BONUSES.get(name, {}).get("accuracy", 0.0)
            hs_rate += GC_PLAYER_BONUSES.get(name, {}).get("hs_rate", 0.0)
            dodge_rate += GC_PLAYER_BONUSES.get(name, {}).get("dodge_rate", 0.0)
            reaction += GC_PLAYER_BONUSES.get(name, {}).get("reaction", 0.0)

        effective[name] = {
            "accuracy": max(0.0, accuracy),
            "hs_rate": max(0.0, min(1.0, hs_rate)),
            "dodge_rate": max(0.0, min(1.0, dodge_rate)),
            "reaction": max(0.0, reaction),
            "ability": GC_ROLE_TO_ABILITY[raw.role],
        }
    return effective


GC_EFFECTIVE_STATS = _compute_gc_effective_stats()

print("[gc_v1] 固定チーム(Attacker/carry) 確定ステータス:")
for _name in GC_ROSTER_ORDER:
    _s = GC_EFFECTIVE_STATS[_name]
    print(
        f"  {_name}: acc={_s['accuracy']:.2f} hs={_s['hs_rate']:.2f} "
        f"dodge={_s['dodge_rate']:.2f} reaction={_s['reaction']:.0f} "
        f"ability={_s['ability']}"
    )


# 報酬パラメータ
STEP_PENALTY = -0.001
PROGRESS_REWARD = 0.03          # 目標プラント地点への接近(ポテンシャル差分)
DAMAGE_TAKEN_PENALTY_SCALE = -0.002  # 被弾ダメージ1あたり
KILL_REWARD = 0.4
DEATH_PENALTY = -1.0            # キャリア死亡=スパイクドロップ(retrieveフェーズへ引き継ぎ)
ABILITY_WHIFF_PENALTY = -0.05
ABILITY_OVERLAP_PENALTY = -0.05
PLANT_WHIFF_PENALTY = -0.05     # サイト外でPLANTを選んだ場合(マスクが機能していれば理論上到達しない保険)
PLANT_TICK_BONUS = 0.15         # PLANT進行1Tickごと。迅速な設置を強く評価
PLANT_SUCCESS_REWARD = 2.5      # プラント完了
PLANTABLE_WAIT_PENALTY = -0.08  # 設置可能なのにPLANTしないTick
PLANTABLE_WAIT_GROWTH = -0.025  # 待つほど追加で悪化（2Tick目以降）
MAX_PLANTABLE_WAIT_PENALTY = -0.30
TIME_EXPIRE_PENALTY = -0.5      # ラウンド時間切れ(未設置=Defender有利)
TEAM_WIPE_PENALTY = -0.2        # 味方(エスコート込み)全滅だがキャリアは生存継続中
WAYPOINT_REACHED_REWARD = 0.10  # サイト別ウェイポイント(6=右/7=左)を初通過した時の一度きりのボーナス


# ============================================================================
# マップ読み込み(map_data.NEW_MAZE_STRのみ参照。パース処理は自前で複製)
# ============================================================================

def _parse_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = _parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
ATTACKER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 3]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
# プラント可能マスは 2(通常) と 5(map_data_carry_gc.pyが付与した優先/代表マーカー)の両方
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) in SITE_VALUES]
# 優先(代表)地点: 本番map_data.pyには存在しない、map_data_carry_gc.py専用のマーカー(5)。
# 学習済みチェックポイントには座標として保存し、本番実行時はgridの値に依存せず
# その座標リストをそのまま使う(learning_attacker_carry.py側で対応予定)。
PRIORITY_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if int(GRID[r, c]) == 5]

# サイト別ウェイポイント(右=6/左=7)。
# 同じ値は複数配置可能。対応サイトでは候補群のどれか1つを通れば達成とする。
# 未配置のサイトはWARNのみ出し、target_plant_posへ直接誘導する。
WAYPOINT_CELLS = {}
for _site, _value in WAYPOINT_VALUE_BY_SITE.items():
    _cells = [
        (r, c)
        for r in range(HEIGHT)
        for c in range(WIDTH)
        if int(GRID[r, c]) == _value
    ]
    if _cells:
        WAYPOINT_CELLS[_site] = list(_cells)
        print(
            f"[MAP] site={_site} waypoint value={_value}: "
            f"{len(_cells)} cells -> {_cells}"
        )
    else:
        print(
            f"[WARN] map_data_carry_gc.py にサイト別ウェイポイント"
            f"(値={_value}, site={_site})が見つかりません。"
            f"このサイトは経路強制なしの従来挙動になります。"
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
    raise RuntimeError("map_data_carry_gc.py にプラント可能マス(2 または 5)が見つかりません。")
if not PRIORITY_CELLS:
    print("[WARN] map_data_carry_gc.py に優先地点マーカー(5)が見つかりません。優先地点ガイダンス特徴量は常に0になります。")


# ============================================================================
# LOS・BFS(abilities_los.py / controllers.py と同等のロジックを複製)
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



def bfs_distance_map_multi(sources):
    """複数始点からのマルチソースBFS距離マップ。
    6/7の複数ウェイポイント候補、およびPRIORITY_CELLSへの誘導に使う。"""
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    queue = deque()
    for gr, gc in sources:
        if GRID[gr, gc] == 1:
            continue
        if dist[gr, gc] == -1:
            dist[gr, gc] = 0
            queue.append((gr, gc))
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def bfs_best_direction(dist_map, r0, c0):
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


# 各PLANT_CELLへの個別BFS距離マップ(target_plant_pos固定用)。
# エピソード開始時にどれか1つをtarget_plant_posとして選び、以降このエピソード中は
# 固定して使う(実ゲームのinit_round()と同じ考え方。毎tick再計算による目標の
# すり替わり・ジグザグ移動を防止する)。
PLANT_DIST_MAPS = {cell: bfs_distance_map(cell) for cell in PLANT_CELLS}

# サイト別ウェイポイントへのBFS距離マップ(WAYPOINT_CELLSに存在するサイトのみ)。
WAYPOINT_DIST_MAPS = {
    site: bfs_distance_map_multi(cells)
    for site, cells in WAYPOINT_CELLS.items()
}




# 優先(代表)地点への誘導特徴量用。PRIORITY_CELLSが空の場合は全マス-1(=未到達)になる。
PRIORITY_DIST_MAP = (
    bfs_distance_map_multi(PRIORITY_CELLS) if PRIORITY_CELLS else np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
)
_PRIORITY_FINITE = PRIORITY_DIST_MAP[PRIORITY_DIST_MAP >= 0]
PRIORITY_MAX_DIST = int(_PRIORITY_FINITE.max()) if _PRIORITY_FINITE.size else (HEIGHT + WIDTH)


def _bfs_next_step(start, goal, occupied, allow_adjacent_goal=True):
    """controllers.BaseController.move_towards_target と同等のロジックを複製。"""
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


def _shortest_path_distance(pos, target):
    start = (int(pos[0]), int(pos[1]))
    goal = (int(target[0]), int(target[1]))
    if start == goal:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        for dr, dc in CARDINAL:
            nxt = (r + dr, c + dc)
            if nxt in visited or not (0 <= nxt[0] < HEIGHT and 0 <= nxt[1] < WIDTH):
                continue
            if GRID[nxt[0], nxt[1]] == 1:
                continue
            if nxt == goal:
                return d + 1
            visited.add(nxt)
            queue.append((nxt, d + 1))
    return float("inf")


# ============================================================================
# ユニットスタブ(game_core.Characterの必要最小限の複製。継承・importはしない)
# ============================================================================

class UnitStub:
    def __init__(self, name, team, pos, role, ability, accuracy, dodge_rate, hs_rate, reaction, has_spike=False):
        self.name = name
        self.team = team  # "A"(gc/carry+escort) or "D"(敵)
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
    """gc_v1固定チーム(5人)をAttackerとして生成する。
    通常はATTACKER_SPAWNS(ロースター順=area_3走査順)に配置し、carrier_nameが
    スパイクを持つ。handoff=Trueの場合、拾得後の合流を模した位置バラつきを与える。"""
    attackers = []
    occupied = set()

    if handoff:
        base_positions = random.sample(WALKABLE, N_ATTACKERS)
    else:
        base_positions = list(ATTACKER_SPAWNS[:N_ATTACKERS])

    for i, name in enumerate(GC_ROSTER_ORDER):
        stats = GC_EFFECTIVE_STATS[name]
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
    """敵(Defender)側は当面ヒューリスティック対応のため、DEFENDER_SPAWNSから
    配置する。プラント前提のため常時ランダム索敵移動+近接時アビリティ。"""
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
# controllers.py の Default*Controller と同等ロジックをこのファイル内へ複製
# ============================================================================

def _heuristic_ability_action(unit, visible_enemies):
    """DefaultAttackerController._decide_ability / DefaultDefenderController._decide_ability
    を簡略統合したもの。近接複数視認でSMOKE、近接単体でFLASH、索敵目的でRECON。"""
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
        # 索敵先はプラント候補の中心方向へ適当に投げる(簡易)
        target = random.choice(PLANT_CELLS)
        return ("RECON", target)

    return None


def _apply_ability(unit, ability_name, target_pos, smokes, defenders_or_attackers, smoke_cells):
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
        for other in defenders_or_attackers:
            if other.is_alive and other.team != unit.team and has_los(target_pos, other.pos, smoke_cells):
                other.blind_remaining = max(other.blind_remaining, BLIND_DURATION_TICKS)
    elif ability_name == "RECON":
        for other in defenders_or_attackers:
            if other.is_alive and other.team != unit.team and has_los(target_pos, other.pos, smoke_cells):
                other.reveal_remaining = max(other.reveal_remaining, REVEAL_DURATION_TICKS)


def _escort_move(unit, carrier, all_units, occupied, smoke_cells):
    """DefaultAttackerControllerの護衛ロジック(ケース4)を複製。
    30%でランダム移動、carrierから離れすぎたら追従、それ以外は待機気味。"""
    if carrier is None or not carrier.is_alive:
        return _random_step(tuple(unit.pos), occupied)

    dist = max(abs(carrier.pos[0] - unit.pos[0]), abs(carrier.pos[1] - unit.pos[1]))
    if random.random() < 0.3:
        return _random_step(tuple(unit.pos), occupied)
    if dist > 5:
        return _bfs_next_step(tuple(unit.pos), tuple(carrier.pos), occupied, allow_adjacent_goal=True)
    return _random_step(tuple(unit.pos), occupied)


def _defender_move(unit, occupied):
    """DefaultDefenderController(is_planted=False時)と同等: ランダム移動。"""
    return _random_step(tuple(unit.pos), occupied)


# ============================================================================
# 索敵メモリ(キャリア視点。敵の目撃情報)
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
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, action_dim))

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
# 観測構築(キャリア視点のみ。エスコート・敵はヒューリスティックのため不要)
# ============================================================================

def build_observation(
    carrier, attackers, defenders, sighting, smoke_cells, own_smoke_active,
    elapsed_ticks, dist_map, reached_waypoint=True,
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
    best_dr, best_dc = bfs_best_direction(dist_map, r0, c0)
    obs[10] = float(best_dr)
    obs[11] = float(best_dc)
    obs[12] = 1.0 if int(GRID[r0, c0]) in SITE_VALUES else 0.0

    visible_enemies = [d for d in defenders if d.is_alive and has_los(carrier.pos, d.pos, smoke_cells)]
    obs[13] = 1.0 if visible_enemies else 0.0
    obs[14] = len(visible_enemies) / 5.0
    if visible_enemies:
        nearest = min(
            visible_enemies,
            key=lambda d: max(abs(d.pos[0] - r0), abs(d.pos[1] - c0)),
        )
        obs[15] = (nearest.pos[0] - r0) / HEIGHT
        obs[16] = (nearest.pos[1] - c0) / WIDTH
        dist = max(abs(nearest.pos[0] - r0), abs(nearest.pos[1] - c0))
        obs[17] = min(dist, HEIGHT) / HEIGHT
    obs[18] = 1.0 if any(d.is_alive and (d.blind_remaining > 0 or d.reveal_remaining > 0) for d in defenders) else 0.0
    obs[19] = 1.0 if own_smoke_active else 0.0

    teammates = [a for a in attackers if a is not carrier and a.is_alive]
    obs[20] = len(teammates) / 4.0
    if teammates:
        nearest_d = min(max(abs(t.pos[0] - r0), abs(t.pos[1] - c0)) for t in teammates)
        obs[21] = min(nearest_d, HEIGHT) / HEIGHT
    obs[22] = 1.0  # is_carrier (このエージェント視点では常に1)
    obs[23] = min(elapsed_ticks, MAX_TICKS) / MAX_TICKS
    obs[24] = sum(1 for d in defenders if d.is_alive) / 5.0

    # --- 優先(代表)地点への誘導特徴量。map_data_carry_gc.pyの5マーカー(PRIORITY_CELLS)
    # からのマルチソースBFS(PRIORITY_DIST_MAP)を参照する。target_plant_pos(dist_map)
    # とは独立した、常時提供される追加ガイダンス。---
    p_dist = PRIORITY_DIST_MAP[r0, c0]
    if p_dist < 0:
        p_dist = PRIORITY_MAX_DIST
    obs[25] = min(p_dist, PRIORITY_MAX_DIST) / max(1, PRIORITY_MAX_DIST)
    p_best_dr, p_best_dc = bfs_best_direction(PRIORITY_DIST_MAP, r0, c0)
    obs[26] = float(p_best_dr)
    obs[27] = float(p_best_dc)

    # サイト別ウェイポイント(6/7)を通過済みかどうか。dist_mapがウェイポイント向けか
    # target_plant_pos向けかを区別するためのフラグ(未配置サイトは常に1.0)。
    obs[28] = 1.0 if reached_waypoint else 0.0

    return obs


def decode_action(action_idx):
    """PLANT_ACTION_INDEXの場合は文字列"PLANT"を返す。それ以外は
    (move_delta, use_ability)のタプル。"""
    if int(action_idx) == PLANT_ACTION_INDEX:
        return "PLANT"
    move_idx, use_ability = divmod(int(action_idx), 2)
    return MOVES[move_idx], bool(use_ability)


def build_action_mask(unit, occupied, on_site):
    """on_site: このユニットが現在SITE_VALUESのマスに立っているか。
    立っている場合のみ明示PLANT(index=PLANT_ACTION_INDEX)を選択可能にする。"""
    mask = np.ones(ACTION_DIM, dtype=bool)
    r, c = int(unit.pos[0]), int(unit.pos[1])
    for move_idx, (dr, dc) in enumerate(MOVES):
        nr, nc = r + dr, c + dc
        walkable = (
            0 <= nr < HEIGHT and 0 <= nc < WIDTH
            and GRID[nr, nc] != 1
            and (nr, nc) not in occupied
        )
        if not walkable:
            mask[move_idx * 2] = False
            mask[move_idx * 2 + 1] = False

    if unit.charges <= 0 or unit.ability_name in ("HUNT", "NONE"):
        for move_idx in range(5):
            mask[move_idx * 2 + 1] = False

    mask[PLANT_ACTION_INDEX] = bool(on_site)

    return mask


# ============================================================================
# 環境本体
# ============================================================================

class CarryEnv:
    """carryフェーズを模した簡易環境。学習対象はキャリア1体のみ。
    エスコート4人・敵5人はヒューリスティックで動く。
    プラントはgrid==2マス滞在中に自動進行する(実ゲーム仕様と同一)。"""

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
        self.active_waypoint_site = None
        self.reached_waypoint = True
        # 設置可能マス上でPLANTせず待った連続Tick数
        self.plantable_wait_ticks = 0

    def reset(self):
        self.sighting.reset()
        self.smokes = []
        self.elapsed_ticks = 0
        self.plant_progress = 0
        self.plantable_wait_ticks = 0
        self.match_over_reason = None
        # 意図的に_prev_distを初期化しない。_compute_reward側のgetattr(self,"_prev_dist",cur_dist)
        # が「属性が存在しない場合のみ」cur_distへフォールバックする仕様を利用し、
        # 最初のtickではprev_dist==cur_distとなってdelta=0(報酬なし)になるようにする。
        # ここでNoneを代入すると、getattrが「属性は存在する(値がNone)」と判定してしまい、
        # フォールバックが効かずPROGRESS_REWARDが永久に発火しなくなる。

        handoff = random.random() < HANDOFF_AUGMENT_PROB
        if handoff:
            carrier_name = random.choice(GC_ROSTER_ORDER)
        else:
            carrier_name = GC_SPIKE_HOLDER

        self.attackers = _build_fixed_attackers(carrier_name, handoff=handoff)
        self.defenders = _build_defenders()
        self.carrier = next(a for a in self.attackers if a.name == carrier_name)

        # 実ゲームのinit_round()と同じ考え方: ラウンド開始時に1点だけ選んで固定する。
        # 以降このエピソード中は目標がすり替わらない(target_plant_posに対応)。
        self.target_plant_pos = random.choice(PLANT_CELLS)

        # サイト(左/右)を判定し、対応するウェイポイントが存在すれば
        # まずそこへの距離マップを使う(通過後にstep()側でplant距離マップへ切り替え)。
        # ウェイポイント未配置のサイトは従来通りtarget_plant_posへ直接誘導する。
        self.active_waypoint_site = (
            "left" if self.target_plant_pos[1] < WIDTH // 2 else "right"
        )
        if self.active_waypoint_site in WAYPOINT_DIST_MAPS:
            self.reached_waypoint = False
            self.dist_map = WAYPOINT_DIST_MAPS[self.active_waypoint_site]
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

    def _collect_observation(self):
        smoke_cells = self._smoke_cells()
        obs = build_observation(
            self.carrier, self.attackers, self.defenders, self.sighting,
            smoke_cells, self._own_smoke_active(), self.elapsed_ticks, self.dist_map,
            self.reached_waypoint,
        )
        occupied = {tuple(u.pos) for u in self.attackers + self.defenders if u.is_alive} - {tuple(self.carrier.pos)}
        on_site = (
            self.carrier.is_alive
            and int(GRID[int(self.carrier.pos[0]), int(self.carrier.pos[1])]) in SITE_VALUES
        )
        mask = build_action_mask(self.carrier, occupied, on_site)
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
        on_site_before_action = (
            carrier_alive
            and int(GRID[int(self.carrier.pos[0]), int(self.carrier.pos[1])]) in SITE_VALUES
        )

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

        # --- エスコート4人: ヒューリスティック ---
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
            nxt = _escort_move(a, self.carrier, self.attackers + self.defenders, own_occupied, smoke_cells)
            move_plans.append((a, nxt))

        # --- キャリア: DQNの行動を反映(明示PLANT対応) ---
        plant_action_chosen = False
        if carrier_alive:
            decoded = decode_action(action_idx)
            visible_enemies = [
                d for d in self.defenders if d.is_alive and has_los(self.carrier.pos, d.pos, smoke_cells)
            ]

            if decoded == "PLANT":
                # 行動マスク上、on_site_before_action==Trueの時のみ選択され得る。
                # 移動・アビリティ使用は行わない(実ゲームのbattle_logic.py PLANT分岐と同一仕様)。
                move_plans.append((self.carrier, tuple(self.carrier.pos)))
                plant_action_chosen = True
            else:
                (dr, dc), use_ability = decoded
                nr = self.carrier.pos[0] + dr
                nc = self.carrier.pos[1] + dc
                move_plans.append((self.carrier, (nr, nc)))

                if use_ability:
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

        # --- 移動の適用 ---
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

        # --- 明示PLANT進捗(実ゲームのbattle_logic.py PLANT分岐と同一仕様):
        # PLANTを選び続けている間だけ進み、それ以外の行動を選んだ瞬間0に戻る。---
        plant_tick_progress = False
        plant_completed = False
        if plant_action_chosen and on_site_before_action and self.carrier.is_alive:
            self.plant_progress += 1
            plant_tick_progress = True
            if self.plant_progress >= PLANT_REQUIRED_TICKS:
                plant_completed = True
        else:
            self.plant_progress = 0

        self.sighting.update(self.carrier, self.defenders, self._smoke_cells())

        # ウェイポイント通過判定(未通過の場合のみ)。通過したらdist_mapを
        # target_plant_pos向けのBFSへ切り替える。_prev_distは切り替え直後の
        # 距離で上書きし、マップ切り替えによる急激な差分報酬(スパイク)を防ぐ。
        waypoint_bonus = 0.0
        if self.carrier.is_alive and not self.reached_waypoint:
            wr, wc = int(self.carrier.pos[0]), int(self.carrier.pos[1])
            waypoint_cells = WAYPOINT_CELLS[self.active_waypoint_site]
            waypoint_dist = min(
                max(abs(wr - cell[0]), abs(wc - cell[1]))
                for cell in waypoint_cells
            )
            if waypoint_dist <= 1:
                self.reached_waypoint = True
                waypoint_bonus = WAYPOINT_REACHED_REWARD
                self.dist_map = PLANT_DIST_MAPS[self.target_plant_pos]
                self._prev_dist = self.dist_map[wr, wc]
        # dist_mapはウェイポイント通過時にのみ切り替わる。それ以外は固定のため、
        # ここでの再計算は行わない。

        reward, done = self._compute_reward(
            ability_whiff, ability_overlap, plant_tick_progress, plant_completed,
            plant_action_chosen, on_site_before_action, waypoint_bonus,
        )

        all_units = self.attackers + self.defenders
        self._prev_kills = {u.name: u.kills for u in all_units}
        self._prev_alive = {u.name: u.is_alive for u in all_units}
        self._prev_hp = {u.name: u.hp for u in all_units}

        obs, mask = self._collect_observation() if self.carrier.is_alive else (
            np.zeros(OBS_DIM, dtype=np.float32), np.ones(ACTION_DIM, dtype=bool)
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
        plant_action_chosen, on_site_before_action, waypoint_bonus=0.0,
    ):
        reward = STEP_PENALTY + waypoint_bonus

        was_alive = self._prev_alive.get(self.carrier.name, True)

        # キャリア死亡: このフェーズの終了条件(retrieveフェーズへ引き継ぎ)
        if was_alive and not self.carrier.is_alive:
            reward += DEATH_PENALTY
            self.match_over_reason = "carrier_died"
            return reward, True

        if not self.carrier.is_alive:
            # 既に死亡している状態が続くことは想定していない(即終了するため)
            return reward, True

        # 被ダメージペナルティ
        hp_lost = max(0, self._prev_hp.get(self.carrier.name, self.carrier.hp) - self.carrier.hp)
        reward += DAMAGE_TAKEN_PENALTY_SCALE * hp_lost

        # 目標地点への接近報酬(ポテンシャル差分)。プラント進行中は距離0扱いなので自然に0。
        r0, c0 = int(self.carrier.pos[0]), int(self.carrier.pos[1])
        if self.dist_map is not None:
            cur_dist = self.dist_map[r0, c0]
            prev_dist = getattr(self, "_prev_dist", cur_dist)
            if cur_dist is not None and prev_dist is not None:
                if cur_dist >= 0 and prev_dist >= 0:
                    reward += PROGRESS_REWARD * (prev_dist - cur_dist)
                self._prev_dist = cur_dist

        if ability_whiff:
            reward += ABILITY_WHIFF_PENALTY
        if ability_overlap:
            reward += ABILITY_OVERLAP_PENALTY

        new_kills = self.carrier.kills - self._prev_kills.get(self.carrier.name, self.carrier.kills)
        if new_kills > 0:
            reward += KILL_REWARD * new_kills

        if plant_action_chosen and not on_site_before_action:
            # マスクが正しく機能していれば理論上到達しないが、保険としてペナルティを設ける
            reward += PLANT_WHIFF_PENALTY

        # 設置可能マスに立っているのにPLANTしない行動を明確に罰する。
        # これにより「サイトで時間を使ってから最後に設置」のQ値局所解を抑える。
        if on_site_before_action and not plant_action_chosen:
            self.plantable_wait_ticks += 1
            wait_penalty = (
                PLANTABLE_WAIT_PENALTY
                + PLANTABLE_WAIT_GROWTH * max(0, self.plantable_wait_ticks - 1)
            )
            reward += max(MAX_PLANTABLE_WAIT_PENALTY, wait_penalty)
        elif plant_action_chosen:
            self.plantable_wait_ticks = 0
        else:
            self.plantable_wait_ticks = 0

        if plant_tick_progress:
            reward += PLANT_TICK_BONUS

        if plant_completed:
            reward += PLANT_SUCCESS_REWARD
            self.match_over_reason = "planted"
            return reward, True

        if self.elapsed_ticks >= MAX_TICKS:
            reward += TIME_EXPIRE_PENALTY
            self.match_over_reason = "time_expired"
            return reward, True

        if not any(a.is_alive for a in self.attackers if a is not self.carrier):
            # エスコート全滅だがキャリアは生存継続中: 一度だけ軽いペナルティを与え継続
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
    best_avg_reward = -float("inf")
    episode_reward_history = deque(maxlen=100)
    episode_success_history = deque(maxlen=100)  # match_over_reason=="planted"の割合

    def _save_checkpoint(path, episode_no, success_rate_value):
        """learning_attacker_carry.py(推論側)が期待するdict形式で保存する。
        obs_dim/n_actionsで観測・行動空間の不一致を早期検知できるようにし、
        priority_cellsは本番map_data.pyに5マーカーが存在しないため座標のまま保存する。"""
        torch.save(
            {
                "model_state_dict": policy_net.state_dict(),
                "obs_dim": OBS_DIM,
                "n_actions": ACTION_DIM,
                "episode": episode_no,
                "success_rate": success_rate_value,
                "priority_cells": list(PRIORITY_CELLS),
                "has_priority_cells": bool(PRIORITY_CELLS),
                "waypoint_cells": {
                    site: [tuple(cell) for cell in cells]
                    for site, cells in WAYPOINT_CELLS.items()
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
        episode_success_history.append(1.0 if env.match_over_reason == "planted" else 0.0)
        avg_reward = sum(episode_reward_history) / len(episode_reward_history)
        success_rate = sum(episode_success_history) / len(episode_success_history)

        if episode % 20 == 0:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter();
            print(
                f"[EP {episode}/{episodes}] reward={episode_reward_total:.3f} elapse={elapsed_time:.1f} "
                f"avg100={avg_reward:.3f} success100={success_rate:.3f} "
                f"epsilon={epsilon_by_episode(episode):.3f} "
                f"buffer={len(buffer)} reason={env.match_over_reason}"
            )

        if avg_reward > best_avg_reward and len(episode_reward_history) >= 50:
            best_avg_reward = avg_reward
            _save_checkpoint(MODEL_SAVE_PATH, episode, success_rate)
            print(f"[SAVE] best model updated: avg100={avg_reward:.3f} success100={success_rate:.3f} -> {MODEL_SAVE_PATH}")

        if episode % 100 == 0:
            _save_checkpoint(MODEL_LATEST_PATH, episode, success_rate)

    print("[DONE] training finished.")


if __name__ == "__main__":
    train()