"""touyama_v1/train_attacker_guard.py

固定チーム(いぐるん/夢の街/ろびぃな/Tortlilyan/えんぺん)専用の
Attacker「guard phase」学習スクリプト(プラント後、解除阻止に特化)。

train_defender_search.py(touyama_v1)をベースに、以下を置き換えた:
  - このフェーズでは touyama_v1 固定チームは Attacker側(A)。
    既にスパイクは設置済みという前提から開始する(プラント行動は扱わない)。
  - 敵(Defender)側は当面ヒューリスティック(DEFENDER_SPAWNSから
    プラント地点へBFS接近し、隣接したら解除する)。将来固定敵チームを
    学習する場合は _build_defenders() だけを差し替えれば良い設計。
  - 待機地点(guard position)は map_data_search.py のような専用マーカー
    ファイルが存在しないため、プラント地点からLOSが通る候補マスを
    走査し、farthest-point samplingで分散選出する方式をこのファイル内で
    完結させている(モジュール読み込み時にサイトごと1回だけ事前計算)。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py などのfeatureモジュールは一切importしない。
character_stats_touyama.py / game_core.py / map_data.py は定数専用
ファイルとして参照する(import制限の対象外)。

学習データ・チェックポイントは touyama_v1/data/attacker_guard_touyama_data/
以下に保存する。

--------------------------------------------------------------------------
優先順位ツリー(_compute_rewards / _priority_mode_and_distmap):
    1. 解除進行中(defuse_alert)         -- 最優先。プラント地点へ強く詰め寄る。
    2. 敵目撃情報(sighting)             -- 次点。視認した敵の方向へ寄る。
    3. どちらも無い場合                 -- 担当ガードポジション(LOSが通る
       分散地点)へ向かい、到着後は静止して警戒する。
--------------------------------------------------------------------------
"""

import os
import sys
import random
from collections import deque, namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR
from map_data_guard import NEW_MAZE_STR as GUARD_MAZE_STR
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
    SPIKE_DETONATION_TICKS,
    DEFUSE_REQUIRED_TICKS,
)
from character_stats_touyama import (
    CHARACTER_TABLE as TOUYAMA_STATS_TABLE,
    TOUYAMA_ROSTER_ORDER,
)

EPISODE_COUNT = 8000

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = "data/attacker_guard_touyama_data/"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_attacker_guard_touyama_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_attacker_guard_touyama_latest.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 34
ACTION_DIM = 10  # move_idx(0-4) * 2 + use_ability_flag(0/1)
ROLES = ["FLASH", "SMOKE", "RECON", "HUNT"]  # 参考用(touyama側ロールはステータス表から決定)

N_ATTACKERS = 5  # touyama固定チーム(このフェーズではAttacker)
N_DEFENDERS = 5  # 敵(ヒューリスティック)
MAX_TICKS = SPIKE_DETONATION_TICKS  # 55: プラント後の起爆までの時間と一致させる

ABILITY_RANGE = 8            # FLASH/RECONを即時適用してよい最大距離(簡易化)
GUARD_POS_REACH_RADIUS = 1   # 担当ガードポジションへ「到着した」とみなすBFS距離
SIGHTING_STALENESS_CAP = 20

# 敵(Defender)側の既定ステータス(当面ヒューリスティックのため簡易値のまま)
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0

# ---------------------------------------------------------------------------
# touyama_v1 固定チーム定義(train_defender_search.pyと同一)
# ---------------------------------------------------------------------------
TOUYAMA_ROSTER_ORDER = ["Tortlilyan", "いぐるん", "ろびぃな", "夢の街", "えんぺん"]
TOUYAMA_SPIKE_HOLDER = "ろびぃな"  # このフェーズでは既にプラント済みのため不使用(参照用)

TOUYAMA_COMBO_NAME = "ふわんだりぃず"
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
    """character_stats_touyama.py の生値に、常時発動するチームコンボ
    (ふわんだりぃず)とタイガーパッシブを適用した確定値を返す。
    train_defender_search.pyと同一ロジック(A/D問わず同じ5人・同じ補正)。
    """
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

print("[touyama_v1] 固定チーム(Attacker/guard) 確定ステータス:")
for _name in TOUYAMA_ROSTER_ORDER:
    _s = TOUYAMA_EFFECTIVE_STATS[_name]
    print(
        f"  {_name}: acc={_s['accuracy']:.2f} hs={_s['hs_rate']:.2f} "
        f"dodge={_s['dodge_rate']:.2f} reaction={_s['reaction']:.0f} "
        f"ability={_s['ability']}"
    )


# 報酬パラメータ
# 優先度: defuse_alert > sighting > position の順で明確に重みを引き離す。
STEP_PENALTY = -0.001
DEFUSE_PROGRESS_PENALTY = -0.05      # 敵の解除が1Tick進むごとのペナルティ(全員で共有)
GUARD_POSITION_PULL_REWARD = 0.03    # 担当ガードポジション/解除地点へ近づく(ポテンシャル差分)
HOLD_POSITION_BONUS = 0.04           # 到着後の静止をより強く優遇(射撃は静止側が有利なため)
HOLD_POSITION_PENALTY = -0.02        # 到着後の無駄な動き回りを強めに抑制(ただし移動自体は禁止しない)
ABILITY_WHIFF_PENALTY = -0.05
ABILITY_OVERLAP_PENALTY = -0.05
HOLD_ANGLE_BONUS = 0.02
HOLD_ANGLE_PENALTY = -0.01
SPIKE_WATCH_BONUS = 0.02     # プラント地点にLOSが通っている間、静止して警戒(引き上げ)
KILL_REWARD = 0.5
DEFUSER_KILL_BONUS = 0.4     # 解除中だった敵を倒した場合の追加ボーナス
DEATH_PENALTY = -0.5
ROUND_WIN_REWARD = 1.0       # 起爆 or 敵全滅によるAttacker勝利
DEFUSE_LOSS_PENALTY = -1.0   # 解除完了によるDefender勝利
WIPE_LOSS_PENALTY = -0.5     # 自チーム全滅(解除は時間の問題)による実質敗北


# ============================================================================
# マップ読み込み(map_data.NEW_MAZE_STRのみ参照。パース処理は自前で複製)
# ============================================================================

def _parse_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = _parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 2]

if len(DEFENDER_SPAWNS) < N_DEFENDERS:
    raise RuntimeError(
        f"DEFENDER_SPAWNSが{len(DEFENDER_SPAWNS)}マスしかなく、敵{N_DEFENDERS}人分を配置できません。"
    )


def _cluster_plant_cells(cells, max_sites=2, cluster_radius=6):
    """プラント可能セル群を、単純な距離クラスタリングでサイトごとにまとめる。
    (train_defender_search.py の _extract_site_positions と同じ考え方だが、
    こちらは各クラスタの実セル一覧も保持する)。"""
    if not cells:
        return []
    clusters = []
    for cell in cells:
        placed = False
        for cluster in clusters:
            cr, cc = cluster["centroid"]
            if max(abs(cell[0] - cr), abs(cell[1] - cc)) <= cluster_radius:
                cluster["cells"].append(cell)
                rs = [c[0] for c in cluster["cells"]]
                cs = [c[1] for c in cluster["cells"]]
                cluster["centroid"] = (sum(rs) / len(rs), sum(cs) / len(cs))
                placed = True
                break
        if not placed:
            clusters.append({"cells": [cell], "centroid": (float(cell[0]), float(cell[1]))})
    clusters.sort(key=lambda c: -len(c["cells"]))
    return clusters[:max_sites]


SITE_CLUSTERS = _cluster_plant_cells(PLANT_CELLS, max_sites=2)
if not SITE_CLUSTERS:
    print("[WARN] map_data.py にプラント可能マス(2)が見つかりません。マップ中央で代用します。")
    SITE_CLUSTERS = [{"cells": [(HEIGHT // 2, WIDTH // 2)], "centroid": (HEIGHT / 2.0, WIDTH / 2.0)}]


def _pick_canonical_plant_cell(cluster):
    """クラスタの重心に最も近い実セルを、そのサイトの代表プラント地点とする。"""
    cr, cc = cluster["centroid"]
    return min(cluster["cells"], key=lambda cell: (cell[0] - cr) ** 2 + (cell[1] - cc) ** 2)


SITE_PLANT_POS = [_pick_canonical_plant_cell(cluster) for cluster in SITE_CLUSTERS]


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


SITE_SPIKE_DIST_MAPS = [bfs_distance_map(pos) for pos in SITE_PLANT_POS]


def _bfs_next_step(start, goal, occupied, allow_adjacent_goal=True):
    """敵(Defender)ヒューリスティック用。controllers.BaseController.move_towards_target
    と同等のロジックをこのファイル内で複製したもの。"""
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


# ============================================================================
# ガードポジション生成(map_data_searchのような専用マーカーが無いため、
# プラント地点からLOSが通る候補マスをこのファイル内で走査して決定する)
# ============================================================================

# ============================================================================
# ガードポジション(map_data_guard.py の 6 マーカーをそのまま使用する。
# NEW_MAZE_STRと同一サイズであることを前提に、プラント地点(SITE_PLANT_POS)
# ごとに距離が近い順でN_ATTACKERS個を割り当てる)
# ============================================================================

_GUARD_GRID = _parse_grid(GUARD_MAZE_STR)
if _GUARD_GRID.shape != GRID.shape:
    raise RuntimeError(
        f"map_data_guard.pyのマップサイズ{_GUARD_GRID.shape}が"
        f"map_data.pyの{GRID.shape}と一致しません。"
    )

GUARD_POSITION_CELLS = [
    (r, c) for r in range(HEIGHT) for c in range(WIDTH) if _GUARD_GRID[r, c] == 6
]
if len(GUARD_POSITION_CELLS) < N_ATTACKERS:
    raise RuntimeError(
        f"map_data_guard.pyの6(ガードポジション)が{len(GUARD_POSITION_CELLS)}個しかなく、"
        f"固定チーム{N_ATTACKERS}人分を配置できません。"
    )


def _assign_guard_positions_for_site(plant_pos, num_positions=N_ATTACKERS):
    """プラント地点にChebyshev距離が近い順でガードポジション(6)をnum_positions個選ぶ。"""
    ordered = sorted(
        GUARD_POSITION_CELLS,
        key=lambda cell: max(abs(cell[0] - plant_pos[0]), abs(cell[1] - plant_pos[1])),
    )
    return ordered[:num_positions]


SITE_GUARD_POSITIONS = [
    _assign_guard_positions_for_site(pos, num_positions=N_ATTACKERS) for pos in SITE_PLANT_POS
]
SITE_GUARD_DIST_MAPS = [
    [bfs_distance_map(gp) for gp in guard_positions]
    for guard_positions in SITE_GUARD_POSITIONS
]

print("[touyama_v1] サイト別プラント地点・ガードポジション:")
for _idx, _plant in enumerate(SITE_PLANT_POS):
    print(f"  site{_idx}: plant={_plant} guard_positions={SITE_GUARD_POSITIONS[_idx]}")


# ============================================================================
# ユニットスタブ(game_core.Characterの必要最小限の複製。継承・importはしない)
# ============================================================================

class UnitStub:
    def __init__(self, name, team, pos, role, has_spike=False):
        self.name = name
        self.team = team  # "A"(touyama/guard) or "D"(敵/解除側)
        self.pos = list(pos)
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.is_alive = True
        self.role = role
        self.ability_name = role
        self.charges = 0 if role in ("HUNT", "NONE") else 1
        self.blind_remaining = 0
        self.reveal_remaining = 0
        self.moved_this_tick = False
        self.has_spike = has_spike
        self.kills = 0
        self.accuracy = DEFAULT_ACCURACY
        self.dodge_rate = DEFAULT_DODGE
        self.hs_rate = DEFAULT_HS_RATE
        self.reaction = DEFAULT_REACTION + random.uniform(-10, 10)

        # Defender(敵)専用: 解除進捗Tick数
        self.defuse_timer = 0

        # Attacker(touyama/guard)専用: 割り当てられたガードポジションとBFS距離マップ
        self.assigned_guard_pos = None
        self.assigned_guard_dist_map = None

        # 優先モード管理(報酬のポテンシャル差分計算用)
        self.prev_priority_mode = None
        self.prev_priority_dist = None
        self.prev_priority_target_key = None


def _resolve_spawn_collision(pos, occupied):
    """ガードポジション候補が重複/壁だった場合に、BFSで最寄りの空きマスへ逃がす。"""
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
    return pos  # フォールバック(通常到達しない)


def _build_fixed_attackers(site_idx, guard_positions):
    """touyama_v1固定チーム(5人)をAttacker(guard側)として生成する。
    ロースター順にガードポジションを1人ずつ割り当てる(重複時は隣接空きマスへ逃がす)。
    ステータス・ロールは_compute_touyama_effective_statsで確定済みの値をそのまま使う。"""
    attackers = []
    occupied = set()
    for i, name in enumerate(TOUYAMA_ROSTER_ORDER):
        stats = TOUYAMA_EFFECTIVE_STATS[name]
        base_pos = guard_positions[i % len(guard_positions)]
        spawn_pos = _resolve_spawn_collision(base_pos, occupied)
        occupied.add(spawn_pos)

        unit = UnitStub(name, "A", spawn_pos, stats["ability"])
        unit.accuracy = stats["accuracy"]
        unit.dodge_rate = stats["dodge_rate"]
        unit.hs_rate = stats["hs_rate"]
        unit.reaction = stats["reaction"]
        unit.assigned_guard_pos = base_pos
        unit.assigned_guard_dist_map = SITE_GUARD_DIST_MAPS[site_idx][i % len(guard_positions)]
        attackers.append(unit)
    return attackers


def _build_defenders():
    """敵(Defender)側は当面ヒューリスティック対応のため、DEFENDER_SPAWNSから
    ランダムスポーンさせる。アビリティは使用しない(role="NONE")。
    将来、敵専用の固定チームに差し替える場合はこの関数だけを変更すればよい。"""
    d_spawns = random.sample(DEFENDER_SPAWNS, min(N_DEFENDERS, len(DEFENDER_SPAWNS)))
    return [UnitStub(f"D{i+1}", "D", pos, "NONE") for i, pos in enumerate(d_spawns)]


# ============================================================================
# チーム共有メモリ(touyama/Attacker視点。敵目撃情報のみ管理)
# ============================================================================

class GuardMemory:
    def __init__(self):
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.last_seen_enemy = None

    def update(self, attackers, defenders, smoke_cells):
        alive_attackers = [a for a in attackers if a.is_alive]
        visible_enemies = []
        for a in alive_attackers:
            for d in defenders:
                if not d.is_alive:
                    continue
                if has_los(a.pos, d.pos, smoke_cells) and d not in visible_enemies:
                    visible_enemies.append(d)

        if visible_enemies:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next((d for d in visible_enemies if d.name == tracked_name), None)
            if tracked is None:
                # 解除中の敵がいれば最優先、いなければ最も近い敵を追跡する
                defusing = [d for d in visible_enemies if d.defuse_timer > 0]
                pool = defusing if defusing else visible_enemies
                tracked = min(
                    pool,
                    key=lambda d: min(
                        max(abs(d.pos[0] - a.pos[0]), abs(d.pos[1] - a.pos[1]))
                        for a in alive_attackers
                    ) if alive_attackers else 0,
                )
            self.last_seen_enemy = {"pos": tuple(tracked.pos), "name": tracked.name, "tick_ago": 0}
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ============================================================================
# ネットワーク
# ============================================================================

class AttackerGuardDuelingDQN(nn.Module):
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
# 観測構築
# ============================================================================

def build_observation(
    unit, attackers, defenders, guard_memory, smoke_cells, own_smoke_active,
    detonate_timer, spike_dist_map, sighting_dist_map, unit_has_spike_los,
    active_defuse_info,
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
        d for d in defenders if d.is_alive and has_los(unit.pos, d.pos, smoke_cells)
    ]
    obs[9] = 1.0 if visible_enemies else 0.0

    teammates = [a for a in attackers if a is not unit and a.is_alive]
    obs[10] = len(teammates) / 4.0
    if teammates:
        nearest_d = min(
            max(abs(t.pos[0] - unit.pos[0]), abs(t.pos[1] - unit.pos[1])) for t in teammates
        )
        obs[11] = min(nearest_d, HEIGHT) / HEIGHT

    obs[12] = 1.0 if any(
        d.is_alive and (d.blind_remaining > 0 or d.reveal_remaining > 0) for d in defenders
    ) else 0.0
    obs[13] = 1.0 if own_smoke_active else 0.0

    # プラント地点(常時既知)への距離・方向・LOS
    dist_here = spike_dist_map[r0, c0]
    obs[14] = min(dist_here if dist_here >= 0 else HEIGHT + WIDTH, HEIGHT + WIDTH) / (HEIGHT + WIDTH)
    best_dr, best_dc = bfs_best_direction(spike_dist_map, r0, c0)
    obs[15] = float(best_dr)
    obs[16] = float(best_dc)
    obs[17] = 1.0 if unit_has_spike_los else 0.0

    if guard_memory.last_seen_enemy is not None:
        ls = guard_memory.last_seen_enemy
        obs[18] = 1.0
        best_dr, best_dc = bfs_best_direction(sighting_dist_map, r0, c0)
        obs[19] = float(best_dr)
        obs[20] = float(best_dc)
        obs[21] = min(ls["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP

    obs[22] = len(visible_enemies) / 5.0
    if visible_enemies:
        nearest_enemy = min(
            visible_enemies,
            key=lambda d: max(abs(d.pos[0] - unit.pos[0]), abs(d.pos[1] - unit.pos[1])),
        )
        obs[23] = (nearest_enemy.pos[0] - unit.pos[0]) / HEIGHT
        obs[24] = (nearest_enemy.pos[1] - unit.pos[1]) / WIDTH
        dist = max(abs(nearest_enemy.pos[0] - unit.pos[0]), abs(nearest_enemy.pos[1] - unit.pos[1]))
        obs[25] = min(dist, HEIGHT) / HEIGHT

    # 現在進行中の解除情報(LOS不要、常時把握できる仕様。battle_logic.pyの
    # defender_defuse_infoに準拠)
    if active_defuse_info is not None:
        obs[26] = 1.0
        obs[27] = active_defuse_info["progress_ratio"]
    obs[28] = min(detonate_timer, MAX_TICKS) / MAX_TICKS

    # 担当ガードポジションへの距離・方向
    dist_map = unit.assigned_guard_dist_map
    bfs_dist = dist_map[r0, c0] if dist_map is not None else -1
    if bfs_dist < 0:
        bfs_dist = HEIGHT + WIDTH
    obs[29] = min(bfs_dist, HEIGHT + WIDTH) / (HEIGHT + WIDTH)
    best_dr, best_dc = bfs_best_direction(dist_map, r0, c0)
    obs[30] = float(best_dr)
    obs[31] = float(best_dc)
    obs[32] = 1.0 if bfs_dist <= GUARD_POS_REACH_RADIUS else 0.0

    obs[33] = 0.0  # 予備次元

    return obs


def decode_action(action_idx):
    move_idx, use_ability = divmod(int(action_idx), 2)
    return MOVES[move_idx], bool(use_ability)


def build_action_mask(unit, occupied, lock_movement=False):
    """lock_movement=True の場合、stay(move_idx=0)以外の移動を禁止する。
    敵を視認している間は静止させ、射撃の当たりやすさを優先する。"""
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

    if unit.charges <= 0 or unit.role in ("HUNT", "NONE"):
        for move_idx in range(5):
            mask[move_idx * 2 + 1] = False

    return mask


# ============================================================================
# 環境本体
# ============================================================================

class GuardEnv:
    """プラント後フェーズを模した簡易マルチエージェント環境。

    Attacker = touyama_v1固定チーム(5人、固定ステータス・固定ロール、
    プラント地点周辺のガードポジションに配置)。Defender(敵)側は本物の
    controllers.pyロジックではなく、このファイル内に複製した簡易
    ヒューリスティック(BFSでプラント地点へ接近→隣接で解除)で動かす。
    """

    def __init__(self):
        self.guard_memory = GuardMemory()
        self.attackers = []
        self.defenders = []
        self.smokes = []  # [{"cells": set, "remaining_ticks": int, "team": str}]
        self.detonate_timer = MAX_TICKS
        self.site_idx = 0
        self.planted_pos = None
        self.match_over_reason = None
        self._prev_kills = {}
        self._prev_alive = {}
        self.spike_dist_map = None
        self.sighting_dist_map = None
        self.active_defuser_name = None
        self.last_shots = []

    # -- 初期化 --------------------------------------------------------
    def reset(self):
        self.guard_memory.reset()
        self.smokes = []
        self.detonate_timer = MAX_TICKS
        self.match_over_reason = None
        self.active_defuser_name = None

        self.site_idx = random.randrange(len(SITE_PLANT_POS))
        self.planted_pos = SITE_PLANT_POS[self.site_idx]
        self.spike_dist_map = SITE_SPIKE_DIST_MAPS[self.site_idx]
        guard_positions = SITE_GUARD_POSITIONS[self.site_idx]

        self.attackers = _build_fixed_attackers(self.site_idx, guard_positions)
        self.defenders = _build_defenders()

        self.guard_memory.update(self.attackers, self.defenders, self._smoke_cells())
        self._update_sighting_dist_map()

        self._prev_kills = {u.name: u.kills for u in self.attackers + self.defenders}
        self._prev_alive = {u.name: u.is_alive for u in self.attackers + self.defenders}

        for a in self.attackers:
            a.prev_priority_mode = None
            a.prev_priority_dist = None
            a.prev_priority_target_key = None

        return self._collect_observations()

    def _update_sighting_dist_map(self):
        self.sighting_dist_map = (
            bfs_distance_map(self.guard_memory.last_seen_enemy["pos"])
            if self.guard_memory.last_seen_enemy is not None else None
        )

    def _smoke_cells(self):
        cells = set()
        for s in self.smokes:
            if s["remaining_ticks"] > 0:
                cells.update(s["cells"])
        return cells

    def _own_smoke_active(self, team):
        return any(s["team"] == team and s["remaining_ticks"] > 0 for s in self.smokes)

    def _active_defuse_info(self):
        defuser = next((d for d in self.defenders if d.is_alive and d.defuse_timer > 0), None)
        if defuser is None:
            return None
        return {
            "defuser": defuser,
            "progress_ratio": min(defuser.defuse_timer / DEFUSE_REQUIRED_TICKS, 1.0),
        }

    def _collect_observations(self):
        smoke_cells = self._smoke_cells()
        occupied = {tuple(u.pos) for u in self.attackers + self.defenders if u.is_alive}
        active_defuse = self._active_defuse_info()
        obs_dict, mask_dict = {}, {}
        for a in self.attackers:
            if not a.is_alive:
                continue
            unit_has_spike_los = has_los(a.pos, self.planted_pos, smoke_cells)
            obs_dict[a.name] = build_observation(
                a, self.attackers, self.defenders, self.guard_memory,
                smoke_cells, self._own_smoke_active("A"), self.detonate_timer,
                self.spike_dist_map, self.sighting_dist_map, unit_has_spike_los,
                active_defuse,
            )
            own_occupied = occupied - {tuple(a.pos)}
            has_enemy_los = any(
                d.is_alive and has_los(a.pos, d.pos, smoke_cells) for d in self.defenders
            )
            mask_dict[a.name] = build_action_mask(a, own_occupied, lock_movement=has_enemy_los)
        return obs_dict, mask_dict

    # -- メインステップ ---------------------------------------------------
    def step(self, action_dict):
        for u in self.attackers + self.defenders:
            u.moved_this_tick = False

        pre_tick_flash_recon_active = any(
            d.is_alive and (d.blind_remaining > 0 or d.reveal_remaining > 0)
            for d in self.defenders
        )
        pre_tick_defuse_timers = {d.name: d.defuse_timer for d in self.defenders}

        smoke_cells = self._smoke_cells()
        occupied = {tuple(u.pos) for u in self.attackers + self.defenders if u.is_alive}

        move_plans = []
        ability_requests = []
        ability_whiff = {}
        ability_overlap = {}
        held_angle = {}

        # --- 敵(Defender)側: プラント地点へBFSで接近し、隣接したら解除する ---
        for d in self.defenders:
            if not d.is_alive:
                continue
            dist_to_spike = max(
                abs(d.pos[0] - self.planted_pos[0]), abs(d.pos[1] - self.planted_pos[1])
            )
            if dist_to_spike <= 1:
                if self.active_defuser_name in (None, d.name):
                    self.active_defuser_name = d.name
                    d.defuse_timer += 1
                else:
                    d.defuse_timer = 0
                move_plans.append((d, (0, 0)))
                continue

            if self.active_defuser_name == d.name:
                self.active_defuser_name = None
            d.defuse_timer = 0

            own_occupied = occupied - {tuple(d.pos)}
            if random.random() < 0.1:
                nxt = _random_step(tuple(d.pos), own_occupied)
            else:
                nxt = _bfs_next_step(tuple(d.pos), self.planted_pos, own_occupied, allow_adjacent_goal=True)
            dr, dc = nxt[0] - d.pos[0], nxt[1] - d.pos[1]
            move_plans.append((d, (dr, dc)))

        # --- 味方(touyama Attacker)側: DQNの行動を反映 ---
        for a in self.attackers:
            if not a.is_alive or a.name not in action_dict:
                continue
            (dr, dc), use_ability = decode_action(action_dict[a.name])
            move_plans.append((a, (dr, dc)))

            visible_enemies = [
                d for d in self.defenders if d.is_alive and has_los(a.pos, d.pos, smoke_cells)
            ]
            has_enemy_los = bool(visible_enemies)

            if has_enemy_los and (dr, dc) == (0, 0):
                held_angle[a.name] = "held_with_los"
            elif has_enemy_los:
                held_angle[a.name] = "moved_with_los"
            else:
                held_angle[a.name] = "no_los"

            if use_ability:
                ability_whiff[a.name] = not has_enemy_los
                ability_overlap[a.name] = (
                    pre_tick_flash_recon_active and a.role in ("FLASH", "RECON")
                )
                if a.charges > 0:
                    if visible_enemies:
                        nearest = min(
                            visible_enemies,
                            key=lambda d: max(abs(d.pos[0] - a.pos[0]), abs(d.pos[1] - a.pos[1])),
                        )
                        dist = max(abs(nearest.pos[0] - a.pos[0]), abs(nearest.pos[1] - a.pos[1]))
                        if dist <= ABILITY_RANGE:
                            ability_requests.append((a, tuple(nearest.pos)))
                    elif self.guard_memory.last_seen_enemy is not None:
                        ability_requests.append((a, self.guard_memory.last_seen_enemy["pos"]))
                    else:
                        # 視認情報が全く無い場合はプラント地点周辺を予防的に牽制する
                        ability_requests.append((a, self.planted_pos))

        # --- 移動の適用(壁・マップ外・他ユニット占有マスを回避) ---
        for unit, (dr, dc) in move_plans:
            if not unit.is_alive or (dr, dc) == (0, 0):
                continue
            old_pos = tuple(unit.pos)
            nr, nc = unit.pos[0] + dr, unit.pos[1] + dc
            in_bounds = 0 <= nr < HEIGHT and 0 <= nc < WIDTH
            is_wall = in_bounds and GRID[nr, nc] == 1
            occ = any(
                other is not unit and other.is_alive and tuple(other.pos) == (nr, nc)
                for other in self.attackers + self.defenders
            )
            if in_bounds and not is_wall and not occ:
                unit.pos = [nr, nc]
            unit.moved_this_tick = tuple(unit.pos) != old_pos

        # --- アビリティの適用 ---
        for unit, target_pos in ability_requests:
            unit.charges -= 1
            if unit.role == "SMOKE":
                tr, tc = int(target_pos[0]), int(target_pos[1])
                cells = {
                    (rr, cc)
                    for rr in range(tr - 1, tr + 2)
                    for cc in range(tc - 1, tc + 2)
                    if 0 <= rr < HEIGHT and 0 <= cc < WIDTH and GRID[rr, cc] != 1
                }
                self.smokes.append({
                    "cells": cells, "remaining_ticks": SMOKE_DURATION_TICKS, "team": unit.team,
                })
            elif unit.role == "FLASH":
                for d in self.defenders:
                    if d.is_alive and has_los(target_pos, d.pos, smoke_cells):
                        d.blind_remaining = max(d.blind_remaining, BLIND_DURATION_TICKS)
            elif unit.role == "RECON":
                for d in self.defenders:
                    if d.is_alive and has_los(target_pos, d.pos, smoke_cells):
                        d.reveal_remaining = max(d.reveal_remaining, REVEAL_DURATION_TICKS)

        self._resolve_shots()

        # 解除者が死亡していたらロックを解放する(射撃解決後に判定)
        if self.active_defuser_name is not None:
            active = next((d for d in self.defenders if d.name == self.active_defuser_name), None)
            if active is None or not active.is_alive:
                self.active_defuser_name = None

        for u in self.attackers + self.defenders:
            u.blind_remaining = max(0, u.blind_remaining - 1)
            u.reveal_remaining = max(0, u.reveal_remaining - 1)
        for s in self.smokes:
            s["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

        self.guard_memory.update(self.attackers, self.defenders, self._smoke_cells())
        self._update_sighting_dist_map()
        self.detonate_timer -= 1

        defuse_completed = any(
            d.is_alive and d.defuse_timer >= DEFUSE_REQUIRED_TICKS for d in self.defenders
        )

        rewards = self._compute_rewards(
            pre_tick_defuse_timers, ability_whiff, ability_overlap, held_angle
        )

        self._prev_kills = {u.name: u.kills for u in self.attackers + self.defenders}
        self._prev_alive = {u.name: u.is_alive for u in self.attackers + self.defenders}

        attackers_alive = any(a.is_alive for a in self.attackers)
        defenders_alive = any(d.is_alive for d in self.defenders)

        done = (
            defuse_completed
            or self.detonate_timer <= 0
            or not attackers_alive
            or not defenders_alive
        )

        if done:
            if defuse_completed:
                self.match_over_reason = "defused"
                for a in self.attackers:
                    rewards[a.name] = rewards.get(a.name, 0.0) + DEFUSE_LOSS_PENALTY
            elif not defenders_alive:
                self.match_over_reason = "attacker_win_wipe"
                for a in self.attackers:
                    rewards[a.name] = rewards.get(a.name, 0.0) + ROUND_WIN_REWARD
            elif self.detonate_timer <= 0:
                self.match_over_reason = "attacker_win_detonate"
                for a in self.attackers:
                    rewards[a.name] = rewards.get(a.name, 0.0) + ROUND_WIN_REWARD
            elif not attackers_alive:
                self.match_over_reason = "attacker_wipe"
                for a in self.attackers:
                    rewards[a.name] = rewards.get(a.name, 0.0) + WIPE_LOSS_PENALTY

        obs_dict, mask_dict = self._collect_observations()
        return obs_dict, mask_dict, rewards, done

    def _resolve_shots(self):
        alive = [u for u in self.attackers + self.defenders if u.is_alive]
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
                    max(abs(t.pos[0] - shooter.pos[0]), abs(t.pos[1] - shooter.pos[1])),
                    t.hp, t.name,
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
                    if target.defuse_timer > 0:
                        target.defuse_timer = 0
                        if self.active_defuser_name == target.name:
                            self.active_defuser_name = None
            else:
                self.last_shots.append({"shooter": shooter, "target": target, "hit": False})

    def _priority_mode_and_distmap(self, attacker):
        active_defuse = self._active_defuse_info()
        if active_defuse is not None:
            # 解除中は最優先でプラント地点(=解除者の隣接マス)へ詰め寄る
            return "defuse_alert", self.spike_dist_map, "defuse_alert"
        if self.guard_memory.last_seen_enemy is not None:
            target_key = f"sighting:{self.guard_memory.last_seen_enemy.get('name')}"
            return "sighting", self.sighting_dist_map, target_key
        return "position", attacker.assigned_guard_dist_map, "position"

    def _compute_rewards(self, pre_tick_defuse_timers, ability_whiff, ability_overlap, held_angle):
        rewards = {}
        smoke_cells = self._smoke_cells()

        for a in self.attackers:
            r = STEP_PENALTY

            mode, dist_map, target_key = self._priority_mode_and_distmap(a)
            r0, c0 = int(a.pos[0]), int(a.pos[1])
            bfs_dist = dist_map[r0, c0] if dist_map is not None else None
            if bfs_dist is not None and bfs_dist < 0:
                bfs_dist = None

            if bfs_dist is None:
                a.prev_priority_mode = mode
                a.prev_priority_target_key = target_key
                a.prev_priority_dist = None
            elif (
                mode != a.prev_priority_mode
                or target_key != a.prev_priority_target_key
                or a.prev_priority_dist is None
            ):
                a.prev_priority_mode = mode
                a.prev_priority_target_key = target_key
                a.prev_priority_dist = bfs_dist
            else:
                delta = a.prev_priority_dist - bfs_dist
                a.prev_priority_dist = bfs_dist

                if mode == "defuse_alert":
                    r += GUARD_POSITION_PULL_REWARD * delta * 2.0  # 解除中は接近を強く促す
                elif mode == "sighting":
                    r += GUARD_POSITION_PULL_REWARD * delta
                else:
                    if bfs_dist > GUARD_POS_REACH_RADIUS:
                        r += GUARD_POSITION_PULL_REWARD * delta
                    else:
                        r += HOLD_POSITION_BONUS if not a.moved_this_tick else HOLD_POSITION_PENALTY

            # プラント地点にLOSが通っている間、静止していれば常時警戒ボーナス
            if not a.moved_this_tick and has_los(a.pos, self.planted_pos, smoke_cells):
                r += SPIKE_WATCH_BONUS

            if ability_whiff.get(a.name):
                r += ABILITY_WHIFF_PENALTY
            if ability_overlap.get(a.name):
                r += ABILITY_OVERLAP_PENALTY

            angle_state = held_angle.get(a.name)
            if angle_state == "held_with_los":
                r += HOLD_ANGLE_BONUS
            elif angle_state == "moved_with_los":
                r += HOLD_ANGLE_PENALTY

            new_kills = a.kills - self._prev_kills.get(a.name, a.kills)
            if new_kills > 0:
                r += KILL_REWARD * new_kills
                for shot in self.last_shots:
                    if (
                        shot["shooter"] is a
                        and shot["hit"]
                        and not shot["target"].is_alive
                        and pre_tick_defuse_timers.get(shot["target"].name, 0) > 0
                    ):
                        r += DEFUSER_KILL_BONUS

            was_alive = self._prev_alive.get(a.name, True)
            if was_alive and not a.is_alive:
                r += DEATH_PENALTY

            rewards[a.name] = r

        # 解除進捗ペナルティ(全員で共有、ポテンシャル差分)
        for d in self.defenders:
            prev = pre_tick_defuse_timers.get(d.name, 0)
            if d.is_alive and d.defuse_timer > prev:
                for a in self.attackers:
                    rewards[a.name] = rewards.get(a.name, 0.0) + DEFUSE_PROGRESS_PENALTY

        return rewards


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
    policy_net = AttackerGuardDuelingDQN().to(DEVICE)
    target_net = AttackerGuardDuelingDQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size)
    env = GuardEnv()

    global_step = 0
    best_avg_reward = -float("inf")
    episode_reward_history = deque(maxlen=100)

    start_time = time.perf_counter()
    for episode in range(1, episodes + 1):
        obs_dict, mask_dict = env.reset()
        episode_reward_total = 0.0
        epsilon = epsilon_by_episode(episode)

        for tick in range(MAX_TICKS):

            action_dict = {
                name: select_action(policy_net, obs, mask_dict[name], epsilon)
                for name, obs in obs_dict.items()
            }

            next_obs_dict, next_mask_dict, rewards, done = env.step(action_dict)

            for name, obs in obs_dict.items():
                action = action_dict[name]
                reward = rewards.get(name, 0.0)
                episode_reward_total += reward

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

            optimize(policy_net, target_net, optimizer, buffer, batch_size, gamma)

            if global_step % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done or not obs_dict:
                break

        episode_reward_history.append(episode_reward_total)
        avg_reward = sum(episode_reward_history) / len(episode_reward_history)

        if episode % 20 == 0:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter();
            print(
                f"[EP {episode}/{episodes}] reward={episode_reward_total:.3f} elapse={elapsed_time:.1f} "
                f"avg100={avg_reward:.3f} epsilon={epsilon_by_episode(episode):.3f} "
                f"buffer={len(buffer)} reason={env.match_over_reason}"
            )

        if avg_reward > best_avg_reward and len(episode_reward_history) >= 50:
            best_avg_reward = avg_reward
            torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
            print(f"[SAVE] best model updated: avg100={avg_reward:.3f} -> {MODEL_SAVE_PATH}")

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), MODEL_LATEST_PATH)

    print("[DONE] training finished.")


if __name__ == "__main__":
    train()