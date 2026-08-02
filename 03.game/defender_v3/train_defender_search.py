"""train_defender_search.py

Defender「search phase」学習スクリプト（プラント前限定）。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py などのfeatureモジュールは一切importしない。
LOS判定・BFS・衝突判定・射撃解決・アビリティ適用など必要なロジックは
すべてこのファイル内に複製する。

game_core / map_data / map_data_search からは定数・マップ文字列のみを
参照し、ロジック(関数・クラス)は一切参照しない(train_attacker_retrieve.py
と同じ方針)。

学習データ・チェックポイントは data/defender_search_data/ 以下に保存する。

"""

import os
import sys
import random
import math
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR
from map_data_search import SEARCH_MAZE_STR

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

EPISODE_COUNT = 5000

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
DATA_DIR = "data/defender_search_data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_defender_search_best_by_eval.pt")
MODEL_LATEST_PATH = os.path.join(DATA_DIR, "dqn_defender_search_latest.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 34  # 31(従来) + 3(担当ポジションへの相対方向dr,dc + 到着フラグ)
ACTION_DIM = 10  # move_idx(0-4) * 2 + use_ability_flag(0/1)
ROLES = ["FLASH", "SMOKE", "RECON", "HUNT"]

N_DEFENDERS = 5
N_ATTACKERS = 5
MAX_TICKS = ROUND_DURATION_TICKS  # 90

ABILITY_RANGE = 8       # FLASH/RECONを即時適用してよい最大距離(簡易化)
SIGHTING_STALENESS_CAP = 30
REACH_RADIUS = 1        # 担当ポジションへ「到着した」とみなすChebyshev距離

# デフォルトの戦闘ステータス(character_stats.pyの既定値相当。ロジックではなく
# 数値のみの簡易複製)
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0

# 報酬パラメータ
STEP_PENALTY = -0.001
SPIKE_PULL_REWARD = 0.05
SIGHTING_PULL_REWARD = 0.01
DEFENSE_POSITION_PULL_REWARD = 0.03   # 平常時、担当7地点へ寄る
HOLD_POSITION_BONUS = 0.02            # 担当地点到着後、静止
HOLD_POSITION_PENALTY = -0.01         # 担当地点到着後、無駄にうろつく
ABILITY_WHIFF_PENALTY = -0.05
ABILITY_OVERLAP_PENALTY = -0.05
DEBUFF_KILL_BONUS = 0.3
HOLD_ANGLE_BONUS = 0.02
HOLD_ANGLE_PENALTY = -0.01
KILL_REWARD = 0.5
DEATH_PENALTY = -0.5
ROUND_WIN_REWARD = 1.0          # 時間切れ・全滅によるDefender勝利
PLANT_PENALTY = -0.5            # このフェーズの範囲外(プラント成立)に至った場合


# ============================================================================
# マップ読み込み(map_data.NEW_MAZE_STR / map_data_search.SEARCH_MAZE_STR
# のみ参照。パース処理は自前で複製)
# ============================================================================

def _parse_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = _parse_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
ATTACKER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 3]
DEFENDER_SPAWNS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 2]

_SEARCH_GRID = _parse_grid(SEARCH_MAZE_STR)
DEFENSE_POSITIONS = [
    (r, c) for r in range(_SEARCH_GRID.shape[0]) for c in range(_SEARCH_GRID.shape[1])
    if _SEARCH_GRID[r, c] == 7
]
if not DEFENSE_POSITIONS:
    # 万一7が定義されていない場合のフォールバック(defenderスポーン地点を代用)
    print("[WARN] map_data_search.py に7のポジションが見つかりません。DEFENDER_SPAWNSで代用します。")
    DEFENSE_POSITIONS = list(DEFENDER_SPAWNS) if DEFENDER_SPAWNS else [(HEIGHT // 2, WIDTH // 2)]


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


SITE_DIST_MAPS = [bfs_distance_map(tuple(map(int, s))) for s in SITE_POSITIONS]
DEFENSE_POS_DIST_MAPS = [bfs_distance_map(pos) for pos in DEFENSE_POSITIONS]


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
        # Defender専用: 割り当てられた待機ポジション(7)。Attackerには使わない。
        self.assigned_defense_pos = None


# ============================================================================
# チーム共有メモリ(Defender視点。スパイク確定情報 / 敵目撃情報)
# ============================================================================

class TeamMemory:
    def __init__(self):
        self.spike_pos = None
        self.last_seen_enemy = None  # {"pos": (r, c), "tick_ago": int}

    def reset(self):
        self.spike_pos = None
        self.last_seen_enemy = None

    def update(self, defenders, attackers, smoke_cells):
        visible_enemies = []
        for d in defenders:
            if not d.is_alive:
                continue
            for a in attackers:
                if not a.is_alive:
                    continue
                if has_los(d.pos, a.pos, smoke_cells):
                    visible_enemies.append(a)

        spike_holder = next((a for a in visible_enemies if a.has_spike), None)
        if spike_holder is not None:
            self.spike_pos = tuple(spike_holder.pos)

        if visible_enemies:
            self.last_seen_enemy = {"pos": tuple(visible_enemies[0].pos), "tick_ago": 0}
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ============================================================================
# ネットワーク
# ============================================================================

class DefenderSearchDuelingDQN(nn.Module):
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

def build_observation(unit, defenders, attackers, team_memory, smoke_cells, own_smoke_active, round_timer):
    obs = np.zeros(OBS_DIM, dtype=np.float32)

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
        sp = team_memory.spike_pos
        obs[14] = 1.0
        obs[15] = (sp[0] - unit.pos[0]) / HEIGHT
        obs[16] = (sp[1] - unit.pos[1]) / WIDTH

    if team_memory.last_seen_enemy is not None:
        ls = team_memory.last_seen_enemy
        obs[17] = 1.0
        obs[18] = (ls["pos"][0] - unit.pos[0]) / HEIGHT
        obs[19] = (ls["pos"][1] - unit.pos[1]) / WIDTH
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

    # --- 担当する有利ポジション(7)への相対方向・到着フラグ ---
    if unit.assigned_defense_pos is not None:
        dp = unit.assigned_defense_pos
        obs[31] = (dp[0] - unit.pos[0]) / HEIGHT
        obs[32] = (dp[1] - unit.pos[1]) / WIDTH
        dist_to_dp = max(abs(dp[0] - unit.pos[0]), abs(dp[1] - unit.pos[1]))
        obs[33] = 1.0 if dist_to_dp <= REACH_RADIUS else 0.0

    return obs


def decode_action(action_idx):
    move_idx, use_ability = divmod(int(action_idx), 2)
    return MOVES[move_idx], bool(use_ability)


def build_action_mask(unit, occupied):
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

    if unit.charges <= 0 or unit.role == "HUNT":
        for move_idx in range(5):
            mask[move_idx * 2 + 1] = False

    return mask


# ============================================================================
# 環境本体
# ============================================================================

class SearchEnv:
    """プラント前フェーズを模した簡易マルチエージェント環境。

    Attacker側は本物のcontrollers.pyロジックではなく、このファイル内に
    複製した簡易ヒューリスティック(BFSでサイトへ寄る + 時々姿を見せる)で
    動かす。射撃・アビリティも簡易な即時判定モデルで再実装している
    (本物のプロジェクタイル物理は再現しない。train_attacker_retrieve.py の
    簡易化方針に倣う)。

    Defenderは平常時、map_data_search.py の7地点(DEFENSE_POSITIONS)から
    ラウンド開始時に割り当てられた1箇所へ向かい、到着後は待機する。
    スパイクや敵の目撃情報が入った時点で、その情報がこの待機行動より
    優先される(_compute_rewardsの優先順位ツリーを参照)。
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

    # -- 初期化 --------------------------------------------------------
    def reset(self):
        self.team_memory.reset()
        self.smokes = []
        self.round_timer = MAX_TICKS
        self.planted = False
        self.match_over_reason = None

        d_spawns = random.sample(DEFENDER_SPAWNS, min(N_DEFENDERS, len(DEFENDER_SPAWNS)))
        a_spawns = random.sample(ATTACKER_SPAWNS, min(N_ATTACKERS, len(ATTACKER_SPAWNS)))

        self.defenders = [
            UnitStub(f"D{i+1}", "D", pos, random.choice(ROLES))
            for i, pos in enumerate(d_spawns)
        ]
        self.attackers = [
            UnitStub(f"A{i+1}", "A", pos, random.choice(ROLES))
            for i, pos in enumerate(a_spawns)
        ]
        carrier = random.choice(self.attackers)
        carrier.has_spike = True

        self.carrier_target_site_idx = random.randrange(len(SITE_POSITIONS))

        self._assign_defense_positions()

        self.team_memory.update(self.defenders, self.attackers, self._smoke_cells())
        self._prev_kills = {u.name: u.kills for u in self.defenders + self.attackers}
        self._prev_alive = {u.name: u.is_alive for u in self.defenders + self.attackers}

        return self._collect_observations()

    def _assign_defense_positions(self):
        """各defenderへ、自スポーンから最も近い未割当の7地点を貪欲に割り当てる。

        7地点数がdefender数より少ない場合は、余ったdefenderへランダムに
        (重複を許容して)割り当てる(その場合は移動衝突判定側で隣接マスへ
        自然に押し出される)。
        """
        remaining = list(DEFENSE_POSITIONS)
        order = list(self.defenders)
        random.shuffle(order)
        for d in order:
            if remaining:
                pos = min(
                    remaining,
                    key=lambda p: max(abs(p[0] - d.pos[0]), abs(p[1] - d.pos[1])),
                )
                remaining.remove(pos)
            else:
                pos = random.choice(DEFENSE_POSITIONS)
            d.assigned_defense_pos = pos

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
            obs_dict[d.name] = build_observation(
                d, self.defenders, self.attackers, self.team_memory,
                smoke_cells, self._own_smoke_active("D"), self.round_timer,
            )
            own_occupied = occupied - {tuple(d.pos)}
            mask_dict[d.name] = build_action_mask(d, own_occupied)
        return obs_dict, mask_dict

    # -- Attacker側の簡易ヒューリスティック ------------------------------
    def _attacker_decide_move(self, unit):
        """簡易AI: スパイク保持者はBFSでターゲットサイトへ、それ以外は
        保持者の近くをゆるく追従しつつ、時々ランダムに動いて姿を見せる。"""
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
        """action_dict: {defender_name: action_idx}。1tick進める。"""
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

        move_plans = []  # (unit, (dr, dc))
        ability_requests = []  # (unit, target_pos)
        ability_whiff = {}
        ability_overlap = {}
        held_angle = {}

        carriers = [a for a in self.attackers if a.is_alive and a.has_spike]
        others = [a for a in self.attackers if a.is_alive and not a.has_spike]
        for unit in carriers + others:
            dr, dc = self._attacker_decide_move(unit)
            move_plans.append((unit, (dr, dc)))

        for d in self.defenders:
            if not d.is_alive or d.name not in action_dict:
                continue
            (dr, dc), use_ability = decode_action(action_dict[d.name])
            move_plans.append((d, (dr, dc)))

            visible_enemies = [
                a for a in self.attackers if a.is_alive and has_los(d.pos, a.pos, smoke_cells)
            ]
            has_enemy_los = bool(visible_enemies)

            if has_enemy_los and (dr, dc) == (0, 0):
                held_angle[d.name] = "held_with_los"
            elif has_enemy_los:
                held_angle[d.name] = "moved_with_los"
            else:
                held_angle[d.name] = "no_los"

            if use_ability:
                ability_whiff[d.name] = not has_enemy_los
                ability_overlap[d.name] = (
                    pre_tick_flash_recon_active and d.role in ("FLASH", "RECON")
                )
                if d.charges > 0:
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
                    "cells": cells,
                    "remaining_ticks": SMOKE_DURATION_TICKS,
                    "team": unit.team,
                })
            elif unit.role == "FLASH":
                for a in self.attackers:
                    if a.is_alive and has_los(target_pos, a.pos, smoke_cells):
                        a.blind_remaining = max(a.blind_remaining, BLIND_DURATION_TICKS)
            elif unit.role == "RECON":
                for a in self.attackers:
                    if a.is_alive and has_los(target_pos, a.pos, smoke_cells):
                        a.reveal_remaining = max(a.reveal_remaining, REVEAL_DURATION_TICKS)

        if not any(a.is_alive and a.has_spike for a in self.attackers):
            dropped_holder = next((a for a in self.attackers if a.has_spike), None)
            if dropped_holder is not None:
                nearest_alive = next((a for a in self.attackers if a.is_alive), None)
                if nearest_alive is not None:
                    nearest_alive.has_spike = True
                dropped_holder.has_spike = False

        self._resolve_shots()

        for u in self.defenders + self.attackers:
            u.blind_remaining = max(0, u.blind_remaining - 1)
            u.reveal_remaining = max(0, u.reveal_remaining - 1)
        for s in self.smokes:
            s["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

        self.team_memory.update(self.defenders, self.attackers, self._smoke_cells())
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
            pre_tick_enemy_debuffed, ability_whiff, ability_overlap, held_angle
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
        return obs_dict, mask_dict, rewards, done

    def _resolve_shots(self):
        """battle_logic.BattleLogicMixin._resolve_all_shots の簡易複製。
        反応速度が高い順に、射線が通る最近接の敵を撃つ。"""
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

    def _compute_rewards(self, pre_tick_enemy_debuffed, ability_whiff, ability_overlap, held_angle):
        rewards = {}
        for d in self.defenders:
            r = STEP_PENALTY

            # --- 平常時の寄せ先を優先度順に決める ---
            # 1. スパイク確定情報(最優先)
            # 2. 敵目撃情報(retake準備としてチーム全体で寄る)
            # 3. どちらも無ければ、担当する有利ポジション(7)へ向かい、
            #    到着後は静止する
            if self.team_memory.spike_pos is not None:
                sp = self.team_memory.spike_pos
                dist = max(abs(sp[0]-d.pos[0]), abs(sp[1]-d.pos[1]))
                r += SPIKE_PULL_REWARD * max(0.0, 1.0 - dist / max(HEIGHT, WIDTH))
            elif self.team_memory.last_seen_enemy is not None:
                ls = self.team_memory.last_seen_enemy["pos"]
                dist = max(abs(ls[0]-d.pos[0]), abs(ls[1]-d.pos[1]))
                r += SIGHTING_PULL_REWARD * max(0.0, 1.0 - dist / max(HEIGHT, WIDTH))
            elif d.assigned_defense_pos is not None:
                dp = d.assigned_defense_pos
                dist = max(abs(dp[0]-d.pos[0]), abs(dp[1]-d.pos[1]))
                if dist > REACH_RADIUS:
                    r += DEFENSE_POSITION_PULL_REWARD * max(0.0, 1.0 - dist / max(HEIGHT, WIDTH))
                else:
                    # 到着済み: 動かないことを評価し、無駄なうろつきを抑制する
                    if not d.moved_this_tick:
                        r += HOLD_POSITION_BONUS
                    else:
                        r += HOLD_POSITION_PENALTY

            if ability_whiff.get(d.name):
                r += ABILITY_WHIFF_PENALTY
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

            was_alive = self._prev_alive.get(d.name, True)
            if was_alive and not d.is_alive:
                r += DEATH_PENALTY

            rewards[d.name] = r
        return rewards


# ============================================================================
# 学習ループ
# ============================================================================

def epsilon_by_step(step, eps_start=1.0, eps_end=0.05, eps_decay=60_000):
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * step / eps_decay)


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
    policy_net = DefenderSearchDuelingDQN().to(DEVICE)
    target_net = DefenderSearchDuelingDQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size)
    env = SearchEnv()

    global_step = 0
    best_avg_reward = -float("inf")
    episode_reward_history = deque(maxlen=100)

    for episode in range(1, episodes + 1):
        obs_dict, mask_dict = env.reset()
        episode_reward_total = 0.0

        for tick in range(MAX_TICKS):
            epsilon = epsilon_by_step(global_step)

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
            print(
                f"[EP {episode}/{episodes}] reward={episode_reward_total:.3f} "
                f"avg100={avg_reward:.3f} epsilon={epsilon_by_step(global_step):.3f} "
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