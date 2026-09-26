"""train_defender_retake.py (修正版)

変更点: プラント地点への距離・方向をチェビシェフ距離(直線)ではなく
BFS距離マップ(壁を考慮した実際の経路距離)に基づいて計算するよう修正。
角での迂回行動が正しく報酬付けされるようになる。

以下は前バージョンからの主な差分:
- bfs_site_zone() を削除し、bfs_distance_map() を新設(プラント地点からの
  全マス距離を1回のBFSで計算)。site_zoneはこのマップから派生させる。
- build_observation(): dx, dy(直線方向) を good_dir(上下左右いずれが
  BFS距離を縮めるか、4次元) に置き換え。dist_to_plant もBFS距離ベースに変更。
- snapshot_before() / compute_rewards(): 接近報酬の距離差分をBFS距離ベースに変更。

それ以外(行動空間・行動マスク・アビリティ処理・戦闘解決・保存先パス等)は
前バージョンと同一。完全に自己完結という制約(controllers.py / run_game.py /
battle_logic.py 等をimportしない)も維持している。
"""

import random
import sys
import os
import argparse
import copy
import shutil
from collections import deque, namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    DEFUSE_REQUIRED_TICKS,
    SPIKE_DETONATION_TICKS,
    SMOKE_DURATION_TICKS,
    RECON_REVEAL_SIZE,
    ORB_COLLECT_REQUIRED_TICKS,
    ORB_ULTIMATE_POINTS,
)

try:
    from .character_stats_gc import (
        CHARACTER_TABLE as GC_STATS_TABLE,
        GC_ROSTER_ORDER,
    )
    from .gc_combo_stats import build_combo_bonuses
    from .gc_facing import (
        FACING_DIRS,
        append_facing_onehot,
        facing_towards,
        nearest_alive_enemy_facing,
    )
    from .defender_objectives_gc import (
        RETAKE_COORDINATION_DIM,
        RETAKE_UTILITY_CONTEXT_DIM,
        assign_retake_lane_targets,
        bfs_distance_map as objective_distance_map,
        nearest_orb_assignment,
        path_distance,
        retake_approach_sector,
        retake_coordination_features,
        retake_coordination_state,
        retake_utility_features,
        retake_utility_state,
        shortest_legal_action,
    )
    from .ultimate_tactics_gc import (
        ORB_CONTEXT_DIM,
        can_collect_orb,
        orb_context_features,
        tactical_ultimate_window,
        ultimate_context_features,
    )
except ImportError:
    from character_stats_gc import (
        CHARACTER_TABLE as GC_STATS_TABLE,
        GC_ROSTER_ORDER,
    )
    from gc_combo_stats import build_combo_bonuses
    from gc_facing import (
        FACING_DIRS,
        append_facing_onehot,
        facing_towards,
        nearest_alive_enemy_facing,
    )
    from defender_objectives_gc import (
        RETAKE_COORDINATION_DIM,
        RETAKE_UTILITY_CONTEXT_DIM,
        assign_retake_lane_targets,
        bfs_distance_map as objective_distance_map,
        nearest_orb_assignment,
        path_distance,
        retake_approach_sector,
        retake_coordination_features,
        retake_coordination_state,
        retake_utility_features,
        retake_utility_state,
        shortest_legal_action,
    )
    from ultimate_tactics_gc import (
        ORB_CONTEXT_DIM,
        can_collect_orb,
        orb_context_features,
        tactical_ultimate_window,
        ultimate_context_features,
    )

EPISODE_COUNT = 20000

DEVICE = torch.device("cpu")
SAVE_DIR = Path(__file__).resolve().parent / "data" / "defender_retake_gc_data"
BEST_MODEL_PATH = SAVE_DIR / "dqn_defender_retake_gc_best_by_eval.pt"
FINAL_MODEL_PATH = SAVE_DIR / "dqn_defender_retake_gc_final.pt"
PROMOTION_BACKUP_DIR = SAVE_DIR.parent / "defender_gc_model_backups"
RETAKE_ARCHIVE_DIR = SAVE_DIR / "retake_checkpoint_archive"

# ============================================================
# マップ読み込み
# ============================================================


def load_grid():
    lines = [l.strip() for l in NEW_MAZE_STR.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = load_grid()
HEIGHT, WIDTH = GRID.shape
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
DEFENDER_SPAWNS = [
    (r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 4
]
PLANT_CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] == 2]


# ============================================================
# 移動・LOS等の共通ロジック(controllers.py非依存の自前実装)
# ============================================================

CARDINAL_MOVES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]  # up, down, left, right (行動ID 0-3と対応)

TAU = 0.005


def soft_update(target_net, net, tau=TAU):
    for t_param, param in zip(target_net.parameters(), net.parameters()):
        t_param.data.copy_(tau * param.data + (1.0 - tau) * t_param.data)


def _alive_occupied_positions(chars, moving_char=None):
    occupied = set()
    for other in chars:
        if other is moving_char or not other.is_alive:
            continue
        occupied.add((int(other.pos[0]), int(other.pos[1])))
    return occupied


def _in_bounds(pos):
    r, c = pos
    return 0 <= r < HEIGHT and 0 <= c < WIDTH


def _is_walkable(pos, blocked):
    r, c = pos
    return _in_bounds(pos) and GRID[r, c] != 1 and (r, c) not in blocked


def get_next_pos_random(pos, chars, moving_char=None):
    r, c = int(pos[0]), int(pos[1])
    blocked = _alive_occupied_positions(chars, moving_char)
    valid = []
    for dr, dc in CARDINAL_MOVES:
        cand = (r + dr, c + dc)
        if _is_walkable(cand, blocked):
            valid.append(cand)
    return list(random.choice(valid)) if valid else [r, c]


def _candidate_goals(goal, blocked, allow_adjacent_goal):
    goal = (int(goal[0]), int(goal[1]))
    candidates = []
    if _is_walkable(goal, blocked):
        candidates.append(goal)
    if allow_adjacent_goal or goal in blocked:
        for dr, dc in CARDINAL_MOVES:
            adj = (goal[0] + dr, goal[1] + dc)
            if _is_walkable(adj, blocked):
                candidates.append(adj)
    return list(dict.fromkeys(candidates))


def evaluate_greedy(env, net, obs_dim, num_eval_episodes=100):
    wins = 0
    entered_site_count = 0
    reason_counts = {"defused": 0, "detonated": 0, "defenders_wiped": 0, "timeout": 0}
    zero_obs = np.zeros(obs_dim, dtype=np.float32)

    for _ in range(num_eval_episodes):
        env.reset()
        entered = False
        while True:
            terminal, reason = env.is_terminal()
            if terminal:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                break
            actions = {}
            for char in env.defenders():
                if not char.is_alive:
                    continue
                state = env.build_observation(char)
                mask = env.action_mask(char)
                action, facing = select_action(net, state, mask, epsilon=0.0)
                char.facing = facing
                actions[char.name] = action
                if tuple(char.pos) in env.site_zone:
                    entered = True
            env.step_tick(actions)
        if env.is_defused:
            wins += 1
        if entered:
            entered_site_count += 1

    print(f"    breakdown: {reason_counts}")
    return wins / num_eval_episodes, entered_site_count / num_eval_episodes


def move_towards_target(
    pos, target, chars, moving_char=None, allow_adjacent_goal=False
):
    """BFSで壁・生存キャラクターを避けながらtargetへ1マス進む(controllers.py複製版)。"""
    start = (int(pos[0]), int(pos[1]))
    goal = (int(target[0]), int(target[1]))
    blocked = _alive_occupied_positions(chars, moving_char)
    blocked.discard(start)

    if start == goal:
        return [start[0], start[1]]

    candidate_goals = _candidate_goals(goal, blocked, allow_adjacent_goal)
    if not candidate_goals:
        return get_next_pos_random(start, chars, moving_char)

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
            if nxt in parent or not _is_walkable(nxt, blocked):
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if reached is None:
        return get_next_pos_random(start, chars, moving_char)

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return [start[0], start[1]]
    return [int(step[0]), int(step[1])]


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


def _smoke_allows_line(cells, smoke_cells):
    if not cells or len(cells) <= 2:
        return True
    return not any(cell in smoke_cells for cell in cells)


def has_los(p1, p2, smoke_cells):
    cells = line_cells(p1, p2)
    for r, c in cells:
        if GRID[r, c] == 1:
            return False
    return _smoke_allows_line(cells, smoke_cells)


def bfs_distance_map(goal):
    """goal(プラント地点)から各床マスへの最短距離マップ(壁越え不可)。
    到達不能マスは-1。retrieveモデルのbfs_distance_map()と同一方式。"""
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if GRID[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < HEIGHT
                and 0 <= nc < WIDTH
                and GRID[nr, nc] != 1
                and dist[nr, nc] == -1
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def good_directions(dist_map, r, c):
    """現在地から上下左右のうち、BFS距離マップ上でプラントへの距離を実際に
    縮められる方向を1.0、そうでない方向(壁・行き止まり・遠回りになる方向)を
    0.0とする4次元フラグ。行動ID 0=UP,1=DOWN,2=LEFT,3=RIGHTと対応させる。"""
    good = [0.0, 0.0, 0.0, 0.0]
    raw = dist_map[r, c]
    if raw < 0:
        return good
    for i, (dr, dc) in enumerate(CARDINAL_MOVES):
        nr, nc = r + dr, c + dc
        if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1:
            nd = dist_map[nr, nc]
            if nd != -1 and nd < raw:
                good[i] = 1.0
    return good


# ============================================================
# 軽量キャラクター表現(game_core.Character非依存)
# ============================================================

ROLE_TO_ABILITY = {"フラッシュ": "FLASH", "スモーカー": "SMOKE", "シーカー": "RECON"}
# DEFENDER_ROLE_CYCLE は gc_v1 固定チームでは使用しない
# (ロールは固定ロースターの実効ステータスから決定するため)

BASE_ACCURACY = 0.55
BASE_DODGE = 0.12
BASE_HS_RATE = 0.22
BASE_REACTION = 100.0


# ============================================================
# gc_v1 固定チーム定義(引き継ぎ資料 3〜4節に準拠)
# 自己完結ルールのため、character_stats_gc.py を import せず
# 生データをこのファイル内に直接複製する。
# ============================================================

# GC専用コンボ(player_combos.py準拠)。固定5人ロースターのため毎ラウンド常時発動。

# タイガー固有パッシブ(game_core.Character準拠)。役職がタイガーなら常時発動。
GC_TIGER_BONUS = {"accuracy": 0.10, "hs_rate": 0.05}


GC_COMBO_NAME = "幽霊部員de廃部待ったなし"
GC_COMBO_MEMBERS = set(GC_ROSTER_ORDER)
GC_PLAYER_BONUSES = build_combo_bonuses(GC_ROSTER_ORDER)


def _compute_gc_effective_stats():
    """GC固定ロスターの実効ステータスを算出する。

    player_combos.py の「幽霊部員de廃部待ったなし」と
    タイガー固有パッシブを反映する。
    """
    effective = {}
    combo_active = GC_COMBO_MEMBERS.issubset(set(GC_ROSTER_ORDER))
    for name in GC_ROSTER_ORDER:
        raw = GC_STATS_TABLE[name]
        accuracy = float(raw.hit_pct)
        hs_rate = float(raw.hs_pct)
        dodge_rate = float(raw.dodge_pct)
        iq = float(raw.iq)
        reaction = float(raw.reaction)
        mental = float(getattr(raw, "mental", 0.0))

        if combo_active:
            pb = GC_PLAYER_BONUSES.get(name, {})
            accuracy += float(pb.get("accuracy", 0.0))
            hs_rate += float(pb.get("hs_rate", 0.0))
            dodge_rate += float(pb.get("dodge_rate", 0.0))
            iq += float(pb.get("iq", 0.0))
            reaction += float(pb.get("reaction", 0.0))
            mental += float(pb.get("mental", 0.0))

        if raw.role == "タイガー":
            accuracy += GC_TIGER_BONUS["accuracy"]
            hs_rate += GC_TIGER_BONUS["hs_rate"]

        effective[name] = {
            "accuracy": max(0.0, accuracy),
            "hs_rate": max(0.0, hs_rate),
            "dodge_rate": max(0.0, min(1.0, dodge_rate)),
            "iq": iq,
            "reaction": max(0.0, reaction),
            "mental": mental,
            "role": raw.role,
        }
    return effective


class SimChar:
    def __init__(self, name, team, pos, role=None, override_stats=None):
        self.name = name
        self.team = team
        self.pos = list(pos)
        self.is_alive = True
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.moved_this_tick = False
        self.los_revealed = False
        self.blind_remaining = 0
        self.reveal_remaining = 0
        self.defuse_timer = 0

        self.role = role
        ability = ROLE_TO_ABILITY.get(role, "NONE")
        self.ability_name = ability
        self.smoke_charges = 1 if ability == "SMOKE" else 0
        self.flash_charges = 1 if ability == "FLASH" else 0
        self.recon_charges = 1 if ability == "RECON" else 0
        self.facing = "N" if team == "A" else "S"
        self.ultimate_name = {
            "FLASH": "TUNNEL",
            "SMOKE": "ESCAPE",
            "RECON": "MONITOR",
            "NONE": "RAID",
        }[ability]
        self.ultimate_cost = {
            "RAID": 3,
            "ESCAPE": 6,
            "MONITOR": 8,
            "TUNNEL": 5,
        }[self.ultimate_name]
        self.ultimate_points = (
            self.ultimate_cost
            if random.random() < 0.45
            else random.randrange(self.ultimate_cost)
        )
        self.orb_collect_timer = 0
        self.collecting_orb_pos = None

        if override_stats is not None:
            # gc_v1固定チーム用: 実効ステータスをそのまま使用(ランダム化しない)
            self.accuracy = float(override_stats["accuracy"])
            self.dodge_rate = float(override_stats["dodge_rate"])
            self.hs_rate = float(override_stats["hs_rate"])
            self.reaction = float(override_stats["reaction"])
        else:
            self.accuracy = max(0.0, BASE_ACCURACY + random.uniform(-0.05, 0.05))
            self.dodge_rate = max(0.0, BASE_DODGE + random.uniform(-0.03, 0.03))
            self.hs_rate = max(0.0, BASE_HS_RATE + random.uniform(-0.03, 0.03))
            self.reaction = max(1.0, BASE_REACTION + random.uniform(-10.0, 10.0))

    def own_ability_charge(self):
        if self.ability_name == "SMOKE":
            return self.smoke_charges
        if self.ability_name == "FLASH":
            return self.flash_charges
        if self.ability_name == "RECON":
            return self.recon_charges
        return 0


# ============================================================
# Attacker側 固定AI(簡易・非学習)
# ============================================================


class AttackerStub:
    """サイト付近を保持する固定ロジック。アビリティは使用しない。"""

    def __init__(self, hold_radius=4):
        self.hold_radius = hold_radius

    def decide_move(self, char, chars, plant_pos):
        r, c = int(char.pos[0]), int(char.pos[1])
        dist = max(abs(plant_pos[0] - r), abs(plant_pos[1] - c))
        if dist > self.hold_radius:
            return move_towards_target(
                char.pos, plant_pos, chars, char, allow_adjacent_goal=True
            )
        if random.random() < 0.5:
            return list(char.pos)
        return get_next_pos_random(char.pos, chars, char)


# ============================================================
# 行動空間
# ============================================================

BASE_OBS_DIM = 37
ULTIMATE_CONTEXT_OBS_DIM = BASE_OBS_DIM + 4
ORB_CONTEXT_OBS_DIM = ULTIMATE_CONTEXT_OBS_DIM + ORB_CONTEXT_DIM
LEGACY_OBS_DIM = ORB_CONTEXT_OBS_DIM + len(FACING_DIRS)
RETAKE_UTILITY_RECENT_INDEX = LEGACY_OBS_DIM + RETAKE_COORDINATION_DIM
RETAKE_LANE_TARGET_INDEX = RETAKE_UTILITY_RECENT_INDEX + 1
RETAKE_UTILITY_CONTEXT_INDEX = RETAKE_LANE_TARGET_INDEX + 2
OBS_DIM = RETAKE_UTILITY_CONTEXT_INDEX + RETAKE_UTILITY_CONTEXT_DIM
LEGACY_N_ACTIONS = 7
N_ACTIONS = 9
MOVE_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
ACTION_DEFUSE = 5
ACTION_ABILITY = 6
ACTION_ULTIMATE = 7
ACTION_COLLECT_ORB = 8
FACING_HEAD_VERSION = 2
TRAINING_REVISION = "defender_retake_utility_setup_v8"

SITE_ZONE_RADIUS = 6
ENTRY_READY_RADIUS = 3
UTILITY_SETUP_RADIUS = SITE_ZONE_RADIUS
MIN_ALLIES_FOR_ENTRY = 2
ROLE_INDEX = {"フラッシュ": 0, "スモーカー": 1, "シーカー": 2, "タイガー": 3}

# 💡追加: 「解除が安全かどうか」の判定を、起爆タイマーの割合(detonate_frac)ではなく
# 「解除完了に最低限必要なtick数からの絶対的な残り時間」で判定するための定数。
DEFUSE_SAFETY_MARGIN_TICKS = 4
ENTRY_SAFETY_MARGIN_TICKS = DEFUSE_SAFETY_MARGIN_TICKS + ENTRY_READY_RADIUS
DEFENDER_ORB_APPROACH_RADIUS = 8


# ============================================================
# Retake 環境
# ============================================================


class RetakeEnv:
    def __init__(
        self,
        min_detonate_ticks=15,
        max_detonate_ticks=SPIKE_DETONATION_TICKS,
        attacker_hold_radius=4,
        max_ticks=90,
    ):
        self.min_detonate_ticks = min_detonate_ticks
        self.max_detonate_ticks = max_detonate_ticks
        self.attacker_stub = AttackerStub(hold_radius=attacker_hold_radius)
        self.max_ticks = max_ticks

    def reset(self):
        self.tick = 0
        self.round_over = False
        self.is_defused = False
        self.active_defuser_name = None
        self.available_orbs = set(zip(*np.where(GRID == 5)))
        self.orb_distance_maps = {
            tuple(map(int, orb)): objective_distance_map(GRID, orb)
            for orb in self.available_orbs
        }
        self.orb_collections = 0
        self.ability_uses = 0
        self.retake_utility_push_until = -1
        self.retake_utility_types_used = set()
        self.crossfire_entry_rewarded = set()

        self.planted_pos = (
            random.choice(PLANT_CELLS) if PLANT_CELLS else random.choice(WALKABLE)
        )

        # 💡修正: プラント地点からの全マスBFS距離を1回だけ計算し、
        # site_zoneも観測・報酬の距離計算もすべてこれに基づかせる。
        self.dist_map = bfs_distance_map(self.planted_pos)
        self.site_zone = {
            (r, c)
            for r in range(HEIGHT)
            for c in range(WIDTH)
            if 0 <= self.dist_map[r, c] <= SITE_ZONE_RADIUS
        }

        self.detonate_timer = random.randint(
            self.min_detonate_ticks, self.max_detonate_ticks
        )
        self.smokes = []  # list of {"cells": set, "remaining_ticks": int, "owner": str}

        used = set()
        self.chars = []
        self._build_fixed_defenders(used)
        self._build_attackers(used)
        self.retake_lane_targets = assign_retake_lane_targets(
            GRID,
            self.planted_pos,
            self.defenders(),
            self.dist_map,
            self.detonate_timer,
            entry_radius=ENTRY_READY_RADIUS,
            defuse_ticks=DEFUSE_REQUIRED_TICKS,
            safety_margin=DEFUSE_SAFETY_MARGIN_TICKS,
        )
        self.retake_lane_maps = {
            target: objective_distance_map(GRID, target)
            for target in set(self.retake_lane_targets.values())
        }

    def _build_fixed_defenders(self, used):
        """gc_v1固定ロースターをDEFENDER_SPAWNS順に固定配置し、実効ステータスをセットする。
        ランダム生成は行わない(run_game.pyのarea_4スキャン順と対応させるため)。"""
        effective_stats = _compute_gc_effective_stats()
        spawn_pool = (
            DEFENDER_SPAWNS
            if len(DEFENDER_SPAWNS) >= len(GC_ROSTER_ORDER)
            else WALKABLE
        )
        for i, name in enumerate(GC_ROSTER_ORDER):
            pos = spawn_pool[i]
            used.add(pos)
            stats = effective_stats[name]
            self.chars.append(
                SimChar(name, "D", pos, role=stats["role"], override_stats=stats)
            )

    def _build_attackers(self, used):
        """敵(Attacker)側は従来通りランダムスポーン・既定値のまま
        (将来ここだけ差し替え可能な設計)。"""
        hold_candidates = [
            p for p in self.site_zone if GRID[p[0], p[1]] != 2 and p not in used
        ]
        pool = (
            hold_candidates
            if hold_candidates
            else [p for p in WALKABLE if p not in used]
        )

        candidates = list(pool)
        random.shuffle(candidates)
        chosen = candidates[:5]
        used.update(chosen)
        while len(chosen) < 5:
            extra = random.choice(WALKABLE)
            if extra not in used:
                chosen.append(extra)
                used.add(extra)

        for i, pos in enumerate(chosen):
            self.chars.append(SimChar(f"Attacker{i+1}", "A", pos, role=None))

    def defenders(self):
        return [c for c in self.chars if c.team == "D"]

    def attackers(self):
        return [c for c in self.chars if c.team == "A"]

    def smoke_cells(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def check_line_of_sight(self, c1, c2):
        return has_los(tuple(c1.pos), tuple(c2.pos), self.smoke_cells())

    # -- アビリティ(サイトへ向けて使用する前提) --------------------------
    def apply_ability(self, char):
        ability = char.ability_name
        pr, pc = self.planted_pos

        if ability == "FLASH":
            char.flash_charges -= 1
            for enemy in self.chars:
                if enemy.team != char.team and enemy.is_alive:
                    if has_los((pr, pc), tuple(enemy.pos), self.smoke_cells()):
                        enemy.blind_remaining = max(
                            enemy.blind_remaining, BLIND_DURATION_TICKS
                        )

        elif ability == "RECON":
            char.recon_charges -= 1
            radius = RECON_REVEAL_SIZE // 2
            for enemy in self.chars:
                if enemy.team != char.team and enemy.is_alive:
                    er, ec = enemy.pos
                    if max(abs(er - pr), abs(ec - pc)) <= radius:
                        enemy.reveal_remaining = max(
                            enemy.reveal_remaining, REVEAL_DURATION_TICKS
                        )

        elif ability == "SMOKE":
            char.smoke_charges -= 1
            cells = {
                (rr, cc)
                for rr in range(pr - 1, pr + 2)
                for cc in range(pc - 1, pc + 2)
                if _in_bounds((rr, cc)) and GRID[rr, cc] != 1
            }
            self.smokes.append(
                {
                    "cells": cells,
                    "remaining_ticks": SMOKE_DURATION_TICKS,
                    "owner": char.name,
                }
            )

    def ally_ability_active(self, char):
        for enemy in self.chars:
            if enemy.team != char.team and enemy.is_alive:
                if enemy.blind_remaining > 0 or enemy.reveal_remaining > 0:
                    return True
        defender_names = {d.name for d in self.defenders()}
        for smoke in self.smokes:
            if smoke["owner"] in defender_names and smoke["remaining_ticks"] > 0:
                return True
        return False

    # -- 行動マスク ------------------------------------------------------
    def action_mask(self, char):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = _alive_occupied_positions(self.chars, char)

        for a in range(4):
            dr, dc = MOVE_DELTAS[a]
            nr, nc = r + dr, c + dc
            if _in_bounds((nr, nc)) and GRID[nr, nc] != 1 and (nr, nc) not in occupied:
                mask[a] = True
        mask[4] = True

        pr, pc = self.planted_pos
        dist_to_plant = max(abs(pr - r), abs(pc - c))
        mask[ACTION_DEFUSE] = bool(not self.is_defused and dist_to_plant <= 1)

        has_charge = char.own_ability_charge() > 0
        # An active smoke must not disable a teammate's flash or recon.
        mask[ACTION_ABILITY] = bool(has_charge)
        mask[ACTION_ULTIMATE] = bool(
            char.ultimate_cost > 0 and char.ultimate_points >= char.ultimate_cost
        )
        mask[ACTION_COLLECT_ORB] = can_collect_orb(char, self.available_orbs)

        if self.active_defuser_name == char.name and char.defuse_timer > 0:
            mask[:] = False
            mask[ACTION_DEFUSE] = dist_to_plant <= 1
        elif self.active_defuser_name is not None:
            mask[ACTION_DEFUSE] = False

        # 💡追加: 時間に余裕があり(time_critical_for_entryでない)、かつ敵が視認できている場合、
        # 移動action(0-3)をマスクして足を止めさせる(撃ち合い中の移動は不利なため)。
        return mask

    # -- 観測 --------------------------------------------------------------
    def build_observation(self, char):
        h, w = HEIGHT, WIDTH
        r, c = char.pos
        pr, pc = self.planted_pos

        # 💡修正: 直線方向(dx, dy)ではなく、BFS距離マップ上で実際に
        # 距離を縮められる方向を4次元フラグとして与える。
        good_dir = good_directions(self.dist_map, r, c)
        raw_dist = self.dist_map[r, c]
        dist_to_plant = min(1.0, raw_dist / (h + w)) if raw_dist >= 0 else 1.0

        in_site_zone = 1.0 if (r, c) in self.site_zone else 0.0
        adjacent_to_plant = 1.0 if max(abs(pr - r), abs(pc - c)) <= 1 else 0.0

        own_charge = 1.0 if char.own_ability_charge() > 0 else 0.0
        blind_norm = (
            char.blind_remaining / BLIND_DURATION_TICKS if BLIND_DURATION_TICKS else 0.0
        )
        defuse_norm = char.defuse_timer / DEFUSE_REQUIRED_TICKS
        detonate_norm = self.detonate_timer / SPIKE_DETONATION_TICKS

        allies = [a for a in self.defenders() if a.is_alive]
        enemies = [e for e in self.attackers() if e.is_alive]

        allies_alive_norm = len(allies) / 5.0
        allies_in_zone = sum(1 for a in allies if tuple(a.pos) in self.site_zone) / 5.0
        allies_near_entry = (
            sum(
                1
                for a in allies
                if max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
            )
            / 5.0
        )

        others = [a for a in allies if a is not char]
        if others:
            nearest_ally_dist = min(
                max(abs(a.pos[0] - r), abs(a.pos[1] - c)) for a in others
            ) / max(h, w)
        else:
            nearest_ally_dist = 1.0

        ally_ability_active = 1.0 if self.ally_ability_active(char) else 0.0

        visible_enemies = [e for e in enemies if self.check_line_of_sight(char, e)]
        visible_enemies.sort(key=lambda e: max(abs(e.pos[0] - r), abs(e.pos[1] - c)))

        enemy_feats = []
        for e in visible_enemies[:2]:
            edx = (e.pos[1] - c) / w
            edy = (e.pos[0] - r) / h
            edist = max(abs(e.pos[0] - r), abs(e.pos[1] - c)) / max(h, w)
            ehp = e.hp / e.max_hp
            eblind = 1.0 if e.blind_remaining > 0 else 0.0
            erevealed = 1.0 if (e.reveal_remaining > 0 or e.los_revealed) else 0.0
            enemy_feats.extend([edx, edy, edist, ehp, eblind, erevealed])
        while len(enemy_feats) < 12:
            enemy_feats.append(0.0)

        role_onehot = [0.0, 0.0, 0.0, 0.0]
        role_onehot[ROLE_INDEX.get(char.role, 0)] = 1.0

        obs = (
            [
                r / h,
                c / w,
                char.hp / char.max_hp,
                *good_dir,  # up, down, left, right (4次元。BFSベース)
                dist_to_plant,  # BFS距離ベース
                in_site_zone,
                adjacent_to_plant,
                own_charge,
                blind_norm,
                defuse_norm,
                detonate_norm,
                allies_alive_norm,
                allies_in_zone,
                allies_near_entry,
                nearest_ally_dist,
                ally_ability_active,
                len(visible_enemies) / 5.0,
                len(enemies) / 5.0,
            ]
            + enemy_feats
            + role_onehot
        )

        obs = np.array(obs, dtype=np.float32)
        expanded = np.zeros(OBS_DIM, dtype=np.float32)
        expanded[:BASE_OBS_DIM] = obs
        expanded[BASE_OBS_DIM:ULTIMATE_CONTEXT_OBS_DIM] = ultimate_context_features(
            char,
            engaged=bool(visible_enemies),
            objective_window=True,
            urgent=(
                self.detonate_timer <= 20
                or (char.max_hp > 0 and char.hp / char.max_hp <= 0.45)
            ),
        )
        expanded[ULTIMATE_CONTEXT_OBS_DIM:ORB_CONTEXT_OBS_DIM] = orb_context_features(
            char, self.available_orbs, ORB_COLLECT_REQUIRED_TICKS
        )
        append_facing_onehot(
            expanded[ORB_CONTEXT_OBS_DIM:LEGACY_OBS_DIM], char.facing
        )
        coordination = retake_coordination_state(
            char,
            allies,
            self.dist_map,
            self.detonate_timer,
            entry_radius=ENTRY_READY_RADIUS,
            defuse_ticks=DEFUSE_REQUIRED_TICKS,
            safety_margin=DEFUSE_SAFETY_MARGIN_TICKS,
        )
        expanded[LEGACY_OBS_DIM:RETAKE_UTILITY_RECENT_INDEX] = retake_coordination_features(
            coordination, SPIKE_DETONATION_TICKS
        )
        expanded[RETAKE_UTILITY_RECENT_INDEX] = float(
            self.tick <= self.retake_utility_push_until
        )
        lane_target = self.retake_lane_targets.get(char.name, self.planted_pos)
        expanded[RETAKE_LANE_TARGET_INDEX:RETAKE_UTILITY_CONTEXT_INDEX] = (
            (lane_target[0] - r) / h,
            (lane_target[1] - c) / w,
        )
        utility = retake_utility_state(
            allies, enemies, self.planted_pos, self.smoke_cells(),
            UTILITY_SETUP_RADIUS, self.dist_map,
        )
        expanded[RETAKE_UTILITY_CONTEXT_INDEX:OBS_DIM] = retake_utility_features(
            utility
        )
        return expanded

    # -- 1Tick進行 ---------------------------------------------------------
    def step_tick(self, defender_actions):
        for char in self.chars:
            char.moved_this_tick = False

        next_positions = {}
        pending_defuse = set()
        pending_ability = set()
        pending_ultimate = set()
        pending_orb = set()

        for char in self.chars:
            if not char.is_alive:
                continue

            if char.team == "D":
                action = defender_actions.get(char.name, 4)
                if action in (0, 1, 2, 3, 4):
                    dr, dc = MOVE_DELTAS[action]
                    next_positions[char.name] = [char.pos[0] + dr, char.pos[1] + dc]
                elif action == ACTION_DEFUSE:
                    pending_defuse.add(char.name)
                elif action == ACTION_ABILITY:
                    pending_ability.add(char.name)
                elif action == ACTION_ULTIMATE:
                    pending_ultimate.add(char.name)
                elif action == ACTION_COLLECT_ORB:
                    pending_orb.add(char.name)
            else:
                next_positions[char.name] = self.attacker_stub.decide_move(
                    char, self.chars, self.planted_pos
                )

        for char in self.chars:
            if char.name in pending_ability:
                previous_charge = char.own_ability_charge()
                self.apply_ability(char)
                if char.own_ability_charge() < previous_charge:
                    self.ability_uses += 1
                    self.retake_utility_types_used.add(char.ability_name)
                    self.retake_utility_push_until = max(
                        self.retake_utility_push_until, self.tick + 3
                    )

        for char in self.defenders():
            if char.name not in pending_orb or not can_collect_orb(
                char, self.available_orbs
            ):
                char.orb_collect_timer = 0
                char.collecting_orb_pos = None
                continue
            orb_pos = tuple(map(int, char.pos))
            if char.collecting_orb_pos != orb_pos:
                char.orb_collect_timer = 0
            char.collecting_orb_pos = orb_pos
            char.orb_collect_timer += 1
            if char.orb_collect_timer >= ORB_COLLECT_REQUIRED_TICKS:
                char.ultimate_points = min(
                    char.ultimate_cost,
                    char.ultimate_points + ORB_ULTIMATE_POINTS,
                )
                self.available_orbs.discard(orb_pos)
                self.orb_collections += 1
                char.orb_collect_timer = 0
                char.collecting_orb_pos = None

        for char in self.chars:
            if char.name not in pending_ultimate:
                continue
            if char.ultimate_points < char.ultimate_cost:
                continue
            char.ultimate_points = 0
            if char.ultimate_name == "TUNNEL":
                for enemy in self.attackers():
                    if enemy.is_alive and self.check_line_of_sight(char, enemy):
                        enemy.blind_remaining = max(enemy.blind_remaining, 5)
            elif char.ultimate_name == "MONITOR":
                for enemy in self.attackers():
                    if enemy.is_alive:
                        enemy.reveal_remaining = max(enemy.reveal_remaining, 5)
            elif char.ultimate_name == "ESCAPE":
                next_pos = move_towards_target(
                    char.pos, self.planted_pos, self.chars, char, allow_adjacent_goal=True
                )
                next_positions[char.name] = next_pos
            elif char.ultimate_name == "RAID":
                step = {
                    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
                    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
                }.get(char.facing, (0, 0))
                next_positions[char.name] = [
                    char.pos[0] + step[0], char.pos[1] + step[1]
                ]
            self.retake_utility_push_until = max(
                self.retake_utility_push_until, self.tick + 3
            )

        pr, pc = self.planted_pos
        for char in self.chars:
            if char.name not in pending_defuse:
                if self.active_defuser_name == char.name:
                    self.active_defuser_name = None
                    char.defuse_timer = 0
                continue
            r, c = char.pos
            dist = max(abs(pr - r), abs(pc - c))
            if self.is_defused or dist > 1:
                char.defuse_timer = 0
                continue
            if self.active_defuser_name in (None, char.name):
                self.active_defuser_name = char.name
                char.defuse_timer += 1
            else:
                char.defuse_timer = 0

        move_order = [c for c in self.chars if c.is_alive and c.name in next_positions]
        random.shuffle(move_order)
        for char in move_order:
            if char.name in pending_defuse or char.name in pending_ability:
                continue
            target = next_positions[char.name]
            nr, nc = int(target[0]), int(target[1])
            in_bounds = _in_bounds((nr, nc))
            occupied = any(
                other is not char and other.is_alive and tuple(other.pos) == (nr, nc)
                for other in self.chars
            )
            is_wall = in_bounds and GRID[nr, nc] == 1
            old_pos = tuple(char.pos)
            if in_bounds and not is_wall and not occupied:
                char.pos = [nr, nc]
            char.moved_this_tick = tuple(char.pos) != old_pos

        for char in self.chars:
            char.blind_remaining = max(0, char.blind_remaining - 1)
            char.reveal_remaining = max(0, char.reveal_remaining - 1)
        for smoke in self.smokes:
            smoke["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

        current_los_revealed = set()
        alive = [c for c in self.chars if c.is_alive]
        for i, a in enumerate(alive):
            for b in alive[i + 1 :]:
                if a.team != b.team and self.check_line_of_sight(a, b):
                    current_los_revealed.add(a.name)
                    current_los_revealed.add(b.name)

        self.last_shots = self._resolve_shots(current_los_revealed)

        for char in self.chars:
            char.los_revealed = char.is_alive and char.name in current_los_revealed

        if not self.is_defused:
            completed = [
                c
                for c in self.defenders()
                if c.is_alive and c.defuse_timer >= DEFUSE_REQUIRED_TICKS
            ]
            if completed:
                self.is_defused = True
                self.active_defuser_name = None

        self.detonate_timer -= 1
        self.tick += 1

    def _resolve_shots(self, current_los_revealed):
        alive = [c for c in self.chars if c.is_alive]
        intents = []
        for shooter in alive:
            if shooter.defuse_timer > 0:
                continue
            possible = [
                t
                for t in alive
                if t.team != shooter.team and self.check_line_of_sight(shooter, t)
            ]
            if not possible:
                continue
            defusers = [t for t in possible if t.defuse_timer > 0]
            pool = defusers if defusers else possible
            target = min(
                pool,
                key=lambda t: (
                    max(abs(t.pos[0] - shooter.pos[0]), abs(t.pos[1] - shooter.pos[1])),
                    t.hp,
                    t.name,
                ),
            )
            intents.append({"shooter": shooter, "target": target})

        random.shuffle(intents)
        intents.sort(key=lambda i: i["shooter"].reaction, reverse=True)

        shots = []
        for intent in intents:
            shooter, target = intent["shooter"], intent["target"]
            if not shooter.is_alive or not target.is_alive:
                continue

            shooter_accuracy = (
                MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            )
            if shooter.blind_remaining > 0:
                shooter_accuracy *= BLIND_ACCURACY_MULTIPLIER

            revealed = target.reveal_remaining > 0 or (
                target.los_revealed and target.name in current_los_revealed
            )
            effective_dodge = target.dodge_rate * (
                REVEALED_DODGE_MULTIPLIER if revealed else 1.0
            )
            hit_chance = shooter_accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            headshot = hit and random.random() < shooter.hs_rate
            damage = (HEADSHOT_DAMAGE if headshot else BODY_DAMAGE) if hit else 0

            shots.append(
                {
                    "shooter": shooter,
                    "target": target,
                    "hit": hit,
                    "headshot": headshot,
                    "damage": damage,
                }
            )

            if damage > 0:
                target.hp = max(0, target.hp - damage)
                if target.hp <= 0:
                    target.is_alive = False
                    target.defuse_timer = 0
                    if self.active_defuser_name == target.name:
                        self.active_defuser_name = None

        return shots

    def is_terminal(self):
        if self.is_defused:
            return True, "defused"
        if self.detonate_timer <= 0:
            return True, "detonated"
        if not any(c.is_alive for c in self.defenders()):
            return True, "defenders_wiped"
        if self.tick >= self.max_ticks:
            return True, "timeout"
        return False, None


# ============================================================
# 報酬設計
# ============================================================

APPROACH_REWARD_SCALE = 0.05
ENTRY_WITH_SUPPORT_BONUS = 1.5
ENTRY_ALONE_PENALTY = -0.3
ENTRY_ALONE_LINGER_PENALTY = -0.15
ABILITY_GOOD_USE_BONUS = 0.4
ABILITY_PREMATURE_PENALTY = -0.5
DAMAGE_REWARD_SCALE = 0.01
DEBUFF_HIT_MULTIPLIER = 1.5
KILL_REWARD = 3.0
KILL_ON_DEBUFFED_BONUS = 1.5
DEATH_PENALTY = -3.0
DEFUSE_PROGRESS_REWARD = 0.3
DEFUSE_INTERRUPT_PENALTY = -1.0
UNSAFE_DEFUSE_PENALTY = -0.5
DEFUSE_WIN_REWARD = 10.0
LOSS_PENALTY = -10.0
TICK_TIME_PENALTY = -0.01
ORB_PROGRESS_REWARD = 0.50
ORB_COLLECTION_REWARD = 3.00
ORB_ROUND_MISS_PENALTY = 1.00
RETAKE_EARLY_PEEK_PENALTY = -0.60
RETAKE_UTILITY_PREP_BONUS = 0.60
RETAKE_UTILITY_ENTRY_BONUS = 0.75
RETAKE_UTILITY_FOLLOWUP_REWARD = 0.35
RETAKE_DRY_ENTRY_PENALTY = -0.50
RETAKE_CROSSFIRE_ENTRY_BONUS = 0.50
RETAKE_UTILITY_CHAIN_BONUS = 0.75
RETAKE_UNUSED_FOLLOWUP_ENTRY_PENALTY = -1.25


def snapshot_before(env):
    before = {}
    for char in env.defenders():
        r, c = char.pos
        # 💡修正: 直線距離ではなくBFS距離を保存する。到達不能(-1)の場合は
        # マップ最大距離相当の値でフォールバックする(通常は起こらない想定)。
        raw = env.dist_map[r, c]
        dist_val = raw if raw >= 0 else (HEIGHT + WIDTH)
        coordination = retake_coordination_state(
            char,
            env.defenders(),
            env.dist_map,
            env.detonate_timer,
            entry_radius=ENTRY_READY_RADIUS,
            defuse_ticks=DEFUSE_REQUIRED_TICKS,
            safety_margin=DEFUSE_SAFETY_MARGIN_TICKS,
        )
        ready_allies = [
            ally
            for ally in env.defenders()
            if ally.is_alive
            and 0 <= int(env.dist_map[ally.pos[0], ally.pos[1]]) <= ENTRY_READY_RADIUS
        ]
        utility = retake_utility_state(
            env.defenders(), env.attackers(), env.planted_pos,
            env.smoke_cells(), UTILITY_SETUP_RADIUS, env.dist_map,
        )
        active_utility_types = {
            ability for ability, active_key in (
                ("SMOKE", "smoke_active"),
                ("FLASH", "blind_active"),
                ("RECON", "reveal_active"),
            ) if utility[active_key]
        }
        before[char.name] = {
            "alive": char.is_alive,
            "in_zone": (r, c) in env.site_zone,
            "dist_to_plant": dist_val,
            "defuse_timer": char.defuse_timer,
            "orb_collect_timer": char.orb_collect_timer,
            "ultimate_points": char.ultimate_points,
            "coordination": coordination,
            "ally_ability_active": env.ally_ability_active(char),
            "active_utility_types": active_utility_types,
            "unused_followup_utility": bool(
                (utility["remaining"]["FLASH"] and not utility["blind_active"])
                or (utility["remaining"]["RECON"] and not utility["reveal_active"])
            ),
            "team_utility_available": any(
                ally.own_ability_charge() > 0
                or (
                    ally.ultimate_cost > 0
                    and ally.ultimate_points >= ally.ultimate_cost
                )
                for ally in ready_allies
            ),
        }
    return before


def compute_rewards(env, before, chosen_actions):
    rewards = {}
    detonate_frac = env.detonate_timer / SPIKE_DETONATION_TICKS
    pr, pc = env.planted_pos

    allies_alive = [a for a in env.chars if a.team == "D" and a.is_alive]

    for char in env.chars:
        if char.team != "D":
            continue
        name = char.name
        b = before.get(name)
        if b is None or not b["alive"]:
            continue

        if not char.is_alive:
            rewards[name] = rewards.get(name, 0.0) + DEATH_PENALTY
            continue

        reward = TICK_TIME_PENALTY
        r, c = char.pos
        in_zone_now = (r, c) in env.site_zone

        # 💡修正: 接近報酬もBFS距離の減少量で計算する。
        raw_now = env.dist_map[r, c]
        dist_now = raw_now if raw_now >= 0 else (HEIGHT + WIDTH)

        allies_near_entry = sum(
            1
            for a in allies_alive
            if max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
        )
        support_required = min(MIN_ALLIES_FOR_ENTRY, len(allies_alive))

        # 💡変更: detonate_frac(割合)ではなく、残りtickの絶対値で「時間切れ間近か」を判定する。
        time_critical_for_entry = env.detonate_timer <= ENTRY_SAFETY_MARGIN_TICKS

        if in_zone_now and not b["in_zone"]:
            if allies_near_entry >= support_required or time_critical_for_entry:
                reward += ENTRY_WITH_SUPPORT_BONUS
            else:
                reward += ENTRY_ALONE_PENALTY
        elif (
            in_zone_now
            and allies_near_entry < support_required
            and not time_critical_for_entry
        ):
            reward += ENTRY_ALONE_LINGER_PENALTY
        elif not in_zone_now:
            reward += APPROACH_REWARD_SCALE * (b["dist_to_plant"] - dist_now)

        if (
            env.tick <= env.retake_utility_push_until
            and dist_now < b["dist_to_plant"]
        ):
            reward += RETAKE_UTILITY_FOLLOWUP_REWARD

        if (
            name not in env.crossfire_entry_rewarded
            and b["dist_to_plant"] > ENTRY_READY_RADIUS
            and dist_now <= ENTRY_READY_RADIUS
            and any(enemy.is_alive for enemy in env.attackers())
        ):
            own_sector = retake_approach_sector(char.pos, env.planted_pos)
            if own_sector != "center" and any(
                ally is not char
                and 0 <= int(env.dist_map[ally.pos[0], ally.pos[1]]) <= ENTRY_READY_RADIUS
                and retake_approach_sector(ally.pos, env.planted_pos)
                not in ("center", own_sector)
                for ally in allies_alive
            ):
                reward += RETAKE_CROSSFIRE_ENTRY_BONUS
                env.crossfire_entry_rewarded.add(name)

        action_id = chosen_actions.get(name)
        coordination = b["coordination"]
        if (
            coordination["should_wait"]
            and action_id in (0, 1, 2, 3)
            and in_zone_now
            and not b["in_zone"]
        ):
            reward += RETAKE_EARLY_PEEK_PENALTY
        if action_id in (ACTION_ABILITY, ACTION_ULTIMATE):
            if coordination["team_ready"] or coordination["must_commit"]:
                reward += RETAKE_UTILITY_PREP_BONUS
        if (
            action_id == ACTION_ABILITY
            and b["active_utility_types"]
            and char.ability_name not in b["active_utility_types"]
        ):
            reward += RETAKE_UTILITY_CHAIN_BONUS
        if in_zone_now and not b["in_zone"] and coordination["team_ready"]:
            if b["ally_ability_active"] or env.ally_ability_active(char):
                reward += RETAKE_UTILITY_ENTRY_BONUS
            elif b["team_utility_available"] and not coordination["must_commit"]:
                reward += RETAKE_DRY_ENTRY_PENALTY
        if (
            in_zone_now and not b["in_zone"]
            and b["unused_followup_utility"]
            and b["active_utility_types"]
            and not coordination["must_commit"]
            and any(enemy.is_alive for enemy in env.attackers())
            and not any(
                enemy.is_alive
                and (enemy.blind_remaining > 0 or enemy.reveal_remaining > 0)
                for enemy in env.attackers()
            )
        ):
            reward += RETAKE_UNUSED_FOLLOWUP_ENTRY_PENALTY

        if action_id == ACTION_ABILITY:
            ready = 0 <= raw_now <= UTILITY_SETUP_RADIUS
            if ready or time_critical_for_entry:
                reward += ABILITY_GOOD_USE_BONUS
            else:
                reward += ABILITY_PREMATURE_PENALTY

        if action_id == ACTION_COLLECT_ORB:
            if char.ultimate_points > b["ultimate_points"]:
                reward += ORB_COLLECTION_REWARD
            elif char.orb_collect_timer > b["orb_collect_timer"]:
                reward += ORB_PROGRESS_REWARD
            else:
                reward -= 0.05

        if char.defuse_timer > b["defuse_timer"]:
            reward += DEFUSE_PROGRESS_REWARD
        elif b["defuse_timer"] > 0 and char.defuse_timer == 0:
            reward += DEFUSE_INTERRUPT_PENALTY

        if (
            action_id == ACTION_DEFUSE
            and b["defuse_timer"] == 0
            and char.defuse_timer > 0
        ):
            threat = any(
                e.is_alive and env.check_line_of_sight(char, e) for e in env.attackers()
            )
            # 💡変更: 解除に最低限必要な時間(DEFUSE_REQUIRED_TICKS)+安全マージンを
            # 切っている場合は、脅威がいてもペナルティを科さない(間に合わなくなるのを防ぐ)。
            time_critical_for_defuse = env.detonate_timer <= (
                DEFUSE_REQUIRED_TICKS + DEFUSE_SAFETY_MARGIN_TICKS
            )
            if threat and not time_critical_for_defuse:
                reward += UNSAFE_DEFUSE_PENALTY

        rewards[name] = rewards.get(name, 0.0) + reward

    for shot in env.last_shots:
        shooter, target = shot["shooter"], shot["target"]
        if shooter.team != "D" or not shot["hit"]:
            continue
        if shooter.name not in rewards:
            continue
        add = DAMAGE_REWARD_SCALE * shot["damage"]
        debuffed = (
            target.blind_remaining > 0
            or target.reveal_remaining > 0
            or target.los_revealed
        )
        if debuffed:
            add *= DEBUFF_HIT_MULTIPLIER
        if not target.is_alive:
            add += KILL_REWARD
            if debuffed:
                add += KILL_ON_DEBUFFED_BONUS
        rewards[shooter.name] = rewards.get(shooter.name, 0.0) + add

    return rewards


# ============================================================
# Dueling DQN
# ============================================================


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions)
        )
        facing_hidden = hidden // 2
        self.facing_feature = nn.Sequential(
            nn.Linear(obs_dim, facing_hidden),
            nn.ReLU(),
            nn.Linear(facing_hidden, facing_hidden),
            nn.ReLU(),
        )
        self.facing_output = nn.Linear(facing_hidden + n_actions, len(FACING_DIRS))
        self.action_dim = n_actions
        self.facing_head_version = FACING_HEAD_VERSION

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)

    def facing_values(self, x, actions):
        features = self.facing_feature(x)
        actions = torch.as_tensor(actions, dtype=torch.long, device=x.device).view(-1)
        onehot = F.one_hot(actions, num_classes=self.action_dim).to(features.dtype)
        return self.facing_output(torch.cat((features, onehot), dim=1))


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done", "mask", "next_mask"],
)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def select_action(net, state, mask, epsilon):
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) == 0:
        base_action = 4
    elif random.random() < epsilon:
        base_action = int(random.choice(valid_indices))
    else:
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q_values = net(state_t).squeeze(0).cpu().numpy()
        q_values = np.where(mask, q_values, -np.inf)
        base_action = int(np.argmax(q_values))
    return base_action, select_facing(net, state, base_action, epsilon)


def select_facing(net, state, action, epsilon):
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        facing_values = net.facing_values(state_t, [action]).squeeze(0)
    if random.random() < epsilon:
        facing_idx = random.randrange(len(FACING_DIRS))
    else:
        facing_idx = int(facing_values.argmax().item())
    return FACING_DIRS[facing_idx]


def compute_td_loss(net, target_net, batch, gamma):
    states = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE)
    actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(DEVICE)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
    next_masks = torch.from_numpy(np.stack(batch.next_mask)).bool().to(DEVICE)

    q_values = net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_online = net(next_states)
        next_q_online = next_q_online.masked_fill(~next_masks, float("-inf"))
        next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states).gather(1, next_actions).squeeze(1)
        next_q_target = torch.nan_to_num(next_q_target, neginf=0.0)
        td_target = rewards + gamma * next_q_target * (1.0 - dones)

    return F.smooth_l1_loss(q_values, td_target)


# ============================================================
# 学習ループ
# ============================================================


def observable_facing_target(env, char, action):
    def facing_index(direction):
        if direction in FACING_DIRS:
            return FACING_DIRS.index(direction)
        current = getattr(char, "facing", "S")
        return FACING_DIRS.index(current) if current in FACING_DIRS else FACING_DIRS.index("S")

    facing = nearest_alive_enemy_facing(char, env.attackers())
    if facing is not None:
        return facing_index(facing), 1.0
    if action in MOVE_DELTAS and MOVE_DELTAS[action] != (0, 0):
        dr, dc = MOVE_DELTAS[action]
        facing = facing_towards((0, 0), (dr, dc))
        return facing_index(facing), 0.35
    facing = facing_towards(char.pos, env.planted_pos)
    # At the exact plant cell there is no geometric direction. Keeping the
    # current direction is an observable and stable teacher for that case.
    return facing_index(facing), 0.60 if facing is not None else 0.15


def defender_orb_teacher_action(env, char, mask):
    """Route one ult-hungry defender to an orb only when the defuse still fits."""
    if env.orb_collections > 0 or not env.available_orbs:
        return None
    assignment = nearest_orb_assignment(
        env.defenders(),
        env.available_orbs,
        GRID,
        env.orb_distance_maps,
        DEFENDER_ORB_APPROACH_RADIUS,
    )
    if assignment is None or assignment[3].name != char.name:
        return None
    distance, _name, orb, _collector, orb_map = assignment
    orb_to_plant = int(env.dist_map[orb[0], orb[1]])
    total_required = (
        int(distance)
        + ORB_COLLECT_REQUIRED_TICKS
        + max(0, orb_to_plant - 1)
        + DEFUSE_REQUIRED_TICKS
        + DEFUSE_SAFETY_MARGIN_TICKS
    )
    if orb_to_plant < 0 or total_required > env.detonate_timer:
        return None
    env.orb_teacher_opportunity = True
    if tuple(map(int, char.pos)) == orb:
        return ACTION_COLLECT_ORB if mask[ACTION_COLLECT_ORB] else None
    return shortest_legal_action(orb_map, char.pos, mask, MOVE_DELTAS)


def retake_teacher_utility_actor(env, allies, utility):
    """Choose a useful setup cast before entering the close combat area."""
    enemies = [enemy for enemy in env.attackers() if enemy.is_alive]
    if not enemies or not utility["ready"]:
        return None
    defuse_eta = min(
        max(0, path_distance(env.dist_map, ally) - 1) + DEFUSE_REQUIRED_TICKS
        for ally in allies
    )
    active_defuser = next(
        (ally for ally in allies if ally.name == env.active_defuser_name), None
    )
    if active_defuser is not None:
        defuse_eta = max(0, DEFUSE_REQUIRED_TICKS - active_defuser.defuse_timer)
    if env.detonate_timer <= defuse_eta:
        return None
    # A spike-centred smoke blocks the spike-centred flash LOS. Put the
    # information/flash effects in place first and smoke immediately after.
    for ability, active_key in (
        ("RECON", "reveal_active"),
        ("FLASH", "blind_active"),
        ("SMOKE", "smoke_active"),
    ):
        if utility[active_key] or ability in env.retake_utility_types_used:
            continue
        if ability == "FLASH" and not any(
            has_los(env.planted_pos, tuple(enemy.pos), env.smoke_cells())
            for enemy in enemies
        ):
            continue
        if ability == "RECON" and not any(
            max(abs(enemy.pos[0] - env.planted_pos[0]),
                abs(enemy.pos[1] - env.planted_pos[1])) <= RECON_REVEAL_SIZE // 2
            for enemy in enemies
        ):
            continue
        candidates = [
            ally for ally in utility["ready"]
            if ally.ability_name == ability and ally.own_ability_charge() > 0
            and ally.name != env.active_defuser_name
        ]
        if candidates:
            return min(candidates, key=lambda ally: (path_distance(env.dist_map, ally), ally.name))
    return None


def retake_teacher_action(env, char, mask):
    """Teacher for coordinated entry through distinct reachable approaches."""
    allies = [ally for ally in env.defenders() if ally.is_alive]
    if env.active_defuser_name is not None:
        if env.active_defuser_name == char.name and mask[ACTION_DEFUSE]:
            return ACTION_DEFUSE
        utility = retake_utility_state(
            allies, env.attackers(), env.planted_pos,
            env.smoke_cells(), UTILITY_SETUP_RADIUS, env.dist_map,
        )
        if retake_teacher_utility_actor(env, allies, utility) is char and mask[ACTION_ABILITY]:
            return ACTION_ABILITY
        return 4

    enemies_alive = any(enemy.is_alive for enemy in env.attackers())
    if not enemies_alive:
        adjacent = [
            ally for ally in allies
            if max(
                abs(ally.pos[0] - env.planted_pos[0]),
                abs(ally.pos[1] - env.planted_pos[1]),
            ) <= 1
        ]
        if adjacent:
            designated = min(adjacent, key=lambda ally: (path_distance(env.dist_map, ally), ally.name))
            if designated.name == char.name and mask[ACTION_DEFUSE]:
                return ACTION_DEFUSE
            return 4
        next_action = shortest_legal_action(env.dist_map, char.pos, mask, MOVE_DELTAS)
        return 4 if next_action is None else next_action

    utility_recent = env.tick <= env.retake_utility_push_until
    utility = retake_utility_state(
        allies, env.attackers(), env.planted_pos,
        env.smoke_cells(), UTILITY_SETUP_RADIUS, env.dist_map,
    )
    coordination = retake_coordination_state(
        char,
        allies,
        env.dist_map,
        env.detonate_timer,
        entry_radius=ENTRY_READY_RADIUS,
        defuse_ticks=DEFUSE_REQUIRED_TICKS,
        safety_margin=DEFUSE_SAFETY_MARGIN_TICKS,
    )
    utility_actor = retake_teacher_utility_actor(env, allies, utility)
    if utility_actor is char and mask[ACTION_ABILITY]:
        return ACTION_ABILITY
    if utility_actor is not None and coordination["self_distance"] <= UTILITY_SETUP_RADIUS:
        return 4
    if not utility_recent and not coordination["must_commit"]:
        lane_target = env.retake_lane_targets.get(char.name)
        if (
            lane_target is not None
            and tuple(map(int, char.pos)) != lane_target
            and coordination["self_distance"] >= ENTRY_READY_RADIUS
        ):
            lane_map = env.retake_lane_maps[lane_target]
            next_action = shortest_legal_action(
                lane_map, char.pos, mask, MOVE_DELTAS
            )
            if next_action is not None:
                return next_action
    if coordination["self_distance"] > ENTRY_READY_RADIUS:
        return shortest_legal_action(env.dist_map, char.pos, mask, MOVE_DELTAS)
    if coordination["should_wait"]:
        return 4

    if not coordination["must_commit"]:
        if not utility_recent and not any(
            utility[key] for key in ("smoke_active", "blind_active", "reveal_active")
        ):
            ultimate_options = sorted(
                (ally for ally in utility["ready"]
                 if ally.ultimate_cost > 0
                 and ally.ultimate_points >= ally.ultimate_cost),
                key=lambda ally: ally.name,
            )
            if ultimate_options:
                return (
                    ACTION_ULTIMATE
                    if ultimate_options[0].name == char.name and mask[ACTION_ULTIMATE]
                    else 4
                )

    designated = min(
        allies,
        key=lambda ally: (path_distance(env.dist_map, ally), ally.name),
    )
    if (
        designated.name == char.name
        and coordination["self_distance"] <= 1
        and mask[ACTION_DEFUSE]
    ):
        return ACTION_DEFUSE
    next_action = shortest_legal_action(env.dist_map, char.pos, mask, MOVE_DELTAS)
    return 4 if next_action is None else next_action


def optimize_demonstrations(
    net, optimizer, samples, weight=2.0, batch_size=64, ability_samples=None,
):
    if (not samples and not ability_samples) or weight <= 0:
        return None
    all_samples = list(samples)
    utility_samples = []
    ordinary_samples = []
    for sample in all_samples:
        (utility_samples if sample[1] == ACTION_ABILITY else ordinary_samples).append(
            sample
        )
    if ability_samples is not None:
        utility_samples = list(ability_samples)
    # Teacher rollouts contain many more movement/wait labels than ability
    # labels. Reserve one quarter of each minibatch for utility actions so
    # those demonstrations are not diluted by the common actions.
    target_size = min(batch_size, len(utility_samples) + len(ordinary_samples))
    if target_size == 0:
        return None
    utility_size = min(len(utility_samples), max(1, target_size // 4))
    selected_utility = random.sample(utility_samples, utility_size)
    batch = list(selected_utility)
    ordinary_size = min(len(ordinary_samples), target_size - utility_size)
    batch.extend(random.sample(ordinary_samples, ordinary_size))
    remaining = target_size - len(batch)
    if remaining:
        selected_ids = {id(sample) for sample in selected_utility}
        unused_utility = [
            sample for sample in utility_samples if id(sample) not in selected_ids
        ]
        batch.extend(random.sample(unused_utility, min(remaining, len(unused_utility))))
    states, actions, masks = zip(*batch)
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
    masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=DEVICE)
    q_values = net(states_t)
    competitor = q_values.masked_fill(~masks_t, -torch.inf)
    margin = torch.full_like(q_values, 0.8)
    margin.scatter_(1, actions_t[:, None], 0.0)
    loss = weight * (
        torch.max(competitor + margin, dim=1).values
        - q_values.gather(1, actions_t[:, None]).squeeze(1)
    ).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def scheduled_teacher_probability(
    episode, total_episodes, final=0.10, initial=0.80,
):
    fraction = min(1.0, max(0.0, (episode - 1) / max(1, total_episodes - 1)))
    return float(initial) + fraction * (float(final) - float(initial))


def optimize_facing(net, optimizer, samples, weight=0.35, batch_size=64):
    if not samples:
        return None
    batch = random.sample(list(samples), min(batch_size, len(samples)))
    states, actions, targets, confidences = zip(*batch)
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
    targets_t = torch.as_tensor(targets, dtype=torch.long, device=DEVICE)
    confidence_t = torch.as_tensor(confidences, dtype=torch.float32, device=DEVICE)
    losses = F.cross_entropy(net.facing_values(states_t, actions_t), targets_t, reduction="none")
    loss = weight * (losses * confidence_t).sum() / confidence_t.sum().clamp_min(1e-6)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def optimize_ultimate_classification(net, optimizer, positives, negatives, weight=0.25, batch_size=64):
    if not positives and not negatives:
        return None
    per_class = max(1, batch_size // 2)
    selected = []
    if positives:
        selected.extend((s, m, 1.0) for s, m in random.sample(list(positives), min(per_class, len(positives))))
    if negatives:
        selected.extend((s, m, 0.0) for s, m in random.sample(list(negatives), min(per_class, len(negatives))))
    random.shuffle(selected)
    states, masks, labels = zip(*selected)
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)
    masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=DEVICE)
    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=DEVICE)
    q_values = net(states_t)
    alternatives = masks_t.clone()
    alternatives[:, ACTION_ULTIMATE] = False
    competitor = q_values.masked_fill(~alternatives, -torch.inf).max(dim=1).values
    margin = q_values[:, ACTION_ULTIMATE] - competitor
    loss = weight * torch.where(
        labels_t > 0.5, torch.relu(0.8 - margin), torch.relu(0.8 + margin)
    ).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def run_episode(
    env, net, target_net, replay, epsilon, obs_dim,
    facing_samples=None, ultimate_positives=None, ultimate_negatives=None,
    teacher_samples=None, teacher_probability=0.0,
    ability_teacher_samples=None,
):
    env.reset()
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    zero_mask = np.zeros(N_ACTIONS, dtype=bool)
    episode_teacher_samples = []

    while True:
        terminal, reason = env.is_terminal()
        if terminal:
            break

        before = snapshot_before(env)
        use_team_teacher = (
            teacher_probability > 0.0 and random.random() < teacher_probability
        )

        obs_before, mask_before, chosen_actions = {}, {}, {}
        for char in env.defenders():
            if not char.is_alive:
                continue
            state = env.build_observation(char)
            mask = env.action_mask(char)
            action, facing = select_action(net, state, mask, epsilon)
            if teacher_probability > 0.0:
                teacher_action = retake_teacher_action(env, char, mask)
                if teacher_action == ACTION_ABILITY and ability_teacher_samples is not None:
                    # The teacher checked charge, setup distance, cast effect
                    # and time to defuse. Keep this local label even if the
                    # student's later movement causes the round to be lost.
                    ability_teacher_samples.append(
                        (state.copy(), ACTION_ABILITY, mask.copy())
                    )
                if (
                    teacher_action is not None
                    and use_team_teacher
                ):
                    action = int(teacher_action)
                    facing = select_facing(net, state, action, epsilon)
                    if teacher_samples is not None:
                        episode_teacher_samples.append(
                            (state.copy(), action, mask.copy())
                        )
            target = confidence = None
            in_combat = any(
                enemy.is_alive and env.check_line_of_sight(char, enemy)
                for enemy in env.attackers()
            )
            if in_combat:
                # Combat orientation is owned by the game, not the facing head.
                facing = getattr(char, "facing", "S")
            elif facing_samples is not None:
                target, confidence = observable_facing_target(env, char, action)
                facing = FACING_DIRS[target]
            char.facing = facing
            obs_before[char.name] = state
            mask_before[char.name] = mask
            chosen_actions[char.name] = action
            if facing_samples is not None and not in_combat:
                facing_samples.append((state.copy(), action, target, confidence))
            if mask[ACTION_ULTIMATE]:
                tactical = tactical_ultimate_window(
                    char, state[BASE_OBS_DIM:ULTIMATE_CONTEXT_OBS_DIM]
                )
                sample = (state.copy(), mask.copy())
                if tactical and ultimate_positives is not None:
                    ultimate_positives.append(sample)
                elif not tactical and ultimate_negatives is not None:
                    ultimate_negatives.append(sample)

        env.step_tick(chosen_actions)

        rewards = compute_rewards(env, before, chosen_actions)
        for name, action in chosen_actions.items():
            if action == ACTION_ULTIMATE:
                state = obs_before[name]
                tactical = tactical_ultimate_window(
                    next(char for char in env.defenders() if char.name == name),
                    state[BASE_OBS_DIM:ULTIMATE_CONTEXT_OBS_DIM],
                )
                rewards[name] = rewards.get(name, 0.0) + (0.10 if tactical else -0.08)

        terminal_now, reason_now = env.is_terminal()
        if terminal_now:
            outcome = DEFUSE_WIN_REWARD if reason_now == "defused" else LOSS_PENALTY
            for char in env.defenders():
                if char.is_alive:
                    rewards[char.name] = rewards.get(char.name, 0.0) + outcome

        for name, state in obs_before.items():
            char = next((c for c in env.chars if c.name == name), None)
            done = terminal_now or char is None or not char.is_alive
            if char is not None and char.is_alive:
                next_state = env.build_observation(char)
                next_mask = env.action_mask(char)
            else:
                next_state = zero_obs
                next_mask = zero_mask
            replay.push(
                state,
                chosen_actions[name],
                rewards.get(name, 0.0),
                next_state,
                float(done),
                mask_before[name],
                next_mask,
            )

    # A route or ability choice from a failed retake is not a proven
    # demonstration. Keep the transition in replay with its actual reward,
    # but supervise the policy only with successful complete retakes.
    if env.is_defused and teacher_samples is not None:
        teacher_samples.extend(episode_teacher_samples)
    return env.is_defused, env.tick


def train_step(
    net, target_net, optimizer, replay, batch_size, gamma,
    anchor_replay=None, reference_net=None,
):
    if len(replay) < batch_size:
        return None
    anchor_size = (
        batch_size // 4
        if anchor_replay is not None and len(anchor_replay) >= batch_size // 4
        else 0
    )
    batch = replay.sample(batch_size - anchor_size)
    if anchor_size:
        anchor_batch = anchor_replay.sample(anchor_size)
        batch = Transition(
            *(current + anchor for current, anchor in zip(batch, anchor_batch))
        )
    loss = compute_td_loss(net, target_net, batch, gamma)
    if reference_net is not None:
        states = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE)
        masks = torch.from_numpy(np.stack(batch.mask)).bool().to(DEVICE)
        current_q = net(states)
        with torch.no_grad():
            reference_q = reference_net(states)
        counts = masks.sum(dim=1, keepdim=True).clamp_min(1)
        current_advantage = current_q - (current_q * masks).sum(
            dim=1, keepdim=True
        ) / counts
        reference_advantage = reference_q - (reference_q * masks).sum(
            dim=1, keepdim=True
        ) / counts
        # Keep the incumbent policy as a light anchor, not a hard barrier to
        # learning the teacher-guided retake sequence.
        loss = loss + 1.0 * F.smooth_l1_loss(
            current_advantage[masks], reference_advantage[masks]
        )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def expand_retake_policy_state(checkpoint, net):
    """Preserve existing behavior when appending observation/action features."""
    source = checkpoint.get("model_state_dict", checkpoint)
    target = net.state_dict()
    old_obs = int(source["feature.0.weight"].shape[1])
    old_actions = int(source["adv_head.2.weight"].shape[0])
    new_obs = int(target["feature.0.weight"].shape[1])
    new_actions = int(target["adv_head.2.weight"].shape[0])
    if old_obs > new_obs or old_actions > new_actions:
        raise ValueError(
            f"retake checkpoint is larger than model: "
            f"obs {old_obs}->{new_obs}, actions {old_actions}->{new_actions}"
        )
    for key, value in source.items():
        if key not in target:
            continue
        if target[key].shape == value.shape:
            target[key] = value.detach().clone()
        elif key in ("feature.0.weight", "facing_feature.0.weight"):
            if (
                value.ndim != 2
                or value.shape[0] != target[key].shape[0]
                or value.shape[1] > target[key].shape[1]
            ):
                raise ValueError(f"incompatible retake input layer: {key}")
            expanded = torch.zeros_like(target[key])
            expanded[:, :value.shape[1]] = value
            target[key] = expanded
        elif key in ("adv_head.2.weight", "adv_head.2.bias"):
            if (
                value.shape[0] > target[key].shape[0]
                or value.shape[1:] != target[key].shape[1:]
            ):
                raise ValueError(f"incompatible retake action layer: {key}")
            expanded = target[key].clone()
            expanded[:value.shape[0]] = value
            target[key] = expanded
        elif key == "facing_output.weight":
            hidden = target[key].shape[1] - new_actions
            old_hidden = value.shape[1] - old_actions
            if value.shape[0] != target[key].shape[0] or old_hidden != hidden:
                raise ValueError(f"incompatible retake facing output: {key}")
            expanded = torch.zeros_like(target[key])
            expanded[:, :hidden] = value[:, :hidden]
            expanded[:, hidden : hidden + old_actions] = value[:, hidden:]
            target[key] = expanded
        else:
            raise ValueError(
                f"unsupported retake checkpoint shape: {key} "
                f"{tuple(value.shape)} -> {tuple(target[key].shape)}"
            )
    net.load_state_dict(target)
    return old_obs, old_actions


def train_only_new_retake_context(net, old_obs):
    """Protect the learned action policy while fitting appended inputs."""
    for name, parameter in net.named_parameters():
        if name.startswith(("feature.", "value_head.", "adv_head.")):
            parameter.requires_grad_(name == "feature.0.weight")
    net.feature[0].weight.register_hook(
        lambda gradient: torch.cat(
            (torch.zeros_like(gradient[:, :old_obs]), gradient[:, old_obs:]), dim=1
        )
    )


def checkpoint_payload(net, episode, metrics=None, protected_obs_dim=None):
    payload = {
        "model_state_dict": net.state_dict(),
        "episode": int(episode),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "facing_head_version": FACING_HEAD_VERSION,
        "training_revision": TRAINING_REVISION,
        "evaluation": metrics or {},
    }
    if protected_obs_dim is not None:
        payload["protected_obs_dim"] = protected_obs_dim
    return payload


def has_retake_crossfire(env):
    """Whether two defenders see one attacker from different sides of the spike."""
    defenders = [unit for unit in env.defenders() if unit.is_alive]
    for enemy in env.attackers():
        if not enemy.is_alive:
            continue
        sectors = {
            retake_approach_sector(unit.pos, env.planted_pos)
            for unit in defenders
            if env.check_line_of_sight(unit, enemy)
        }
        sectors.discard("center")
        if len(sectors) >= 2:
            return True
    return False


def evaluate_retake_fixed(net, episodes_per_seed=30, seeds=(3026091700, 5026091700, 6026091700)):
    was_training = net.training
    net.eval()
    rows = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        env = RetakeEnv(
            min_detonate_ticks=15,
            max_detonate_ticks=SPIKE_DETONATION_TICKS,
            attacker_hold_radius=4,
        )
        counts = {"defused": 0, "detonated": 0, "defenders_wiped": 0, "timeout": 0}
        entered = 0
        crossfire = 0
        ability_uses = 0
        multi_utility = 0
        for _ in range(episodes_per_seed):
            env.reset()
            entered_this_round = False
            crossfire_this_round = False
            while True:
                terminal, reason = env.is_terminal()
                if terminal:
                    counts[reason] = counts.get(reason, 0) + 1
                    break
                crossfire_this_round |= has_retake_crossfire(env)
                actions = {}
                for char in env.defenders():
                    if not char.is_alive:
                        continue
                    state = env.build_observation(char)
                    action, facing = select_action(net, state, env.action_mask(char), 0.0)
                    char.facing = facing
                    actions[char.name] = action
                    entered_this_round |= tuple(char.pos) in env.site_zone
                env.step_tick(actions)
            entered += int(entered_this_round)
            crossfire += int(crossfire_this_round)
            ability_uses += env.ability_uses
            multi_utility += int(env.ability_uses >= 2)
        total = float(episodes_per_seed)
        rows.append(
            {
                "seed": seed,
                "defuse_rate": counts["defused"] / total,
                "entry_rate": entered / total,
                "crossfire_rate": crossfire / total,
                "ability_uses_per_round": ability_uses / total,
                "multi_utility_rate": multi_utility / total,
                "wipe_rate": counts["defenders_wiped"] / total,
                "timeout_rate": counts["timeout"] / total,
            }
        )
    if was_training:
        net.train()
    return {
        "episodes": episodes_per_seed * len(seeds),
        "seeds": list(seeds),
        "defuse_rate": float(np.mean([row["defuse_rate"] for row in rows])),
        "worst_defuse_rate": min(row["defuse_rate"] for row in rows),
        "entry_rate": float(np.mean([row["entry_rate"] for row in rows])),
        "crossfire_rate": float(np.mean([row["crossfire_rate"] for row in rows])),
        "ability_uses_per_round": float(np.mean([
            row["ability_uses_per_round"] for row in rows
        ])),
        "multi_utility_rate": float(np.mean([
            row["multi_utility_rate"] for row in rows
        ])),
        "worst_entry_rate": min(row["entry_rate"] for row in rows),
        "wipe_rate": float(np.mean([row["wipe_rate"] for row in rows])),
        "timeout_rate": float(np.mean([row["timeout_rate"] for row in rows])),
        "per_seed": rows,
    }


def verify_expanded_retake_baseline(net, checkpoint_evaluation, episodes_per_seed):
    """Catch a broken warm start before spending hundreds of training rounds."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        metrics = evaluate_retake_fixed(net, episodes_per_seed=episodes_per_seed)
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    print(
        "[WARM START EVAL] "
        f"defuse={metrics['defuse_rate']:.3f} "
        f"entry={metrics['entry_rate']:.3f} "
        f"worst_defuse={metrics['worst_defuse_rate']:.3f} "
        f"abilities={metrics['ability_uses_per_round']:.2f} "
        f"multi_utility={metrics['multi_utility_rate']:.3f}"
    )
    comparable = (
        checkpoint_evaluation.get("episodes") == metrics["episodes"]
        and checkpoint_evaluation.get("seeds") == metrics["seeds"]
        and "defuse_rate" in checkpoint_evaluation
        and "entry_rate" in checkpoint_evaluation
    )
    if comparable and (
        metrics["defuse_rate"] < checkpoint_evaluation["defuse_rate"] - 0.05
        or metrics["entry_rate"] < checkpoint_evaluation["entry_rate"] - 0.05
    ):
        raise RuntimeError(
            "expanded retake warm start regressed before training: "
            f"defuse {checkpoint_evaluation['defuse_rate']:.3f} -> "
            f"{metrics['defuse_rate']:.3f}, entry "
            f"{checkpoint_evaluation['entry_rate']:.3f} -> "
            f"{metrics['entry_rate']:.3f}; best checkpoint was not changed"
        )
    return metrics


def retake_selection_score(metrics):
    return (
        metrics["defuse_rate"],
        metrics["worst_defuse_rate"],
        metrics.get("multi_utility_rate", 0.0),
        metrics.get("crossfire_rate", 0.0),
        metrics["entry_rate"],
        metrics["worst_entry_rate"],
        -metrics["wipe_rate"],
        -metrics["timeout_rate"],
    )


def archive_best_checkpoint():
    if not BEST_MODEL_PATH.exists():
        return None
    RETAKE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RETAKE_ARCHIVE_DIR / (
        f"dqn_defender_retake_gc_{time.time_ns()}.pt"
    )
    shutil.copy2(BEST_MODEL_PATH, archive_path)
    return archive_path


def scheduled_learning_rate(episode, initial_lr, decay_start=1000, decay_end=2500, minimum_scale=0.20):
    if episode <= decay_start:
        return initial_lr
    fraction = min(1.0, (episode - decay_start) / max(1, decay_end - decay_start))
    return initial_lr * (1.0 - fraction * (1.0 - minimum_scale))


def exploration_epsilon(episode, start_episode, total_episodes, warm_start):
    if warm_start:
        elapsed = max(0, episode - start_episode - 1)
        return 0.02 + 0.06 * max(0.0, 1.0 - elapsed / 1000)
    decay_episodes = max(1, int(total_episodes * 0.8))
    return 0.02 + 0.98 * max(0.0, 1.0 - episode / decay_episodes)


def collect_successful_baseline_replay(net, obs_dim, target_wins=20, max_episodes=80):
    """Keep successful old-policy retakes available during fine-tuning."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    was_training = net.training
    baseline = ReplayBuffer(capacity=100_000)
    wins = 0
    attempts = 0
    try:
        random.seed(20260925)
        np.random.seed(20260925)
        net.eval()
        env = RetakeEnv(
            min_detonate_ticks=15,
            max_detonate_ticks=SPIKE_DETONATION_TICKS,
            attacker_hold_radius=4,
        )
        for attempts in range(1, max_episodes + 1):
            episode_replay = ReplayBuffer()
            defused, _ticks = run_episode(
                env, net, None, episode_replay, 0.0, obs_dim
            )
            if defused:
                baseline.buffer.extend(episode_replay.buffer)
                wins += 1
                if wins >= target_wins:
                    break
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        net.train(was_training)
    if wins == 0:
        raise RuntimeError("warm-start policy produced no successful retake episodes")
    print(
        f"[BASELINE REPLAY] wins={wins}/{attempts} "
        f"transitions={len(baseline)}"
    )
    return baseline


def main(
    episodes=EPISODE_COUNT,
    eval_every=None,
    eval_episodes_per_seed=30,
    learning_rate=None,
    early_stop_patience=0,
    fresh_start=False,
):
    print(f"[INIT] device = {DEVICE}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    env = RetakeEnv(
        min_detonate_ticks=15,
        max_detonate_ticks=SPIKE_DETONATION_TICKS,
        attacker_hold_radius=4,
    )
    env.reset()
    sample_char = env.defenders()[0]
    obs_dim = env.build_observation(sample_char).shape[0]
    print(f"[INIT] obs_dim = {obs_dim} / n_actions = {N_ACTIONS}")

    net = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
    target_net = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
    checkpoint = None
    start_episode = 0
    best_score = None
    baseline_evaluation = {}
    protected_obs_dim = None
    candidate_paths = []
    if fresh_start:
        print("[FRESH START] model weights will be randomly initialized")
    else:
        if BEST_MODEL_PATH.exists():
            candidate_paths.append(BEST_MODEL_PATH)
        candidate_paths.extend(sorted(PROMOTION_BACKUP_DIR.glob(
            "*/dqn_defender_retake_gc_best_by_eval.pt"
        )))
    selected_path = None
    current_score = None
    incumbent_floor_score = None
    if fresh_start and BEST_MODEL_PATH.exists():
        old_checkpoint = torch.load(
            BEST_MODEL_PATH, map_location=DEVICE, weights_only=False
        )
        old_evaluation = old_checkpoint.get("evaluation") or {}
        expected_episodes = eval_episodes_per_seed * 3
        if (
            old_checkpoint.get("training_revision") == TRAINING_REVISION
            and old_evaluation.get("episodes") == expected_episodes
            and old_evaluation.get("seeds") == [3026091700, 5026091700, 6026091700]
            and "multi_utility_rate" in old_evaluation
        ):
            incumbent_floor_score = retake_selection_score(old_evaluation)
        else:
            incumbent_net = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
            expand_retake_policy_state(old_checkpoint, incumbent_net)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            try:
                incumbent_metrics = evaluate_retake_fixed(
                    incumbent_net, episodes_per_seed=eval_episodes_per_seed
                )
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            incumbent_floor_score = retake_selection_score(incumbent_metrics)
        print(
            "[FRESH START FLOOR] existing best remains protected until beaten; "
            f"defuse={incumbent_floor_score[0]:.3f} "
            f"worst_defuse={incumbent_floor_score[1]:.3f}"
        )
    for candidate_path in candidate_paths:
        try:
            candidate_checkpoint = torch.load(
                candidate_path, map_location=DEVICE, weights_only=False
            )
            candidate_net = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
            old_obs, old_actions = expand_retake_policy_state(
                candidate_checkpoint, candidate_net
            )
            print(
                f"[WARM START CANDIDATE] {candidate_path} "
                f"obs={old_obs} actions={old_actions}"
            )
            evaluation = candidate_checkpoint.get("evaluation") or {}
            revision = candidate_checkpoint.get("training_revision")
            comparable = evaluation if revision == TRAINING_REVISION else {}
            if revision != TRAINING_REVISION:
                print(
                    f"[BASELINE REVISION] {revision!r} -> "
                    f"{TRAINING_REVISION}; using fresh evaluation as baseline"
                )
            metrics = verify_expanded_retake_baseline(
                candidate_net, comparable, eval_episodes_per_seed
            )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            if candidate_path == BEST_MODEL_PATH:
                raise
            print(f"[WARM START SKIP] {candidate_path}: {exc}")
            continue
        score = retake_selection_score(metrics)
        if candidate_path == BEST_MODEL_PATH:
            current_score = score
            if (
                revision != TRAINING_REVISION
                and evaluation.get("episodes") == metrics["episodes"]
                and evaluation.get("seeds") == metrics["seeds"]
                and "defuse_rate" in evaluation
                and "worst_defuse_rate" in evaluation
            ):
                # Do not overwrite a stronger prior revision simply because
                # the changed action mask lowered its fresh warm-start score.
                incumbent_floor_score = retake_selection_score(evaluation)
                print(
                    "[INCUMBENT FLOOR] "
                    f"defuse={evaluation['defuse_rate']:.3f} "
                    f"worst_defuse={evaluation['worst_defuse_rate']:.3f}"
                )
        if best_score is None or score > best_score:
            best_score = score
            baseline_evaluation = metrics
            checkpoint = candidate_checkpoint
            net = candidate_net
            selected_path = candidate_path
            start_episode = int(candidate_checkpoint.get("episode", 0))
    if candidate_paths and selected_path is None:
        raise RuntimeError("no usable defender-retake warm-start checkpoint")
    if selected_path is not None:
        print(
            f"[WARM START SELECTED] {selected_path} episode={start_episode} "
            f"defuse={baseline_evaluation['defuse_rate']:.3f} "
            f"worst_defuse={baseline_evaluation['worst_defuse_rate']:.3f}"
        )
        if selected_path != BEST_MODEL_PATH and (
            current_score is None or best_score > current_score
        ) and (
            incumbent_floor_score is None or best_score > incumbent_floor_score
        ):
            archive_path = archive_best_checkpoint()
            if archive_path is not None:
                print(f"[WARM START ARCHIVE] {archive_path}")
            torch.save(
                checkpoint_payload(net, start_episode, baseline_evaluation),
                BEST_MODEL_PATH,
            )
            print(f"[WARM START RECOVERED] {BEST_MODEL_PATH}")
        # Expansion starts with zero weights on new features. Subsequent
        # updates train the complete policy so movement and defuse can change.
    evaluation_interval = (
        int(eval_every) if eval_every is not None
        else (100 if checkpoint is not None else 250)
    )
    if evaluation_interval <= 0:
        raise ValueError("eval_every must be positive")
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    best_train_state = copy.deepcopy(net.state_dict())
    best_train_episode = start_episode
    best_train_metrics = baseline_evaluation

    initial_lr = float(
        learning_rate
        if learning_rate is not None
        else 1e-5
    )
    print(
        f"[INIT] learning_rate={initial_lr:.2e} "
        f"start_episode={start_episode} eval_every={evaluation_interval}"
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in net.parameters() if parameter.requires_grad),
        lr=initial_lr,
    )
    replay = ReplayBuffer(capacity=100_000)
    reference_net = copy.deepcopy(net).eval() if checkpoint is not None else None
    if reference_net is not None:
        for parameter in reference_net.parameters():
            parameter.requires_grad_(False)
    anchor_replay = (
        collect_successful_baseline_replay(net, obs_dim)
        if checkpoint is not None else None
    )
    facing_samples = deque(maxlen=50_000)
    ultimate_positive_samples = deque(maxlen=20_000)
    ultimate_negative_samples = deque(maxlen=20_000)
    teacher_samples = deque(maxlen=50_000)
    ability_teacher_samples = deque(maxlen=10_000)

    num_episodes = episodes
    batch_size = 256
    gamma = 0.99
    global_step = 0
    win_history = deque(maxlen=200)
    no_improvement_evals = 0
    last_episode = start_episode

    start_time = time.perf_counter()
    for episode in range(start_episode + 1, num_episodes + 1):
        last_episode = episode
        if checkpoint is not None:
            current_lr = scheduled_learning_rate(
                episode,
                initial_lr,
                decay_start=start_episode + 2000,
                decay_end=start_episode + 8000,
                minimum_scale=0.5,
            )
        else:
            current_lr = scheduled_learning_rate(episode, initial_lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        epsilon = exploration_epsilon(
            episode, start_episode, num_episodes, checkpoint is not None
        )
        if checkpoint is not None:
            # Continuing a working policy with 75% teacher overrides makes
            # most experience come from the teacher instead of that policy.
            teacher_probability = scheduled_teacher_probability(
                episode - start_episode,
                max(1, num_episodes - start_episode),
                initial=0.35,
                final=0.10,
            )
        else:
            teacher_probability = scheduled_teacher_probability(
                episode, num_episodes, final=0.15
            )

        defused, ticks_used = run_episode(
            env,
            net,
            target_net,
            replay,
            epsilon,
            obs_dim,
            facing_samples,
            ultimate_positive_samples,
            ultimate_negative_samples,
            teacher_samples,
            teacher_probability,
            ability_teacher_samples,
        )
        win_history.append(1 if defused else 0)

        for _ in range(max(1, ticks_used)):
            loss = train_step(
                net, target_net, optimizer, replay, batch_size, gamma,
                anchor_replay=anchor_replay, reference_net=reference_net,
            )
            if loss is not None:
                soft_update(target_net, net)
            if global_step % 4 == 0:
                optimize_facing(net, optimizer, facing_samples)
            if global_step % 4 == 0:
                optimize_demonstrations(
                    net, optimizer, teacher_samples, weight=1.0, batch_size=32,
                    ability_samples=ability_teacher_samples,
                )
            if global_step % 8 == 0:
                optimize_ultimate_classification(
                    net,
                    optimizer,
                    ultimate_positive_samples,
                    ultimate_negative_samples,
                )
            global_step += 1

        if episode % evaluation_interval == 0:
            metrics = evaluate_retake_fixed(
                net, episodes_per_seed=eval_episodes_per_seed
            )
            score = retake_selection_score(metrics)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter()
            ability_demos = len(ability_teacher_samples)

            print(
                f"[FIXED-SEED EVAL EP {episode}/{num_episodes}] "
                f"defuse={metrics['defuse_rate']:.3f} "
                f"worst_defuse={metrics['worst_defuse_rate']:.3f} "
                f"entry={metrics['entry_rate']:.3f} "
                f"worst_entry={metrics['worst_entry_rate']:.3f} "
                f"crossfire={metrics['crossfire_rate']:.3f} "
                f"abilities={metrics['ability_uses_per_round']:.2f} "
                f"multi_utility={metrics['multi_utility_rate']:.3f} "
                f"wipe={metrics['wipe_rate']:.3f} "
                f"teacher={teacher_probability:.3f} epsilon={epsilon:.3f} "
                f"winning_demos={len(teacher_samples)} "
                f"ability_demos={ability_demos} "
                f"lr={current_lr:.2e} "
                f"elapsed={elapsed_time:.1f}s"
            )
            if episode - start_episode >= 1000 and ability_demos < 32:
                raise RuntimeError(
                    "utility teacher produced fewer than 32 usable ability labels "
                    "after 1000 episodes; training stopped before spending more "
                    "episodes on an empty utility demonstration buffer; "
                    "saved best checkpoints were not changed by this check"
                )
            if (
                (best_score is None or score > best_score)
                and (
                    incumbent_floor_score is None
                    or score > incumbent_floor_score
                )
            ):
                best_score = score
                no_improvement_evals = 0
                best_train_state = copy.deepcopy(net.state_dict())
                best_train_episode = episode
                best_train_metrics = metrics
                if reference_net is not None:
                    reference_net.load_state_dict(best_train_state)
                archive_path = archive_best_checkpoint()
                if archive_path is not None:
                    print(f"  -> archived previous best: {archive_path}")
                torch.save(
                    checkpoint_payload(net, episode, metrics, protected_obs_dim),
                    BEST_MODEL_PATH,
                )
                print(f"  -> fixed-seed winner updated: {BEST_MODEL_PATH}")
            else:
                no_improvement_evals += 1
                patience = (
                    f"/{early_stop_patience}" if early_stop_patience > 0 else ""
                )
                print(
                    f"  -> no improvement: {no_improvement_evals}{patience}; "
                    "best checkpoint kept, training continues"
                )
                if (
                    early_stop_patience > 0
                    and no_improvement_evals >= early_stop_patience
                ):
                    print(
                        f"[EARLY STOP] no fixed-seed improvement for "
                        f"{no_improvement_evals} evaluations"
                    )
                    break

    if (
        incumbent_floor_score is None
        or (best_score is not None and best_score > incumbent_floor_score)
    ):
        net.load_state_dict(best_train_state)
        torch.save(
            checkpoint_payload(
                net, best_train_episode, best_train_metrics, protected_obs_dim
            ),
            FINAL_MODEL_PATH,
        )
    else:
        print("[FINAL] incumbent was not beaten; final checkpoint kept")
    print("[DONE] training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train current GC defender-retake policy")
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes-per-seed", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop after this many evaluations without improvement (0 disables).",
    )
    parser.add_argument(
        "--fresh-start", action="store_true",
        help="Ignore all warm-start checkpoints and train from random weights.",
    )
    args = parser.parse_args()
    main(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes_per_seed=args.eval_episodes_per_seed,
        learning_rate=args.learning_rate,
        early_stop_patience=args.early_stop_patience,
        fresh_start=args.fresh_start,
    )
