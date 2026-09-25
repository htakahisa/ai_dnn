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

import argparse
import math
import random
import sys
import os
from collections import deque, namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ov1_map_data_retake import RETAKE_MAZE_STR
from ov1_map_data_retake_ability import RETAKE_MAZE_STR as RETAKE_ABILITY_MAZE_STR

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
    FACING_VECTORS,
    SHOOTING_SITE_DIGREE,
)

from character_stats import CHARACTER_TABLE as STATS_TABLE
import ov1_common_rl
from ov1_common_rl import DEVICE, DuelingQNet, ReplayBuffer, select_action, soft_update
from ov1_common_defender import ROSTER_ORDER, ROLE_TO_ABILITY, compute_effective_stats
from ov1_ultimate_training import (
    ORB_COLLECT_REQUIRED_TICKS,
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

EPISODE_COUNT = 4000
EVAL_MIN_EPISODE = int(EPISODE_COUNT * 0.7)
DEFUSE_LERNING_EPISODE_COUNT = EPISODE_COUNT * 0.3  #解除学習を強制するエピソード。減衰させ最後は0。

SAVE_DIR = "data/defender_retake_data"

# ============================================================
# マップ読み込み
# ============================================================

def _parse_retake_map(maze_str, marker_terrain):
    rows = [row.strip() for row in maze_str.strip().splitlines() if row.strip()]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Retake map rows must have a consistent width")

    grid = np.zeros((len(rows), len(rows[0])), dtype=np.int32)
    markers = {}
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell.isdigit():
                value = int(cell)
                if value == 5:
                    markers.setdefault("5", []).append((r, c))
                grid[r, c] = 0 if value in (3, 4, 5) else value
            elif cell in marker_terrain:
                grid[r, c] = marker_terrain[cell]
                markers.setdefault(cell, []).append((r, c))
            else:
                raise ValueError(f"Unsupported retake map marker {cell!r} at {(r, c)}")
    return grid, markers


GRID, RETAKE_MARKERS = _parse_retake_map(
    RETAKE_MAZE_STR, {"A": 0, "B": 0, "T": 0, "a": 2, "b": 2}
)
HEIGHT, WIDTH = GRID.shape
SITE_BOUNDARY_COL = WIDTH // 2
WALKABLE = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if GRID[r, c] != 1]
DEFENDER_SPAWNS = RETAKE_MARKERS.get("5", [])
ATTACKER_SPAWNS = {
    "left": RETAKE_MARKERS.get("A", []),
    "right": RETAKE_MARKERS.get("B", []),
}
PLANT_POSITIONS = {
    "left": RETAKE_MARKERS["a"][0],
    "right": RETAKE_MARKERS["b"][0],
}
ENTRY_POINTS = {
    "left": [p for p in RETAKE_MARKERS.get("T", []) if p[1] < SITE_BOUNDARY_COL],
    "right": [p for p in RETAKE_MARKERS.get("T", []) if p[1] >= SITE_BOUNDARY_COL],
}


def _parse_ability_markers(maze_str):
    rows = [row.strip() for row in maze_str.strip().splitlines() if row.strip()]
    if len(rows) != HEIGHT or any(len(row) != WIDTH for row in rows):
        raise ValueError("Retake ability map must match the retake map dimensions")
    points = {"SMOKE": [], "FLASH": [], "RECON": []}
    marker_abilities = {"S": "SMOKE", "F": "FLASH", "R": "RECON"}
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell in marker_abilities:
                if GRID[r, c] == 1:
                    raise ValueError(f"Ability marker {cell} is on a wall at {(r, c)}")
                points[marker_abilities[cell]].append((r, c))
            elif not cell.isdigit():
                raise ValueError(f"Unsupported ability map marker {cell!r} at {(r, c)}")
    if not all(points.values()):
        raise ValueError("Retake ability map must contain S, F, and R markers")
    return points


ABILITY_TARGET_POINTS = _parse_ability_markers(RETAKE_ABILITY_MAZE_STR)
ABILITY_TARGET_RANGE = 6

if len(DEFENDER_SPAWNS) < 5:
    raise ValueError("Retake map must provide at least five defender start cells marked 5")
if len(ATTACKER_SPAWNS["left"]) < 5 or len(ATTACKER_SPAWNS["right"]) < 5:
    raise ValueError("Retake map must provide at least five A and five B enemy start cells")
if len(RETAKE_MARKERS.get("a", [])) != 1 or len(RETAKE_MARKERS.get("b", [])) != 1:
    raise ValueError("Retake map must contain exactly one a and one b spike location")
if not ENTRY_POINTS["left"] or not ENTRY_POINTS["right"]:
    raise ValueError("Retake map must provide at least one T entry point on each site")

# 💡追加: 左右サイトのプラント位置サンプリング用。
# 既知プラント位置(実際に攻撃側が狙いやすい座標)を優先的に学習させつつ、
# 汎化のためサイト範囲内のランダムPLANT_CELLSも一定割合混ぜる。
# 境界はWIDTH//2(col基準)。実際のサイト境界とズレる場合は要調整。
# 敵の侵入経路(通路・角)として既知の座標。carry側のSMOKE_LINEUP_CELLS_BY_SITEと
# 同じ考え方: 実際に視認していなくても「ここから敵が来るはず」という構造的な
# 予測情報として使う。facing整合shapingと、アビリティの無効射撃ペナルティ免除
# (事前投げの正当化)の両方に使う。未設定(空リスト)の間は発火しない。
# マップを見ながら座標を埋めること(サイト判定はplanted_posの列基準で自動)。


# CLIから上書き可能(main()内でargparseにより再代入)
SITE_LEFT_PROB = 0.5


# ============================================================
# 行動空間
# ============================================================

N_ACTIONS = 13
MOVE_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
ACTION_DEFUSE = 5
ACTION_ABILITY = 6
TURN_DIRS = ["N", "S", "E", "W"]
ACTION_TURN_BASE = 7  # 7,8,9,10 = N/S/E/W への向き変更のみ(移動・DEFUSE・ABILITYなし)
ACTION_ULTIMATE = 11
ACTION_ORB = 12
# 観測用。被弾時の強制向き(_facing_towards)は斜め8方向を返しうるため、
# 行動としてのTURN(4方向)とは別に、観測エンコードは8方向で持つ。
# game_core.FACING_VECTORSの定義順(N,NE,E,SE,S,SW,W,NW)と一致させること。
ALL_FACINGS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SITE_ZONE_RADIUS = 6
ENTRY_READY_RADIUS = 3
MIN_ALLIES_FOR_ENTRY = 1
SMOKE_READY_BFS_RADIUS = ABILITY_TARGET_RANGE   # スモークを撃つ距離の最大距離
ROLE_INDEX = {"フラッシュ": 0, "スモーカー": 1, "シーカー": 2, "タイガー": 3}

# 💡追加: 「解除が安全かどうか」の判定を、起爆タイマーの割合(detonate_frac)ではなく
# 「解除完了に最低限必要なtick数からの絶対的な残り時間」で判定するための定数。
DEFUSE_SAFETY_MARGIN_TICKS = 4
ENTRY_SAFETY_MARGIN_TICKS = DEFUSE_SAFETY_MARGIN_TICKS + ENTRY_READY_RADIUS


# ============================================================
# 報酬設計
# ============================================================

APPROACH_REWARD_SCALE = 0.20
APPROACH_WRONG_DIRECTION_PENALTY = -0.35
ENTRY_WITH_SUPPORT_BONUS = 1.5
ENTRY_ALONE_PENALTY = -0.3
ENTRY_ALONE_LINGER_PENALTY = -0.15
ALLY_GATHER_REWARD = 0.1
ABILITY_GOOD_USE_BONUS = 0.4
ABILITY_PREMATURE_PENALTY = -0.5
# 💡追加: タイミングは適切でも「誰にも効果がなかった」場合のペナルティと、
# ability種別ごとの効果量に応じた追加ボーナス。
ABILITY_NO_EFFECT_PENALTY = -0.3
FLASH_EFFECT_BONUS_PER_ENEMY = 0.5
RECON_EFFECT_BONUS_PER_ENEMY = 0.3
SMOKE_EFFECT_BONUS_PER_BLOCKED = 0.4
SMOKE_EARLY_CAST_BONUS = 1.2
# 💡追加: 既知の侵入経路(KNOWN_ENTRY_POINTS_*)付近からの投げは、SMOKEと同様に
# 「即時効果が0でも無効射撃ペナルティを免除」し、さらに事前投げ自体への
# 小さなボーナスを与える。この免除がこれまでFLASH/RECONに無かったことが、
# SMOKEだけ事前投げを覚えてFLASH/RECONが覚えなかった主因。
SIGHTING_STALENESS_CAP = 20         # チーム共有の目撃情報を保持する最大tick数(carry/escort/guardと同一方針)
TEAM_SIGHTING_ALIGN_WEIGHT = 0.02   # 自分が直接視認していない時のみ有効。チーム共有の目撃位置を向くほど+
CORRIDOR_WATCH_ALIGN_WEIGHT = 0.02  # 敵の目撃情報(自分・チーム共有とも)が無い時のみ有効。
                                     # 最寄りの既知侵入経路を向くほど+
SMOKE_COVER_DEFUSE_COMPLETE_BONUS = 2.0
SMOKE_COVER_SPIKE_BONUS = 1.0  # 解除完了の瞬間、スモークに覆われていれば加算
SMOKE_COVER_MIN_REMAIN_TICKS = DEFUSE_REQUIRED_TICKS - DEFUSE_SAFETY_MARGIN_TICKS  # 開始時に要求する最低スモーク残りtick
DAMAGE_REWARD_SCALE = 0.01
DEBUFF_HIT_MULTIPLIER = 1.5
KILL_REWARD = 3.0
KILL_ON_DEBUFFED_BONUS = 1.5
DEATH_PENALTY = -3.0
DEFUSE_PROGRESS_REWARD = 0.3
UNSAFE_DEFUSE_PENALTY = -0.5
DEFUSE_WIN_REWARD = 10.0
LOSS_PENALTY = -10.0
TICK_TIME_PENALTY = -0.01


def _sample_planted_pos():
    side = "left" if random.random() < SITE_LEFT_PROB else "right"
    return side, PLANT_POSITIONS[side]


def _jitter_spawn(base_pos, occupied):
    """Randomize a marked enemy start by at most one cell in either axis."""
    br, bc = base_pos
    candidates = [
        (br + dr, bc + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if _in_bounds((br + dr, bc + dc))
        and GRID[br + dr, bc + dc] != 1
        and (br + dr, bc + dc) not in occupied
    ]
    if not candidates:
        raise ValueError(f"No free jitter cell around retake enemy start {base_pos}")
    return random.choice(candidates)


def _select_enemy_starts(side, count, occupied):
    candidates = ATTACKER_SPAWNS[side]
    if len(candidates) < count:
        raise ValueError(f"Retake map has fewer {side} enemy starts than requested players")
    # A candidate can become unusable after earlier jitter choices. Retry the
    # candidate selection as a group so valid layouts are re-drawn on collision.
    for _ in range(100):
        selected = random.sample(candidates, count)
        starts = []
        reserved = set(occupied)
        try:
            for base_pos in selected:
                pos = _jitter_spawn(base_pos, reserved)
                reserved.add(pos)
                starts.append(pos)
        except ValueError:
            continue
        occupied.update(starts)
        return starts
    raise ValueError(f"Could not place {count} jittered enemies for {side} retake")


def _line_cells(p1, p2):
    r0, c0 = map(int, p1)
    r1, c1 = map(int, p2)
    dx, dy = abs(c1 - c0), -abs(r1 - r0)
    sx = 1 if c0 < c1 else -1
    sy = 1 if r0 < r1 else -1
    error = dx + dy
    while True:
        yield r0, c0
        if (r0, c0) == (r1, c1):
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            c0 += sx
        if twice <= dx:
            error += dx
            r0 += sy


def has_wall_los(grid, p1, p2):
    """Map LOS ignores smoke; walls alone block ability trajectories."""
    return all(grid[r, c] != 1 for r, c in _line_cells(p1, p2))


def _chebyshev_distance(p1, p2):
    return max(abs(int(p1[0]) - int(p2[0])), abs(int(p1[1]) - int(p2[1])))


def choose_ability_target(char, grid, visible_enemies, planted_pos, side):
    """Choose a fixed lineup or visible enemy target with stable tie-breaking."""
    ability = str(getattr(char, "ability_name", "")).upper()
    points = ABILITY_TARGET_POINTS.get(ability, [])
    side_points = [p for p in points if (p[1] < SITE_BOUNDARY_COL) == (side == "left")]
    fixed_targets = [
        p for p in side_points
        if _chebyshev_distance(char.pos, p) <= ABILITY_TARGET_RANGE
        and (ability == "SMOKE" or has_wall_los(grid, char.pos, p))
    ]
    enemy_targets = [
        tuple(map(int, enemy.pos)) for enemy in visible_enemies
        if getattr(enemy, "is_alive", True)
        and _chebyshev_distance(char.pos, enemy.pos) <= ABILITY_TARGET_RANGE
        and (ability == "SMOKE" or has_wall_los(grid, char.pos, enemy.pos))
    ]
    candidates = list(dict.fromkeys(fixed_targets + enemy_targets))
    if not candidates:
        return None

    if ability == "SMOKE":
        def smoke_utility(target):
            cells = {
                (r, c)
                for r in range(target[0] - 1, target[0] + 2)
                for c in range(target[1] - 1, target[1] + 2)
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1
            }
            blocked_lines = sum(
                1 for enemy in visible_enemies
                if ov1_common_rl.has_los(grid, tuple(enemy.pos), planted_pos, set())
                and not ov1_common_rl.has_los(grid, tuple(enemy.pos), planted_pos, cells)
            )
            covers_spike = int(tuple(planted_pos) in cells)
            near_spike = _chebyshev_distance(target, planted_pos) <= 2
            return covers_spike * 10 + blocked_lines * 2 + int(near_spike)

        fixed_smokes = fixed_targets
        candidates = list(dict.fromkeys(fixed_smokes + enemy_targets))
        if not candidates:
            return None
        return min(candidates, key=lambda p: (-smoke_utility(p), _chebyshev_distance(char.pos, p), p))

    def utility(target):
        if ability == "FLASH":
            return sum(
                1 for enemy in visible_enemies
                if has_wall_los(grid, target, enemy.pos)
            )
        if ability == "RECON":
            radius = RECON_REVEAL_SIZE // 2
            return sum(1 for enemy in visible_enemies if _chebyshev_distance(target, enemy.pos) <= radius)
        return 0

    return min(candidates, key=lambda p: (-utility(p), _chebyshev_distance(char.pos, p), p))


# ============================================================
# 移動・LOS等の共通ロジック(controllers.py非依存の自前実装)
# ============================================================

CARDINAL_MOVES = ov1_common_rl.CARDINAL_MOVES  # up, down, left, right (行動ID 0-3と対応)
# soft_update は ov1_common_rl.soft_update を使用(importで解決済み)

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
    blocked = _alive_occupied_positions(chars, moving_char)
    return list(ov1_common_rl.random_step(GRID, (int(pos[0]), int(pos[1])), blocked))

# _candidate_goals は ov1_common_rl.bfs_next_step 内部に統合されたため削除

def evaluate_greedy(env, net, obs_dim, num_eval_episodes=100):
    wins = 0
    entered_site_count = 0
    reason_counts = {"defused": 0, "detonated": 0, "defenders_wiped": 0, "timeout": 0}
    # 💡追加: defused / defenders_wiped の左右サイト別内訳。
    site_reason_counts = {
        "defused": {"L": 0, "R": 0},
        "defenders_wiped": {"L": 0, "R": 0},
    }
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    defuse_started_count = 0
    
    defuse_progress_3_count = 0
    defuse_progress_5_count = 0
    defuse_progress_required_count = 0
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    defuse_eligible_count = 0
    defuse_selected_count = 0

    for eval_episode in range(num_eval_episodes):
        env.reset()
        entered = False
        episode_max_defuse_progress = 0

        # DEFUSE試行の診断用
        defuse_attempt_char = None
        defuse_attempt_start_tick = None
        defuse_attempt_max_progress = 0
        defuse_attempt_log_count = 0


        while True:
            terminal, reason = env.is_terminal()
            if terminal:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if reason in site_reason_counts:
                    site = "L" if env.planted_pos[1] < SITE_BOUNDARY_COL else "R"
                    site_reason_counts[reason][site] += 1
                break

            actions = {}
            for char in env.defenders():
                if not char.is_alive:
                    continue
                state = env.build_observation(char)
                mask = env.action_mask(char)

                q_values = net(
                    torch.from_numpy(state)
                    .float()
                    .unsqueeze(0)
                ).detach().cpu().numpy()[0]

                masked_q_values = np.where(mask, q_values, -np.inf)

                action = int(np.argmax(masked_q_values))
                action_counts[action] += 1
                defuse_eligible_count += int(bool(mask[ACTION_DEFUSE]))
                defuse_selected_count += int(action == ACTION_DEFUSE)

                actions[char.name] = action

                if action == ACTION_DEFUSE:
                    if defuse_attempt_char is None:
                        defuse_attempt_char = char.name
                        defuse_attempt_start_tick = env.tick
                        defuse_attempt_max_progress = char.defuse_timer

                        if defuse_attempt_log_count < 10:
                            pr, pc = env.planted_pos
                            dist_check = max(abs(pr - char.pos[0]), abs(pc - char.pos[1]))
                            # print(
                            #     f"      [DEFUSE START] "
                            #     f"eval_ep={eval_episode + 1}, "
                            #     f"char={char.name}, "
                            #     f"tick={env.tick}, "
                            #     f"pos={tuple(char.pos)}, "
                            #     f"detonate={env.detonate_timer}, "
                            #     f"progress={char.defuse_timer}, "
                            #     f"dist_to_plant={dist_check}, "
                            #     f"mask_defuse_valid={bool(mask[ACTION_DEFUSE])}"
                            # )
                            defuse_attempt_log_count += 1

                if tuple(char.pos) in env.site_zone:
                    entered = True

            env.step_tick(actions)

            for char in env.defenders():
                episode_max_defuse_progress = max(
                    episode_max_defuse_progress,
                    char.defuse_timer,
                )

            # DEFUSE試行中の経過を記録
            if defuse_attempt_char is not None:
                attempt_char = next(
                    (c for c in env.defenders()
                     if c.name == defuse_attempt_char),
                    None
                )

                if attempt_char is not None:
                    progress = attempt_char.defuse_timer
                    defuse_attempt_max_progress = max(
                        defuse_attempt_max_progress,
                        progress,
                    )

                    if progress > 0 and progress < DEFUSE_REQUIRED_TICKS:
                        if defuse_attempt_log_count < 10:
                            # print(
                            #     f"      [DEFUSE CONTINUE] "
                            #     f"eval_ep={eval_episode + 1}, "
                            #     f"char={attempt_char.name}, "
                            #     f"tick={env.tick}, "
                            #     f"detonate={env.detonate_timer}, "
                            #     f"progress={progress}"
                            # )
                            defuse_attempt_log_count += 1

                    elif progress == 0 and not env.is_defused:
                        if defuse_attempt_log_count < 10:
                            pr, pc = env.planted_pos
                            dist_check = max(abs(pr - attempt_char.pos[0]), abs(pc - attempt_char.pos[1]))
                            # print(
                            #     f"      [DEFUSE INTERRUPT] "
                            #     f"eval_ep={eval_episode + 1}, "
                            #     f"char={attempt_char.name}, "
                            #     f"tick={env.tick}, "
                            #     f"detonate={env.detonate_timer}, "
                            #     f"max_progress={defuse_attempt_max_progress}, "
                            #     f"alive={attempt_char.is_alive}, "
                            #     f"dist_to_plant={dist_check}"
                            # )
                            defuse_attempt_log_count += 1

                        defuse_attempt_char = None
                        defuse_attempt_start_tick = None

        if env.is_defused:
            wins += 1

            # print(
            #     f"      [DEFUSE SUCCESS] "
            #     f"eval_ep={eval_episode + 1}, "
            #     f"tick={env.tick}, "
            #     f"detonate={env.detonate_timer}, "
            #     f"char={defuse_attempt_char}, "
            #     f"max_progress={defuse_attempt_max_progress}"
            # )

        if entered:
            entered_site_count += 1

        if episode_max_defuse_progress >= 1:
            defuse_started_count += 1
        if episode_max_defuse_progress >= 3:
            defuse_progress_3_count += 1
        if episode_max_defuse_progress >= 5:
            defuse_progress_5_count += 1
        if episode_max_defuse_progress >= DEFUSE_REQUIRED_TICKS:
            defuse_progress_required_count += 1

    defused_l = site_reason_counts["defused"]["L"]
    defused_r = site_reason_counts["defused"]["R"]
    wiped_l = site_reason_counts["defenders_wiped"]["L"]
    wiped_r = site_reason_counts["defenders_wiped"]["R"]
    print(
        f"    breakdown: {{'defused': {reason_counts['defused']}(L={defused_l},R={defused_r}), "
        f"'detonated': {reason_counts['detonated']}, "
        f"'defenders_wiped': {reason_counts['defenders_wiped']}(L={wiped_l},R={wiped_r}), "
        f"'timeout': {reason_counts['timeout']}}}"
    )
    print(
        f"    defuse_progress: "
        f"started={defuse_started_count}, "
        f">=3={defuse_progress_3_count}, "
        f">=5={defuse_progress_5_count}, "
        f">=required={defuse_progress_required_count}"
    )
    print(
        f"    eval_actions: move={int(action_counts[0:4].sum())}, "
        f"stay={int(action_counts[4])}, "
        f"defuse={int(action_counts[ACTION_DEFUSE])}, "
        f"ability={int(action_counts[ACTION_ABILITY])}, "
        f"turn={int(action_counts[ACTION_TURN_BASE:ACTION_ULTIMATE].sum())}, "
        f"ultimate={int(action_counts[ACTION_ULTIMATE])}, "
        f"defuse_eligible={defuse_eligible_count}, "
        f"defuse_selected={defuse_selected_count}"
    )
    return wins / num_eval_episodes, entered_site_count / num_eval_episodes

def move_towards_target(pos, target, chars, moving_char=None, allow_adjacent_goal=False):
    """BFSで壁・生存キャラクターを避けながらtargetへ1マス進む(ov1_common_rl.bfs_next_step利用版)。"""
    start = (int(pos[0]), int(pos[1]))
    blocked = _alive_occupied_positions(chars, moving_char)
    blocked.discard(start)
    step = ov1_common_rl.bfs_next_step(GRID, start, target, blocked, allow_adjacent_goal=allow_adjacent_goal)
    return [int(step[0]), int(step[1])]

def has_los(p1, p2, smoke_cells):
    return ov1_common_rl.has_los(GRID, p1, p2, smoke_cells)


def _facing_from_delta(dr, dc, fallback):
    """battle_logic.py._facing_from_deltaと同一ロジック(自己完結ルールにより複製)。"""
    if dr == 0 and dc == 0:
        return fallback
    if dr != 0:
        return "N" if dr < 0 else "S"
    return "W" if dc < 0 else "E"


def _facing_accuracy_multiplier(shooter_facing, shooter_pos, target_pos):
    """battle_logic.py._facing_accuracy_multiplierと同一ロジック(自己完結ルールにより複製)。
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


def _facing_towards(from_pos, to_pos):
    """battle_logic.py._facing_towardsと同一ロジック(自己完結ルールにより複製)。
    from_posからto_posへ最も近い8方向のfacingを返す。"""
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


def _planned_route_facing(dist_map, pos):
    """最短経路の2手目を先読みした移動 facing を返す。"""
    if dist_map is None:
        return None
    r0, c0 = int(pos[0]), int(pos[1])
    first = ov1_common_rl.bfs_best_direction(dist_map, r0, c0)
    if first == (0, 0):
        return None
    nr, nc = r0 + first[0], c0 + first[1]
    if not _in_bounds((nr, nc)):
        return _facing_from_delta(first[0], first[1], None)
    second = ov1_common_rl.bfs_best_direction(dist_map, nr, nc)
    desired = second if second != (0, 0) and second != first else first
    return _facing_from_delta(desired[0], desired[1], None)


def _forced_combat_facing(unit, visible_enemies, team_sighting):
    """視認中の最寄り敵、または一時的な直前 combat facing を返す。"""
    if visible_enemies:
        target = min(
            visible_enemies,
            key=lambda enemy: max(
                abs(enemy.pos[0] - unit.pos[0]),
                abs(enemy.pos[1] - unit.pos[1]),
            ),
        )
        facing = _facing_towards(tuple(unit.pos), tuple(target.pos))
        if facing is not None:
            unit._combat_facing = facing
            unit._combat_facing_staleness = SIGHTING_STALENESS_CAP
        return getattr(unit, "_combat_facing", None)

    if team_sighting.last_seen_enemy is not None:
        remaining = getattr(unit, "_combat_facing_staleness", 0)
        if remaining > 0:
            unit._combat_facing_staleness = remaining - 1
            return getattr(unit, "_combat_facing", None)
    return None


class _TeamSightingMemory:
    """5人のDefender全体で共有する、敵の最新目撃情報(carry/escort/guardの
    SightingMemory/TeamMemoryと同一方針)。誰か一人でも視認していれば共有され、
    視認が途切れてもSIGHTING_STALENESS_CAP tickの間は保持される。"""

    def __init__(self):
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.last_seen_enemy = None

    def update(self, defenders, attackers, smoke_cells):
        defenders_alive = [d for d in defenders if d.is_alive]
        attackers_alive = [a for a in attackers if a.is_alive]

        visible = []
        for a in attackers_alive:
            for d in defenders_alive:
                if has_los(tuple(d.pos), tuple(a.pos), smoke_cells) and a not in visible:
                    visible.append(a)

        if visible:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next((a for a in visible if a.name == tracked_name), None)
            if tracked is None:
                tracked = min(
                    visible,
                    key=lambda a: min(
                        max(abs(a.pos[0] - d.pos[0]), abs(a.pos[1] - d.pos[1])) for d in defenders_alive
                    ) if defenders_alive else 0,
                )
            self.last_seen_enemy = {
                "pos": tuple(map(int, tracked.pos)), "name": tracked.name, "tick_ago": 0,
            }
        elif self.last_seen_enemy is not None:
            if not any(a.name == self.last_seen_enemy.get("name") for a in attackers_alive):
                self.last_seen_enemy = None
            else:
                self.last_seen_enemy["tick_ago"] += 1
                if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                    self.last_seen_enemy = None


def bfs_distance_map(goal):
    return ov1_common_rl.bfs_distance_map(GRID, goal)


good_directions = ov1_common_rl.good_directions

# ============================================================
# 軽量キャラクター表現(game_core.Character非依存)
# ============================================================

BASE_ACCURACY = 0.70
BASE_DODGE = 0.42
BASE_HS_RATE = 0.30
BASE_REACTION = 150.0

# 💡バグ修正: 元実装は未定義の RAW_STATS を参照しておりNameErrorに
# なる不具合があった。common_defender.compute_effective_stats
# (character_stats.CHARACTER_TABLE を正しく参照する実装)に置き換える。
# ROLE_TO_ABILITY / COMBO_MEMBERS / COMBO_BONUS /
# TIGER_BONUS は common_defender 側に統合されたため削除。
EFFECTIVE_STATS = compute_effective_stats(STATS_TABLE)

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
        # Defenderは下向き(南)、Attackerは上向き(北)スポーンでgame_core.Characterと合わせる。
        self.facing = "S" if team == "D" else "N"
        self._combat_facing = None
        self._combat_facing_staleness = 0
        # battle_logic.py同様、被弾した次のTickだけ相手方向へ強制的に向く仕組みを再現する。
        self.forced_facing_next_tick = None
        self.facing_forced_this_tick = False

        self.role = role
        ability = ROLE_TO_ABILITY.get(role, "NONE")
        self.ability_name = ability
        self.smoke_charges = 1 if ability == "SMOKE" else 0
        self.flash_charges = 1 if ability == "FLASH" else 0
        self.recon_charges = 1 if ability == "RECON" else 0
        if ability in ("SMOKE", "FLASH", "RECON", "HUNT"):
            initialize_ultimate(self, ability)
        else:
            self.ultimate_name = ""
            self.ultimate_cost = 1
            self.ultimate_points = 0

        if override_stats is not None:
            # omoko_v1固定チーム用: 実効ステータスをそのまま使用(ランダム化しない)
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
            return move_towards_target(char.pos, plant_pos, chars, char, allow_adjacent_goal=True)
        if random.random() < 0.5:
            return list(char.pos)
        return get_next_pos_random(char.pos, chars, char)



# ============================================================
# Retake 環境
# ============================================================

class RetakeEnv:
    def __init__(self, min_detonate_ticks=SPIKE_DETONATION_TICKS, max_detonate_ticks=SPIKE_DETONATION_TICKS,
                 attacker_hold_radius=4, max_ticks=100):
        self.min_detonate_ticks = min_detonate_ticks
        self.max_detonate_ticks = max_detonate_ticks
        self.attacker_stub = AttackerStub(hold_radius=attacker_hold_radius)
        self.max_ticks = max_ticks
        self.team_sighting = _TeamSightingMemory()
        self.active_entry_points = []  # このラウンドのサイト(左/右)に対応する既知侵入経路
        self.available_orbs = set()

    def reset(self):
        self.tick = 0
        self.round_over = False
        self.is_defused = False
        self.active_defuser_name = None
        self.combat_started = False
        self.player_count = random.randint(1, 5)
        self.site_side, self.planted_pos = _sample_planted_pos()

        self.dist_map = bfs_distance_map(self.planted_pos)
        self.site_zone = {
            (r, c)
            for r in range(HEIGHT)
            for c in range(WIDTH)
            if 0 <= self.dist_map[r, c] <= SITE_ZONE_RADIUS
        }

        self.detonate_timer = random.randint(self.min_detonate_ticks, self.max_detonate_ticks)
        self.smokes = []
        self.available_orbs = valid_orb_cells(GRID)
        self.last_orb_rewards = {}
        self.last_ability_effects = {}
        self.active_entry_points = ENTRY_POINTS[self.site_side]
        self.team_sighting.reset()

        used = set()
        self.chars = []
        self._build_fixed_defenders(used, self.player_count)
        self._build_attackers(used, self.player_count, self.site_side)
        self.team_sighting.update(self.defenders(), self.attackers(), self.smoke_cells())

    def _build_fixed_defenders(self, used, count):
        if len(DEFENDER_SPAWNS) < count:
            raise ValueError("Retake map has fewer defender start cells than requested players")
        names = random.sample(ROSTER_ORDER, count)
        starts = random.sample(DEFENDER_SPAWNS, count)
        for name, pos in zip(names, starts):
            used.add(pos)
            stats = EFFECTIVE_STATS[name]
            self.chars.append(
                SimChar(name, "D", pos, role=stats["role"], override_stats=stats)
            )

    def _build_attackers(self, used, count, side):
        starts = _select_enemy_starts(side, count, used)
        for i, pos in enumerate(starts):
            self.chars.append(SimChar(f"Attacker{i + 1}", "A", pos, role=None))

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

    def _nearest_visible_entry_point(self, pos):
        """既知侵入経路(active_entry_points)のうち、posから視認できるものだけを
        候補にし、最も近い1点を返す(視認できるものが無ければNone)。"""
        smoke_cells = self.smoke_cells()
        visible = [p for p in self.active_entry_points if has_los(pos, p, smoke_cells)]
        if not visible:
            return None
        return min(visible, key=lambda p: max(abs(p[0] - pos[0]), abs(p[1] - pos[1])))

    # -- アビリティ(サイトへ向けて使用する前提) --------------------------
    def apply_ability(self, char):
        ability = char.ability_name
        pr, pc = self.planted_pos
        visible_enemies = [
            enemy for enemy in self.attackers()
            if enemy.is_alive and self.check_line_of_sight(char, enemy)
        ]
        target_pos = choose_ability_target(
            char, GRID, visible_enemies, self.planted_pos, self.site_side
        )
        if target_pos is None:
            return
        tr, tc = target_pos
        # 💡追加: 効果量を記録(報酬側で参照)。valueの意味はability種別ごとに異なる
        # (FLASH/RECON=新たに効果を受けた敵の人数, SMOKE=LOSが遮断された生存attacker人数)。
        effect = {"type": ability, "value": 0}

        if ability == "FLASH":
            char.flash_charges -= 1
            newly_blinded = 0
            for enemy in self.chars:
                if enemy.team != char.team and enemy.is_alive:
                    if has_los((tr, tc), tuple(enemy.pos), self.smoke_cells()):
                        if enemy.blind_remaining <= 0:
                            newly_blinded += 1
                        enemy.blind_remaining = max(enemy.blind_remaining, BLIND_DURATION_TICKS)
            effect["value"] = newly_blinded

        elif ability == "RECON":
            char.recon_charges -= 1
            radius = RECON_REVEAL_SIZE // 2
            newly_revealed = 0
            for enemy in self.chars:
                if enemy.team != char.team and enemy.is_alive:
                    er, ec = enemy.pos
                    if max(abs(er - tr), abs(ec - tc)) <= radius:
                        if enemy.reveal_remaining <= 0:
                            newly_revealed += 1
                        enemy.reveal_remaining = max(enemy.reveal_remaining, REVEAL_DURATION_TICKS)
            effect["value"] = newly_revealed

        elif ability == "SMOKE":
            char.smoke_charges -= 1
            cells = {
                (rr, cc)
                for rr in range(tr - 1, tr + 2)
                for cc in range(tc - 1, tc + 2)
                if _in_bounds((rr, cc)) and GRID[rr, cc] != 1
            }
            # 💡追加: 設置前後でplanted_posへのLOSが通っていた生存attacker数を比較し、
            # このスモークによって新たに遮断できた人数を効果量とする。
            pre_smoke_cells = self.smoke_cells()
            pre_los_count = sum(
                1 for a in self.attackers()
                if a.is_alive and has_los(tuple(a.pos), (pr, pc), pre_smoke_cells)
            )
            self.smokes.append({"cells": cells, "remaining_ticks": SMOKE_DURATION_TICKS, "owner": char.name})
            post_los_count = sum(
                1 for a in self.attackers()
                if a.is_alive and has_los(tuple(a.pos), (pr, pc), self.smoke_cells())
            )
            effect["value"] = max(0, pre_los_count - post_los_count)
            effect["covers_spike"] = tuple(self.planted_pos) in cells

        self.last_ability_effects[char.name] = effect

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

        # Keep the retake moving whenever an unoccupied step shortens the
        # walkable path to the spike. Ability and ultimate actions stay enabled.
        raw_dist = self.dist_map[r, c] if self.dist_map is not None else -1
        under_direct_threat = any(
            enemy.is_alive and self.check_line_of_sight(char, enemy)
            for enemy in self.attackers()
        )
        if under_direct_threat:
            # Preserve the learned combat choice: staying fires automatically,
            # while movement, defuse, and tactical actions remain selectable.
            mask[4] = True
        elif dist_to_plant <= 1:
            # At the spike with no visible threat, prevent idle/walk-away
            # actions while leaving defuse and tactical choices to the policy.
            mask[:5] = False
        elif raw_dist > 1:
            advancing_actions = []
            for action in range(4):
                if not mask[action]:
                    continue
                dr, dc = MOVE_DELTAS[action]
                next_dist = self.dist_map[r + dr, c + dc]
                if 0 <= next_dist < raw_dist:
                    advancing_actions.append(action)
            if advancing_actions:
                for action in range(4):
                    mask[action] = action in advancing_actions
                mask[4] = False
            else:
                # If allies temporarily block the route, wait for the next tick.
                mask[4] = True
        else:
            mask[4] = True

        has_charge = char.own_ability_charge() > 0
        visible_enemies = [
            enemy for enemy in self.attackers()
            if enemy.is_alive and self.check_line_of_sight(char, enemy)
        ]
        has_target = choose_ability_target(
            char, GRID, visible_enemies, self.planted_pos, self.site_side
        ) is not None
        mask[ACTION_ABILITY] = bool(has_charge and self.combat_started and has_target)

        # facing は経路・combat 状態から自動決定するため、TURN action は使わない。
        mask[ACTION_TURN_BASE:ACTION_ULTIMATE] = False
        mask[ACTION_ULTIMATE] = ultimate_ready(char)
        mask[ACTION_ORB] = (
            tuple(char.pos) in self.available_orbs
            and orb_priority(char, self.defenders())
        )
        if (
            dist_to_plant <= 1
            and self.detonate_timer <= (
                DEFUSE_REQUIRED_TICKS + ORB_COLLECT_REQUIRED_TICKS
                + DEFUSE_SAFETY_MARGIN_TICKS
            )
        ):
            # Orb collection takes five stationary ticks; preserve that time
            # for an available defuse.
            mask[ACTION_ORB] = False

        if dist_to_plant <= 1 and not under_direct_threat:
            # Once the site is safe, do not let utility, ultimates, or orbs
            # postpone the objective. Smoke should be set before this point.
            mask[:] = False
            mask[ACTION_DEFUSE] = not self.is_defused

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
        blind_norm = char.blind_remaining / BLIND_DURATION_TICKS if BLIND_DURATION_TICKS else 0.0
        defuse_norm = char.defuse_timer / DEFUSE_REQUIRED_TICKS
        detonate_norm = self.detonate_timer / SPIKE_DETONATION_TICKS

        allies = [a for a in self.defenders() if a.is_alive]
        enemies = [e for e in self.attackers() if e.is_alive]

        allies_alive_norm = len(allies) / 5.0
        allies_in_zone = sum(1 for a in allies if tuple(a.pos) in self.site_zone) / 5.0
        allies_near_entry = sum(
            1 for a in allies
            if a is not char
            and max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
        ) / 5.0

        others = [a for a in allies if a is not char]
        if others:
            nearest_ally_dist = min(
                max(abs(a.pos[0] - r), abs(a.pos[1] - c)) for a in others
            ) / max(h, w)
        else:
            nearest_ally_dist = 1.0

        ally_ability_active = 1.0 if self.ally_ability_active(char) else 0.0
        # 💡追加: 自分が「現在」味方スモークに覆われているかの直接フラグ。
        # team_sighting_featsは敵を見失った後もSIGHTING_STALENESS_CAP tickの間
        # 古い目撃情報を保持し続けるため、スモークで実際には安全になっていても
        # 観測上「敵がいる」ように見えて解除をためらう原因になっていた。
        # この直接フラグで「スモークで守られている」ことを明示し、
        # 古い目撃情報より優先して解除して良いと学習できるようにする。
        self_smoke_covered = 1.0 if _smoke_covering(self, tuple(char.pos)) is not None else 0.0

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

        # 自身の向き(8方向)。facingが命中率へ直接影響するため、
        # 観測に含めないとネットワークから見て部分観測(POMDP)になり
        # TD学習が不安定化する。被弾直後は斜め向き(forced_facing)にもなり得るため
        # TURN_DIRS(4方向の操作アクション)ではなくALL_FACINGS(8方向)でエンコードする。
        facing_onehot = [0.0] * len(ALL_FACINGS)
        if char.facing in ALL_FACINGS:
            facing_onehot[ALL_FACINGS.index(char.facing)] = 1.0

        # チーム共有の目撃情報(自分が直接視認していなくても、他の味方が
        # 見ていれば共有される。carry/escort/guardと同一方針)。
        last_seen = self.team_sighting.last_seen_enemy
        if last_seen is not None:
            tr, tc = last_seen["pos"]
            team_sighting_feats = [
                1.0,
                max(-1.0, min(1.0, (tr - r) / max(h, w))),
                max(-1.0, min(1.0, (tc - c) / max(h, w))),
                min(last_seen["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP,
            ]
        else:
            team_sighting_feats = [0.0, 0.0, 0.0, 0.0]

        # 既知侵入経路(active_entry_points)のうち視認できる最寄り点への方向。
        # 敵の実位置とは無関係な、構造的な予測情報(carryのSMOKE_LINEUP、
        # guardの警戒ポイントと同一方針)。
        nearest_entry = self._nearest_visible_entry_point((r, c))
        if nearest_entry is not None:
            entry_feats = [1.0, (nearest_entry[0] - r) / h, (nearest_entry[1] - c) / w]
        else:
            entry_feats = [0.0, 0.0, 0.0]

        obs = [
            r / h, c / w,
            char.hp / char.max_hp,
            *good_dir,                 # up, down, left, right (4次元。BFSベース)
            dist_to_plant,              # BFS距離ベース
            in_site_zone, adjacent_to_plant,
            own_charge, blind_norm, defuse_norm, detonate_norm,
            allies_alive_norm, allies_in_zone, allies_near_entry, nearest_ally_dist,
            ally_ability_active, self_smoke_covered,
            len(visible_enemies) / 5.0,
            len(enemies) / 5.0,
        ] + enemy_feats + role_onehot + facing_onehot + team_sighting_feats + entry_feats

        obs.extend(ultimate_context(
            char,
            self_sighting=bool(visible_enemies),
            team_sighting=last_seen is not None,
            tactical=bool(visible_enemies) or last_seen is not None,
        ))
        obs.extend(orb_context(char, self.available_orbs, GRID, allies))

        return np.array(obs, dtype=np.float32)

    # -- 1Tick進行 ---------------------------------------------------------
    def step_tick(self, defender_actions):
        for char in self.chars:
            char.moved_this_tick = False
            # battle_logic.py同様、前Tickで被弾していれば今Tickだけ強制的に
            # 相手方向(8方向)を向く。TURN行動・移動方向による上書きは抑止する。
            char.facing_forced_this_tick = False
            if char.forced_facing_next_tick:
                char.facing = char.forced_facing_next_tick
                char.facing_forced_this_tick = True
            char.forced_facing_next_tick = None

        # 💡追加: このtickのability効果記録をクリア(前tick分の値が
        # compute_rewards側に残らないようにする)。
        self.last_ability_effects = {}
        self.last_orb_rewards = {}

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
                elif action == ACTION_ORB:
                    pending_orb.add(char.name)
                elif action >= ACTION_TURN_BASE:
                    if not char.facing_forced_this_tick:
                        char.facing = TURN_DIRS[action - ACTION_TURN_BASE]

                visible_enemies = [
                    enemy for enemy in self.attackers()
                    if enemy.is_alive and self.check_line_of_sight(char, enemy)
                ]
                forced_facing = _forced_combat_facing(
                    char, visible_enemies, self.team_sighting
                )
                if forced_facing is None and action in (0, 1, 2, 3):
                    dr, dc = MOVE_DELTAS[action]
                    first_step = ov1_common_rl.bfs_best_direction(
                        self.dist_map, int(char.pos[0]), int(char.pos[1])
                    )
                    if (dr, dc) == first_step:
                        forced_facing = _planned_route_facing(self.dist_map, char.pos)
                    if forced_facing is None:
                        forced_facing = _facing_from_delta(dr, dc, char.facing)
                if forced_facing is not None:
                    char.facing = forced_facing
                    char.facing_forced_this_tick = True
            else:
                next_positions[char.name] = self.attacker_stub.decide_move(char, self.chars, self.planted_pos)

        for char in self.chars:
            if char.name in pending_ability:
                self.apply_ability(char)
            elif char.name in pending_ultimate and spend_ultimate(char):
                visible_enemies = [
                    enemy for enemy in self.attackers()
                    if enemy.is_alive and self.check_line_of_sight(char, enemy)
                ]
                if char.ultimate_name == "MONITOR":
                    for enemy in self.attackers():
                        if enemy.is_alive:
                            enemy.reveal_remaining = max(
                                enemy.reveal_remaining, REVEAL_DURATION_TICKS
                            )
                elif char.ultimate_name == "TUNNEL":
                    for enemy in visible_enemies:
                        enemy.blind_remaining = max(
                            enemy.blind_remaining, BLIND_DURATION_TICKS
                        )
            elif char.name in pending_orb:
                _completed, reward = collect_orb_tick(char, self.available_orbs)
                self.last_orb_rewards[char.name] = reward

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

        # battle_logic.py._move_order()と一致させる。リテイクフェーズはスパイク
        # 設置済み(has_spike常にFalse)のため、実際は self.chars の元の並び順
        # (run_game.py.init_round(): Attacker→Defenderロースター順)で毎tick
        # 固定的に処理される。ランダムシャッフルは実戦と異なる順序を学習させて
        # しまうため廃止し、固定順に統一する。
        fixed_order = self.attackers() + self.defenders()
        move_order = [c for c in fixed_order if c.is_alive and c.name in next_positions]
        for char in move_order:
            if (
                char.name in pending_defuse
                or char.name in pending_ability
                or char.name in pending_ultimate
                or char.name in pending_orb
            ):
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
                if not char.facing_forced_this_tick:
                    char.facing = _facing_from_delta(nr - old_pos[0], nc - old_pos[1], char.facing)
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
            for b in alive[i + 1:]:
                if a.team != b.team and self.check_line_of_sight(a, b):
                    current_los_revealed.add(a.name)
                    current_los_revealed.add(b.name)

        self.last_shots = self._resolve_shots(current_los_revealed)
        if self.last_shots:
            self.combat_started = True

        for char in self.chars:
            char.los_revealed = char.is_alive and char.name in current_los_revealed

        # 移動・戦闘が確定した後の位置関係でチーム共有目撃情報を更新する。
        # 次tickのbuild_observation/compute_rewardsはこの更新後の状態を参照する。
        self.team_sighting.update(self.defenders(), self.attackers(), self.smoke_cells())

        if not self.is_defused:
            completed = [
                c for c in self.defenders()
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
            possible = [t for t in alive if t.team != shooter.team and self.check_line_of_sight(shooter, t)]
            if not possible:
                continue
            defusers = [t for t in possible if t.defuse_timer > 0]
            pool = defusers if defusers else possible
            target = min(
                pool,
                key=lambda t: (max(abs(t.pos[0] - shooter.pos[0]), abs(t.pos[1] - shooter.pos[1])), t.hp, t.name),
            )
            intents.append({"shooter": shooter, "target": target})

        random.shuffle(intents)
        intents.sort(key=lambda i: i["shooter"].reaction, reverse=True)

        shots = []
        for intent in intents:
            shooter, target = intent["shooter"], intent["target"]
            if not shooter.is_alive or not target.is_alive:
                continue

            # battle_logic.py同様、命中・被弾を問わず撃たれたら次のTickだけ
            # 相手の方向を強制的に向く(斜め8方向を含む)。
            target.forced_facing_next_tick = _facing_towards(target.pos, shooter.pos)

            shooter_accuracy = MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            if shooter.team == "D":
                shooter_accuracy *= _facing_accuracy_multiplier(shooter.facing, shooter.pos, target.pos)
            if shooter.blind_remaining > 0:
                shooter_accuracy *= BLIND_ACCURACY_MULTIPLIER

            revealed = target.reveal_remaining > 0 or (
                target.los_revealed and target.name in current_los_revealed
            )
            effective_dodge = target.dodge_rate * (REVEALED_DODGE_MULTIPLIER if revealed else 1.0)
            hit_chance = shooter_accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            headshot = hit and random.random() < shooter.hs_rate
            damage = (HEADSHOT_DAMAGE if headshot else BODY_DAMAGE) if hit else 0

            shots.append({
                "shooter": shooter, "target": target, "hit": hit,
                "headshot": headshot, "damage": damage,
            })

            if damage > 0:
                target.hp = max(0, target.hp - damage)
                if target.hp <= 0:
                    target.is_alive = False
                    target.defuse_timer = 0
                    if shooter.role == "タイガー":
                        shooter.hp = min(shooter.max_hp, shooter.hp + 50)
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





def snapshot_before(env):
    before = {}
    for char in env.defenders():
        r, c = char.pos
        # 💡修正: 直線距離ではなくBFS距離を保存する。到達不能(-1)の場合は
        # マップ最大距離相当の値でフォールバックする(通常は起こらない想定)。
        raw = env.dist_map[r, c]
        dist_val = raw if raw >= 0 else (HEIGHT + WIDTH)
        visible_enemies = [
            enemy for enemy in env.attackers()
            if enemy.is_alive and env.check_line_of_sight(char, enemy)
        ]
        target_pos = None
        if visible_enemies:
            target_pos = tuple(min(
                visible_enemies,
                key=lambda enemy: max(
                    abs(enemy.pos[0] - char.pos[0]),
                    abs(enemy.pos[1] - char.pos[1]),
                ),
            ).pos)
        elif env.team_sighting.last_seen_enemy is not None:
            target_pos = env.team_sighting.last_seen_enemy["pos"]
        before[char.name] = {
            "alive": char.is_alive,
            "in_zone": (r, c) in env.site_zone,
            "dist_to_plant": dist_val,
            "defuse_timer": char.defuse_timer,
            "ability_charge": char.own_ability_charge(),
            "ability_target": choose_ability_target(
                char, GRID, visible_enemies, env.planted_pos, env.site_side
            ),
            "combat_started": env.combat_started,
            "self_sighting": bool(visible_enemies),
            "team_sighting": env.team_sighting.last_seen_enemy is not None,
            "ultimate_target": target_pos,
        }
    return before

def _smoke_covering(env, pos):
    """posを覆っている、味方(Defender)が設置した有効なスモークのうち、
    remaining_ticksが最大のものを返す(無ければNone)。"""
    defender_names = {d.name for d in env.defenders()}
    pos_t = tuple(pos)
    covering = [
        s for s in env.smokes
        if pos_t in s["cells"] and s["owner"] in defender_names and s["remaining_ticks"] > 0
    ]
    if not covering:
        return None
    return max(covering, key=lambda s: s["remaining_ticks"])


def compute_rewards(env, before, chosen_actions):
    rewards = {}
    detonate_frac = env.detonate_timer / SPIKE_DETONATION_TICKS
    pr, pc = env.planted_pos

    allies_alive = [a for a in env.chars if a.team == "D" and a.is_alive]
    enemies_alive = any(a.is_alive for a in env.attackers())

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
            1 for a in allies_alive
            if a is not char
            and max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
        )

        if allies_near_entry > 0:
            reward += ALLY_GATHER_REWARD * allies_near_entry

        # 💡変更: detonate_frac(割合)ではなく、残りtickの絶対値で「時間切れ間近か」を判定する。
        time_critical_for_entry = env.detonate_timer <= ENTRY_SAFETY_MARGIN_TICKS

        # 💡追加: 自分がラスト1人(他に生存defenderがいない)場合、援護は
        # 原理的に来ないため、援護不足ペナルティを課すと「サイトを離れる」
        # ことが最適行動になってしまう。この場合はtime_critical同様に免除する。
        is_last_defender = len(allies_alive) <= 1

        if in_zone_now and not b["in_zone"]:
            if (
                allies_near_entry >= MIN_ALLIES_FOR_ENTRY
                or time_critical_for_entry
                or is_last_defender
                or not enemies_alive
            ):
                reward += ENTRY_WITH_SUPPORT_BONUS
            else:
                reward += ENTRY_ALONE_PENALTY
        elif (
            in_zone_now
            and allies_near_entry < MIN_ALLIES_FOR_ENTRY
            and not time_critical_for_entry
            and not is_last_defender
            and enemies_alive
        ):
            reward += ENTRY_ALONE_LINGER_PENALTY
        elif not in_zone_now:
            reward += APPROACH_REWARD_SCALE * (b["dist_to_plant"] - dist_now)
            if dist_now > b["dist_to_plant"]:
                reward += APPROACH_WRONG_DIRECTION_PENALTY

        action_id = chosen_actions.get(name)
        smoke_ready_before = (
            char.ability_name == "SMOKE"
            and b["ability_charge"] > 0
            and b["combat_started"]
            and b["ability_target"] is not None
        )
        if action_id == ACTION_ABILITY:
            ready = b["combat_started"] and b["ability_target"] is not None
            if ready:
                reward += ABILITY_GOOD_USE_BONUS
                if smoke_ready_before:
                    target_dist = _chebyshev_distance(b["ability_target"], env.planted_pos)
                    reward += SMOKE_EARLY_CAST_BONUS * max(
                        0.0, 1.0 - target_dist / (ABILITY_TARGET_RANGE + 1.0)
                    )
                effect = env.last_ability_effects.get(name, {"type": None, "value": 0})
                if effect["type"] == "FLASH":
                    reward += FLASH_EFFECT_BONUS_PER_ENEMY * effect["value"]
                elif effect["type"] == "RECON":
                    reward += RECON_EFFECT_BONUS_PER_ENEMY * effect["value"]
                elif effect["type"] == "SMOKE":
                    reward += SMOKE_EFFECT_BONUS_PER_BLOCKED * effect["value"]
                    if effect.get("covers_spike"):
                        reward += SMOKE_COVER_SPIKE_BONUS
                if effect["value"] == 0 and effect["type"] != "SMOKE":
                    reward += ABILITY_NO_EFFECT_PENALTY
            else:
                reward += ABILITY_PREMATURE_PENALTY
        elif action_id == ACTION_ULTIMATE:
            reward += ultimate_use_reward(
                b["self_sighting"] or b["team_sighting"]
            )
            if b["ultimate_target"] is not None:
                reward += 0.25 * _facing_alignment(
                    char.facing, tuple(char.pos), b["ultimate_target"]
                )
        elif action_id == ACTION_ORB:
            reward += env.last_orb_rewards.get(name, 0.0)
        if char.defuse_timer > b["defuse_timer"]:
            reward += DEFUSE_PROGRESS_REWARD
            # 💡変更: tickごと/開始時のボーナスは中断してもリセットされずに得られてしまい
            # (完走せずに稼げる)ファーミングの抜け道になっていたため撤回。
            # 「解除タイマーが完了ラインに到達した瞬間、スモークに覆われていたか」
            # だけを見る一括ボーナスに変更する(完走しないと得られない)。
            if (
                b["defuse_timer"] < DEFUSE_REQUIRED_TICKS
                and char.defuse_timer >= DEFUSE_REQUIRED_TICKS
                and _smoke_covering(env, char.pos) is not None
            ):
                reward += SMOKE_COVER_DEFUSE_COMPLETE_BONUS

        if action_id == ACTION_DEFUSE and b["defuse_timer"] == 0 and char.defuse_timer > 0:
            threat = any(
                e.is_alive and env.check_line_of_sight(char, e) for e in env.attackers()
            )
            time_critical_for_defuse = env.detonate_timer <= (DEFUSE_REQUIRED_TICKS + DEFUSE_SAFETY_MARGIN_TICKS)
            if threat and not time_critical_for_defuse:
                reward += UNSAFE_DEFUSE_PENALTY

        # 💡追加: facing整合の弱いshaping報酬。自分が実際の敵を直接視認して
        # いない時のみ有効(直接視認時は通常の交戦報酬(命中率経由)に委ねる)。
        # 優先順位: チーム共有の目撃情報 > 既知の侵入経路(視認できるもののみ)。
        has_direct_los_to_enemy = any(
            e.is_alive and env.check_line_of_sight(char, e) for e in env.attackers()
        )
        if not has_direct_los_to_enemy:
            watch_target, watch_weight = None, 0.0
            last_seen = env.team_sighting.last_seen_enemy
            if last_seen is not None:
                watch_target, watch_weight = last_seen["pos"], TEAM_SIGHTING_ALIGN_WEIGHT
            else:
                nearest_entry = env._nearest_visible_entry_point(tuple(char.pos))
                if nearest_entry is not None:
                    watch_target, watch_weight = nearest_entry, CORRIDOR_WATCH_ALIGN_WEIGHT
            if watch_target is not None:
                reward += watch_weight * _facing_alignment(char.facing, tuple(char.pos), watch_target)

        rewards[name] = rewards.get(name, 0.0) + reward

    for shot in env.last_shots:
        shooter, target = shot["shooter"], shot["target"]
        if shooter.team != "D" or not shot["hit"]:
            continue
        if shooter.name not in rewards:
            continue
        add = DAMAGE_REWARD_SCALE * shot["damage"]
        debuffed = target.blind_remaining > 0 or target.reveal_remaining > 0 or target.los_revealed
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

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "mask", "next_mask"])
# DuelingQNet / ReplayBuffer は ov1_common_rl に統合。

# select_action は ov1_common_rl.select_action(net, state, mask, epsilon,
# fallback_action=4) を使用(このファイルはSTAY=4がフォールバック)。
# compute_td_loss は ov1_common_rl.compute_double_dqn_loss に統合。

# ============================================================
# 学習ループ
# ============================================================

def run_episode(env, net, target_net, replay, epsilon, obs_dim, defuse_force_prob=0.0):
    env.reset()
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    zero_mask = np.zeros(N_ACTIONS, dtype=bool)

    while True:
        terminal, reason = env.is_terminal()
        if terminal:
            break

        before = snapshot_before(env)

        obs_before, mask_before, chosen_actions = {}, {}, {}
        for char in env.defenders():
            if not char.is_alive:
                continue
            
            state = env.build_observation(char)
            mask = env.action_mask(char)
            action = select_action(net, state, mask, epsilon, fallback_action=4)
            # 解除カリキュラム: 序盤はDEFUSE可能かつ「敵に見られていない」
            # 状況に限って強制的にDEFUSEを選ばせ、成功体験を先にreplayへ
            # 積ませる。視認されている状況まで無条件に強制すると、
            # 「DEFUSE=即死」という相関をむしろ強化してしまうため、
            # 危険な状況では強制しない(=ここでは学習済みQ値の判断に任せる)。
            if mask[ACTION_DEFUSE] and random.random() < defuse_force_prob:
                under_threat = any(
                    e.is_alive and env.check_line_of_sight(char, e)
                    for e in env.attackers()
                )
                if not under_threat:
                    action = ACTION_DEFUSE
            obs_before[char.name] = state
            mask_before[char.name] = mask
            chosen_actions[char.name] = action

        env.step_tick(chosen_actions)

        rewards = compute_rewards(env, before, chosen_actions)

        terminal_now, reason_now = env.is_terminal()
        if terminal_now:
            outcome = DEFUSE_WIN_REWARD if reason_now == "defused" else LOSS_PENALTY
            for char in env.defenders():
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
            replay.push(state, chosen_actions[name], rewards.get(name, 0.0), next_state, float(done), mask_before[name], next_mask)

    return env.is_defused, env.tick


# 💡追加: retake専用のtargetクリップ付きDouble DQN更新。
# ov1_common_rl.optimize_double_dqn_step は他のtrain_*.pyからも共有される
# ため、そちらは変更せず、retakeだけで完結する独自実装をここに持つ。
# 稀な状態(スモーク越し解除など)でQ値が自己参照的に吊り上がる問題への対処として、
# TD targetを理論上あり得る範囲にクリップしてから学習する。
TARGET_CLIP_MIN = -15.0
TARGET_CLIP_MAX = 15.0


def _optimize_double_dqn_step_clipped(
    policy_net, target_net, optimizer, state, action, reward, next_state, done, next_mask,
    gamma, device=DEVICE, max_grad_norm=1.0,
):
    states = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.array(action), dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(np.array(done), dtype=torch.float32, device=device)
    next_masks = torch.as_tensor(np.array(next_mask), dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        next_q_policy = next_q_policy.masked_fill(~next_masks, -float("inf"))
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states).gather(1, next_actions).squeeze(1)
        next_q_target = torch.nan_to_num(next_q_target, neginf=0.0)
        target = rewards + gamma * next_q_target * (1.0 - dones)
        target = torch.clamp(target, TARGET_CLIP_MIN, TARGET_CLIP_MAX)

    loss = F.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    q_stats = (
        float(q_values.detach().mean().item()),
        float(q_values.detach().max().item()),
        float(target.detach().mean().item()),
    )
    return float(loss.item()), q_stats


def train_step(net, target_net, optimizer, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return None
    batch = replay.sample(batch_size)
    return _optimize_double_dqn_step_clipped(
        net, target_net, optimizer,
        batch.state, batch.action, batch.reward, batch.next_state, batch.done, batch.next_mask,
        gamma, max_grad_norm=1.0,
    )  # -> (loss, (q_mean, q_max, target_mean))


def main():
    global SITE_LEFT_PROB

    parser = argparse.ArgumentParser()
    parser.add_argument("--site-left-prob", type=float, default=SITE_LEFT_PROB,
                         help="左サイトを選ぶ確率(残りは右サイト)")
    args = parser.parse_args()
    SITE_LEFT_PROB = args.site_left_prob

    print(f"[INIT] device = {DEVICE}")
    print(f"[INIT] site_left_prob = {SITE_LEFT_PROB}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    env = RetakeEnv(min_detonate_ticks=SPIKE_DETONATION_TICKS, max_detonate_ticks=SPIKE_DETONATION_TICKS, attacker_hold_radius=4)
    env.reset()
    sample_char = env.defenders()[0]
    obs_dim = env.build_observation(sample_char).shape[0]
    print(f"[INIT] obs_dim = {obs_dim} / n_actions = {N_ACTIONS}")

    net = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
    target_net = DuelingQNet(obs_dim, N_ACTIONS).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-5)
    replay = ReplayBuffer(Transition, capacity=100_000)

    num_episodes = EPISODE_COUNT
    batch_size = 256
    gamma = 0.99
    target_update_every = 1000
    epsilon_start, epsilon_end, epsilon_decay_episodes = 1.0, 0.02, int(EPISODE_COUNT * 0.8)
    # 解除カリキュラム: 最初はDEFUSE可能なら必ず選ばせ、EPISODE_COUNTの30%を
    # かけて0まで線形に減衰させる(epsilonの減衰より短め)。
    defuse_force_prob_start, defuse_force_prob_decay_episodes = 1.0, int(DEFUSE_LERNING_EPISODE_COUNT)

    global_step = 0
    win_history = deque(maxlen=200)
    best_win_rate = -1.0
    # 💡追加: 単発evalの結果は分散が大きい(200エピソードでも±0.1以上振れる)ため、
    # search phase(ov1_train_defender_search.py)と同様に直近数回のeval win_rateを
    # 平滑化してからbest判定する。これにより「たまたま良かった1回」を誤ってbestとして
    # 保存したり、逆に一時的な崩壊の谷でbest更新を止めてしまうことを防ぐ。
    eval_win_rate_history = deque(maxlen=5)


    # 診断用: 起動時に一度だけ出力
    for name, pos in zip(ROSTER_ORDER, DEFENDER_SPAWNS):
        for label, cell in (("LEFT", PLANT_POSITIONS["left"]), ("RIGHT", PLANT_POSITIONS["right"])):
            dmap = ov1_common_rl.bfs_distance_map(GRID, cell)
            print(f"{name}@{pos} -> {label}{cell}: dist={dmap[pos[0], pos[1]]}")
    print(f"retake spawns={len(DEFENDER_SPAWNS)}, A={len(ATTACKER_SPAWNS['left'])}, "
          f"B={len(ATTACKER_SPAWNS['right'])}, ability_targets="
          f"{ {key: len(value) for key, value in ABILITY_TARGET_POINTS.items()} }")
    print(f"min_detonate_ticks={SPIKE_DETONATION_TICKS}, DEFUSE_REQUIRED_TICKS={DEFUSE_REQUIRED_TICKS}")

    start_time = time.perf_counter()

    log_loss_sum = 0.0
    log_loss_count = 0
    log_train_steps = 0
    log_q_mean_sum = 0.0
    log_q_max_max = -float("inf")
    log_target_mean_sum = 0.0

    for episode in range(1, num_episodes + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0,
            1.0 - episode / epsilon_decay_episodes
        )

        defuse_force_prob = defuse_force_prob_start * max(
            0.0,
            1.0 - episode / defuse_force_prob_decay_episodes
        )
        defused, ticks_used = run_episode(env, net, target_net, replay, epsilon, obs_dim, defuse_force_prob)
        win_history.append(1 if defused else 0)

        for _ in range(max(1, ticks_used)):
            result = train_step(net, target_net, optimizer, replay, batch_size, gamma)
            if result is not None:
                loss, (q_mean, q_max, target_mean) = result
                soft_update(target_net, net, tau=0.001)
                log_loss_sum += loss
                log_loss_count += 1
                log_train_steps += 1
                log_q_mean_sum += q_mean
                log_q_max_max = max(log_q_max_max, q_max)
                log_target_mean_sum += target_mean
            global_step += 1

        num_eval_episodes = 200
        if episode % 200 == 0:
            print(f"[EVAL EP {episode}/{EPISODE_COUNT}]")

            eval_win_rate, eval_entered_rate = evaluate_greedy(
                env, net, obs_dim, num_eval_episodes=200
            )

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter()

            avg_loss = (
                log_loss_sum / log_loss_count
                if log_loss_count > 0
                else 0.0
            )
            avg_q_mean = (
                log_q_mean_sum / log_loss_count
                if log_loss_count > 0
                else 0.0
            )
            avg_target_mean = (
                log_target_mean_sum / log_loss_count
                if log_loss_count > 0
                else 0.0
            )
            # 直近区間(500ep)ごとの平均に戻す。累積のままだと発散の実態が薄まって見える。
            log_loss_sum = 0.0
            log_loss_count = 0
            q_max_snapshot = log_q_max_max
            log_q_mean_sum = 0.0
            log_q_max_max = -float("inf")
            log_target_mean_sum = 0.0

            print(
                f"greedy win_rate(200 episodes) = {eval_win_rate:.3f}, "
                f"eval_entered_rate={eval_entered_rate:.3f} "
                f"elapse={elapsed_time:.1f}"
            )
            print(
                f"    train: epsilon={epsilon:.4f}, "
                f"replay={len(replay)}, "
                f"train_steps={log_train_steps}, "
                f"avg_loss={avg_loss:.6f}, "
                f"global_step={global_step}"
            )
            print(
                f"    qvalues: avg_q_mean={avg_q_mean:.3f}, "
                f"q_max(区間内)={q_max_snapshot:.3f}, "
                f"avg_target_mean={avg_target_mean:.3f}"
            )
            eval_win_rate_history.append(eval_win_rate)
            eval_win_rate_smoothed = sum(eval_win_rate_history) / len(eval_win_rate_history)
            print(f"    smoothed(last<={eval_win_rate_history.maxlen}) win_rate = {eval_win_rate_smoothed:.3f}")

            if episode < EVAL_MIN_EPISODE:
                print(f"[SAVE skip] episode={episode} < EVAL_MIN_EPISODE={EVAL_MIN_EPISODE}")
            elif eval_win_rate_smoothed > best_win_rate:
                best_win_rate = eval_win_rate_smoothed
                torch.save(net.state_dict(), os.path.join(SAVE_DIR, "dqn_defender_retake_best_by_eval.pt"))
                print(f"  -> best model updated (smoothed win_rate={best_win_rate:.3f})")

    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "dqn_defender_retake_final.pt"))
    print("[DONE] training finished.")


if __name__ == "__main__":
    main()
