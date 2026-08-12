"""
gc_v1/train_attacker_macro_gc.py

Ghost Champions 専用 Attacker Macro / Strategy DQN。

目的:
    既存の Carry / Escort / Retrieve / Guard より上位で、
    「ラウンド全体の戦術」を学習する。

学習するMacro action:
    A_RUSH
    B_RUSH
    MID_TO_B
    A_SPLIT
    B_SPLIT
    DEFAULT
    FAKE_A_TO_B
    FAKE_B_TO_A
    ROTATE_A_TO_B
    ROTATE_B_TO_A
    REHIT_A
    REHIT_B

重要:
- 5人の1マス単位の移動をDQNに直接学習させない。
- DQNは「今の状況で次にどの戦術状態へ移るか」を決める。
- 各戦術内での担当レーン・Staging・Split Entry・Rotate・Lurk等への
  低レベル移動はMacro mapを使ったBFSナビゲーションで実行する。
- 将来の実戦統合時は、Macro controllerが戦術・各選手のmacro_targetを決め、
  必要な局面で既存Carry/Escortへ委譲できる構造を想定する。

この学習環境はrun_game.py / battle_logic.py / controllers.pyをimportしない。
map_data.py / map_data_macro_gc.py / macro_config_gc.py /
character_stats_gc.py / game_core.py の定数・データだけを参照する。

保存先:
    data/attacker_macro_gc_data/
"""

from __future__ import annotations
from itertools import permutations

import math
import os
import random
import sys
import time
from collections import Counter, deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# import path
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from map_data import NEW_MAZE_STR
from map_data_macro_gc import (
    MACRO_ZONE_STR,
    MACRO_TACTICAL_STR,
    MACRO_ROTATE_INFO_STR,
    MACRO_CONTROL_STR,
    MACRO_ROLE_STR,
    ZONE_MARKERS,
    TACTICAL_MARKERS,
    ROTATE_INFO_MARKERS,
    CONTROL_MARKERS,
    ROLE_MARKERS,
    parse_layer,
    marker_groups,
    validate_layers,
)
from macro_config_gc import (
    GC_MACRO_STRATEGY_WEIGHTS,
    GC_MACRO_ROUTE_BIAS,
    GC_MACRO_GROUP_SIZES,
    GC_MACRO_DECISION_THRESHOLDS,
    GC_MACRO_TRANSITIONS,
    validate_macro_config,
)
from character_stats_gc import GC_ROSTER_ORDER
from game_core import ROUND_DURATION_TICKS


# ============================================================================
# Training config
# ============================================================================

EPISODE_COUNT = 20000
EVAL_INTERVAL = 250
EVAL_EPISODES = 200
PRINT_INTERVAL = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data") / "attacker_macro_gc_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BEST_PATH = DATA_DIR / "dqn_attacker_macro_gc_best_by_eval.pt"
MODEL_LATEST_PATH = DATA_DIR / "dqn_attacker_macro_gc_latest.pt"
MODEL_FINAL_PATH = DATA_DIR / "dqn_attacker_macro_gc_final.pt"

SEED = 20260811

MAX_MACRO_STEPS = 24
LOW_LEVEL_TICKS_PER_MACRO_STEP = 3

GAMMA = 0.985
LR = 2e-4
BATCH_SIZE = 256
REPLAY_CAPACITY = 200000
LEARNING_START = 4000
TARGET_SYNC = 1200

EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY_STEPS = 140000

HIDDEN = 192

# Defender配置の多様性。
# 実戦のSearch modelそのものをここへ埋め込むのではなく、
# A/B/Midへの人数配分を毎episode変え、Macro判断の入力を学ばせる。
DEFENDER_SETUP_WEIGHTS = {
    "BALANCED": 1.0,      # 2-1-2
    "A_HEAVY": 1.0,       # 3-1-1
    "B_HEAVY": 1.0,       # 1-1-3
    "MID_HEAVY": 0.8,     # 1-3-1
    "A_STACK": 0.45,      # 4-1-0
    "B_STACK": 0.45,      # 0-1-4
}

# Macro環境の戦闘・情報モデル。
INFO_DECAY = 0.11
INFO_GAIN_IN_AREA = 0.22
INFO_GAIN_SITE = 0.40
INFO_GAIN_VISIBLE = 0.32

CONTROL_GAIN = 0.25
CONTROL_DECAY = 0.06

# Defenderは見られた/圧力を受けた方面へ少し寄る。
DEFENDER_ROTATE_PROB = 0.22
FAKE_PULL_BONUS = 0.28

# 実戦の「敵を引きつけて逆サイトへ」を評価するための最低圧力。
FAKE_PRESSURE_MIN = 0.45

# ---------------------------------------------------------------------------
# Macro reward shaping v3
# ---------------------------------------------------------------------------
# v2ではFake/Control等の途中報酬を稼ぎ続ける局所解が出たため、
# v3では「途中成果は小さく、Plantまでつながった戦術履歴を大きく評価」する。

MACRO_STEP_COST = -0.10                 # 長考・無限Default/Fakeを抑える
STRATEGY_SWITCH_COST = -0.04            # 不要な頻繁切替を軽く抑える

INFO_PROGRESS_REWARD_SCALE = 0.20       # 情報取得は小さな途中報酬
CONTROL_PROGRESS_REWARD_SCALE = 0.30    # Control取得も小さな途中報酬
DISTRIBUTION_REWARD = 0.04              # 分散の小さな補助報酬

SMART_ROTATE_INTERMEDIATE = 0.35        # 正しいRotate判断の途中報酬
BAD_COMMIT_PENALTY = -1.25              # 厚い側へ明確にコミット
BAD_ROTATE_PENALTY = -0.65              # 情報と逆方向のRotate
RUSH_REPEAT_START = 3
RUSH_REPEAT_PENALTY = -0.10

# ---------------------------------------------------------------------------
# v10: situation-aware strategy selection
# ---------------------------------------------------------------------------
# 戦術を「選んだ瞬間」だけ現在の情報との適合度を小さく評価する。
# Plant報酬を主役に保つため、最大でも±0.38。
STRATEGY_SUITABILITY_REWARD_SCALE = 0.42
STRATEGY_SUITABILITY_SCORE_CLIP = 1.0

# v11:
# 新戦術の絶対適性ではなく、
#   new_suitability - current_suitability
# の改善量だけを評価する。
#
# 改善が小さい切替では報酬を出さず、既存のSTRATEGY_SWITCH_COSTだけ払う。
SUITABILITY_SWITCH_DEADZONE = 0.15

# Rush / Default / MID_TO_B / Re-hit などにも短い最低継続時間を設ける。
# Split / Fake / Rotateは既存の専用Option lockをそのまま使う。
NORMAL_STRATEGY_COMMIT_STEPS = 3

# 明確なSmart Rotate条件が成立した時だけ、通常戦術の短期lockを解除できる。
ALLOW_URGENT_SMART_ROTATE_BREAK = True

# ---------------------------------------------------------------------------
# v12: "simplest sufficient tactic" preference
# ---------------------------------------------------------------------------
# サイトが明確に薄いなら、複雑なSplit/Fakeより直接Rushを優先する。
DIRECT_RUSH_LIGHT_BONUS = 0.22

# Splitは「サイトが空いている時」ではなく、
# 正面に1～2人いてMidから挟む価値がある時に最も評価する。
SPLIT_LIGHT_SITE_COMPLEXITY_PENALTY = -0.22
SPLIT_MODERATE_SITE_BONUS = 0.28
SPLIT_MID_OPEN_BONUS = 0.20
SPLIT_MID_CONTROL_BONUS = 0.12

# Fakeは本命サイトが既にガラ空きなら、わざわざ逆側で時間を使う価値を下げる。
FAKE_DIRECT_EXECUTE_OPPORTUNITY_PENALTY = -0.32

# Fakeが本当に有効なのは、
# Fake側に人数がいて、本命側が「薄いが完全な空きではない」時。
FAKE_PULL_VALUE_BONUS = 0.22

# Rush / Split / Fake の選択傾向をEVALで直接確認するための診断。
# 実際の報酬値には影響しない。
COMPLEXITY_DIAG_ENABLED = True

# v13: Rushで十分な状況では、理由のない複雑戦術に切替時1回だけコスト。
UNNECESSARY_COMPLEXITY_COST = {
    "A_SPLIT": -0.15,
    "B_SPLIT": -0.15,
    "FAKE_A_TO_B": -0.25,
    "FAKE_B_TO_A": -0.25,
    "MID_TO_B": -0.10,
}

# 本番EVALはFREE固定。Option確認用EVALは小さく別枠で回す。
AUX_EVAL_EPISODES = 50

# v14: DEFAULTを本当の情報収集フェーズにする。
DEFAULT_INFO_HOLD_ENABLED = True
BOTH_SITE_INFO_ACQUIRED_REWARD = 0.45
DEFAULT_INFO_HOLD_MAX_MACRO_STEPS = 7

# v15: 1人Scoutでもdecay(0.11/tick)を上回って情報を蓄積できる最低gain。
INFO_MIN_GAIN_WHILE_OCCUPIED = 0.18

# DEFAULTを短い情報収集Optionとして扱う最大継続step。
# A/B両情報が揃えばこの上限より前でも解除する。
# v16: 専用Scoutが実際にINFOへ到達して取得する時間を確保。
# v15のFREE評価ではboth-site初取得が平均10～13 Macro step付近だった。
DEFAULT_INFO_OPTION_MAX_AGE = 12

# A/B Scout取得後はLurkへ移る確率。
DEFAULT_SCOUT_TO_LURK_PROB = 0.55

# v17: Scout pair selection / movement priority
# A/B Scoutを別々に貪欲選択せず、2人組全候補から
# max(A距離, B距離) を最小化し、同点なら合計距離を最小化する。
DEFAULT_SCOUT_PAIR_MINIMAX = True

# DEFAULT情報収集中はScoutをMainより先に移動させる。
DEFAULT_SCOUT_MOVEMENT_PRIORITY = True

# v18: ScoutのINFO targetは確率選択せず、BFS最短セルへ完全固定する。
DEFAULT_SCOUT_USE_EXACT_NEAREST_INFO = True

# v20: DEFAULTを「本命側Lead + 逆側Scout + Mid + Main x2」に再設計。
DEFAULT_USE_SINGLE_OPPOSITE_SCOUT = True
DEFAULT_MAIN_LEAD_HOLD_INFO = True

# 逆側ScoutはINFO取得後LURKへ移る。
DEFAULT_OPPOSITE_SCOUT_TO_LURK_PROB = 0.80

# Plant時にまとめて与える「戦術完遂」ボーナス。
PLANT_BASE_REWARD = 8.0
PLANT_AFTER_DEFAULT_BONUS = 1.2
PLANT_AFTER_SMART_ROTATE_BONUS = 2.5
PLANT_AFTER_FAKE_BONUS = 2.2
PLANT_AFTER_SPLIT_BONUS = 2.4
PLANT_AFTER_LURK_BONUS = 0.8
PLANT_AFTER_FLANK_CUT_BONUS = 1.0
PLANT_FAST_BONUS_MAX = 1.2              # 速くPlantできたほど追加
PLANT_FAST_BONUS_MIN_TIME_RATIO = 0.35

# Split成立判定:
# main側とsupport側が別経路から同じサイトへ到達し、
# 近い時間幅で双方がサイト/入口へ入ったことを条件にする。
SPLIT_MAIN_READY_PRESSURE = 0.28
SPLIT_SUPPORT_READY_MID_CONTROL = 0.28

# Fake成立判定:
# fake側の圧力が閾値を超えた後、逆サイトへtargetを切り替えたことを記録。
FAKE_TRIGGER_PRESSURE = 0.35

# Default成立判定:
# 十分な情報を取ってから別戦術へ切り替えたか。
DEFAULT_INFO_TRIGGER = 0.60

# v4: イベント履歴を一定期間保持する。
# Aを見た後にBを確認するまでにA情報が減衰してもRotate判断へ使える。
INFO_MEMORY_MAX_TICKS = 24
INFO_MEMORY_MIN_CONF = 0.55

# v21: staged information confidence
INFO_CONF_LOW = 0.28
INFO_CONF_MEDIUM = 0.42
INFO_CONF_HIGH = 0.62

INFO_GAIN_FORWARD_CONTROL = 0.085
INFO_GAIN_DEEP_CONTROL = 0.145
INFO_GAIN_INFO_AREA_V21 = 0.22
INFO_GAIN_SITE_V21 = 0.18

INFO_EST_ALPHA_LOW = 0.12
INFO_EST_ALPHA_MEDIUM = 0.24
INFO_EST_ALPHA_HIGH = 0.42

# Rotate: both sides need at least MEDIUM. Normally one side must be HIGH;
# two MEDIUM observations are enough only when the estimated count gap is large.
ROTATE_MIN_CONF = INFO_CONF_MEDIUM
ROTATE_STRONG_CONF = INFO_CONF_HIGH
ROTATE_MEDIUM_COUNT_GAP = 2.0

# v22: 片側HIGH情報 + Mid controlから逆siteの人数上限を推論する。
ROTATE_INFER_SOURCE_CONF = INFO_CONF_HIGH
ROTATE_INFER_MID_FORWARD_MIN = 0.55
ROTATE_INFER_MID_DEEP_MIN = 0.35
ROTATE_INFER_HEAVY_MIN = 2.5
ROTATE_INFER_LIGHT_MAX = 2.0
ROTATE_INFER_CONF = 0.52

# Fakeは圧力を掛けた後、別のMacro actionへ切り替えても追跡する。
FAKE_FOLLOWUP_WINDOW_MACRO_STEPS = 6

# Splitは横挟み・深い180度挟みのどちらのSplit Entryでも、
# 「Entryへ実際に到達した時刻」を記録する。
# 深いEntryは時間が掛かるのでv3より同期窓を少し広げる。
SPLIT_ENTRY_WINDOW_MACRO_STEPS = 4

# v5: 情報不足のまま深くコミットすることへの軽いコスト。
# 開幕Rush自体は残すため、序盤は免除する。
UNCERTAIN_COMMIT_GRACE_MACRO_STEPS = 2
UNCERTAIN_COMMIT_PENALTY_ONE_SIDE = -0.12
UNCERTAIN_COMMIT_PENALTY_NO_INFO = -0.30
UNCERTAIN_DEEP_CONTROL_THRESHOLD = 0.35

# v5: 戦術練習シナリオ。
# すべてを強制せず、一定割合だけ明確な練習episodeにする。
CURRICULUM_FREE_PROB = 0.55
CURRICULUM_SPLIT_PROB = 0.15
CURRICULUM_FAKE_PROB = 0.15
CURRICULUM_ROTATE_PROB = 0.15

# 練習シナリオ中もDQNは毎step自由にactionを選べる。
# ただし初期状況/初期戦術を成功しやすい条件に寄せる。
CURRICULUM_BONUS_SCALE = 0.55

# ===========================================================================
# v6: temporally extended Macro Options
# ===========================================================================
# Split/Fake/Rotateは「1stepだけの命令」ではなく、一連の戦術として完遂させる。
# LOW_LEVEL_TICKS_PER_MACRO_STEP=3なので、14step = 最大42 low-level ticks。
SPLIT_OPTION_COMMIT_STEPS = 14
FAKE_OPTION_COMMIT_STEPS = 12
ROTATE_OPTION_COMMIT_STEPS = 6

# ---------------------------------------------------------------------------
# v7 Fake Option: SELL -> ROTATE -> EXECUTE
# ---------------------------------------------------------------------------
# Pressureだけでなく「前目を取って一定時間見せた」場合もFake成立準備とする。
FAKE_FORWARD_TRIGGER_CONTROL = 0.25
FAKE_SELL_MIN_PLAYERS = 2
FAKE_SELL_DWELL_STEPS = 2

# 逆サイトへ2人以上再展開したらROTATE完了 -> EXECUTEへ。
FAKE_REDEPLOY_MIN_PLAYERS = 2

# EXECUTE移行直後に別Macro戦術へ上書きされない最低時間。
FAKE_EXECUTE_LOCK_STEPS = 3

# Split時、MID_STAGING全体からランダムに選ぶと反対サイト側の4へ行く場合がある。
# 先に今回使うSplit Entry(5/6)を決め、そのEntryへ近い4だけを候補にする。
SPLIT_STAGING_TOP_K = 8

# Fakeのtriggerと実際の再展開開始に同じpressure基準を使う。
# v5は履歴trigger=0.35、実移動=FAKE_PRESSURE_MIN(0.45)で不一致だった。
FAKE_PRESSURE_MIN = FAKE_TRIGGER_PRESSURE

# v6ではremembered infoもDQN観測へ直接入れる。
# 既存55次元の末尾に count/confidence/age を A/B/MID 各3個追加する。
MEMORY_FEATURE_DIM = 9

# 1episode内イベント履歴を使い、Plant時にまとめて評価する。


# ============================================================================
# Strategy/action definitions
# ============================================================================

STRATEGIES = [
    "A_RUSH",
    "B_RUSH",
    "MID_TO_B",
    "A_SPLIT",
    "B_SPLIT",
    "DEFAULT",
    "FAKE_A_TO_B",
    "FAKE_B_TO_A",
    "ROTATE_A_TO_B",
    "ROTATE_B_TO_A",
    "REHIT_A",
    "REHIT_B",
]
STRATEGY_TO_INDEX = {name: i for i, name in enumerate(STRATEGIES)}
N_ACTIONS = len(STRATEGIES)

SIDE_A = "A"
SIDE_B = "B"
SIDE_MID = "MID"


# ============================================================================
# Map helpers
# ============================================================================

def _parse_base_grid(text: str) -> np.ndarray:
    lines = [
        line.strip()
        for line in text.strip("\n").split("\n")
        if line.strip()
    ]
    return np.asarray([[int(ch) for ch in line] for line in lines], dtype=np.int8)


BASE_GRID = _parse_base_grid(NEW_MAZE_STR)
HEIGHT, WIDTH = BASE_GRID.shape

validate_layers()
validate_macro_config()

ZONE_GRID = np.asarray(parse_layer(MACRO_ZONE_STR), dtype=np.int8)
TACTICAL_GRID = np.asarray(parse_layer(MACRO_TACTICAL_STR), dtype=np.int8)
ROTATE_INFO_GRID = np.asarray(parse_layer(MACRO_ROTATE_INFO_STR), dtype=np.int8)
CONTROL_GRID = np.asarray(parse_layer(MACRO_CONTROL_STR), dtype=np.int8)
ROLE_GRID = np.asarray(parse_layer(MACRO_ROLE_STR), dtype=np.int8)

for name, grid in {
    "ZONE": ZONE_GRID,
    "TACTICAL": TACTICAL_GRID,
    "ROTATE_INFO": ROTATE_INFO_GRID,
    "CONTROL": CONTROL_GRID,
    "ROLE": ROLE_GRID,
}.items():
    if grid.shape != BASE_GRID.shape:
        raise RuntimeError(
            f"{name} macro map shape={grid.shape} != base map={BASE_GRID.shape}"
        )

ZONE_CELLS = marker_groups(MACRO_ZONE_STR, ZONE_MARKERS)
TACTICAL_CELLS = marker_groups(MACRO_TACTICAL_STR, TACTICAL_MARKERS)
ROTATE_INFO_CELLS = marker_groups(MACRO_ROTATE_INFO_STR, ROTATE_INFO_MARKERS)
CONTROL_CELLS = marker_groups(MACRO_CONTROL_STR, CONTROL_MARKERS)
ROLE_CELLS = marker_groups(MACRO_ROLE_STR, ROLE_MARKERS)

ATTACKER_SPAWNS = list(zip(*np.where(BASE_GRID == 3)))
DEFENDER_SPAWNS = list(zip(*np.where(BASE_GRID == 4)))
PLANT_CELLS = list(zip(*np.where(BASE_GRID == 2)))

if len(ATTACKER_SPAWNS) < 5:
    raise RuntimeError(f"Attacker spawn cells不足: {len(ATTACKER_SPAWNS)}")
if len(DEFENDER_SPAWNS) < 5:
    raise RuntimeError(f"Defender spawn cells不足: {len(DEFENDER_SPAWNS)}")


CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def walkable(pos):
    r, c = int(pos[0]), int(pos[1])
    return 0 <= r < HEIGHT and 0 <= c < WIDTH and BASE_GRID[r, c] != 1


def bfs_distance_map_multi(goals):
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    q = deque()

    for goal in goals:
        r, c = int(goal[0]), int(goal[1])
        if walkable((r, c)) and dist[r, c] < 0:
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < HEIGHT
                and 0 <= nc < WIDTH
                and BASE_GRID[nr, nc] != 1
                and dist[nr, nc] < 0
            ):
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def bfs_distance_map(goal):
    return bfs_distance_map_multi([goal])


DIST_CACHE = {}


def dist_map_for_cells(key, cells):
    cache_key = (key, tuple(cells))
    if cache_key not in DIST_CACHE:
        DIST_CACHE[cache_key] = bfs_distance_map_multi(cells)
    return DIST_CACHE[cache_key]


def nearest_distance(pos, cells):
    if not cells:
        return 999
    dm = dist_map_for_cells("nearest", tuple(cells))
    d = int(dm[int(pos[0]), int(pos[1])])
    return d if d >= 0 else 999


def choose_weighted_candidate(cells, origin, bias=0.0):
    """候補数可変。BFS距離順とbiasから候補を確率選択する。

    bias > 0: 近い候補寄り
    bias < 0: 遠い候補寄り
    bias == 0: 均等
    """
    cells = list(cells)
    if not cells:
        return None
    if len(cells) == 1:
        return tuple(cells[0])

    scored = []
    for cell in cells:
        d = nearest_distance(origin, [cell])
        scored.append((tuple(cell), d))

    scored.sort(key=lambda x: x[1])

    if abs(float(bias)) < 1e-9:
        return random.choice([cell for cell, _ in scored])

    n = len(scored)
    weights = []
    for rank, (_cell, _d) in enumerate(scored):
        x = 0.0 if n <= 1 else rank / (n - 1)
        # positive bias -> low rank(near) is heavier
        exponent = (-float(bias) * 2.0 * x)
        weights.append(math.exp(exponent))

    return random.choices(
        [cell for cell, _ in scored],
        weights=weights,
        k=1,
    )[0]


def step_toward(pos, target, occupied):
    """壁+BFS。occupiedを避けられる範囲で1歩進む。"""
    start = (int(pos[0]), int(pos[1]))
    goal = (int(target[0]), int(target[1]))
    if start == goal:
        return start

    dm = bfs_distance_map(goal)
    current_d = int(dm[start[0], start[1]])
    candidates = []

    for dr, dc in CARDINAL:
        nxt = (start[0] + dr, start[1] + dc)
        if not walkable(nxt) or nxt in occupied:
            continue
        d = int(dm[nxt[0], nxt[1]])
        if d >= 0:
            candidates.append((d, random.random(), nxt))

    if not candidates:
        return start

    candidates.sort()
    if current_d >= 0:
        better = [row for row in candidates if row[0] < current_d]
        if better:
            return better[0][2]
    return candidates[0][2]


def normalized_bfs_distance(pos, cells):
    d = nearest_distance(pos, cells)
    if d >= 999:
        return 1.0
    return min(1.0, d / 60.0)


# ============================================================================
# Team state
# ============================================================================

class MacroUnit:
    def __init__(self, name, team, pos, has_spike=False):
        self.name = str(name)
        self.team = team
        self.pos = tuple(pos)
        self.is_alive = True
        self.has_spike = bool(has_spike)
        self.group = "MAIN"
        self.role = "NORMAL"


# ============================================================================
# Strategy planner
# ============================================================================

def _lane_cells(side):
    return {
        SIDE_A: ZONE_CELLS["A_LANE"],
        SIDE_B: ZONE_CELLS["B_LANE"],
        SIDE_MID: ZONE_CELLS["MID_LANE"],
    }[side]


def _site_cells(side):
    return {
        SIDE_A: ZONE_CELLS["A_SITE"],
        SIDE_B: ZONE_CELLS["B_SITE"],
    }[side]


def _staging_cells(side):
    return {
        SIDE_A: TACTICAL_CELLS["A_STAGING"],
        SIDE_B: TACTICAL_CELLS["B_STAGING"],
        SIDE_MID: TACTICAL_CELLS["MID_STAGING"],
    }[side]


def _info_cells(side):
    return {
        SIDE_A: ROTATE_INFO_CELLS["A_INFO"],
        SIDE_B: ROTATE_INFO_CELLS["B_INFO"],
        SIDE_MID: ROTATE_INFO_CELLS["MID_INFO"],
    }[side]


def _forward_control_cells(side):
    return {
        SIDE_A: CONTROL_CELLS["A_FORWARD_CONTROL"],
        SIDE_B: CONTROL_CELLS["B_FORWARD_CONTROL"],
        SIDE_MID: CONTROL_CELLS["MID_FORWARD_CONTROL"],
    }[side]


def _deep_control_cells(side):
    return {
        SIDE_A: CONTROL_CELLS["A_DEEP_CONTROL"],
        SIDE_B: CONTROL_CELLS["B_DEEP_CONTROL"],
        SIDE_MID: CONTROL_CELLS["MID_DEEP_CONTROL"],
    }[side]


def _lurk_cells(side):
    return {
        SIDE_A: ROLE_CELLS["A_LURK"],
        SIDE_B: ROLE_CELLS["B_LURK"],
        SIDE_MID: ROLE_CELLS["MID_LURK"],
    }[side]


def _cut_cells(side):
    return {
        SIDE_A: ROLE_CELLS["A_FLANK_CUT"],
        SIDE_B: ROLE_CELLS["B_FLANK_CUT"],
        SIDE_MID: ROLE_CELLS["MID_FLANK_CUT"],
    }[side]


def choose_split_staging(target_side, origin, planned_entry):
    """planned_entryへ近いMID_STAGING候補から選ぶ。

    A/Bそれぞれ専用の数字を増やさなくても、地理的にEntryへ近い4を
    自動的に同じSplit routeのStagingとして扱う。
    """
    staging = list(_staging_cells(SIDE_MID))
    if not staging:
        return None
    if planned_entry is None:
        return choose_weighted_candidate(staging, origin, 0.0)

    entry_dm = bfs_distance_map(tuple(planned_entry))
    scored = []
    for cell in staging:
        d_entry = int(entry_dm[int(cell[0]), int(cell[1])])
        if d_entry < 0:
            d_entry = 999
        scored.append((d_entry, tuple(cell)))

    scored.sort(key=lambda row: row[0])
    top_k = max(1, min(int(SPLIT_STAGING_TOP_K), len(scored)))
    good = [cell for _d, cell in scored[:top_k]]

    # そのEntryに近い4の中では、現在地から極端に遠いものを避ける。
    return choose_weighted_candidate(good, origin, +0.8)


def _strategy_is_split(strategy):
    return strategy in {"A_SPLIT", "B_SPLIT"}


def _strategy_is_fake(strategy):
    return strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}


def _strategy_is_rotate(strategy):
    return strategy in {"ROTATE_A_TO_B", "ROTATE_B_TO_A"}


def target_for_side(side, phase, origin):
    """Macro mapから低レベル目標地点を選ぶ。"""
    if phase == "STAGING":
        key = f"{side}_STAGING" if side != SIDE_MID else "MID_STAGING"
        return choose_weighted_candidate(
            _staging_cells(side),
            origin,
            GC_MACRO_ROUTE_BIAS.get(key, 0.0),
        )

    if phase == "LANE":
        cells = _lane_cells(side)
        return choose_weighted_candidate(cells, origin, +1.0)

    if phase == "INFO":
        return choose_weighted_candidate(_info_cells(side), origin, +0.7)

    if phase == "FORWARD":
        return choose_weighted_candidate(_forward_control_cells(side), origin, +0.7)

    if phase == "DEEP":
        return choose_weighted_candidate(_deep_control_cells(side), origin, +0.5)

    if phase == "SITE":
        return choose_weighted_candidate(_site_cells(side), origin, +0.8)

    if phase == "LURK":
        return choose_weighted_candidate(_lurk_cells(side), origin, 0.0)

    if phase == "CUT":
        return choose_weighted_candidate(_cut_cells(side), origin, 0.0)

    if phase == "RESET":
        return choose_weighted_candidate(
            ROTATE_INFO_CELLS["SAFE_RESET"],
            origin,
            GC_MACRO_ROUTE_BIAS.get("SAFE_RESET", 0.0),
        )

    return None


# ============================================================================
# Defender setup / pseudo-visibility
# ============================================================================

DEFENDER_SETUP_COUNTS = {
    "BALANCED": {SIDE_A: 2, SIDE_MID: 1, SIDE_B: 2},
    "A_HEAVY": {SIDE_A: 3, SIDE_MID: 1, SIDE_B: 1},
    "B_HEAVY": {SIDE_A: 1, SIDE_MID: 1, SIDE_B: 3},
    "MID_HEAVY": {SIDE_A: 1, SIDE_MID: 3, SIDE_B: 1},
    "A_STACK": {SIDE_A: 4, SIDE_MID: 1, SIDE_B: 0},
    "B_STACK": {SIDE_A: 0, SIDE_MID: 1, SIDE_B: 4},
}


def weighted_choice(weight_dict):
    names = list(weight_dict)
    weights = [max(0.0, float(weight_dict[n])) for n in names]
    return random.choices(names, weights=weights, k=1)[0]


def side_of_pos(pos):
    r, c = int(pos[0]), int(pos[1])

    # site is strongest
    if ZONE_GRID[r, c] == ZONE_MARKERS["A_SITE"]:
        return SIDE_A
    if ZONE_GRID[r, c] == ZONE_MARKERS["B_SITE"]:
        return SIDE_B

    value = int(ZONE_GRID[r, c])
    if value == ZONE_MARKERS["A_LANE"]:
        return SIDE_A
    if value == ZONE_MARKERS["B_LANE"]:
        return SIDE_B
    if value in {ZONE_MARKERS["MID"], ZONE_MARKERS["MID_LANE"]}:
        return SIDE_MID

    # fallback by nearest lane
    choices = []
    for side in (SIDE_A, SIDE_MID, SIDE_B):
        choices.append((nearest_distance(pos, _lane_cells(side)), side))
    return min(choices)[1]



# ============================================================================
# v5 Curriculum helpers
# ============================================================================

def choose_curriculum_mode():
    names = ["FREE", "SPLIT", "FAKE", "ROTATE"]
    weights = [
        CURRICULUM_FREE_PROB,
        CURRICULUM_SPLIT_PROB,
        CURRICULUM_FAKE_PROB,
        CURRICULUM_ROTATE_PROB,
    ]
    return random.choices(names, weights=weights, k=1)[0]


def choose_curriculum_strategy(mode):
    if mode == "SPLIT":
        return random.choice(["A_SPLIT", "B_SPLIT"])
    if mode == "FAKE":
        return random.choice(["FAKE_A_TO_B", "FAKE_B_TO_A"])
    if mode == "ROTATE":
        # Start from Default so the model has to convert information into rotate.
        return "DEFAULT"
    return None


# ============================================================================
# Environment
# ============================================================================

class MacroEnv:
    def __init__(self):
        self.tick = 0
        self.macro_step = 0
        self.attackers = []
        self.defenders = []
        self.current_strategy = "DEFAULT"
        self.previous_strategy = "DEFAULT"
        self.strategy_age = 0
        self.last_switch_step = -999
        self.target_site = SIDE_A
        self.initial_strategy = "DEFAULT"

        self.info_conf = {SIDE_A: 0.0, SIDE_B: 0.0, SIDE_MID: 0.0}
        self.enemy_est = {SIDE_A: 0.0, SIDE_B: 0.0, SIDE_MID: 0.0}
        self.control = {
            f"{SIDE_A}_FORWARD": 0.0,
            f"{SIDE_A}_DEEP": 0.0,
            f"{SIDE_B}_FORWARD": 0.0,
            f"{SIDE_B}_DEEP": 0.0,
            f"{SIDE_MID}_FORWARD": 0.0,
            f"{SIDE_MID}_DEEP": 0.0,
        }
        self.pressure = {SIDE_A: 0.0, SIDE_B: 0.0, SIDE_MID: 0.0}

        self.defender_setup = "BALANCED"
        self.planted = False
        self.plant_site = None
        self.done = False
        self.success = False
        self.reason = ""
        self.rotate_count = 0
        self.rehit_count = 0
        self.fake_value = 0.0
        self.split_sync_score = 0.0
        self.map_control_score = 0.0

        self.assignment = {}
        self.targets = {}

        # v5 curriculum scenario
        self.curriculum_mode = "FREE"
        self.curriculum_target = None
        self.curriculum_bonus_given = False

        # reward shaping state
        self._rewarded_events = set()
        self._rush_streak = 0
        self._last_rush_name = None

        # v3 episode-level tactical history
        self._default_had_good_info = False
        self._default_to_decision = False

        self._smart_rotate_completed = False
        self._smart_rotate_direction = None

        # v4: 最後に信頼できた各方面の敵情報を保持
        self._info_memory_count = {
            SIDE_A: 0.0,
            SIDE_B: 0.0,
            SIDE_MID: 0.0,
        }
        self._info_memory_conf = {
            SIDE_A: 0.0,
            SIDE_B: 0.0,
            SIDE_MID: 0.0,
        }
        self._info_memory_age = {
            SIDE_A: 10**9,
            SIDE_B: 10**9,
            SIDE_MID: 10**9,
        }

        self._fake_triggered = False
        self._fake_completed = False
        self._fake_direction = None
        self._fake_trigger_step = None

        # v7: Fake専用3段階Option
        self._fake_phase = "NONE"      # NONE / SELL / ROTATE / EXECUTE
        self._fake_side = None
        self._fake_real_side = None
        self._fake_sell_dwell = 0
        self._fake_group_names = set()
        self._fake_rotate_names = set()
        self._fake_execute_start_step = None
        # v9: Fake全体のstrategy_ageとは別に、各内部phaseの経過stepを持つ。
        self._fake_phase_start_step = None

        self._split_main_ready_step = None
        self._split_support_ready_step = None
        self._split_completed = False
        self._split_site = None
        self._split_tracking_site = None

        # v6: Supportごとに今回使うSplit Entryを固定する。
        # 横Entry / 180度背面Entryが複数あっても一貫して同じEntryへ向かう。
        self._split_planned_entry = {}

        self._lurk_touched = False
        self._cut_touched = False

        # --------------------------------------------------------------
        # v5 diagnostic only: learning behavior is NOT changed.
        # episode-level stage counters / flags
        # --------------------------------------------------------------
        self._diag_fake_selected = 0
        self._diag_fake_triggered = False
        self._diag_fake_opposite_redeploy = False
        self._diag_fake_expired = False
        self._diag_fake_max_opposite_count = 0

        self._diag_split_selected = 0
        self._diag_split_main_ready = False
        self._diag_split_support_entry = False
        self._diag_split_both_ready = False
        self._diag_split_window_miss = False
        self._diag_split_min_step_gap = None

        self._diag_rotate_selected = 0
        self._diag_rotate_both_info = False
        self._diag_rotate_heavy_light = False
        self._diag_rotate_wrong_heavy_light = False

        # v10: situation-aware selection diagnostics
        self._diag_suitability_sum = 0.0
        self._diag_suitability_count = 0
        self._diag_suitability_positive = 0
        self._diag_suitability_negative = 0

        # v11 delta-based switching diagnostics
        self._diag_switch_delta_sum = 0.0
        self._diag_switch_delta_count = 0
        self._diag_switch_improved = 0
        self._diag_switch_worse = 0
        self._diag_switch_deadzone = 0

        # v12 simple-vs-complex diagnostics
        self._diag_direct_rush_preferred = 0
        self._diag_split_preferred = 0
        self._diag_fake_preferred = 0
        self._diag_complex_when_rush_was_sufficient = 0
        self._both_site_info_rewarded = False
        self._diag_both_site_info_acquired = False
        self._diag_both_site_info_first_step = None
        self._diag_default_info_hold_ticks = 0
        self._diag_default_option_hold_steps = 0
        self._diag_default_option_info_complete = 0
        self._default_info_option_completion_counted = False

        # v16 deterministic DEFAULT roles
        self._default_a_scout_name = None
        self._default_b_scout_name = None
        self._default_mid_name = None
        self._default_main_names = []

        # v20 DEFAULT role layout
        self._default_main_lead_name = None
        self._default_opposite_scout_name = None
        self._default_main_side = None
        self._default_opposite_side = None
        self._default_main_lead_target = None
        self._default_opposite_scout_target = None

        self._diag_main_lead_info_reached = False
        self._diag_opposite_scout_info_reached = False
        self._diag_default_role_assignments = 0
        self._diag_a_scout_info_reached = False
        self._diag_b_scout_info_reached = False

        # v17 scout distance diagnostics
        self._diag_a_scout_start_dist = None
        self._diag_b_scout_start_dist = None
        self._diag_a_scout_final_dist = None
        self._diag_b_scout_final_dist = None

        # v18 actual chosen target diagnostics
        self._default_a_scout_target = None
        self._default_b_scout_target = None
        self._diag_a_scout_actual_target_start_dist = None
        self._diag_b_scout_actual_target_start_dist = None

        # v19 diagnostic only: DEFAULT Scout travel / termination reason
        self._diag_default_start_remaining_steps_sum = 0.0
        self._diag_default_start_count = 0

        self._diag_a_scout_move_attempts = 0
        self._diag_a_scout_forward_moves = 0
        self._diag_a_scout_stopped_moves = 0
        self._diag_b_scout_move_attempts = 0
        self._diag_b_scout_forward_moves = 0
        self._diag_b_scout_stopped_moves = 0

        self._diag_a_scout_died = False
        self._diag_b_scout_died = False

        self._diag_default_exit_info_complete = 0
        self._diag_default_exit_max_age = 0
        self._diag_default_exit_round_end = 0
        self._diag_default_exit_switched = 0

        # v23: inferred Rotate gate diagnostics
        self._diag_infer_any_high = 0
        self._diag_infer_heavy_est = 0
        self._diag_infer_mid_forward_ok = 0
        self._diag_infer_mid_deep_ok = 0
        self._diag_infer_mid_any_ok = 0
        self._diag_infer_opposite_bound_ok = 0
        self._diag_infer_final = 0

        self._default_diag_session_active = False
        self._default_diag_exit_recorded = False

    def reset(self, forced_strategy=None, forced_curriculum_mode=None):
        self.tick = 0
        self.macro_step = 0
        self.done = False
        self.success = False
        self.reason = ""
        self.planted = False
        self.plant_site = None
        self.rotate_count = 0
        self.rehit_count = 0
        self.fake_value = 0.0
        self.split_sync_score = 0.0
        self.map_control_score = 0.0
        self.curriculum_mode = "FREE"
        self.curriculum_target = None
        self.curriculum_bonus_given = False
        self._rewarded_events = set()
        self._rush_streak = 0
        self._last_rush_name = None

        self._default_had_good_info = False
        self._default_to_decision = False

        self._smart_rotate_completed = False
        self._smart_rotate_direction = None

        self._info_memory_count = {
            SIDE_A: 0.0,
            SIDE_B: 0.0,
            SIDE_MID: 0.0,
        }
        self._info_memory_conf = {
            SIDE_A: 0.0,
            SIDE_B: 0.0,
            SIDE_MID: 0.0,
        }
        self._info_memory_age = {
            SIDE_A: 10**9,
            SIDE_B: 10**9,
            SIDE_MID: 10**9,
        }

        self._fake_triggered = False
        self._fake_completed = False
        self._fake_direction = None
        self._fake_trigger_step = None

        # v7: Fake専用3段階Option
        self._fake_phase = "NONE"      # NONE / SELL / ROTATE / EXECUTE
        self._fake_side = None
        self._fake_real_side = None
        self._fake_sell_dwell = 0
        self._fake_group_names = set()
        self._fake_rotate_names = set()
        self._fake_execute_start_step = None
        # v9: Fake全体のstrategy_ageとは別に、各内部phaseの経過stepを持つ。
        self._fake_phase_start_step = None

        self._split_main_ready_step = None
        self._split_support_ready_step = None
        self._split_completed = False
        self._split_site = None
        self._split_tracking_site = None

        # v6: Supportごとに今回使うSplit Entryを固定する。
        # 横Entry / 180度背面Entryが複数あっても一貫して同じEntryへ向かう。
        self._split_planned_entry = {}

        self._lurk_touched = False
        self._cut_touched = False

        self._diag_fake_selected = 0
        self._diag_fake_triggered = False
        self._diag_fake_opposite_redeploy = False
        self._diag_fake_expired = False
        self._diag_fake_max_opposite_count = 0

        self._diag_split_selected = 0
        self._diag_split_main_ready = False
        self._diag_split_support_entry = False
        self._diag_split_both_ready = False
        self._diag_split_window_miss = False
        self._diag_split_min_step_gap = None

        self._diag_rotate_selected = 0
        self._diag_rotate_both_info = False
        self._diag_rotate_heavy_light = False
        self._diag_rotate_wrong_heavy_light = False

        # v10: situation-aware selection diagnostics
        self._diag_suitability_sum = 0.0
        self._diag_suitability_count = 0
        self._diag_suitability_positive = 0
        self._diag_suitability_negative = 0

        # v11 delta-based switching diagnostics
        self._diag_switch_delta_sum = 0.0
        self._diag_switch_delta_count = 0
        self._diag_switch_improved = 0
        self._diag_switch_worse = 0
        self._diag_switch_deadzone = 0

        # v12 simple-vs-complex diagnostics
        self._diag_direct_rush_preferred = 0
        self._diag_split_preferred = 0
        self._diag_fake_preferred = 0
        self._diag_complex_when_rush_was_sufficient = 0
        self._both_site_info_rewarded = False
        self._diag_both_site_info_acquired = False
        self._diag_both_site_info_first_step = None
        self._diag_default_info_hold_ticks = 0
        self._diag_default_option_hold_steps = 0
        self._diag_default_option_info_complete = 0
        self._default_info_option_completion_counted = False

        # v16 deterministic DEFAULT roles
        self._default_a_scout_name = None
        self._default_b_scout_name = None
        self._default_mid_name = None
        self._default_main_names = []

        # v20 DEFAULT role layout
        self._default_main_lead_name = None
        self._default_opposite_scout_name = None
        self._default_main_side = None
        self._default_opposite_side = None
        self._default_main_lead_target = None
        self._default_opposite_scout_target = None

        self._diag_main_lead_info_reached = False
        self._diag_opposite_scout_info_reached = False
        self._diag_default_role_assignments = 0
        self._diag_a_scout_info_reached = False
        self._diag_b_scout_info_reached = False

        # v17 scout distance diagnostics
        self._diag_a_scout_start_dist = None
        self._diag_b_scout_start_dist = None
        self._diag_a_scout_final_dist = None
        self._diag_b_scout_final_dist = None

        # v18 actual chosen target diagnostics
        self._default_a_scout_target = None
        self._default_b_scout_target = None
        self._diag_a_scout_actual_target_start_dist = None
        self._diag_b_scout_actual_target_start_dist = None

        # v19 diagnostic only: DEFAULT Scout travel / termination reason
        self._diag_default_start_remaining_steps_sum = 0.0
        self._diag_default_start_count = 0

        self._diag_a_scout_move_attempts = 0
        self._diag_a_scout_forward_moves = 0
        self._diag_a_scout_stopped_moves = 0
        self._diag_b_scout_move_attempts = 0
        self._diag_b_scout_forward_moves = 0
        self._diag_b_scout_stopped_moves = 0

        self._diag_a_scout_died = False
        self._diag_b_scout_died = False

        self._diag_default_exit_info_complete = 0
        self._diag_default_exit_max_age = 0
        self._diag_default_exit_round_end = 0
        self._diag_default_exit_switched = 0

        # v23: inferred Rotate gate diagnostics
        self._diag_infer_any_high = 0
        self._diag_infer_heavy_est = 0
        self._diag_infer_mid_forward_ok = 0
        self._diag_infer_mid_deep_ok = 0
        self._diag_infer_mid_any_ok = 0
        self._diag_infer_opposite_bound_ok = 0
        self._diag_infer_final = 0

        self._default_diag_session_active = False
        self._default_diag_exit_recorded = False

        self.info_conf = {SIDE_A: 0.0, SIDE_B: 0.0, SIDE_MID: 0.0}
        self.enemy_est = {SIDE_A: 0.0, SIDE_B: 0.0, SIDE_MID: 0.0}
        self.pressure = {SIDE_A: 0.0, SIDE_B: 0.0, SIDE_MID: 0.0}
        for key in self.control:
            self.control[key] = 0.0

        self.curriculum_mode = (
            forced_curriculum_mode
            if forced_curriculum_mode is not None
            else choose_curriculum_mode()
        )
        self.curriculum_target = None

        # 練習modeでは、狙った戦術が成立しやすいDefender配置を作る。
        if self.curriculum_mode == "ROTATE":
            if random.random() < 0.5:
                self.defender_setup = random.choice(["A_HEAVY", "A_STACK"])
                self.curriculum_target = "A_TO_B"
            else:
                self.defender_setup = random.choice(["B_HEAVY", "B_STACK"])
                self.curriculum_target = "B_TO_A"
        elif self.curriculum_mode == "FAKE":
            # Fake元サイトにある程度敵がいる方がpullを学びやすい。
            self.defender_setup = random.choice(
                ["BALANCED", "A_HEAVY", "B_HEAVY", "MID_HEAVY"]
            )
        elif self.curriculum_mode == "SPLIT":
            self.defender_setup = random.choice(
                ["BALANCED", "A_HEAVY", "B_HEAVY", "MID_HEAVY"]
            )
        else:
            self.defender_setup = weighted_choice(DEFENDER_SETUP_WEIGHTS)

        counts = DEFENDER_SETUP_COUNTS[self.defender_setup]

        # Attackers
        starts = list(ATTACKER_SPAWNS[:5])
        carrier_name = "Absol" if "Absol" in GC_ROSTER_ORDER else GC_ROSTER_ORDER[0]
        self.attackers = [
            MacroUnit(
                name,
                "A",
                starts[i % len(starts)],
                has_spike=(name == carrier_name),
            )
            for i, name in enumerate(GC_ROSTER_ORDER)
        ]

        # Defenders are placed in representative area cells.
        self.defenders = []
        defender_id = 0
        for side, count in counts.items():
            candidate_pool = list(_deep_control_cells(side)) or list(_forward_control_cells(side))
            if side in {SIDE_A, SIDE_B}:
                candidate_pool += list(_site_cells(side))
            if not candidate_pool:
                candidate_pool = list(DEFENDER_SPAWNS)
            random.shuffle(candidate_pool)
            for i in range(count):
                pos = candidate_pool[i % len(candidate_pool)]
                self.defenders.append(MacroUnit(f"D{defender_id}", "D", pos))
                defender_id += 1

        while len(self.defenders) < 5:
            self.defenders.append(
                MacroUnit(
                    f"D{defender_id}",
                    "D",
                    random.choice(DEFENDER_SPAWNS),
                )
            )
            defender_id += 1

        if forced_strategy is None:
            curriculum_initial = choose_curriculum_strategy(self.curriculum_mode)
            if curriculum_initial is not None:
                initial = curriculum_initial
            else:
                initial = weighted_choice(GC_MACRO_STRATEGY_WEIGHTS)
        else:
            initial = forced_strategy

        self.current_strategy = initial
        self.previous_strategy = initial
        self.initial_strategy = initial
        self.strategy_age = 0
        self.last_switch_step = -999

        self._apply_strategy_assignments(initial, initial=True)
        self._update_information()

        # ROTATE練習episodeだけ、片側が厚いという「直近情報」を少し与える。
        # action自体は強制せず、観測から正しいRotateを選ぶことを学ばせる。
        if self.curriculum_mode == "ROTATE":
            if self.curriculum_target == "A_TO_B":
                self._info_memory_count[SIDE_A] = 3.5
                self._info_memory_conf[SIDE_A] = 0.85
                self._info_memory_age[SIDE_A] = 0
                self._info_memory_count[SIDE_B] = 0.5
                self._info_memory_conf[SIDE_B] = 0.80
                self._info_memory_age[SIDE_B] = 0
            else:
                self._info_memory_count[SIDE_B] = 3.5
                self._info_memory_conf[SIDE_B] = 0.85
                self._info_memory_age[SIDE_B] = 0
                self._info_memory_count[SIDE_A] = 0.5
                self._info_memory_conf[SIDE_A] = 0.80
                self._info_memory_age[SIDE_A] = 0

        return self.build_observation()

    # ------------------------------------------------------------------
    # assignments
    # ------------------------------------------------------------------

    def _carrier(self):
        return next((a for a in self.attackers if a.is_alive and a.has_spike), None)

    def _living_attackers(self):
        return [a for a in self.attackers if a.is_alive]

    def _split_names(self, main_count):
        carrier = self._carrier()
        names = [a.name for a in self._living_attackers()]
        if carrier and carrier.name in names:
            names.remove(carrier.name)
            main = [carrier.name]
        else:
            main = []
        random.shuffle(names)
        while names and len(main) < main_count:
            main.append(names.pop())
        support = names
        return main, support

    def _fake_names(self, fake_count):
        """Fake役とRotate役を分ける。

        v6以前の_split_names()はcarrierをmain側へ固定するため、
        Fake時にはSpike carrierまでFake側へ送っていた。
        v7ではcarrierを原則Rotate/real-site側に残す。
        """
        living = self._living_attackers()
        carrier = self._carrier()

        non_carrier = [
            a.name for a in living
            if carrier is None or a.name != carrier.name
        ]
        random.shuffle(non_carrier)

        fake_count = max(0, min(int(fake_count), len(non_carrier)))
        fake = non_carrier[:fake_count]

        rotate = [
            a.name for a in living
            if a.name not in fake
        ]
        return fake, rotate

    def _fake_sides(self):
        if self._fake_direction == "A_TO_B":
            return SIDE_A, SIDE_B
        if self._fake_direction == "B_TO_A":
            return SIDE_B, SIDE_A
        return None, None

    def _set_fake_phase_targets(self, phase):
        """Fake内部フェーズに応じて5人のassignment/targetを更新する。"""
        fake_side, real_side = self._fake_sides()
        if fake_side is None or real_side is None:
            return

        living = self._living_attackers()
        self._fake_phase = phase
        self._fake_phase_start_step = self.macro_step

        if phase == "SELL":
            # Fake役はFake側Forwardへ。
            # Carrierを含むRotate役はMid stagingで待機してSpikeを守る。
            for a in living:
                if a.name in self._fake_group_names:
                    self.assignment[a.name] = (
                        fake_side, "FORWARD", "FAKE_SELL"
                    )
                    self.targets[a.name] = (
                        target_for_side(fake_side, "FORWARD", a.pos) or a.pos
                    )
                else:
                    self.assignment[a.name] = (
                        SIDE_MID, "STAGING", "FAKE_WAIT"
                    )
                    self.targets[a.name] = (
                        target_for_side(SIDE_MID, "STAGING", a.pos) or a.pos
                    )

        elif phase == "ROTATE":
            # Fakeを十分見せたら全員を反対サイト側へ移す。
            # まずStagingを経由し、まとまって再展開する。
            for a in living:
                self.assignment[a.name] = (
                    real_side, "STAGING", "FAKE_ROTATE"
                )
                self.targets[a.name] = (
                    target_for_side(real_side, "STAGING", a.pos)
                    or target_for_side(real_side, "SITE", a.pos)
                    or a.pos
                )

        elif phase == "EXECUTE":
            self._fake_execute_start_step = self.macro_step
            for a in living:
                self.assignment[a.name] = (
                    real_side, "SITE", "FAKE_EXECUTE"
                )
                self.targets[a.name] = (
                    target_for_side(real_side, "SITE", a.pos) or a.pos
                )

    def _fake_phase_age(self):
        if self._fake_phase_start_step is None:
            return 0
        return max(0, self.macro_step - self._fake_phase_start_step)

    def _fake_sell_trigger_ready(self):
        """Pressure OR Forward Control + 人数 + 滞在でSELL成立を判定する。"""
        fake_side, _real_side = self._fake_sides()
        if fake_side is None:
            return False

        fake_side_players = sum(
            1
            for a in self._living_attackers()
            if a.name in self._fake_group_names
            and side_of_pos(a.pos) == fake_side
        )

        if fake_side_players >= FAKE_SELL_MIN_PLAYERS:
            self._fake_sell_dwell += 1
        else:
            self._fake_sell_dwell = max(0, self._fake_sell_dwell - 1)

        pressure_ready = (
            self.pressure[fake_side] >= FAKE_TRIGGER_PRESSURE
        )
        control_ready = (
            self.control[f"{fake_side}_FORWARD"]
            >= FAKE_FORWARD_TRIGGER_CONTROL
            and fake_side_players >= FAKE_SELL_MIN_PLAYERS
            and self._fake_sell_dwell >= FAKE_SELL_DWELL_STEPS
        )
        return bool(pressure_ready or control_ready)

    def _advance_fake_rotate_targets(self):
        """v8: Fake ROTATE中、Staging到着後はForwardへ自動進行する。

        v7では全員をreal-side STAGINGへ送った後、そのtargetが固定されるため、
        Stagingに着いて止まり続けるケースがあった。
        v8では各キャラがStagingへ十分近づいたら、
        同じreal sideのFORWARDへtargetを更新する。
        """
        if self._fake_phase != "ROTATE":
            return

        _fake_side, real_side = self._fake_sides()
        if real_side is None:
            return

        late_rotate = (
            self._fake_phase_age()
            >= max(2, FAKE_OPTION_COMMIT_STEPS // 2)
        )

        for a in self._living_attackers():
            side, phase, role = self.assignment.get(
                a.name,
                (real_side, "STAGING", "FAKE_ROTATE"),
            )

            if role != "FAKE_ROTATE":
                continue

            target = self.targets.get(a.name)
            if target is None:
                continue

            # Staging到達（隣接を含む）でForwardへ進行。
            # v9: ROTATE時間の半分を使ってもStaging未到達なら、
            # 戦術停滞を避けてForwardへ直接切り替える。
            if phase == "STAGING" and (
                nearest_distance(a.pos, [target]) <= 1
                or late_rotate
            ):
                self.assignment[a.name] = (
                    real_side,
                    "FORWARD",
                    "FAKE_ROTATE",
                )
                self.targets[a.name] = (
                    target_for_side(real_side, "FORWARD", a.pos)
                    or target_for_side(real_side, "SITE", a.pos)
                    or a.pos
                )

    def _update_fake_option_phase(self):
        """低レベルtickごとにFake Optionを進行させる。"""
        if self.current_strategy not in {"FAKE_A_TO_B", "FAKE_B_TO_A"}:
            return

        fake_side, real_side = self._fake_sides()
        if fake_side is None or real_side is None:
            return

        if self._fake_phase == "SELL":
            if self._fake_sell_trigger_ready():
                self._fake_triggered = True
                self._fake_trigger_step = self.macro_step
                self._diag_fake_triggered = True
                self._set_fake_phase_targets("ROTATE")

        elif self._fake_phase == "ROTATE":
            # v8: Stagingで止めず、到着した選手からreal-side Forwardへ進める。
            self._advance_fake_rotate_targets()

            real_forward = set(_forward_control_cells(real_side))
            real_deep = set(_deep_control_cells(real_side))
            real_site = set(_site_cells(real_side))

            real_count = sum(
                1
                for a in self._living_attackers()
                if (
                    side_of_pos(a.pos) == real_side
                    or a.pos in real_forward
                    or a.pos in real_deep
                    or a.pos in real_site
                )
            )

            self._diag_fake_max_opposite_count = max(
                self._diag_fake_max_opposite_count,
                int(real_count),
            )

            if real_count >= FAKE_REDEPLOY_MIN_PLAYERS:
                self._diag_fake_opposite_redeploy = True
                self._fake_completed = True
                self.fake_value = max(self.fake_value, 1.0)
                self._set_fake_phase_targets("EXECUTE")

        elif self._fake_phase == "EXECUTE":
            # targetsはSITEのまま維持。
            pass

    def _pick_nearest_name(self, units, cells, excluded=None):
        """cellsへのBFS距離が最短の生存unit名を返す。"""
        excluded = set(excluded or ())
        candidates = [
            a for a in units
            if a.name not in excluded
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda a: (
                nearest_distance(a.pos, cells),
                a.name,
            )
        )
        return candidates[0].name

    def _nearest_info_cell_exact(self, origin, side):
        """指定sideのINFO候補からBFS距離が最短のセルを決定論的に返す。"""
        cells = list(_info_cells(side))
        if not cells:
            return None, None

        scored = []
        for cell in cells:
            d = nearest_distance(origin, [cell])
            scored.append((int(d), int(cell[0]), int(cell[1]), tuple(cell)))

        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        best_d, _r, _c, best_cell = scored[0]
        return best_cell, best_d

    def _pick_default_scout_pair_v17(self, living):
        """A/B INFOへの2人組をminimaxで選ぶ。

        非carrierを優先し、
          1) max(A距離, B距離)
          2) A距離+B距離
        の順に最小化する。
        """
        carrier = self._carrier()
        non_carrier = [
            a for a in living
            if carrier is None or a.name != carrier.name
        ]

        pool = non_carrier if len(non_carrier) >= 2 else list(living)
        if len(pool) < 2:
            return None, None

        best = None

        # ordered pair: first=A scout, second=B scout
        for a_unit, b_unit in permutations(pool, 2):
            a_dist = nearest_distance(a_unit.pos, _info_cells(SIDE_A))
            b_dist = nearest_distance(b_unit.pos, _info_cells(SIDE_B))

            score = (
                max(a_dist, b_dist),
                a_dist + b_dist,
                a_dist,
                b_dist,
                a_unit.name,
                b_unit.name,
            )

            if best is None or score < best[0]:
                best = (
                    score,
                    a_unit.name,
                    b_unit.name,
                    int(a_dist),
                    int(b_dist),
                )

        if best is None:
            return None, None

        _score, a_name, b_name, a_dist, b_dist = best
        self._diag_a_scout_start_dist = a_dist
        self._diag_b_scout_start_dist = b_dist

        return a_name, b_name

    def _assign_default_roles_v20(self, living):
        """DEFAULT = Main Lead + Opposite Scout + Mid + Main x2.

        本命側の情報はLeadが攻撃準備と並行して取得し、
        逆側だけ専任Scoutを送る。
        """
        if not living:
            return

        carrier = self._carrier()

        # 本命sideはランダム性を残す。
        main_side = random.choice([SIDE_A, SIDE_B])
        opposite_side = SIDE_B if main_side == SIDE_A else SIDE_A

        self.target_site = main_side
        self._default_main_side = main_side
        self._default_opposite_side = opposite_side

        used = set()

        # 逆側Scoutはcarrierを避け、逆INFOに最短のunit。
        non_carrier = [
            a for a in living
            if carrier is None or a.name != carrier.name
        ]
        opposite_scout = self._pick_nearest_name(
            non_carrier,
            _info_cells(opposite_side),
            used,
        )
        if opposite_scout is None:
            opposite_scout = self._pick_nearest_name(
                living,
                _info_cells(opposite_side),
                used,
            )
        if opposite_scout is not None:
            used.add(opposite_scout)

        # Main Leadもcarrierを避け、本命INFOへ最短。
        lead_pool = [
            a for a in non_carrier
            if a.name not in used
        ]
        if not lead_pool:
            lead_pool = [
                a for a in living
                if a.name not in used
            ]
        main_lead = self._pick_nearest_name(
            lead_pool,
            _info_cells(main_side),
            used,
        )
        if main_lead is not None:
            used.add(main_lead)

        # Mid担当
        mid_name = self._pick_nearest_name(
            living,
            _forward_control_cells(SIDE_MID),
            used,
        )
        if mid_name is not None:
            used.add(mid_name)

        # 残り2人がMain。carrierは通常ここに残る。
        main_names = [
            a.name for a in living
            if a.name not in used
        ]

        self._default_main_lead_name = main_lead
        self._default_opposite_scout_name = opposite_scout
        self._default_mid_name = mid_name
        self._default_main_names = list(main_names)

        # v16/v17 legacy fieldsもsideに合わせて埋め、既存診断を壊さない。
        self._default_a_scout_name = None
        self._default_b_scout_name = None
        if main_side == SIDE_A:
            self._default_a_scout_name = main_lead
            self._default_b_scout_name = opposite_scout
        else:
            self._default_b_scout_name = main_lead
            self._default_a_scout_name = opposite_scout

        self._diag_default_role_assignments += 1

        # Lead / ScoutのINFO targetを最短セルへ固定。
        self._default_main_lead_target = None
        self._default_opposite_scout_target = None

        if main_lead is not None:
            unit = next((x for x in living if x.name == main_lead), None)
            if unit is not None:
                cell, dist = self._nearest_info_cell_exact(unit.pos, main_side)
                self._default_main_lead_target = cell
                if main_side == SIDE_A:
                    self._default_a_scout_target = cell
                    self._diag_a_scout_start_dist = int(dist)
                    self._diag_a_scout_actual_target_start_dist = int(dist)
                else:
                    self._default_b_scout_target = cell
                    self._diag_b_scout_start_dist = int(dist)
                    self._diag_b_scout_actual_target_start_dist = int(dist)

        if opposite_scout is not None:
            unit = next((x for x in living if x.name == opposite_scout), None)
            if unit is not None:
                cell, dist = self._nearest_info_cell_exact(
                    unit.pos,
                    opposite_side,
                )
                self._default_opposite_scout_target = cell
                if opposite_side == SIDE_A:
                    self._default_a_scout_target = cell
                    self._diag_a_scout_start_dist = int(dist)
                    self._diag_a_scout_actual_target_start_dist = int(dist)
                else:
                    self._default_b_scout_target = cell
                    self._diag_b_scout_start_dist = int(dist)
                    self._diag_b_scout_actual_target_start_dist = int(dist)

        if main_lead is not None:
            self.assignment[main_lead] = (
                main_side,
                "INFO",
                "MAIN_LEAD",
            )

        if opposite_scout is not None:
            self.assignment[opposite_scout] = (
                opposite_side,
                "INFO",
                "OPPOSITE_SCOUT",
            )

        if mid_name is not None:
            self.assignment[mid_name] = (
                SIDE_MID,
                "FORWARD",
                "MID_CONTROL",
            )

        for name in main_names:
            self.assignment[name] = (
                main_side,
                "STAGING",
                "DEFAULT_MAIN",
            )

    def _start_default_diag_session_v19(self):
        """DEFAULTへ入った瞬間の残りMacro budgetを記録する。"""
        remaining = max(0, MAX_MACRO_STEPS - self.macro_step)
        self._diag_default_start_remaining_steps_sum += float(remaining)
        self._diag_default_start_count += 1
        self._default_diag_session_active = True
        self._default_diag_exit_recorded = False

    def _record_default_diag_exit_v19(self, reason):
        """1回のDEFAULT sessionにつき終了理由を1つだけ記録。"""
        if not self._default_diag_session_active:
            return
        if self._default_diag_exit_recorded:
            return

        if reason == "info_complete":
            self._diag_default_exit_info_complete += 1
        elif reason == "max_age":
            self._diag_default_exit_max_age += 1
        elif reason == "round_end":
            self._diag_default_exit_round_end += 1
        elif reason == "switched":
            self._diag_default_exit_switched += 1

        self._default_diag_exit_recorded = True
        self._default_diag_session_active = False

    def _assign_default_roles_v16(self, living):
        """必ず A Scout / B Scout / Mid Control / Main x2 を作る。

        - Spike carrierは原則Scoutにしない。
        - A/B Scoutは各INFOへ最短の非carrierを選ぶ。
        - Mid担当は残りからMID_FORWARDへ最短。
        - carrierを含む残り2人はMainとして候補サイト側へ進む。
        """
        carrier = self._carrier()
        non_carrier = [
            a for a in living
            if carrier is None or a.name != carrier.name
        ]

        used = set()

        if DEFAULT_SCOUT_PAIR_MINIMAX:
            a_scout, b_scout = self._pick_default_scout_pair_v17(living)
        else:
            a_scout = None
            b_scout = None

        # fallback
        if a_scout is None:
            a_scout = self._pick_nearest_name(
                non_carrier,
                _info_cells(SIDE_A),
                used,
            )
        if a_scout is not None:
            used.add(a_scout)

        if b_scout is None:
            b_scout = self._pick_nearest_name(
                non_carrier,
                _info_cells(SIDE_B),
                used,
            )
        if b_scout is not None:
            used.add(b_scout)

        # 万一non-carrierが不足する構成でも必ず役割を埋める。
        if a_scout is None:
            a_scout = self._pick_nearest_name(
                living,
                _info_cells(SIDE_A),
                used,
            )
            if a_scout is not None:
                used.add(a_scout)

        if b_scout is None:
            b_scout = self._pick_nearest_name(
                living,
                _info_cells(SIDE_B),
                used,
            )
            if b_scout is not None:
                used.add(b_scout)

        mid_name = self._pick_nearest_name(
            living,
            _forward_control_cells(SIDE_MID),
            used,
        )
        if mid_name is not None:
            used.add(mid_name)

        main_names = [
            a.name for a in living
            if a.name not in used
        ]

        # Main側はcarrierがいる側を優先しつつ、A/Bをランダム化して
        # 毎回同じ初手にはしない。
        preferred = random.choice([SIDE_A, SIDE_B])
        self.target_site = preferred

        self._default_a_scout_name = a_scout
        self._default_b_scout_name = b_scout

        if a_scout is not None and self._diag_a_scout_start_dist is None:
            au = next((x for x in living if x.name == a_scout), None)
            if au is not None:
                self._diag_a_scout_start_dist = int(
                    nearest_distance(au.pos, _info_cells(SIDE_A))
                )

        if b_scout is not None and self._diag_b_scout_start_dist is None:
            bu = next((x for x in living if x.name == b_scout), None)
            if bu is not None:
                self._diag_b_scout_start_dist = int(
                    nearest_distance(bu.pos, _info_cells(SIDE_B))
                )
        self._default_mid_name = mid_name
        self._default_main_names = list(main_names)
        self._diag_default_role_assignments += 1

        # v18: Scout targetを役割決定時点で最短INFOセルへ固定。
        self._default_a_scout_target = None
        self._default_b_scout_target = None

        if a_scout is not None:
            au = next((x for x in living if x.name == a_scout), None)
            if au is not None:
                cell, dist = self._nearest_info_cell_exact(au.pos, SIDE_A)
                self._default_a_scout_target = cell
                self._diag_a_scout_actual_target_start_dist = (
                    int(dist) if dist is not None else None
                )

        if b_scout is not None:
            bu = next((x for x in living if x.name == b_scout), None)
            if bu is not None:
                cell, dist = self._nearest_info_cell_exact(bu.pos, SIDE_B)
                self._default_b_scout_target = cell
                self._diag_b_scout_actual_target_start_dist = (
                    int(dist) if dist is not None else None
                )

        if a_scout is not None:
            self.assignment[a_scout] = (
                SIDE_A,
                "INFO",
                "A_SCOUT",
            )

        if b_scout is not None:
            self.assignment[b_scout] = (
                SIDE_B,
                "INFO",
                "B_SCOUT",
            )

        if mid_name is not None:
            self.assignment[mid_name] = (
                SIDE_MID,
                "FORWARD",
                "MID_CONTROL",
            )

        for name in main_names:
            # Mainは情報Scoutと同じ道へ重ねず、攻め候補側のStagingへ。
            self.assignment[name] = (
                preferred,
                "STAGING",
                "DEFAULT_MAIN",
            )

    def _apply_strategy_assignments(self, strategy, initial=False):
        living = self._living_attackers()
        if not living:
            return

        self.assignment = {}
        self.targets = {}

        if strategy == "A_RUSH":
            self.target_site = SIDE_A
            for a in living:
                self.assignment[a.name] = (SIDE_A, "SITE", "MAIN")

        elif strategy == "B_RUSH":
            self.target_site = SIDE_B
            for a in living:
                self.assignment[a.name] = (SIDE_B, "SITE", "MAIN")

        elif strategy == "MID_TO_B":
            self.target_site = SIDE_B
            main, support = self._split_names(3)
            for name in main:
                self.assignment[name] = (SIDE_MID, "DEEP", "MID")
            for name in support:
                self.assignment[name] = (SIDE_B, "STAGING", "B")

        elif strategy in {"A_SPLIT", "B_SPLIT"}:
            side = SIDE_A if strategy == "A_SPLIT" else SIDE_B
            self.target_site = side

            # v4: A/B別に新しいSplit sequenceを追跡。
            if self._split_tracking_site != side:
                self._split_tracking_site = side
                self._split_main_ready_step = None
                self._split_support_ready_step = None

            cfg = GC_MACRO_GROUP_SIZES[strategy]
            main, support = self._split_names(int(cfg["main"]))

            # 新しいSplit sequenceならplanned Entryも作り直す。
            self._split_planned_entry = {}

            entry_name = (
                "A_SPLIT_ENTRY" if side == SIDE_A else "B_SPLIT_ENTRY"
            )
            entry_cells = list(TACTICAL_CELLS[entry_name])

            for name in main:
                self.assignment[name] = (side, "DEEP", "MAIN")

            for name in support:
                unit = next((x for x in living if x.name == name), None)
                origin = unit.pos if unit is not None else ATTACKER_SPAWNS[0]

                planned_entry = choose_weighted_candidate(
                    entry_cells,
                    origin,
                    GC_MACRO_ROUTE_BIAS.get(entry_name, 0.0),
                )
                self._split_planned_entry[name] = planned_entry

                self.assignment[name] = (SIDE_MID, "STAGING", "SUPPORT")

        elif strategy == "DEFAULT":
            self._start_default_diag_session_v19()
            if DEFAULT_USE_SINGLE_OPPOSITE_SCOUT:
                self._assign_default_roles_v20(living)
            else:
                self._assign_default_roles_v16(living)

        elif strategy == "FAKE_A_TO_B":
            self.target_site = SIDE_B
            self._fake_direction = "A_TO_B"
            self._fake_side = SIDE_A
            self._fake_real_side = SIDE_B
            self._fake_triggered = False
            self._fake_completed = False
            self._fake_trigger_step = None
            self._fake_sell_dwell = 0
            self._fake_execute_start_step = None
            self._fake_phase_start_step = self.macro_step

            cfg = GC_MACRO_GROUP_SIZES["FAKE_A_TO_B"]
            fake, rotate = self._fake_names(int(cfg["fake"]))
            self._fake_group_names = set(fake)
            self._fake_rotate_names = set(rotate)
            self._set_fake_phase_targets("SELL")

        elif strategy == "FAKE_B_TO_A":
            self.target_site = SIDE_A
            self._fake_direction = "B_TO_A"
            self._fake_side = SIDE_B
            self._fake_real_side = SIDE_A
            self._fake_triggered = False
            self._fake_completed = False
            self._fake_trigger_step = None
            self._fake_sell_dwell = 0
            self._fake_execute_start_step = None
            self._fake_phase_start_step = self.macro_step

            cfg = GC_MACRO_GROUP_SIZES["FAKE_B_TO_A"]
            fake, rotate = self._fake_names(int(cfg["fake"]))
            self._fake_group_names = set(fake)
            self._fake_rotate_names = set(rotate)
            self._set_fake_phase_targets("SELL")

        elif strategy == "ROTATE_A_TO_B":
            self.target_site = SIDE_B
            for a in living:
                self.assignment[a.name] = (SIDE_B, "SITE", "ROTATE")
            self.rotate_count += 1

        elif strategy == "ROTATE_B_TO_A":
            self.target_site = SIDE_A
            for a in living:
                self.assignment[a.name] = (SIDE_A, "SITE", "ROTATE")
            self.rotate_count += 1

        elif strategy == "REHIT_A":
            self.target_site = SIDE_A
            for a in living:
                self.assignment[a.name] = (SIDE_A, "RESET", "REHIT")
            self.rehit_count += 1

        elif strategy == "REHIT_B":
            self.target_site = SIDE_B
            for a in living:
                self.assignment[a.name] = (SIDE_B, "RESET", "REHIT")
            self.rehit_count += 1

        # Per-unit concrete target.
        # Fakeは専用3段階Option側でtargetを管理する。
        if strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}:
            if not initial:
                self.last_switch_step = self.macro_step
            return

        for a in living:
            side, phase, role = self.assignment.get(
                a.name, (self.target_site, "SITE", "MAIN")
            )

            if (
                role == "SUPPORT"
                and phase == "STAGING"
                and strategy in {"A_SPLIT", "B_SPLIT"}
            ):
                target = choose_split_staging(
                    self.target_site,
                    a.pos,
                    self._split_planned_entry.get(a.name),
                )
            elif (
                strategy == "DEFAULT"
                and role in {
                    "A_SCOUT",
                    "B_SCOUT",
                    "MAIN_LEAD",
                    "OPPOSITE_SCOUT",
                }
                and phase == "INFO"
            ):
                if role == "MAIN_LEAD":
                    target = self._default_main_lead_target
                elif role == "OPPOSITE_SCOUT":
                    target = self._default_opposite_scout_target
                elif role == "A_SCOUT":
                    target = self._default_a_scout_target
                else:
                    target = self._default_b_scout_target

                if target is None:
                    target, _dist = self._nearest_info_cell_exact(
                        a.pos,
                        side,
                    )
            elif phase == "RESET":
                target = target_for_side(side, "RESET", a.pos)
            else:
                target = target_for_side(side, phase, a.pos)

            if target is None:
                target = target_for_side(side, "SITE", a.pos) if side != SIDE_MID else a.pos
            self.targets[a.name] = target

        if not initial:
            self.last_switch_step = self.macro_step

    # ------------------------------------------------------------------
    # progression
    # ------------------------------------------------------------------

    def _advance_assignment_phase_if_needed(self, a):
        side, phase, role = self.assignment.get(
            a.name, (self.target_site, "SITE", "MAIN")
        )
        target = self.targets.get(a.name)
        if target is None:
            return

        reached = nearest_distance(a.pos, [target]) <= 1
        if not reached:
            return

        strategy = self.current_strategy

        # Split: support waits until main pressure/control is sufficient.
        if strategy in {"A_SPLIT", "B_SPLIT"}:
            target_side = SIDE_A if strategy == "A_SPLIT" else SIDE_B

            if role == "MAIN":
                if phase == "DEEP":
                    self.assignment[a.name] = (target_side, "SITE", role)

            elif role == "SUPPORT":
                main_ready = (
                    self.pressure[target_side] >= SPLIT_MAIN_READY_PRESSURE
                    or self.control[f"{target_side}_FORWARD"] >= 0.45
                )
                if phase == "STAGING" and main_ready:
                    entry_name = (
                        "A_SPLIT_ENTRY"
                        if target_side == SIDE_A
                        else "B_SPLIT_ENTRY"
                    )

                    entry = self._split_planned_entry.get(a.name)
                    if entry is None:
                        entry = choose_weighted_candidate(
                            TACTICAL_CELLS[entry_name],
                            a.pos,
                            GC_MACRO_ROUTE_BIAS.get(entry_name, 0.0),
                        )
                        self._split_planned_entry[a.name] = entry

                    self.assignment[a.name] = (
                        target_side,
                        "SPLIT_ENTRY",
                        role,
                    )
                    self.targets[a.name] = entry
                    return
                elif phase == "SPLIT_ENTRY":
                    # v4: 実際にA/B_SPLIT_ENTRYへ到達した時刻を記録する。
                    # 同じmarkerが横入口と180度背面入口の複数箇所にあっても、
                    # どれか1つへ到達すればsupport entry成立。
                    if self._split_tracking_site == target_side:
                        if self._split_support_ready_step is None:
                            self._split_support_ready_step = self.macro_step
                        self._diag_split_support_entry = True
                    self.assignment[a.name] = (target_side, "SITE", role)

        elif strategy == "MID_TO_B":
            if side == SIDE_MID and phase == "DEEP":
                self.assignment[a.name] = (SIDE_B, "SITE", role)
            elif side == SIDE_B and phase == "STAGING":
                if self.control[f"{SIDE_MID}_FORWARD"] >= 0.45:
                    self.assignment[a.name] = (SIDE_B, "SITE", role)

        elif strategy == "DEFAULT":
            if role == "MAIN_LEAD" and phase == "INFO":
                known, _count = self._remembered_enemy_info(side)

                if side == SIDE_A:
                    self._diag_a_scout_info_reached = True
                else:
                    self._diag_b_scout_info_reached = True
                self._diag_main_lead_info_reached = True

                if (
                    DEFAULT_MAIN_LEAD_HOLD_INFO
                    and DEFAULT_INFO_HOLD_ENABLED
                    and not known
                    and self.strategy_age < DEFAULT_INFO_HOLD_MAX_MACRO_STEPS
                ):
                    self._diag_default_info_hold_ticks += 1
                    return

                # Leadは情報取得後、そのまま本命側Forwardへ進む。
                self.assignment[a.name] = (
                    side,
                    "FORWARD",
                    "MAIN_LEAD",
                )

            elif role == "OPPOSITE_SCOUT" and phase == "INFO":
                known, _count = self._remembered_enemy_info(side)

                if side == SIDE_A:
                    self._diag_a_scout_info_reached = True
                else:
                    self._diag_b_scout_info_reached = True
                self._diag_opposite_scout_info_reached = True

                if (
                    DEFAULT_INFO_HOLD_ENABLED
                    and not known
                    and self.strategy_age < DEFAULT_INFO_HOLD_MAX_MACRO_STEPS
                ):
                    self._diag_default_info_hold_ticks += 1
                    return

                if random.random() < DEFAULT_OPPOSITE_SCOUT_TO_LURK_PROB:
                    self.assignment[a.name] = (
                        side,
                        "LURK",
                        "OPPOSITE_SCOUT_LURK",
                    )
                else:
                    self.assignment[a.name] = (
                        side,
                        "DEEP",
                        "OPPOSITE_SCOUT_DEEP",
                    )

            elif role == "MID_CONTROL" and phase == "FORWARD":
                self.assignment[a.name] = (
                    SIDE_MID,
                    "DEEP",
                    "MID_CONTROL",
                )

            elif role == "DEFAULT_MAIN" and phase == "STAGING":
                self.assignment[a.name] = (
                    side,
                    "FORWARD",
                    "DEFAULT_MAIN",
                )

        elif strategy in {"REHIT_A", "REHIT_B"}:
            if phase == "RESET":
                self.assignment[a.name] = (side, "SITE", role)

        # Fake v7 targets are managed by _update_fake_option_phase().
        if strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}:
            return

        # v18: DEFAULT ScoutがINFO待機中なら固定targetを維持する。
        if (
            strategy == "DEFAULT"
            and role in {
                "A_SCOUT",
                "B_SCOUT",
                "MAIN_LEAD",
                "OPPOSITE_SCOUT",
            }
            and phase == "INFO"
        ):
            if role == "MAIN_LEAD":
                fixed = self._default_main_lead_target
            elif role == "OPPOSITE_SCOUT":
                fixed = self._default_opposite_scout_target
            elif role == "A_SCOUT":
                fixed = self._default_a_scout_target
            else:
                fixed = self._default_b_scout_target

            if fixed is not None:
                self.targets[a.name] = fixed
            return

        # Generic target rebuild
        side, phase, role = self.assignment[a.name]
        if phase == "SPLIT_ENTRY":
            return
        self.targets[a.name] = target_for_side(
            side,
            phase if phase in {
                "STAGING", "LANE", "INFO", "FORWARD", "DEEP",
                "SITE", "LURK", "CUT", "RESET"
            } else "SITE",
            a.pos,
        ) or a.pos

    def _move_attackers_one_tick(self):
        occupied = {a.pos for a in self.attackers if a.is_alive}
        order = self._living_attackers()

        if (
            DEFAULT_SCOUT_MOVEMENT_PRIORITY
            and self.current_strategy == "DEFAULT"
        ):
            priority = {
                self._default_main_lead_name: 0,
                self._default_opposite_scout_name: 1,
                self._default_mid_name: 2,
            }

            # Scout/Midを先に、Main同士だけランダム性を残す。
            random_keys = {a.name: random.random() for a in order}
            order.sort(
                key=lambda a: (
                    priority.get(a.name, 3),
                    random_keys[a.name],
                )
            )
        else:
            random.shuffle(order)

        for a in order:
            self._advance_assignment_phase_if_needed(a)
            target = self.targets.get(a.name, a.pos)

            is_a_scout = (
                self.current_strategy == "DEFAULT"
                and a.name == self._default_a_scout_name
            )
            is_b_scout = (
                self.current_strategy == "DEFAULT"
                and a.name == self._default_b_scout_name
            )

            old_pos = a.pos
            old_target_dist = None

            if is_a_scout and self._default_a_scout_target is not None:
                self._diag_a_scout_move_attempts += 1
                old_target_dist = nearest_distance(
                    a.pos,
                    [self._default_a_scout_target],
                )
            elif is_b_scout and self._default_b_scout_target is not None:
                self._diag_b_scout_move_attempts += 1
                old_target_dist = nearest_distance(
                    a.pos,
                    [self._default_b_scout_target],
                )

            occupied.discard(a.pos)
            new_pos = step_toward(a.pos, target, occupied)
            a.pos = new_pos
            occupied.add(a.pos)

            if is_a_scout and old_target_dist is not None:
                new_d = nearest_distance(
                    a.pos,
                    [self._default_a_scout_target],
                )
                if new_d < old_target_dist:
                    self._diag_a_scout_forward_moves += 1
                elif a.pos == old_pos:
                    self._diag_a_scout_stopped_moves += 1

            elif is_b_scout and old_target_dist is not None:
                new_d = nearest_distance(
                    a.pos,
                    [self._default_b_scout_target],
                )
                if new_d < old_target_dist:
                    self._diag_b_scout_forward_moves += 1
                elif a.pos == old_pos:
                    self._diag_b_scout_stopped_moves += 1

            # v17: 現在のINFOまでの残距離を常に記録。
            if a.name == self._default_a_scout_name:
                self._diag_a_scout_final_dist = int(
                    nearest_distance(a.pos, _info_cells(SIDE_A))
                )
            elif a.name == self._default_b_scout_name:
                self._diag_b_scout_final_dist = int(
                    nearest_distance(a.pos, _info_cells(SIDE_B))
                )

    def _defender_side_counts(self):
        counts = {SIDE_A: 0, SIDE_B: 0, SIDE_MID: 0}
        for d in self.defenders:
            if d.is_alive:
                counts[side_of_pos(d.pos)] += 1
        return counts

    def _move_defenders_one_tick(self):
        """圧力を受けた側へ少しだけ寄る簡易ローテーション。"""
        if random.random() > DEFENDER_ROTATE_PROB:
            return

        side = max(self.pressure, key=self.pressure.get)
        if self.pressure[side] < 0.25:
            return

        living = [d for d in self.defenders if d.is_alive]
        if not living:
            return

        d = random.choice(living)
        targets = _forward_control_cells(side)
        if not targets:
            return
        target = random.choice(targets)
        occupied = {x.pos for x in living if x is not d}
        d.pos = step_toward(d.pos, target, occupied)

    # ------------------------------------------------------------------
    # information/control/pressure
    # ------------------------------------------------------------------

    def _area_coverage(self, cells):
        if not cells:
            return 0.0
        attackers = self._living_attackers()
        if not attackers:
            return 0.0

        in_area = sum(
            1
            for a in attackers
            if tuple(a.pos) in set(cells)
        )
        # 人数ベース + 少しでも到達していれば価値あり
        return min(1.0, in_area / max(1.0, min(3.0, len(attackers))))

    def _update_information(self):
        """v21: FORWARD -> INFO -> DEEP/SITE staged information."""
        defender_counts = self._defender_side_counts()

        for side in (SIDE_A, SIDE_B, SIDE_MID):
            self._info_memory_age[side] += 1

        for side in (SIDE_A, SIDE_B, SIDE_MID):
            self.info_conf[side] = max(
                0.0,
                self.info_conf[side] - INFO_DECAY,
            )

            forward_cov = self._area_coverage(_forward_control_cells(side))
            info_cov = self._area_coverage(_info_cells(side))
            deep_cov = self._area_coverage(_deep_control_cells(side))
            site_cov = (
                self._area_coverage(_site_cells(side))
                if side in {SIDE_A, SIDE_B}
                else 0.0
            )

            gain = (
                forward_cov * INFO_GAIN_FORWARD_CONTROL
                + info_cov * INFO_GAIN_INFO_AREA_V21
                + deep_cov * INFO_GAIN_DEEP_CONTROL
                + site_cov * INFO_GAIN_SITE_V21
            )

            floor = 0.0
            if forward_cov > 0.0:
                floor = max(floor, INFO_CONF_LOW)
            if info_cov > 0.0:
                floor = max(floor, INFO_CONF_HIGH)
            if deep_cov > 0.0:
                floor = max(floor, 0.74)
            if site_cov > 0.0:
                floor = max(floor, 0.82)

            if gain > 0.0 or floor > 0.0:
                self.info_conf[side] = min(
                    1.0,
                    max(floor, self.info_conf[side] + gain),
                )

                true_count = float(defender_counts[side])
                conf = float(self.info_conf[side])

                if conf >= INFO_CONF_HIGH:
                    alpha = INFO_EST_ALPHA_HIGH
                elif conf >= INFO_CONF_MEDIUM:
                    alpha = INFO_EST_ALPHA_MEDIUM
                else:
                    alpha = INFO_EST_ALPHA_LOW

                self.enemy_est[side] = (
                    self.enemy_est[side] * (1.0 - alpha)
                    + true_count * alpha
                )

                self._info_memory_count[side] = float(self.enemy_est[side])
                self._info_memory_conf[side] = conf
                self._info_memory_age[side] = 0

            for depth_name, cells in (
                ("FORWARD", _forward_control_cells(side)),
                ("DEEP", _deep_control_cells(side)),
            ):
                key = f"{side}_{depth_name}"
                self.control[key] = max(
                    0.0,
                    self.control[key] - CONTROL_DECAY,
                )
                cov = self._area_coverage(cells)
                self.control[key] = min(
                    1.0,
                    self.control[key] + CONTROL_GAIN * cov,
                )

            self.pressure[side] = min(
                1.0,
                0.35 * forward_cov
                + 0.45 * deep_cov
                + 0.70 * site_cov,
            )

        self.map_control_score = float(
            np.mean(
                [
                    self.control[f"{SIDE_A}_FORWARD"],
                    self.control[f"{SIDE_B}_FORWARD"],
                    self.control[f"{SIDE_MID}_FORWARD"],
                    self.control[f"{SIDE_MID}_DEEP"],
                ]
            )
        )

    # ------------------------------------------------------------------
    # combat / plant outcome
    # ------------------------------------------------------------------

    def _resolve_contact(self):
        """Macro学習用の簡易接敵モデル。

        目的は射撃DQNを再学習することではなく、
        厚いサイトへ5人で突っ込むことと、
        Split/Fake/Rotateで有利を作ることの価値差をMacroへ返すこと。
        """
        for side in (SIDE_A, SIDE_B, SIDE_MID):
            attackers = [
                a for a in self._living_attackers()
                if side_of_pos(a.pos) == side
            ]
            defenders = [
                d for d in self.defenders
                if d.is_alive and side_of_pos(d.pos) == side
            ]

            if not attackers or not defenders:
                continue

            # Split/Deep controlでAttacker側の局所効率を上げる。
            deep_bonus = self.control[f"{side}_DEEP"] if f"{side}_DEEP" in self.control else 0
            attack_strength = len(attackers) * (1.0 + 0.22 * deep_bonus)

            # 同時に別方面へpressureがあるとDefenderの集中力を少し削る。
            cross_pressure = sum(
                self.pressure[s]
                for s in (SIDE_A, SIDE_B, SIDE_MID)
                if s != side
            )
            attack_strength *= (1.0 + 0.10 * min(1.0, cross_pressure))

            defend_strength = len(defenders)

            # probabilistic one casualty max per low-level tick
            p_defender_down = min(
                0.32,
                0.08 + 0.045 * max(0.0, attack_strength - defend_strength + 1.0),
            )
            p_attacker_down = min(
                0.28,
                0.07 + 0.040 * max(0.0, defend_strength - attack_strength + 1.0),
            )

            if random.random() < p_defender_down and defenders:
                random.choice(defenders).is_alive = False

            if random.random() < p_attacker_down and attackers:
                victim = random.choice(attackers)

                # v19: did a DEFAULT scout die before completing its job?
                if victim.name == self._default_a_scout_name:
                    self._diag_a_scout_died = True
                if victim.name == self._default_b_scout_name:
                    self._diag_b_scout_died = True

                victim.is_alive = False
                if victim.has_spike:
                    # Macro trainingではspike handoffを簡略化。
                    survivors = [a for a in self._living_attackers() if a is not victim]
                    if survivors:
                        min(survivors, key=lambda a: nearest_distance(a.pos, [victim.pos])).has_spike = True

    def _check_plant(self):
        carrier = self._carrier()
        if carrier is None:
            return False

        side = side_of_pos(carrier.pos)
        if side not in {SIDE_A, SIDE_B}:
            return False

        if tuple(carrier.pos) not in set(_site_cells(side)):
            return False

        # 周辺Defenderが少ないほどplant成功しやすい。
        nearby_defenders = sum(
            1
            for d in self.defenders
            if d.is_alive and side_of_pos(d.pos) == side
        )
        local_attackers = sum(
            1
            for a in self._living_attackers()
            if side_of_pos(a.pos) == side
        )

        if local_attackers >= max(1, nearby_defenders):
            self.planted = True
            self.plant_site = side
            self.success = True
            self.done = True
            self.reason = "planted"
            return True

        return False

    # ------------------------------------------------------------------
    # valid macro actions / transitions
    # ------------------------------------------------------------------

    def _urgent_smart_rotate_index(self):
        """明確なheavy/light情報がある時の緊急Rotate候補。"""
        if not ALLOW_URGENT_SMART_ROTATE_BREAK:
            return None

        heavy = float(
            GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"]
        )
        light = float(
            GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"]
        )

        decision = self._rotate_decision_v22()
        if decision is None:
            return None

        if decision["direction"] == "A_TO_B":
            return STRATEGY_TO_INDEX["ROTATE_A_TO_B"]
        if decision["direction"] == "B_TO_A":
            return STRATEGY_TO_INDEX["ROTATE_B_TO_A"]

        return None

    def _default_info_option_should_lock(self):
        """DEFAULTはA/B情報取得完了または最大6stepまで継続する。"""
        if self.current_strategy != "DEFAULT":
            self._default_info_option_completion_counted = False
            return False

        usable, a_conf, _a_count, b_conf, _b_count = (
            self._rotate_info_pair()
        )

        if usable:
            if not self._default_info_option_completion_counted:
                self._diag_default_option_info_complete += 1
                self._default_info_option_completion_counted = True
            self._record_default_diag_exit_v19("info_complete")
            return False

        if self.strategy_age < DEFAULT_INFO_OPTION_MAX_AGE:
            self._diag_default_option_hold_steps += 1
            return True

        self._record_default_diag_exit_v19("max_age")
        return False

    def _normal_strategy_should_lock(self):
        """通常戦術の短時間継続lock。

        Fake / Split / Rotateは専用Option lockを別途使用する。
        """
        normal = {
            "A_RUSH",
            "B_RUSH",
            "MID_TO_B",
            "DEFAULT",
            "REHIT_A",
            "REHIT_B",
        }
        return (
            self.current_strategy in normal
            and self.strategy_age < NORMAL_STRATEGY_COMMIT_STEPS
        )

    def _option_lock_action_index(self):
        """進行中の複合戦術を途中で別Macro actionに上書きさせない。"""
        strategy = self.current_strategy

        if _strategy_is_split(strategy):
            if (
                not self._split_completed
                and self.strategy_age < SPLIT_OPTION_COMMIT_STEPS
            ):
                return STRATEGY_TO_INDEX[strategy]

        if _strategy_is_fake(strategy):
            phase_age = self._fake_phase_age()

            # v9: SELLとROTATEで独立した時間枠を与える。
            # SELLに10step使ってもROTATEの持ち時間は減らない。
            if self._fake_phase == "SELL":
                if phase_age < FAKE_OPTION_COMMIT_STEPS:
                    return STRATEGY_TO_INDEX[strategy]

            if self._fake_phase == "ROTATE":
                if phase_age < FAKE_OPTION_COMMIT_STEPS:
                    return STRATEGY_TO_INDEX[strategy]

            # EXECUTE移行直後も最低数stepはそのままサイトへ入る。
            if (
                self._fake_phase == "EXECUTE"
                and phase_age < FAKE_EXECUTE_LOCK_STEPS
            ):
                return STRATEGY_TO_INDEX[strategy]

        if _strategy_is_rotate(strategy):
            if self.strategy_age < ROTATE_OPTION_COMMIT_STEPS:
                return STRATEGY_TO_INDEX[strategy]

        return None

    def action_mask(self):
        mask = np.ones(N_ACTIONS, dtype=bool)

        # v6: Split/Fake/Rotateはtemporally-extended option。
        locked = self._option_lock_action_index()
        if locked is not None:
            mask[:] = False
            mask[locked] = True
            return mask

        # v15: DEFAULTは情報収集Option。
        # A/B両情報が揃うか最大6stepまで保持する。
        if self._default_info_option_should_lock():
            mask[:] = False
            mask[STRATEGY_TO_INDEX["DEFAULT"]] = True
            return mask

        # v11: 通常戦術も最低3 Macro stepは継続。
        # ただし明確なheavy/light情報によるSmart Rotateだけは割り込める。
        if self._normal_strategy_should_lock():
            urgent_rotate = self._urgent_smart_rotate_index()

            mask[:] = False
            mask[STRATEGY_TO_INDEX[self.current_strategy]] = True

            if urgent_rotate is not None:
                mask[urgent_rotate] = True

            return mask

        # 同じ戦術への連打は許可するが、無意味なRotateを抑える。
        time_ratio = max(
            0.0,
            (ROUND_DURATION_TICKS - self.tick) / max(1.0, ROUND_DURATION_TICKS),
        )
        rotate_min = float(
            GC_MACRO_DECISION_THRESHOLDS["ROTATE_MIN_TIME_RATIO"]
        )

        if time_ratio < rotate_min:
            for name in ("ROTATE_A_TO_B", "ROTATE_B_TO_A"):
                mask[STRATEGY_TO_INDEX[name]] = False

        transitions = GC_MACRO_TRANSITIONS.get(self.current_strategy, {})
        if not transitions.get("allow_rotate", True):
            for name in ("ROTATE_A_TO_B", "ROTATE_B_TO_A"):
                mask[STRATEGY_TO_INDEX[name]] = False
        if not transitions.get("allow_rehit", True):
            for name in ("REHIT_A", "REHIT_B"):
                mask[STRATEGY_TO_INDEX[name]] = False

        # v6: remembered heavy/light情報が揃っている場合は、
        # source側のpressure/controlが低くても「情報に基づくRotate」を許可する。
        heavy = float(
            GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"]
        )
        light = float(
            GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"]
        )
        rotate_usable, a_conf, a_count, b_conf, b_count = (
            self._rotate_info_pair()
        )

        rotate_decision = self._rotate_decision_v22()
        smart_a_to_b = (
            rotate_decision is not None
            and rotate_decision["direction"] == "A_TO_B"
        )
        smart_b_to_a = (
            rotate_decision is not None
            and rotate_decision["direction"] == "B_TO_A"
        )

        if (
            not smart_a_to_b
            and self.pressure[SIDE_A] < 0.12
            and self.control[f"{SIDE_A}_FORWARD"] < 0.25
        ):
            mask[STRATEGY_TO_INDEX["ROTATE_A_TO_B"]] = False

        if (
            not smart_b_to_a
            and self.pressure[SIDE_B] < 0.12
            and self.control[f"{SIDE_B}_FORWARD"] < 0.25
        ):
            mask[STRATEGY_TO_INDEX["ROTATE_B_TO_A"]] = False

        if not mask.any():
            mask[STRATEGY_TO_INDEX["DEFAULT"]] = True
        return mask

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def build_observation(self):
        """Team-level fixed-size observation.

        0-11  current strategy onehot
        12-14 info confidence A/B/Mid
        15-17 estimated enemy count A/B/Mid /5
        18-23 control A/B/Mid forward/deep
        24-26 pressure A/B/Mid
        27-31 living attacker positions summary / team status
        32-34 living defender count per side /5
        35     round time ratio
        36     strategy age
        37     switch recency
        38-40 carrier normalized position + target-site
        41-43 group dispersion A/B/Mid
        44-46 lurk occupancy A/B/Mid
        47-49 flank-cut occupancy A/B/Mid
        50     rotate count
        51     rehit count
        52     fake value
        53     split sync
        54     map control score
        55-57 remembered A: enemy count / confidence / age
        58-60 remembered B: enemy count / confidence / age
        61-63 remembered Mid: enemy count / confidence / age
        64-67 Fake phase one-hot: NONE / SELL / ROTATE / EXECUTE
        """
        obs = []

        # strategy
        for name in STRATEGIES:
            obs.append(float(name == self.current_strategy))

        # info
        obs += [
            float(self.info_conf[SIDE_A]),
            float(self.info_conf[SIDE_B]),
            float(self.info_conf[SIDE_MID]),
        ]
        obs += [
            float(np.clip(self.enemy_est[SIDE_A] / 5.0, 0, 1)),
            float(np.clip(self.enemy_est[SIDE_B] / 5.0, 0, 1)),
            float(np.clip(self.enemy_est[SIDE_MID] / 5.0, 0, 1)),
        ]

        # control
        obs += [
            self.control[f"{SIDE_A}_FORWARD"],
            self.control[f"{SIDE_A}_DEEP"],
            self.control[f"{SIDE_B}_FORWARD"],
            self.control[f"{SIDE_B}_DEEP"],
            self.control[f"{SIDE_MID}_FORWARD"],
            self.control[f"{SIDE_MID}_DEEP"],
        ]

        # pressure
        obs += [
            self.pressure[SIDE_A],
            self.pressure[SIDE_B],
            self.pressure[SIDE_MID],
        ]

        living_a = self._living_attackers()
        living_d = [d for d in self.defenders if d.is_alive]

        # team status: living, average row/col, spread, carrier alive
        if living_a:
            avg_r = np.mean([a.pos[0] for a in living_a]) / max(1, HEIGHT - 1)
            avg_c = np.mean([a.pos[1] for a in living_a]) / max(1, WIDTH - 1)
            spread = np.mean(
                [
                    abs(a.pos[0] / max(1, HEIGHT - 1) - avg_r)
                    + abs(a.pos[1] / max(1, WIDTH - 1) - avg_c)
                    for a in living_a
                ]
            )
        else:
            avg_r = avg_c = spread = 0.0

        carrier = self._carrier()
        obs += [
            len(living_a) / 5.0,
            float(avg_r),
            float(avg_c),
            float(np.clip(spread, 0, 1)),
            float(carrier is not None),
        ]

        dcounts = self._defender_side_counts()
        obs += [
            dcounts[SIDE_A] / 5.0,
            dcounts[SIDE_B] / 5.0,
            dcounts[SIDE_MID] / 5.0,
        ]

        time_ratio = max(
            0.0,
            (ROUND_DURATION_TICKS - self.tick) / max(1.0, ROUND_DURATION_TICKS),
        )
        obs += [
            float(time_ratio),
            min(1.0, self.strategy_age / 8.0),
            min(1.0, max(0, self.macro_step - self.last_switch_step) / 8.0),
        ]

        if carrier:
            obs += [
                carrier.pos[0] / max(1, HEIGHT - 1),
                carrier.pos[1] / max(1, WIDTH - 1),
                0.0 if self.target_site == SIDE_A else 1.0,
            ]
        else:
            obs += [0.0, 0.0, 0.5]

        # group dispersion/occupancy by strategic area
        for side in (SIDE_A, SIDE_B, SIDE_MID):
            count = sum(1 for a in living_a if side_of_pos(a.pos) == side)
            obs.append(count / 5.0)

        for side in (SIDE_A, SIDE_B, SIDE_MID):
            cells = set(_lurk_cells(side))
            obs.append(
                sum(1 for a in living_a if a.pos in cells) / 5.0
            )

        for side in (SIDE_A, SIDE_B, SIDE_MID):
            cells = set(_cut_cells(side))
            obs.append(
                sum(1 for a in living_a if a.pos in cells) / 5.0
            )

        obs += [
            min(1.0, self.rotate_count / 3.0),
            min(1.0, self.rehit_count / 3.0),
            float(np.clip(self.fake_value, 0, 1)),
            float(np.clip(self.split_sync_score, 0, 1)),
            float(np.clip(self.map_control_score, 0, 1)),
        ]

        # v6 memory features [55:64]:
        # A/B/MIDそれぞれ remembered_count, confidence, normalized_age。
        for side in (SIDE_A, SIDE_B, SIDE_MID):
            count = float(
                np.clip(self._info_memory_count[side] / 5.0, 0.0, 1.0)
            )
            conf = float(
                np.clip(self._info_memory_conf[side], 0.0, 1.0)
            )
            age = float(
                np.clip(
                    self._info_memory_age[side]
                    / max(1.0, INFO_MEMORY_MAX_TICKS),
                    0.0,
                    1.0,
                )
            )
            obs += [count, conf, age]

        fake_phase = self._fake_phase
        obs += [
            1.0 if fake_phase == "NONE" else 0.0,
            1.0 if fake_phase == "SELL" else 0.0,
            1.0 if fake_phase == "ROTATE" else 0.0,
            1.0 if fake_phase == "EXECUTE" else 0.0,
        ]

        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape != (68,):
            raise RuntimeError(f"Macro OBS_DIM mismatch: {arr.shape}")
        return arr

    def _remembered_enemy_info_detail(self, side):
        """v21: return (confidence, enemy_count, source)."""
        live_conf = float(self.info_conf[side])

        memory_conf = 0.0
        if self._info_memory_age[side] <= INFO_MEMORY_MAX_TICKS:
            memory_conf = float(self._info_memory_conf[side])

        if live_conf >= memory_conf:
            return live_conf, float(self.enemy_est[side]), "live"

        return (
            memory_conf,
            float(self._info_memory_count[side]),
            "memory",
        )

    def _remembered_enemy_info(self, side):
        """Legacy API: MEDIUM or higher counts as known."""
        conf, count, _source = self._remembered_enemy_info_detail(side)
        return conf >= INFO_CONF_MEDIUM, count

    def _info_level(self, side):
        conf, count, source = self._remembered_enemy_info_detail(side)
        if conf >= INFO_CONF_HIGH:
            level = "HIGH"
        elif conf >= INFO_CONF_MEDIUM:
            level = "MEDIUM"
        elif conf >= INFO_CONF_LOW:
            level = "LOW"
        else:
            level = "NONE"
        return level, conf, count, source

    def _rotate_info_pair(self):
        """v21 confidence-aware A/B information gate for Rotate."""
        a_conf, a_count, _ = self._remembered_enemy_info_detail(SIDE_A)
        b_conf, b_count, _ = self._remembered_enemy_info_detail(SIDE_B)

        both_medium = (
            a_conf >= ROTATE_MIN_CONF
            and b_conf >= ROTATE_MIN_CONF
        )
        one_strong = (
            a_conf >= ROTATE_STRONG_CONF
            or b_conf >= ROTATE_STRONG_CONF
        )
        clear_gap = abs(a_count - b_count) >= ROTATE_MEDIUM_COUNT_GAP

        usable = both_medium and (one_strong or clear_gap)
        return usable, a_conf, a_count, b_conf, b_count

    def _infer_opposite_site_for_rotate(self):
        """v23 diagnostic: same v22 behavior, but count every gate."""
        a_conf, a_count, _ = self._remembered_enemy_info_detail(SIDE_A)
        b_conf, b_count, _ = self._remembered_enemy_info_detail(SIDE_B)
        m_conf, m_count, _ = self._remembered_enemy_info_detail(SIDE_MID)

        any_high = (
            a_conf >= ROTATE_INFER_SOURCE_CONF
            or b_conf >= ROTATE_INFER_SOURCE_CONF
        )
        if any_high:
            self._diag_infer_any_high += 1

        a_heavy = (
            a_conf >= ROTATE_INFER_SOURCE_CONF
            and a_count >= ROTATE_INFER_HEAVY_MIN
        )
        b_heavy = (
            b_conf >= ROTATE_INFER_SOURCE_CONF
            and b_count >= ROTATE_INFER_HEAVY_MIN
        )
        if a_heavy or b_heavy:
            self._diag_infer_heavy_est += 1

        mid_forward = float(self.control.get(f"{SIDE_MID}_FORWARD", 0.0))
        mid_deep = float(self.control.get(f"{SIDE_MID}_DEEP", 0.0))

        forward_ok = mid_forward >= ROTATE_INFER_MID_FORWARD_MIN
        deep_ok = mid_deep >= ROTATE_INFER_MID_DEEP_MIN

        if forward_ok:
            self._diag_infer_mid_forward_ok += 1
        if deep_ok:
            self._diag_infer_mid_deep_ok += 1
        if forward_ok or deep_ok:
            self._diag_infer_mid_any_ok += 1

        if not (forward_ok or deep_ok):
            return None

        mid_seen = m_count if m_conf >= INFO_CONF_MEDIUM else 0.0
        candidates = []

        if a_heavy:
            b_max = max(0.0, 5.0 - a_count - mid_seen)
            if b_max <= ROTATE_INFER_LIGHT_MAX:
                self._diag_infer_opposite_bound_ok += 1
                candidates.append(("A_TO_B", a_count, b_max, a_conf))

        if b_heavy:
            a_max = max(0.0, 5.0 - b_count - mid_seen)
            if a_max <= ROTATE_INFER_LIGHT_MAX:
                self._diag_infer_opposite_bound_ok += 1
                candidates.append(("B_TO_A", b_count, a_max, b_conf))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[3], -x[2]), reverse=True)
        direction, heavy_count, opposite_max, source_conf = candidates[0]

        self._diag_infer_final += 1

        return {
            "direction": direction,
            "source": "INFERRED",
            "heavy_count": float(heavy_count),
            "opposite_max": float(opposite_max),
            "source_conf": float(source_conf),
            "mid_conf": float(m_conf),
            "mid_count": float(mid_seen),
            "mid_forward": mid_forward,
            "mid_deep": mid_deep,
            "inferred_conf": ROTATE_INFER_CONF,
        }

    def _rotate_decision_v22(self):
        """DIRECTを優先し、なければ保守的なINFERRED Rotateを許可。"""
        usable, a_conf, a_count, b_conf, b_count = self._rotate_info_pair()
        heavy = float(GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"])
        light = float(GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"])

        if usable:
            if a_count >= heavy and b_count <= light:
                return {"direction": "A_TO_B", "source": "DIRECT"}
            if b_count >= heavy and a_count <= light:
                return {"direction": "B_TO_A", "source": "DIRECT"}

        return self._infer_opposite_site_for_rotate()

    # ------------------------------------------------------------------
    # tactical history (v4)
    # ------------------------------------------------------------------

    def _update_tactical_history(self):
        """v4: action変更後もイベント履歴を保持して戦術完遂を追跡する。"""
        heavy = float(GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"])
        light = float(GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"])

        rotate_usable, a_conf, a_count, b_conf, b_count = (
            self._rotate_info_pair()
        )

        a_heavy = rotate_usable and a_count >= heavy
        b_heavy = rotate_usable and b_count >= heavy
        a_light = rotate_usable and a_count <= light
        b_light = rotate_usable and b_count <= light

        # --------------------------------------------------------------
        # Default -> information -> decision
        # --------------------------------------------------------------
        if self.current_strategy == "DEFAULT":
            if max(
                self.info_conf[SIDE_A],
                self.info_conf[SIDE_B],
                self.info_conf[SIDE_MID],
                self._info_memory_conf[SIDE_A]
                if self._info_memory_age[SIDE_A] <= INFO_MEMORY_MAX_TICKS else 0.0,
                self._info_memory_conf[SIDE_B]
                if self._info_memory_age[SIDE_B] <= INFO_MEMORY_MAX_TICKS else 0.0,
                self._info_memory_conf[SIDE_MID]
                if self._info_memory_age[SIDE_MID] <= INFO_MEMORY_MAX_TICKS else 0.0,
            ) >= DEFAULT_INFO_TRIGGER:
                self._default_had_good_info = True
        elif self._default_had_good_info:
            self._default_to_decision = True

        # --------------------------------------------------------------
        # Smart rotate: A/Bの情報が同時にliveでなくても直近memoryを使う。
        # --------------------------------------------------------------
        if self.current_strategy == "ROTATE_A_TO_B" and a_heavy and b_light:
            self._smart_rotate_completed = True
            self._smart_rotate_direction = "A_TO_B"
        elif self.current_strategy == "ROTATE_B_TO_A" and b_heavy and a_light:
            self._smart_rotate_completed = True
            self._smart_rotate_direction = "B_TO_A"

        # --------------------------------------------------------------
        # Fake v7:
        # SELL -> ROTATE -> EXECUTE は _update_fake_option_phase() が管理する。
        # ここでは既に成立した履歴を保持するだけ。
        # --------------------------------------------------------------
        if self._fake_completed:
            self.fake_value = max(self.fake_value, 1.0)

        # Diagnostic expiry:
        # SELLが長く続きOption上限へ近づいてもtriggerしなかったケース。
        if (
            self.current_strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}
            and self._fake_phase == "SELL"
            and self._fake_phase_age() >= FAKE_OPTION_COMMIT_STEPS - 1
            and not self._fake_triggered
        ):
            self._diag_fake_expired = True

        # --------------------------------------------------------------
        # Split:
        # main readinessと「supportが実際のSplit Entryへ到達した時刻」を使う。
        # supportがEntryを越えてMid Zoneから出ても成立履歴は消えない。
        # 横Entryでも180度背面Entryでも同じmarkerなので両方有効。
        # --------------------------------------------------------------
        if self.current_strategy in {"A_SPLIT", "B_SPLIT"}:
            site = SIDE_A if self.current_strategy == "A_SPLIT" else SIDE_B

            if self._split_tracking_site != site:
                self._split_tracking_site = site
                self._split_main_ready_step = None
                self._split_support_ready_step = None

            main_ready = (
                self.pressure[site] >= SPLIT_MAIN_READY_PRESSURE
                or self.control[f"{site}_FORWARD"] >= 0.45
            )
            if main_ready:
                self._diag_split_main_ready = True
            if main_ready and self._split_main_ready_step is None:
                self._split_main_ready_step = self.macro_step

        if (
            self._split_main_ready_step is not None
            and self._split_support_ready_step is not None
        ):
            gap = abs(
                self._split_main_ready_step
                - self._split_support_ready_step
            )
            if (
                self._diag_split_min_step_gap is None
                or gap < self._diag_split_min_step_gap
            ):
                self._diag_split_min_step_gap = int(gap)

            if gap <= SPLIT_ENTRY_WINDOW_MACRO_STEPS:
                self._diag_split_both_ready = True
            else:
                self._diag_split_window_miss = True

        if (
            self._split_tracking_site in {SIDE_A, SIDE_B}
            and self._split_main_ready_step is not None
            and self._split_support_ready_step is not None
            and abs(
                self._split_main_ready_step
                - self._split_support_ready_step
            ) <= SPLIT_ENTRY_WINDOW_MACRO_STEPS
        ):
            self._split_completed = True
            self._split_site = self._split_tracking_site
            self.split_sync_score = max(self.split_sync_score, 1.0)

        # --------------------------------------------------------------
        # Lurk / flank-cut
        # --------------------------------------------------------------
        living = self._living_attackers()
        for side in (SIDE_A, SIDE_B, SIDE_MID):
            lurk_set = set(_lurk_cells(side))
            cut_set = set(_cut_cells(side))
            if any(a.pos in lurk_set for a in living):
                self._lurk_touched = True
            if any(a.pos in cut_set for a in living):
                self._cut_touched = True

    def _uncertain_commit_penalty(self):
        """情報不足のままDeep/Siteへコミットしている時だけ軽く罰する。

        開幕RushはUNCERTAIN_COMMIT_GRACE_MACRO_STEPSまでは免除。
        A/B両方の情報がないまま深く入るほどコストが大きい。
        """
        if self.macro_step <= UNCERTAIN_COMMIT_GRACE_MACRO_STEPS:
            return 0.0

        side = None
        if self.current_strategy in {"A_RUSH", "A_SPLIT", "REHIT_A"}:
            side = SIDE_A
        elif self.current_strategy in {
            "B_RUSH", "B_SPLIT", "MID_TO_B", "REHIT_B"
        }:
            side = SIDE_B

        if side is None:
            return 0.0

        deep = self.control[f"{side}_DEEP"]
        pressure = self.pressure[side]
        committed = (
            deep >= UNCERTAIN_DEEP_CONTROL_THRESHOLD
            or pressure >= 0.45
        )
        if not committed:
            return 0.0

        a_known, _ = self._remembered_enemy_info(SIDE_A)
        b_known, _ = self._remembered_enemy_info(SIDE_B)

        if not a_known and not b_known:
            return UNCERTAIN_COMMIT_PENALTY_NO_INFO
        if not a_known or not b_known:
            return UNCERTAIN_COMMIT_PENALTY_ONE_SIDE
        return 0.0

    def _curriculum_progress_bonus(self):
        """練習episodeの成立を小さく補助する。一度きり。"""
        if self.curriculum_bonus_given:
            return 0.0

        bonus = 0.0

        if self.curriculum_mode == "SPLIT" and self._split_completed:
            bonus = CURRICULUM_BONUS_SCALE

        elif self.curriculum_mode == "FAKE" and self._fake_completed:
            bonus = CURRICULUM_BONUS_SCALE

        elif self.curriculum_mode == "ROTATE" and self._smart_rotate_completed:
            bonus = CURRICULUM_BONUS_SCALE

        if bonus > 0.0:
            self.curriculum_bonus_given = True
        return bonus

    # ------------------------------------------------------------------
    # v10: situation-aware strategy selection
    # ------------------------------------------------------------------

    def _strategy_suitability(self, strategy):
        """v12: 現在の情報に対する戦術適性を[-1,1]で評価する。

        方針:
        1. 情報が足りない -> DEFAULT
        2. サイトが明確に薄い -> まずRush
        3. サイトに守備がいる + Midが使える -> Split
        4. Fake側に守備を引きつける価値がある -> Fake
        5. Heavy/Lightが明確 -> Rotate

        真のDefender配置は参照せず、DQN観測にも入っている
        live / remembered info と取得Controlだけを使う。
        """
        heavy = float(
            GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"]
        )
        light = float(
            GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"]
        )

        a_known, a_count = self._remembered_enemy_info(SIDE_A)
        b_known, b_count = self._remembered_enemy_info(SIDE_B)
        m_known, m_count = self._remembered_enemy_info(SIDE_MID)
        both = a_known and b_known

        def site_state(known, count):
            if not known:
                return "UNKNOWN"
            if count <= light:
                return "LIGHT"
            if count >= heavy:
                return "HEAVY"
            return "MODERATE"

        a_state = site_state(a_known, a_count)
        b_state = site_state(b_known, b_count)

        mid_open = (
            (m_known and m_count <= light)
            or self.control[f"{SIDE_MID}_FORWARD"] >= 0.25
        )

        # --------------------------------------------------------------
        # DEFAULT: まだA/Bの状況が分からないなら情報を取る。
        # --------------------------------------------------------------
        if strategy == "DEFAULT":
            if not a_known and not b_known:
                return 0.78
            if not both:
                return 0.38
            return -0.22

        # --------------------------------------------------------------
        # RUSH: 「薄いならシンプルに行く」を明確に評価。
        # --------------------------------------------------------------
        if strategy == "A_RUSH":
            if not a_known:
                return 0.0

            if a_state == "LIGHT":
                score = 0.72 + DIRECT_RUSH_LIGHT_BONUS
            elif a_state == "MODERATE":
                score = 0.12
            else:
                score = -0.78

            if both:
                if a_count + 0.75 < b_count:
                    score += 0.18
                elif a_count > b_count + 0.75:
                    score -= 0.18

            return float(np.clip(score, -1.0, 1.0))

        if strategy == "B_RUSH":
            if not b_known:
                return 0.0

            if b_state == "LIGHT":
                score = 0.72 + DIRECT_RUSH_LIGHT_BONUS
            elif b_state == "MODERATE":
                score = 0.12
            else:
                score = -0.78

            if both:
                if b_count + 0.75 < a_count:
                    score += 0.18
                elif b_count > a_count + 0.75:
                    score -= 0.18

            return float(np.clip(score, -1.0, 1.0))

        # --------------------------------------------------------------
        # SPLIT:
        # 空きサイトならRushの方が簡単なので少し下げる。
        # 1～2人守備 + Midが開いている時を本命条件にする。
        # --------------------------------------------------------------
        if strategy == "A_SPLIT":
            if not a_known:
                return 0.0

            if a_state == "LIGHT":
                score = (
                    0.55
                    + SPLIT_LIGHT_SITE_COMPLEXITY_PENALTY
                )
            elif a_state == "MODERATE":
                score = 0.40 + SPLIT_MODERATE_SITE_BONUS
            else:
                score = -0.48

            if m_known and m_count <= light:
                score += SPLIT_MID_OPEN_BONUS
            if self.control[f"{SIDE_MID}_FORWARD"] >= 0.25:
                score += SPLIT_MID_CONTROL_BONUS

            if both and a_count > b_count + 1.0:
                score -= 0.18

            return float(np.clip(score, -1.0, 1.0))

        if strategy == "B_SPLIT":
            if not b_known:
                return 0.0

            if b_state == "LIGHT":
                score = (
                    0.55
                    + SPLIT_LIGHT_SITE_COMPLEXITY_PENALTY
                )
            elif b_state == "MODERATE":
                score = 0.40 + SPLIT_MODERATE_SITE_BONUS
            else:
                score = -0.48

            if m_known and m_count <= light:
                score += SPLIT_MID_OPEN_BONUS
            if self.control[f"{SIDE_MID}_FORWARD"] >= 0.25:
                score += SPLIT_MID_CONTROL_BONUS

            if both and b_count > a_count + 1.0:
                score -= 0.18

            return float(np.clip(score, -1.0, 1.0))

        # --------------------------------------------------------------
        # REHIT:
        # 初回Executeほど高くはしない。
        # 厚いサイトへのRe-hitは悪い。
        # --------------------------------------------------------------
        if strategy == "REHIT_A":
            if not a_known:
                return 0.0
            if a_state == "LIGHT":
                return 0.35
            if a_state == "MODERATE":
                return 0.10
            return -0.65

        if strategy == "REHIT_B":
            if not b_known:
                return 0.0
            if b_state == "LIGHT":
                return 0.35
            if b_state == "MODERATE":
                return 0.10
            return -0.65

        # --------------------------------------------------------------
        # MID_TO_B
        # --------------------------------------------------------------
        if strategy == "MID_TO_B":
            score = 0.0
            if m_known:
                if m_count <= light:
                    score += 0.42
                elif m_count >= heavy:
                    score -= 0.48

            if b_known:
                if b_state == "LIGHT":
                    # Bが完全に薄いなら普通のB_RUSHよりは低くする。
                    score += 0.14
                elif b_state == "MODERATE":
                    score += 0.32
                else:
                    score -= 0.48

            return float(np.clip(score, -1.0, 1.0))

        # --------------------------------------------------------------
        # ROTATE
        # --------------------------------------------------------------
        if strategy == "ROTATE_A_TO_B":
            if not both:
                return 0.0
            if a_state == "HEAVY" and b_state == "LIGHT":
                return 1.0
            if b_state == "HEAVY" and a_state == "LIGHT":
                return -1.0
            if b_count + 1.0 < a_count:
                return 0.45
            if a_count + 1.0 < b_count:
                return -0.45
            return 0.0

        if strategy == "ROTATE_B_TO_A":
            if not both:
                return 0.0
            if b_state == "HEAVY" and a_state == "LIGHT":
                return 1.0
            if a_state == "HEAVY" and b_state == "LIGHT":
                return -1.0
            if a_count + 1.0 < b_count:
                return 0.45
            if b_count + 1.0 < a_count:
                return -0.45
            return 0.0

        # --------------------------------------------------------------
        # FAKE:
        # v11では「本命側が薄い」こと自体が大きな加点だったため、
        # 空きサイトへ直接Rushできる状況でもFakeが高得点になった。
        #
        # v12では:
        # - 本命が完全にLIGHTなら機会損失ペナルティ
        # - 本命がMODERATEでFake側に守備が多い時を最も評価
        # --------------------------------------------------------------
        if strategy == "FAKE_A_TO_B":
            if not both:
                return 0.0

            score = 0.0

            # A側に「見せる価値」があるか
            if a_state == "HEAVY":
                score += 0.42
            elif a_state == "MODERATE":
                score += 0.25
            else:
                score -= 0.18

            # 本命B
            if b_state == "LIGHT":
                # Bが既に空いているなら普通にB_RUSHする方が速い。
                score += 0.28
                score += FAKE_DIRECT_EXECUTE_OPPORTUNITY_PENALTY
            elif b_state == "MODERATE":
                score += 0.34 + FAKE_PULL_VALUE_BONUS
            else:
                score -= 0.60

            if a_count >= b_count + 1.5:
                score += 0.10

            return float(np.clip(score, -1.0, 1.0))

        if strategy == "FAKE_B_TO_A":
            if not both:
                return 0.0

            score = 0.0

            if b_state == "HEAVY":
                score += 0.42
            elif b_state == "MODERATE":
                score += 0.25
            else:
                score -= 0.18

            if a_state == "LIGHT":
                score += 0.28
                score += FAKE_DIRECT_EXECUTE_OPPORTUNITY_PENALTY
            elif a_state == "MODERATE":
                score += 0.34 + FAKE_PULL_VALUE_BONUS
            else:
                score -= 0.60

            if b_count >= a_count + 1.5:
                score += 0.10

            return float(np.clip(score, -1.0, 1.0))

        return 0.0

    def _direct_rush_opportunity(self):
        """現在の情報でA/Bどちらかが明確にRush向きかを返す。"""
        light = float(
            GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"]
        )
        a_known, a_count = self._remembered_enemy_info(SIDE_A)
        b_known, b_count = self._remembered_enemy_info(SIDE_B)

        a_open = a_known and a_count <= light
        b_open = b_known and b_count <= light

        if a_open and not b_open:
            return "A"
        if b_open and not a_open:
            return "B"
        if a_open and b_open:
            return "A" if a_count <= b_count else "B"
        return None

    def _unnecessary_complexity_cost(self, new_strategy):
        """Rushで十分な取得済み情報がある時だけ複雑戦術へコスト。"""
        if self._direct_rush_opportunity() is None:
            return 0.0
        return float(UNNECESSARY_COMPLEXITY_COST.get(new_strategy, 0.0))

    def _strategy_selection_reward(self, old_strategy, new_strategy):
        """v11: 「今より良くなる切替」だけを評価する。

        例:
            A_SPLIT suitability = +0.60
            B_SPLIT suitability = +0.50
            delta = -0.10
            -> 切り替える価値なし。適性報酬0、switch costだけ支払う。

            A_RUSH suitability = -0.70
            ROTATE_A_TO_B = +1.00
            delta = +1.70
            -> 強く評価。
        """
        old_score = float(np.clip(
            self._strategy_suitability(old_strategy),
            -STRATEGY_SUITABILITY_SCORE_CLIP,
            STRATEGY_SUITABILITY_SCORE_CLIP,
        ))
        new_score = float(np.clip(
            self._strategy_suitability(new_strategy),
            -STRATEGY_SUITABILITY_SCORE_CLIP,
            STRATEGY_SUITABILITY_SCORE_CLIP,
        ))

        delta = float(new_score - old_score)

        # v12 diagnostic: 「Rushで十分なのに複雑戦術へ切替」が減るかを見る。
        if COMPLEXITY_DIAG_ENABLED:
            rush_side = self._direct_rush_opportunity()

            if new_strategy in {"A_RUSH", "B_RUSH"}:
                self._diag_direct_rush_preferred += 1

            if new_strategy in {"A_SPLIT", "B_SPLIT"}:
                self._diag_split_preferred += 1

            if new_strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}:
                self._diag_fake_preferred += 1

            if rush_side is not None:
                expected_rush = (
                    "A_RUSH" if rush_side == "A" else "B_RUSH"
                )
                complex_to_same_or_other_site = new_strategy in {
                    "A_SPLIT", "B_SPLIT",
                    "FAKE_A_TO_B", "FAKE_B_TO_A",
                    "MID_TO_B",
                }
                if (
                    complex_to_same_or_other_site
                    and new_score <= self._strategy_suitability(expected_rush)
                ):
                    self._diag_complex_when_rush_was_sufficient += 1

        # 既存の絶対適性診断も残す。
        self._diag_suitability_sum += new_score
        self._diag_suitability_count += 1
        if new_score > 0.05:
            self._diag_suitability_positive += 1
        elif new_score < -0.05:
            self._diag_suitability_negative += 1

        # v11 switch delta diagnostics
        self._diag_switch_delta_sum += delta
        self._diag_switch_delta_count += 1

        if delta > SUITABILITY_SWITCH_DEADZONE:
            self._diag_switch_improved += 1
            shaped_delta = delta - SUITABILITY_SWITCH_DEADZONE
            return STRATEGY_SUITABILITY_REWARD_SCALE * shaped_delta

        if delta < -SUITABILITY_SWITCH_DEADZONE:
            self._diag_switch_worse += 1
            shaped_delta = delta + SUITABILITY_SWITCH_DEADZONE
            return STRATEGY_SUITABILITY_REWARD_SCALE * shaped_delta

        # ほぼ同等の戦術への切替は報酬0。
        # STRATEGY_SWITCH_COSTがあるため、無意味な切替は純損になる。
        self._diag_switch_deadzone += 1
        return 0.0

    def _both_site_info_reward(self):
        """A/B両方の有効なremembered infoを初取得した瞬間だけ報酬。"""
        a_known, _ = self._remembered_enemy_info(SIDE_A)
        b_known, _ = self._remembered_enemy_info(SIDE_B)
        if not (a_known and b_known):
            return 0.0
        if not self._diag_both_site_info_acquired:
            self._diag_both_site_info_acquired = True
            self._diag_both_site_info_first_step = int(self.macro_step)
        if self._both_site_info_rewarded:
            return 0.0
        self._both_site_info_rewarded = True
        return BOTH_SITE_INFO_ACQUIRED_REWARD

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def _reward_before_after(self, before):
        reward = MACRO_STEP_COST

        # 情報/Controlは「途中報酬」として小さくする。
        info_gain = sum(self.info_conf.values()) - sum(before["info"].values())
        control_gain = sum(self.control.values()) - sum(before["control"].values())
        reward += INFO_PROGRESS_REWARD_SCALE * max(0.0, info_gain)
        reward += CONTROL_PROGRESS_REWARD_SCALE * max(0.0, control_gain)

        living = self._living_attackers()
        occupied_sides = {side_of_pos(a.pos) for a in living}
        if len(occupied_sides) >= 2:
            reward += DISTRIBUTION_REWARD
        if len(occupied_sides) >= 3:
            reward += DISTRIBUTION_REWARD

        self._update_tactical_history()

        # v5: 情報不足コミットと、練習scenarioの小さな成立補助。
        reward += self._uncertain_commit_penalty()
        reward += self._curriculum_progress_bonus()

        # 情報と一致するRotateだけ小さな途中報酬。
        heavy = float(GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"])
        light = float(GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"])

        a_known, a_count = self._remembered_enemy_info(SIDE_A)
        b_known, b_count = self._remembered_enemy_info(SIDE_B)
        a_heavy = a_known and a_count >= heavy
        b_heavy = b_known and b_count >= heavy
        a_light = a_known and a_count <= light
        b_light = b_known and b_count <= light

        if self.current_strategy == "ROTATE_A_TO_B":
            if a_heavy and b_light:
                reward += SMART_ROTATE_INTERMEDIATE
            elif b_heavy and a_light:
                reward += BAD_ROTATE_PENALTY

        elif self.current_strategy == "ROTATE_B_TO_A":
            if b_heavy and a_light:
                reward += SMART_ROTATE_INTERMEDIATE
            elif a_heavy and b_light:
                reward += BAD_ROTATE_PENALTY

        # 情報が十分なのに厚い側へコミットし続けるのは明確に罰する。
        if self.current_strategy in {"A_RUSH", "A_SPLIT", "REHIT_A"}:
            if a_heavy and b_light:
                reward += BAD_COMMIT_PENALTY

        if self.current_strategy in {"B_RUSH", "B_SPLIT", "MID_TO_B", "REHIT_B"}:
            if b_heavy and a_light:
                reward += BAD_COMMIT_PENALTY

        # 同じRush固定だけ軽く抑える。
        if self.current_strategy in {"A_RUSH", "B_RUSH"}:
            if self._last_rush_name == self.current_strategy:
                self._rush_streak += 1
            else:
                self._last_rush_name = self.current_strategy
                self._rush_streak = 1

            if self._rush_streak >= RUSH_REPEAT_START:
                reward += RUSH_REPEAT_PENALTY * (
                    self._rush_streak - RUSH_REPEAT_START + 1
                )
        else:
            self._last_rush_name = None
            self._rush_streak = 0

        return reward

    def _plant_completion_bonus(self):
        """Plantまでつながった戦術履歴をまとめて評価する。"""
        bonus = PLANT_BASE_REWARD

        if self._default_to_decision:
            bonus += PLANT_AFTER_DEFAULT_BONUS

        if self._smart_rotate_completed:
            bonus += PLANT_AFTER_SMART_ROTATE_BONUS

        if self._fake_completed:
            bonus += PLANT_AFTER_FAKE_BONUS

        if self._split_completed:
            bonus += PLANT_AFTER_SPLIT_BONUS

        if self._lurk_touched:
            bonus += PLANT_AFTER_LURK_BONUS

        if self._cut_touched:
            bonus += PLANT_AFTER_FLANK_CUT_BONUS

        # v5: 練習episodeで狙った戦術を本当にPlantまで完遂できた場合のみ追加。
        if self.curriculum_mode == "SPLIT" and self._split_completed:
            bonus += 0.8
        elif self.curriculum_mode == "FAKE" and self._fake_completed:
            bonus += 0.8
        elif self.curriculum_mode == "ROTATE" and self._smart_rotate_completed:
            bonus += 0.8

        # 速いPlantほど少し得。ただし最速Rush一択にならない程度に上限を低くする。
        time_ratio = max(
            0.0,
            (ROUND_DURATION_TICKS - self.tick)
            / max(1.0, ROUND_DURATION_TICKS),
        )
        if time_ratio > PLANT_FAST_BONUS_MIN_TIME_RATIO:
            denom = max(
                1e-6,
                1.0 - PLANT_FAST_BONUS_MIN_TIME_RATIO,
            )
            speed_factor = (
                time_ratio - PLANT_FAST_BONUS_MIN_TIME_RATIO
            ) / denom
            bonus += PLANT_FAST_BONUS_MAX * float(
                np.clip(speed_factor, 0.0, 1.0)
            )

        return bonus

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action_idx):
        if self.done:
            return self.build_observation(), 0.0, True, {}

        action_idx = int(action_idx)
        strategy = STRATEGIES[action_idx]

        # --------------------------------------------------------------
        # v5 diagnostic: record stage conditions at the moment the DQN
        # selects Fake / Split / Rotate. This does not alter the action.
        # --------------------------------------------------------------
        if strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}:
            self._diag_fake_selected += 1

        if strategy in {"A_SPLIT", "B_SPLIT"}:
            self._diag_split_selected += 1

        if strategy in {"ROTATE_A_TO_B", "ROTATE_B_TO_A"}:
            self._diag_rotate_selected += 1

            heavy = float(
                GC_MACRO_DECISION_THRESHOLDS["HEAVY_SITE_ENEMY_COUNT"]
            )
            light = float(
                GC_MACRO_DECISION_THRESHOLDS["LIGHT_SITE_ENEMY_COUNT"]
            )
            a_known, a_count = self._remembered_enemy_info(SIDE_A)
            b_known, b_count = self._remembered_enemy_info(SIDE_B)

            if a_known and b_known:
                self._diag_rotate_both_info = True

            if strategy == "ROTATE_A_TO_B":
                if a_known and b_known and a_count >= heavy and b_count <= light:
                    self._diag_rotate_heavy_light = True
                if a_known and b_known and b_count >= heavy and a_count <= light:
                    self._diag_rotate_wrong_heavy_light = True
            else:
                if a_known and b_known and b_count >= heavy and a_count <= light:
                    self._diag_rotate_heavy_light = True
                if a_known and b_known and a_count >= heavy and b_count <= light:
                    self._diag_rotate_wrong_heavy_light = True

        mask = self.action_mask()
        invalid = not bool(mask[action_idx])
        if invalid:
            strategy = self.current_strategy

        before = {
            "info": dict(self.info_conf),
            "control": dict(self.control),
            "fake": self.fake_value,
            "split_sync": self.split_sync_score,
            "living_a": len(self._living_attackers()),
            "living_d": sum(d.is_alive for d in self.defenders),
        }

        reward = 0.0

        if strategy != self.current_strategy:
            old_strategy = self.current_strategy

            # v19 diagnostic: if DEFAULT is being left before another explicit
            # diagnostic exit reason was recorded, classify it as switched.
            if old_strategy == "DEFAULT":
                self._record_default_diag_exit_v19("switched")

            # v11: currentを上書きする前にold/newの適性差を評価する。
            suitability_reward = self._strategy_selection_reward(
                old_strategy,
                strategy,
            )

            self.previous_strategy = old_strategy
            self.current_strategy = strategy
            self.strategy_age = 0
            self._default_info_option_completion_counted = False
            self._apply_strategy_assignments(strategy, initial=False)

            reward += STRATEGY_SWITCH_COST
            reward += suitability_reward
            # v13: 切替時だけ。戦術継続中には繰り返し課さない。
            reward += self._unnecessary_complexity_cost(strategy)
        else:
            self.strategy_age += 1

        if invalid:
            reward -= 0.35

        for _ in range(LOW_LEVEL_TICKS_PER_MACRO_STEP):
            if self.done:
                break

            self._move_attackers_one_tick()
            self._move_defenders_one_tick()
            self._resolve_contact()
            self.tick += 1
            self._update_information()

            # v7: SELL -> ROTATE -> EXECUTEを内部フェーズとして進行。
            self._update_fake_option_phase()

            # fake pressure pulls one defender toward fake side more often
            if (
                self.current_strategy in {"FAKE_A_TO_B", "FAKE_B_TO_A"}
                and self._fake_phase == "SELL"
            ):
                fake_side = (
                    SIDE_A
                    if self.current_strategy == "FAKE_A_TO_B"
                    else SIDE_B
                )
                if (
                    self.pressure[fake_side] >= FAKE_PRESSURE_MIN
                    and random.random() < FAKE_PULL_BONUS
                ):
                    others = [
                        d for d in self.defenders
                        if d.is_alive and side_of_pos(d.pos) != fake_side
                    ]
                    if others:
                        d = random.choice(others)
                        cells = _forward_control_cells(fake_side)
                        if cells:
                            d.pos = step_toward(
                                d.pos,
                                random.choice(cells),
                                {x.pos for x in self.defenders if x is not d and x.is_alive},
                            )

            if self._check_plant():
                break

            if not self._living_attackers():
                self.done = True
                self.success = False
                self.reason = "attackers_wiped"
                break

            if self.tick >= ROUND_DURATION_TICKS:
                self.done = True
                self.success = False
                self.reason = "timeout"
                break

        self.macro_step += 1
        reward += self._reward_before_after(before)
        reward += self._both_site_info_reward()

        living_a_now = len(self._living_attackers())
        living_d_now = sum(d.is_alive for d in self.defenders)
        reward += 0.18 * max(0, before["living_d"] - living_d_now)
        reward -= 0.14 * max(0, before["living_a"] - living_a_now)

        if self.done:
            if self.success:
                reward += self._plant_completion_bonus()
            else:
                reward -= 4.5
                if self.reason in {"timeout", "macro_timeout"}:
                    reward -= 1.5

        if self.macro_step >= MAX_MACRO_STEPS and not self.done:
            self.done = True
            self.reason = "macro_timeout"
            reward -= 4.0

        if self.done and self.current_strategy == "DEFAULT":
            self._record_default_diag_exit_v19("round_end")

        info = {
            "reason": self.reason,
            "strategy": self.current_strategy,
            "initial_strategy": self.initial_strategy,
            "success": self.success,
            "rotate_count": self.rotate_count,
            "rehit_count": self.rehit_count,
            "fake_value": self.fake_value,
            "split_sync": self.split_sync_score,
            "map_control": self.map_control_score,
            "defender_setup": self.defender_setup,
            "invalid_action": invalid,
            "default_to_decision": self._default_to_decision,
            "smart_rotate_completed": self._smart_rotate_completed,
            "fake_completed": self._fake_completed,
            "split_completed": self._split_completed,
            "lurk_touched": self._lurk_touched,
            "cut_touched": self._cut_touched,
            "info_memory_age_A": self._info_memory_age[SIDE_A],
            "info_memory_age_B": self._info_memory_age[SIDE_B],
            "split_main_ready_step": self._split_main_ready_step,
            "split_support_entry_step": self._split_support_ready_step,
            "fake_trigger_step": self._fake_trigger_step,
            "curriculum_mode": self.curriculum_mode,
            "curriculum_target": self.curriculum_target,

            "diag_fake_selected": self._diag_fake_selected,
            "diag_fake_triggered": self._diag_fake_triggered,
            "diag_fake_opposite_redeploy": self._diag_fake_opposite_redeploy,
            "diag_fake_expired": self._diag_fake_expired,
            "diag_fake_max_opposite_count": self._diag_fake_max_opposite_count,

            "diag_split_selected": self._diag_split_selected,
            "diag_split_main_ready": self._diag_split_main_ready,
            "diag_split_support_entry": self._diag_split_support_entry,
            "diag_split_both_ready": self._diag_split_both_ready,
            "diag_split_window_miss": self._diag_split_window_miss,
            "diag_split_min_step_gap": self._diag_split_min_step_gap,

            "diag_rotate_selected": self._diag_rotate_selected,
            "diag_rotate_both_info": self._diag_rotate_both_info,
            "diag_rotate_heavy_light": self._diag_rotate_heavy_light,
            "diag_rotate_wrong_heavy_light": self._diag_rotate_wrong_heavy_light,
            "diag_option_locked": self._option_lock_action_index() is not None,
            "diag_split_planned_entries": len(self._split_planned_entry),
            "fake_phase": self._fake_phase,
            "fake_sell_dwell": self._fake_sell_dwell,
            "fake_group_size": len(self._fake_group_names),
            "fake_rotate_group_size": len(self._fake_rotate_names),
            "fake_rotate_forward_count": sum(
                1
                for a in self._living_attackers()
                if self.assignment.get(a.name, (None, None, None))[1] == "FORWARD"
                and self.assignment.get(a.name, (None, None, None))[2] == "FAKE_ROTATE"
            ),
            "fake_phase_age": self._fake_phase_age(),
            "strategy_suitability_avg": (
                self._diag_suitability_sum / self._diag_suitability_count
                if self._diag_suitability_count > 0 else 0.0
            ),
            "strategy_suitability_count": self._diag_suitability_count,
            "strategy_suitability_positive": self._diag_suitability_positive,
            "strategy_suitability_negative": self._diag_suitability_negative,
            "direct_rush_choices": self._diag_direct_rush_preferred,
            "split_choices": self._diag_split_preferred,
            "fake_choices": self._diag_fake_preferred,
            "complex_when_rush_sufficient": (
                self._diag_complex_when_rush_was_sufficient
            ),
            "both_site_info_acquired": self._diag_both_site_info_acquired,
            "both_site_info_first_step": self._diag_both_site_info_first_step,
            "default_info_hold_ticks": self._diag_default_info_hold_ticks,
            "default_option_hold_steps": self._diag_default_option_hold_steps,
            "default_option_info_complete": self._diag_default_option_info_complete,
            "default_a_scout": self._default_a_scout_name,
            "default_b_scout": self._default_b_scout_name,
            "default_mid": self._default_mid_name,
            "default_main_side": self._default_main_side,
            "default_opposite_side": self._default_opposite_side,
            "default_main_lead": self._default_main_lead_name,
            "default_opposite_scout": self._default_opposite_scout_name,
            "main_lead_info_reached": self._diag_main_lead_info_reached,
            "opposite_scout_info_reached": (
                self._diag_opposite_scout_info_reached
            ),
            "default_role_assignments": self._diag_default_role_assignments,
            "a_scout_info_reached": self._diag_a_scout_info_reached,
            "b_scout_info_reached": self._diag_b_scout_info_reached,
            "a_scout_start_dist": self._diag_a_scout_start_dist,
            "b_scout_start_dist": self._diag_b_scout_start_dist,
            "a_scout_final_dist": self._diag_a_scout_final_dist,
            "b_scout_final_dist": self._diag_b_scout_final_dist,
            "a_scout_actual_target_start_dist": (
                self._diag_a_scout_actual_target_start_dist
            ),
            "b_scout_actual_target_start_dist": (
                self._diag_b_scout_actual_target_start_dist
            ),
            "default_start_remaining_steps_avg": (
                self._diag_default_start_remaining_steps_sum
                / self._diag_default_start_count
                if self._diag_default_start_count > 0
                else None
            ),
            "default_start_count": self._diag_default_start_count,

            "a_scout_move_attempts": self._diag_a_scout_move_attempts,
            "a_scout_forward_moves": self._diag_a_scout_forward_moves,
            "a_scout_stopped_moves": self._diag_a_scout_stopped_moves,
            "b_scout_move_attempts": self._diag_b_scout_move_attempts,
            "b_scout_forward_moves": self._diag_b_scout_forward_moves,
            "b_scout_stopped_moves": self._diag_b_scout_stopped_moves,

            "a_scout_died": self._diag_a_scout_died,
            "b_scout_died": self._diag_b_scout_died,

            "default_exit_info_complete": self._diag_default_exit_info_complete,
            "default_exit_max_age": self._diag_default_exit_max_age,
            "default_exit_round_end": self._diag_default_exit_round_end,
            "default_exit_switched": self._diag_default_exit_switched,
            "infer_any_high": self._diag_infer_any_high,
            "infer_heavy_est": self._diag_infer_heavy_est,
            "infer_mid_forward_ok": self._diag_infer_mid_forward_ok,
            "infer_mid_deep_ok": self._diag_infer_mid_deep_ok,
            "infer_mid_any_ok": self._diag_infer_mid_any_ok,
            "infer_opposite_bound_ok": self._diag_infer_opposite_bound_ok,
            "infer_final": self._diag_infer_final,
            "switch_delta_avg": (
                self._diag_switch_delta_sum / self._diag_switch_delta_count
                if self._diag_switch_delta_count > 0 else 0.0
            ),
            "switch_delta_count": self._diag_switch_delta_count,
            "switch_improved": self._diag_switch_improved,
            "switch_worse": self._diag_switch_worse,
            "switch_deadzone": self._diag_switch_deadzone,
        }

        return self.build_observation(), float(reward), self.done, info


# ============================================================================
# Dueling DQN
# ============================================================================

OBS_DIM = 68


class MacroDuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=HIDDEN):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        h = self.feature(x)
        value = self.value_head(h)
        adv = self.advantage_head(h)
        return value + adv - adv.mean(dim=1, keepdim=True)


Transition = namedtuple(
    "Transition",
    "obs action reward next_obs done next_mask",
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=int(capacity))

    def append(self, *args):
        self.data.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)


# ============================================================================
# DQN helpers
# ============================================================================

def epsilon_by_step(step):
    frac = min(1.0, step / max(1, EPS_DECAY_STEPS))
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(model, obs, mask, epsilon):
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return STRATEGY_TO_INDEX["DEFAULT"]

    if random.random() < epsilon:
        return int(random.choice(valid))

    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q = model(obs_t).squeeze(0).cpu().numpy()

    q = q.copy()
    q[~mask] = -1e9
    return int(np.argmax(q))


def optimize(model, target_model, optimizer, replay):
    if len(replay) < max(BATCH_SIZE, LEARNING_START):
        return None

    batch = replay.sample(BATCH_SIZE)

    obs = torch.from_numpy(np.stack([x.obs for x in batch])).float().to(DEVICE)
    actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=DEVICE)
    rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=DEVICE)
    next_obs = torch.from_numpy(np.stack([x.next_obs for x in batch])).float().to(DEVICE)
    dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=DEVICE)
    next_masks = torch.from_numpy(np.stack([x.next_mask for x in batch])).bool().to(DEVICE)

    q = model(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN + action mask
        next_online = model(next_obs)
        next_online[~next_masks] = -1e9
        next_actions = next_online.argmax(dim=1)

        next_target = target_model(next_obs)
        next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + GAMMA * (1.0 - dones) * next_q

    loss = nn.functional.smooth_l1_loss(q, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return float(loss.item())


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, episodes=EVAL_EPISODES, forced_curriculum_mode="FREE"):
    model.eval()

    successes = 0
    rewards = []
    reasons = Counter()
    strategy_counts = Counter()
    initial_counts = Counter()
    setup_stats = {}
    rotate_successes = 0
    fake_successes = 0
    split_successes = 0
    invalid_total = 0
    curriculum_stats = {}

    # --------------------------------------------------------------
    # v5 diagnostic totals.
    # "episodes" counters mean: at least once in that episode.
    # selection counters are actual number of action selections.
    # --------------------------------------------------------------
    diag = {
        "fake_selected_actions": 0,
        "fake_selected_episodes": 0,
        "fake_triggered_episodes": 0,
        "fake_redeploy_episodes": 0,
        "fake_expired_episodes": 0,
        "fake_completed_episodes": 0,
        "fake_max_opposite_sum": 0,

        "split_selected_actions": 0,
        "split_selected_episodes": 0,
        "split_main_ready_episodes": 0,
        "split_support_entry_episodes": 0,
        "split_both_ready_episodes": 0,
        "split_window_miss_episodes": 0,
        "split_completed_episodes": 0,
        "split_gap_sum": 0,
        "split_gap_n": 0,

        "rotate_selected_actions": 0,
        "rotate_selected_episodes": 0,
        "rotate_both_info_episodes": 0,
        "rotate_heavy_light_episodes": 0,
        "rotate_wrong_condition_episodes": 0,
        "rotate_completed_episodes": 0,

        "suitability_sum": 0.0,
        "suitability_count": 0,
        "suitability_positive": 0,
        "suitability_negative": 0,

        "direct_rush_choices": 0,
        "split_choices": 0,
        "fake_choices": 0,
        "complex_when_rush_sufficient": 0,
        "both_site_info_episodes": 0,
        "both_site_info_step_sum": 0.0,
        "both_site_info_step_n": 0,
        "default_info_hold_ticks": 0,
        "default_option_hold_steps": 0,
        "default_option_info_complete": 0,
        "default_role_assignments": 0,
        "a_scout_info_reached_episodes": 0,
        "b_scout_info_reached_episodes": 0,
        "both_scouts_reached_episodes": 0,
        "main_lead_info_reached_episodes": 0,
        "opposite_scout_info_reached_episodes": 0,
        "both_v20_roles_reached_episodes": 0,
        "a_low_info_episodes": 0,
        "a_medium_info_episodes": 0,
        "a_high_info_episodes": 0,
        "b_low_info_episodes": 0,
        "b_medium_info_episodes": 0,
        "b_high_info_episodes": 0,
        "rotate_usable_info_episodes": 0,
        "rotate_direct_decision_episodes": 0,
        "rotate_inferred_decision_episodes": 0,
        "rotate_inferred_a_to_b": 0,
        "rotate_inferred_b_to_a": 0,
        "a_scout_start_dist_sum": 0.0,
        "a_scout_start_dist_n": 0,
        "b_scout_start_dist_sum": 0.0,
        "b_scout_start_dist_n": 0,
        "a_scout_final_dist_sum": 0.0,
        "a_scout_final_dist_n": 0,
        "b_scout_final_dist_sum": 0.0,
        "b_scout_final_dist_n": 0,
        "a_scout_actual_target_start_dist_sum": 0.0,
        "a_scout_actual_target_start_dist_n": 0,
        "b_scout_actual_target_start_dist_sum": 0.0,
        "b_scout_actual_target_start_dist_n": 0,

        "default_start_remaining_steps_sum": 0.0,
        "default_start_remaining_steps_n": 0,

        "a_scout_move_attempts": 0,
        "a_scout_forward_moves": 0,
        "a_scout_stopped_moves": 0,
        "b_scout_move_attempts": 0,
        "b_scout_forward_moves": 0,
        "b_scout_stopped_moves": 0,

        "a_scout_died_episodes": 0,
        "b_scout_died_episodes": 0,

        "default_exit_info_complete": 0,
        "default_exit_max_age": 0,
        "default_exit_round_end": 0,
        "default_exit_switched": 0,

        "infer_any_high": 0,
        "infer_heavy_est": 0,
        "infer_mid_forward_ok": 0,
        "infer_mid_deep_ok": 0,
        "infer_mid_any_ok": 0,
        "infer_opposite_bound_ok": 0,
        "infer_final": 0,

        "switch_delta_sum": 0.0,
        "switch_delta_count": 0,
        "switch_improved": 0,
        "switch_worse": 0,
        "switch_deadzone": 0,
    }

    curriculum_diag = {}

    with torch.no_grad():
        for ep in range(episodes):
            env = MacroEnv()
            obs = env.reset(
                forced_curriculum_mode=forced_curriculum_mode
            )
            total_reward = 0.0

            for _ in range(MAX_MACRO_STEPS):
                mask = env.action_mask()
                action = select_action(model, obs, mask, epsilon=0.0)
                strategy_counts[STRATEGIES[action]] += 1

                obs, reward, done, info = env.step(action)
                total_reward += reward
                invalid_total += int(info.get("invalid_action", False))

                if done:
                    break

            rewards.append(total_reward)
            reasons[env.reason] += 1
            initial_counts[env.initial_strategy] += 1

            crow = curriculum_stats.setdefault(
                env.curriculum_mode,
                {"episodes": 0, "wins": 0, "split": 0, "fake": 0, "rotate": 0},
            )
            crow["episodes"] += 1
            if env.success:
                crow["wins"] += 1
            if env._split_completed:
                crow["split"] += 1
            if env._fake_completed:
                crow["fake"] += 1
            if env._smart_rotate_completed:
                crow["rotate"] += 1

            # Per-curriculum diagnostic funnel.
            cd = curriculum_diag.setdefault(
                env.curriculum_mode,
                {
                    "episodes": 0,
                    "fake_selected": 0,
                    "fake_triggered": 0,
                    "fake_redeploy": 0,
                    "fake_completed": 0,
                    "split_selected": 0,
                    "split_main": 0,
                    "split_entry": 0,
                    "split_both": 0,
                    "split_completed": 0,
                    "rotate_selected": 0,
                    "rotate_info": 0,
                    "rotate_condition": 0,
                    "rotate_completed": 0,
                },
            )
            cd["episodes"] += 1

            # Fake diagnostics
            diag["fake_selected_actions"] += env._diag_fake_selected
            diag["fake_max_opposite_sum"] += env._diag_fake_max_opposite_count
            if env._diag_fake_selected > 0:
                diag["fake_selected_episodes"] += 1
                cd["fake_selected"] += 1
            if env._diag_fake_triggered:
                diag["fake_triggered_episodes"] += 1
                cd["fake_triggered"] += 1
            if env._diag_fake_opposite_redeploy:
                diag["fake_redeploy_episodes"] += 1
                cd["fake_redeploy"] += 1
            if env._diag_fake_expired:
                diag["fake_expired_episodes"] += 1
            if env._fake_completed:
                diag["fake_completed_episodes"] += 1
                cd["fake_completed"] += 1

            # Split diagnostics
            diag["split_selected_actions"] += env._diag_split_selected
            if env._diag_split_selected > 0:
                diag["split_selected_episodes"] += 1
                cd["split_selected"] += 1
            if env._diag_split_main_ready:
                diag["split_main_ready_episodes"] += 1
                cd["split_main"] += 1
            if env._diag_split_support_entry:
                diag["split_support_entry_episodes"] += 1
                cd["split_entry"] += 1
            if env._diag_split_both_ready:
                diag["split_both_ready_episodes"] += 1
                cd["split_both"] += 1
            if env._diag_split_window_miss:
                diag["split_window_miss_episodes"] += 1
            if env._split_completed:
                diag["split_completed_episodes"] += 1
                cd["split_completed"] += 1
            if env._diag_split_min_step_gap is not None:
                diag["split_gap_sum"] += int(env._diag_split_min_step_gap)
                diag["split_gap_n"] += 1

            # Rotate diagnostics
            diag["rotate_selected_actions"] += env._diag_rotate_selected
            if env._diag_rotate_selected > 0:
                diag["rotate_selected_episodes"] += 1
                cd["rotate_selected"] += 1
            if env._diag_rotate_both_info:
                diag["rotate_both_info_episodes"] += 1
                cd["rotate_info"] += 1
            if env._diag_rotate_heavy_light:
                diag["rotate_heavy_light_episodes"] += 1
                cd["rotate_condition"] += 1
            if env._diag_rotate_wrong_heavy_light:
                diag["rotate_wrong_condition_episodes"] += 1
            if env._smart_rotate_completed:
                diag["rotate_completed_episodes"] += 1
                cd["rotate_completed"] += 1

            diag["suitability_sum"] += env._diag_suitability_sum
            diag["suitability_count"] += env._diag_suitability_count
            diag["suitability_positive"] += env._diag_suitability_positive
            diag["suitability_negative"] += env._diag_suitability_negative

            diag["direct_rush_choices"] += env._diag_direct_rush_preferred
            diag["split_choices"] += env._diag_split_preferred
            diag["fake_choices"] += env._diag_fake_preferred
            diag["complex_when_rush_sufficient"] += (
                env._diag_complex_when_rush_was_sufficient
            )
            if env._diag_both_site_info_acquired:
                diag["both_site_info_episodes"] += 1
                if env._diag_both_site_info_first_step is not None:
                    diag["both_site_info_step_sum"] += float(env._diag_both_site_info_first_step)
                    diag["both_site_info_step_n"] += 1
            diag["default_info_hold_ticks"] += env._diag_default_info_hold_ticks
            diag["default_option_hold_steps"] += env._diag_default_option_hold_steps
            diag["default_option_info_complete"] += env._diag_default_option_info_complete
            diag["default_role_assignments"] += env._diag_default_role_assignments
            if env._diag_a_scout_info_reached:
                diag["a_scout_info_reached_episodes"] += 1
            if env._diag_b_scout_info_reached:
                diag["b_scout_info_reached_episodes"] += 1
            if (
                env._diag_a_scout_info_reached
                and env._diag_b_scout_info_reached
            ):
                diag["both_scouts_reached_episodes"] += 1
            if env._diag_main_lead_info_reached:
                diag["main_lead_info_reached_episodes"] += 1
            if env._diag_opposite_scout_info_reached:
                diag["opposite_scout_info_reached_episodes"] += 1
            if (
                env._diag_main_lead_info_reached
                and env._diag_opposite_scout_info_reached
            ):
                diag["both_v20_roles_reached_episodes"] += 1

            a_level, _ac, _an, _as = env._info_level(SIDE_A)
            b_level, _bc, _bn, _bs = env._info_level(SIDE_B)

            if a_level in {"LOW", "MEDIUM", "HIGH"}:
                diag["a_low_info_episodes"] += 1
            if a_level in {"MEDIUM", "HIGH"}:
                diag["a_medium_info_episodes"] += 1
            if a_level == "HIGH":
                diag["a_high_info_episodes"] += 1

            if b_level in {"LOW", "MEDIUM", "HIGH"}:
                diag["b_low_info_episodes"] += 1
            if b_level in {"MEDIUM", "HIGH"}:
                diag["b_medium_info_episodes"] += 1
            if b_level == "HIGH":
                diag["b_high_info_episodes"] += 1

            usable, *_ = env._rotate_info_pair()
            if usable:
                diag["rotate_usable_info_episodes"] += 1

            rotate_decision = env._rotate_decision_v22()
            if rotate_decision is not None:
                if rotate_decision["source"] == "DIRECT":
                    diag["rotate_direct_decision_episodes"] += 1
                else:
                    diag["rotate_inferred_decision_episodes"] += 1
                    if rotate_decision["direction"] == "A_TO_B":
                        diag["rotate_inferred_a_to_b"] += 1
                    else:
                        diag["rotate_inferred_b_to_a"] += 1
            if env._diag_a_scout_start_dist is not None:
                diag["a_scout_start_dist_sum"] += env._diag_a_scout_start_dist
                diag["a_scout_start_dist_n"] += 1
            if env._diag_b_scout_start_dist is not None:
                diag["b_scout_start_dist_sum"] += env._diag_b_scout_start_dist
                diag["b_scout_start_dist_n"] += 1
            if env._diag_a_scout_final_dist is not None:
                diag["a_scout_final_dist_sum"] += env._diag_a_scout_final_dist
                diag["a_scout_final_dist_n"] += 1
            if env._diag_b_scout_final_dist is not None:
                diag["b_scout_final_dist_sum"] += env._diag_b_scout_final_dist
                diag["b_scout_final_dist_n"] += 1
            if env._diag_a_scout_actual_target_start_dist is not None:
                diag["a_scout_actual_target_start_dist_sum"] += (
                    env._diag_a_scout_actual_target_start_dist
                )
                diag["a_scout_actual_target_start_dist_n"] += 1
            if env._diag_b_scout_actual_target_start_dist is not None:
                diag["b_scout_actual_target_start_dist_sum"] += (
                    env._diag_b_scout_actual_target_start_dist
                )
                diag["b_scout_actual_target_start_dist_n"] += 1
            if env._diag_default_start_count > 0:
                diag["default_start_remaining_steps_sum"] += (
                    env._diag_default_start_remaining_steps_sum
                )
                diag["default_start_remaining_steps_n"] += (
                    env._diag_default_start_count
                )

            diag["a_scout_move_attempts"] += env._diag_a_scout_move_attempts
            diag["a_scout_forward_moves"] += env._diag_a_scout_forward_moves
            diag["a_scout_stopped_moves"] += env._diag_a_scout_stopped_moves
            diag["b_scout_move_attempts"] += env._diag_b_scout_move_attempts
            diag["b_scout_forward_moves"] += env._diag_b_scout_forward_moves
            diag["b_scout_stopped_moves"] += env._diag_b_scout_stopped_moves

            if env._diag_a_scout_died:
                diag["a_scout_died_episodes"] += 1
            if env._diag_b_scout_died:
                diag["b_scout_died_episodes"] += 1

            diag["default_exit_info_complete"] += (
                env._diag_default_exit_info_complete
            )
            diag["default_exit_max_age"] += env._diag_default_exit_max_age
            diag["default_exit_round_end"] += env._diag_default_exit_round_end
            diag["default_exit_switched"] += env._diag_default_exit_switched
            diag["infer_any_high"] += env._diag_infer_any_high
            diag["infer_heavy_est"] += env._diag_infer_heavy_est
            diag["infer_mid_forward_ok"] += env._diag_infer_mid_forward_ok
            diag["infer_mid_deep_ok"] += env._diag_infer_mid_deep_ok
            diag["infer_mid_any_ok"] += env._diag_infer_mid_any_ok
            diag["infer_opposite_bound_ok"] += env._diag_infer_opposite_bound_ok
            diag["infer_final"] += env._diag_infer_final

            diag["switch_delta_sum"] += env._diag_switch_delta_sum
            diag["switch_delta_count"] += env._diag_switch_delta_count
            diag["switch_improved"] += env._diag_switch_improved
            diag["switch_worse"] += env._diag_switch_worse
            diag["switch_deadzone"] += env._diag_switch_deadzone

            row = setup_stats.setdefault(
                env.defender_setup,
                {"episodes": 0, "wins": 0},
            )
            row["episodes"] += 1

            if env.success:
                successes += 1
                row["wins"] += 1
                if env._smart_rotate_completed:
                    rotate_successes += 1
                if env._fake_completed:
                    fake_successes += 1
                if env._split_completed:
                    split_successes += 1

    model.train()

    diag["fake_avg_max_opposite"] = (
        diag["fake_max_opposite_sum"] / max(1, diag["fake_selected_episodes"])
    )
    diag["split_avg_min_step_gap"] = (
        diag["split_gap_sum"] / max(1, diag["split_gap_n"])
        if diag["split_gap_n"] > 0
        else None
    )
    diag["suitability_avg"] = (
        diag["suitability_sum"] / diag["suitability_count"]
        if diag["suitability_count"] > 0
        else 0.0
    )
    diag["a_scout_avg_start_dist"] = (
        diag["a_scout_start_dist_sum"] / diag["a_scout_start_dist_n"]
        if diag["a_scout_start_dist_n"] > 0 else None
    )
    diag["b_scout_avg_start_dist"] = (
        diag["b_scout_start_dist_sum"] / diag["b_scout_start_dist_n"]
        if diag["b_scout_start_dist_n"] > 0 else None
    )
    diag["a_scout_avg_final_dist"] = (
        diag["a_scout_final_dist_sum"] / diag["a_scout_final_dist_n"]
        if diag["a_scout_final_dist_n"] > 0 else None
    )
    diag["b_scout_avg_final_dist"] = (
        diag["b_scout_final_dist_sum"] / diag["b_scout_final_dist_n"]
        if diag["b_scout_final_dist_n"] > 0 else None
    )

    diag["default_avg_start_remaining_steps"] = (
        diag["default_start_remaining_steps_sum"]
        / diag["default_start_remaining_steps_n"]
        if diag["default_start_remaining_steps_n"] > 0
        else None
    )

    diag["a_scout_forward_rate"] = (
        diag["a_scout_forward_moves"] / diag["a_scout_move_attempts"]
        if diag["a_scout_move_attempts"] > 0
        else None
    )
    diag["b_scout_forward_rate"] = (
        diag["b_scout_forward_moves"] / diag["b_scout_move_attempts"]
        if diag["b_scout_move_attempts"] > 0
        else None
    )
    diag["a_scout_stop_rate"] = (
        diag["a_scout_stopped_moves"] / diag["a_scout_move_attempts"]
        if diag["a_scout_move_attempts"] > 0
        else None
    )
    diag["b_scout_stop_rate"] = (
        diag["b_scout_stopped_moves"] / diag["b_scout_move_attempts"]
        if diag["b_scout_move_attempts"] > 0
        else None
    )

    diag["a_scout_avg_actual_target_start_dist"] = (
        diag["a_scout_actual_target_start_dist_sum"]
        / diag["a_scout_actual_target_start_dist_n"]
        if diag["a_scout_actual_target_start_dist_n"] > 0
        else None
    )
    diag["b_scout_avg_actual_target_start_dist"] = (
        diag["b_scout_actual_target_start_dist_sum"]
        / diag["b_scout_actual_target_start_dist_n"]
        if diag["b_scout_actual_target_start_dist_n"] > 0
        else None
    )

    diag["both_site_info_avg_step"] = (
        diag["both_site_info_step_sum"] / diag["both_site_info_step_n"]
        if diag["both_site_info_step_n"] > 0
        else None
    )
    diag["switch_delta_avg"] = (
        diag["switch_delta_sum"] / diag["switch_delta_count"]
        if diag["switch_delta_count"] > 0
        else 0.0
    )

    return {
        "success_rate": successes / episodes,
        "avg_reward": float(np.mean(rewards)),
        "reasons": dict(reasons),
        "strategy_counts": dict(strategy_counts),
        "initial_counts": dict(initial_counts),
        "setup_stats": setup_stats,
        "rotate_success_rate": rotate_successes / episodes,
        "fake_success_rate": fake_successes / episodes,
        "split_success_rate": split_successes / episodes,
        "invalid_per_episode": invalid_total / episodes,
        "curriculum_stats": curriculum_stats,
        "diagnostic": diag,
        "curriculum_diagnostic": curriculum_diag,
    }


# ============================================================================
# Checkpoint
# ============================================================================

def checkpoint_dict(model, episode, eval_result=None):
    return {
        "model_state_dict": model.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "strategies": list(STRATEGIES),
        "episode": int(episode),
        "eval_result": dict(eval_result or {}),
        "macro_map_version": 19,
        "roster_order": list(GC_ROSTER_ORDER),
    }


def save_checkpoint(path, model, episode, eval_result=None):
    torch.save(
        checkpoint_dict(model, episode, eval_result),
        str(path),
    )


# ============================================================================
# Main
# ============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 78)
    print("Ghost Champions - Attacker Macro DQN Training v24")
    print("=" * 78)
    print(f"device       : {DEVICE}")
    print(f"episodes     : {EPISODE_COUNT}")
    print(f"obs_dim      : {OBS_DIM}")
    print(f"actions      : {N_ACTIONS}")
    print(f"map size     : {HEIGHT} x {WIDTH}")
    print(f"roster       : {GC_ROSTER_ORDER}")
    print(f"save dir     : {DATA_DIR}")
    print("-" * 78)

    print("[STRATEGY WEIGHTS]")
    for name in STRATEGIES:
        print(f"  {name:16s}: {GC_MACRO_STRATEGY_WEIGHTS.get(name, 0.0):.3f}")

    print("[MAP COUNTS]")
    for label, groups in (
        ("ZONE", ZONE_CELLS),
        ("TACTICAL", TACTICAL_CELLS),
        ("ROTATE_INFO", ROTATE_INFO_CELLS),
        ("CONTROL", CONTROL_CELLS),
        ("ROLE", ROLE_CELLS),
    ):
        print(f"  {label}: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    model = MacroDuelingDQN().to(DEVICE)
    target_model = MacroDuelingDQN().to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    global_step = 0
    best_success = -1.0
    best_reward = -1e18

    rolling_rewards = deque(maxlen=100)
    rolling_success = deque(maxlen=100)
    rolling_invalid = deque(maxlen=100)
    rolling_strategy = Counter()

    start_time = time.time()

    for episode in range(1, EPISODE_COUNT + 1):
        env = MacroEnv()
        obs = env.reset()
        ep_reward = 0.0
        losses = []
        invalid_count = 0

        for _ in range(MAX_MACRO_STEPS):
            mask = env.action_mask()
            eps = epsilon_by_step(global_step)
            action = select_action(model, obs, mask, eps)

            next_obs, reward, done, info = env.step(action)
            next_mask = env.action_mask() if not done else np.ones(N_ACTIONS, dtype=bool)

            replay.append(
                obs,
                action,
                reward,
                next_obs,
                float(done),
                next_mask,
            )

            rolling_strategy[STRATEGIES[action]] += 1
            invalid_count += int(info.get("invalid_action", False))
            ep_reward += reward
            obs = next_obs
            global_step += 1

            loss = optimize(
                model,
                target_model,
                optimizer,
                replay,
            )
            if loss is not None:
                losses.append(loss)

            if global_step % TARGET_SYNC == 0:
                target_model.load_state_dict(model.state_dict())

            if done:
                break

        rolling_rewards.append(ep_reward)
        rolling_success.append(float(env.success))
        rolling_invalid.append(invalid_count)

        if episode % PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            eps_now = epsilon_by_step(global_step)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            print(
                f"[EP {episode:5d}/{EPISODE_COUNT}] "
                f"reward={ep_reward:7.3f} "
                f"avg100={np.mean(rolling_rewards):7.3f} "
                f"success100={np.mean(rolling_success):.3f} "
                f"invalid100={np.mean(rolling_invalid):.3f} "
                f"eps={eps_now:.3f} "
                f"buffer={len(replay)} "
                f"loss={avg_loss:.5f} "
                f"reason={env.reason or '-'} "
                f"strategy={env.current_strategy} "
                f"elapsed={elapsed:.1f}s"
            )

        if episode % EVAL_INTERVAL == 0:
            result = evaluate(model, EVAL_EPISODES, forced_curriculum_mode="FREE")

            print(
                f"[EVAL FREE EP {episode}/{EPISODE_COUNT}] "
                f"success={result['success_rate']:.3f} "
                f"reward={result['avg_reward']:.3f} "
                f"rotate={result['rotate_success_rate']:.3f} "
                f"fake={result['fake_success_rate']:.3f} "
                f"split={result['split_success_rate']:.3f} "
                f"invalid={result['invalid_per_episode']:.3f}"
            )
            print(f"  reasons={result['reasons']}")
            print(f"  action_counts={result['strategy_counts']}")
            print(
                "  setup_win="
                + " / ".join(
                    f"{name}:{row['wins']}/{row['episodes']}"
                    for name, row in sorted(result["setup_stats"].items())
                )
            )
            print(
                "  eval_mode="
                + " / ".join(
                    (
                        f"{name}:W{row['wins']}/{row['episodes']}"
                        f",S{row['split']},F{row['fake']},R{row['rotate']}"
                    )
                    for name, row in sorted(result["curriculum_stats"].items())
                )
            )

            d = result["diagnostic"]
            print(
                "  [FAKE DIAG] "
                f"selected_actions={d['fake_selected_actions']} "
                f"selected_ep={d['fake_selected_episodes']} "
                f"triggered={d['fake_triggered_episodes']} "
                f"redeploy2+={d['fake_redeploy_episodes']} "
                f"expired={d['fake_expired_episodes']} "
                f"completed={d['fake_completed_episodes']} "
                f"avg_max_opp={d['fake_avg_max_opposite']:.2f}"
            )
            print(
                "  [SPLIT DIAG] "
                f"selected_actions={d['split_selected_actions']} "
                f"selected_ep={d['split_selected_episodes']} "
                f"main_ready={d['split_main_ready_episodes']} "
                f"support_entry={d['split_support_entry_episodes']} "
                f"both_in_window={d['split_both_ready_episodes']} "
                f"window_miss={d['split_window_miss_episodes']} "
                f"completed={d['split_completed_episodes']} "
                f"avg_min_gap={d['split_avg_min_step_gap']}"
            )
            print(
                "  [ROTATE DIAG] "
                f"selected_actions={d['rotate_selected_actions']} "
                f"selected_ep={d['rotate_selected_episodes']} "
                f"both_info={d['rotate_both_info_episodes']} "
                f"heavy_light={d['rotate_heavy_light_episodes']} "
                f"wrong_condition={d['rotate_wrong_condition_episodes']} "
                f"completed={d['rotate_completed_episodes']}"
            )
            print(
                "  [SUITABILITY DIAG] "
                f"avg={d['suitability_avg']:.3f} "
                f"choices={d['suitability_count']} "
                f"positive={d['suitability_positive']} "
                f"negative={d['suitability_negative']}"
            )
            print(
                "  [INFO DIAG] "
                f"both_site={d['both_site_info_episodes']}/{EVAL_EPISODES} "
                f"avg_first_step={d['both_site_info_avg_step']} "
                f"default_hold_ticks={d['default_info_hold_ticks']} "
                f"option_hold={d['default_option_hold_steps']} "
                f"option_complete={d['default_option_info_complete']} "
                f"A_scout_reached={d['a_scout_info_reached_episodes']} "
                f"B_scout_reached={d['b_scout_info_reached_episodes']} "
                f"both_scouts_reached={d['both_scouts_reached_episodes']} "
                f"A_start={d['a_scout_avg_start_dist']} "
                f"B_start={d['b_scout_avg_start_dist']} "
                f"A_final={d['a_scout_avg_final_dist']} "
                f"B_final={d['b_scout_avg_final_dist']} "
                f"A_target_start={d['a_scout_avg_actual_target_start_dist']} "
                f"B_target_start={d['b_scout_avg_actual_target_start_dist']} "
                f"lead_reached={d['main_lead_info_reached_episodes']} "
                f"opp_scout_reached={d['opposite_scout_info_reached_episodes']} "
                f"both_v20_roles={d['both_v20_roles_reached_episodes']}"
            )
            print(
                "  [INFO LEVEL DIAG] "
                f"A_low={d['a_low_info_episodes']} "
                f"A_med={d['a_medium_info_episodes']} "
                f"A_high={d['a_high_info_episodes']} "
                f"B_low={d['b_low_info_episodes']} "
                f"B_med={d['b_medium_info_episodes']} "
                f"B_high={d['b_high_info_episodes']} "
                f"rotate_usable={d['rotate_usable_info_episodes']}"
            )
            print(
                "  [ROTATE INFERENCE DIAG] "
                f"direct={d['rotate_direct_decision_episodes']} "
                f"inferred={d['rotate_inferred_decision_episodes']} "
                f"A_to_B={d['rotate_inferred_a_to_b']} "
                f"B_to_A={d['rotate_inferred_b_to_a']}"
            )
            print(
                "  [ROTATE INFER GATE DIAG] "
                f"any_high={d['infer_any_high']} "
                f"heavy_est={d['infer_heavy_est']} "
                f"mid_forward={d['infer_mid_forward_ok']} "
                f"mid_deep={d['infer_mid_deep_ok']} "
                f"mid_any={d['infer_mid_any_ok']} "
                f"opp_bound={d['infer_opposite_bound_ok']} "
                f"final={d['infer_final']}"
            )
            print(
                "  [SCOUT MOVE DIAG] "
                f"default_start_remaining={d['default_avg_start_remaining_steps']} "
                f"A_attempt={d['a_scout_move_attempts']} "
                f"A_forward={d['a_scout_forward_moves']} "
                f"A_stop={d['a_scout_stopped_moves']} "
                f"A_forward_rate={d['a_scout_forward_rate']} "
                f"A_stop_rate={d['a_scout_stop_rate']} "
                f"A_died_ep={d['a_scout_died_episodes']} "
                f"B_attempt={d['b_scout_move_attempts']} "
                f"B_forward={d['b_scout_forward_moves']} "
                f"B_stop={d['b_scout_stopped_moves']} "
                f"B_forward_rate={d['b_scout_forward_rate']} "
                f"B_stop_rate={d['b_scout_stop_rate']} "
                f"B_died_ep={d['b_scout_died_episodes']}"
            )
            print(
                "  [DEFAULT EXIT DIAG] "
                f"info_complete={d['default_exit_info_complete']} "
                f"max_age={d['default_exit_max_age']} "
                f"round_end={d['default_exit_round_end']} "
                f"switched={d['default_exit_switched']}"
            )
            print(
                "  [COMPLEXITY DIAG] "
                f"rush_choices={d['direct_rush_choices']} "
                f"split_choices={d['split_choices']} "
                f"fake_choices={d['fake_choices']} "
                f"complex_when_rush_sufficient="
                f"{d['complex_when_rush_sufficient']}"
            )
            print(
                "  [SWITCH DIAG] "
                f"avg_delta={d['switch_delta_avg']:.3f} "
                f"switches={d['switch_delta_count']} "
                f"improved={d['switch_improved']} "
                f"worse={d['switch_worse']} "
                f"deadzone={d['switch_deadzone']}"
            )
            print(
                "  [CURRICULUM DIAG] "
                + " / ".join(
                    (
                        f"{name}:"
                        f"F({row['fake_selected']}->{row['fake_triggered']}"
                        f"->{row['fake_redeploy']}->{row['fake_completed']}),"
                        f"S({row['split_selected']}->{row['split_main']}"
                        f"->{row['split_entry']}->{row['split_both']}"
                        f"->{row['split_completed']}),"
                        f"R({row['rotate_selected']}->{row['rotate_info']}"
                        f"->{row['rotate_condition']}->{row['rotate_completed']})"
                    )
                    for name, row in sorted(
                        result["curriculum_diagnostic"].items()
                    )
                )
            )

            # v13: 実戦FREEとOption練習性能を分離して表示。
            aux_parts = []
            for aux_mode in ("FAKE", "SPLIT", "ROTATE"):
                aux_result = evaluate(
                    model,
                    AUX_EVAL_EPISODES,
                    forced_curriculum_mode=aux_mode,
                )
                aux_parts.append(
                    f"{aux_mode}:W{aux_result['success_rate']:.3f}"
                    f",F{aux_result['fake_success_rate']:.3f}"
                    f",S{aux_result['split_success_rate']:.3f}"
                    f",R{aux_result['rotate_success_rate']:.3f}"
                )
            print("  [AUX CURRICULUM EVAL] " + " / ".join(aux_parts))

            success = float(result["success_rate"])
            avg_reward = float(result["avg_reward"])

            if (
                success > best_success
                or (
                    abs(success - best_success) < 1e-12
                    and avg_reward > best_reward
                )
            ):
                best_success = success
                best_reward = avg_reward
                save_checkpoint(
                    MODEL_BEST_PATH,
                    model,
                    episode,
                    result,
                )
                print(
                    f"  -> best saved: "
                    f"success={best_success:.3f} reward={best_reward:.3f}"
                )

            save_checkpoint(
                MODEL_LATEST_PATH,
                model,
                episode,
                result,
            )

    final_eval = evaluate(model, max(EVAL_EPISODES, 500))
    save_checkpoint(
        MODEL_FINAL_PATH,
        model,
        EPISODE_COUNT,
        final_eval,
    )
    save_checkpoint(
        MODEL_LATEST_PATH,
        model,
        EPISODE_COUNT,
        final_eval,
    )

    print("=" * 78)
    print("[DONE] Macro training finished.")
    print(
        f"final success={final_eval['success_rate']:.3f} "
        f"reward={final_eval['avg_reward']:.3f}"
    )
    print(f"best  : {MODEL_BEST_PATH}")
    print(f"latest: {MODEL_LATEST_PATH}")
    print(f"final : {MODEL_FINAL_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
