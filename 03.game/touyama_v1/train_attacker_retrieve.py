"""touyama_v1/train_attacker_retrieve.py

固定チーム(Tortlilyan/いぐるん/ろびぃな/夢の街/えんぺん)専用の
Attacker「retrieve phase」学習スクリプト(落下スパイク回収)。

【全員対称設計への変更】
retrieveフェーズは「キャリアーが死亡してスパイクが地面に落ちた後、
生存している誰かがそれを拾いに行く」状況で発生する。1マス1キャラの
衝突ルールはstep()側の移動解決(シャッフル順で早い者勝ち)が既に
担保しているため、特定の1人を「リトリーバー」役として選出する必要は
ないと判断し、全員が対称に「スパイクへのBFS距離を縮める」ことを
学習する設計に変更した。エピソードごとにTOUYAMA_ROSTER_ORDERから
1〜5人をランダムに抽選し、各キャラの実効ステータス・アビリティ種別
(character_stats_touyama.py + コンボ「ふわんだりぃず」+ タイガーパッシブ)
を適用する。

【HUNT(タイガー/Tortlilyan)対応】
アビリティonehotにHUNTを追加(3種→4種、OBS_DIM: 20→21)。HUNTは
アビリティ行動を持たないため、チャージ0として初期化し
(train_attacker_escort.pyと同じ方針)、_action_mask()が自動的に
ABILITY行動を弾く。

【敵ステータスの統一】
汎用版train_attacker_retrieve.pyはこのファイル固有のENEMY_ACCURACY等の
値を使っていたが、他のtouyama_v1学習ファイル(train_attacker_carry.py /
train_attacker_guard.py / train_attacker_escort.py)と同じ
DEFAULT_ACCURACY等の共通値に統一した。

完全に自己完結。他のfeatureモジュール(controllers.py, battle_logic.py等)は
importせず、必要なロジックはこのファイル内に持つ。map_data.py /
character_stats_touyama.py / game_core.py は定数専用ファイルとして参照する
(import制限の対象外)。run_game.py / controllers.py は変更しない。

保存先: touyama_v1/data/attacker_retrieve_touyama_data/
チェックポイントは{"model_state_dict","obs_dim","n_actions","episode",
"success_rate","roster_order"} を含むdict形式で保存する
(train_attacker_carry.py以降と同一方針)。
"""

import random
import sys
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR
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
)
from character_stats_touyama import (
    CHARACTER_TABLE as TOUYAMA_STATS_TABLE,
    TOUYAMA_ROSTER_ORDER,
)

EPISODE_COUNT = 4000

# ---------------------------------------------------------------------------
# 保存先
# ---------------------------------------------------------------------------
import os
DATA_DIR = "data/attacker_retrieve_touyama_data/"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "dqn_attacker_retrieve_touyama_best_by_eval.pt")
MODEL_FINAL_PATH = os.path.join(DATA_DIR, "dqn_attacker_retrieve_touyama_final.pt")

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

MAX_TICKS = 70
CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "ABILITY"]
N_ACTIONS = len(ACTIONS)
# HUNT(タイガー/Tortlilyan)を含む4種。HUNTはチャージ0で初期化されるため
# _action_mask()が自動的にABILITY行動を弾く。
ROLES = ["FLASH", "SMOKE", "RECON", "HUNT"]

# 敵スタブ(学習用の簡易ディフェンダー)。他のtouyama_v1学習ファイルと
# 共通のDEFAULT_*値に統一(汎用版はこのファイル固有の値を使っていた)。
ENEMY_SPAWN_PROB = 0.6          # そのエピソードで敵が出現するか
ENEMY_PRE_AFFECTED_PROB = 0.3   # 出現時、すでに味方が炙り出し済みという想定
DEFAULT_ACCURACY = 0.50
DEFAULT_DODGE = 0.12
DEFAULT_HS_RATE = 0.20
DEFAULT_REACTION = 100.0

# 全員が対称に「スパイクへの最短距離を縮める」ことを学習する設計に変更。
# 1マス1キャラの衝突ルールはstep()側の移動解決(シャッフル順で早い者勝ち)が
# 既に担保しているため、役割分担(retriever/blocker)は不要と判断し廃止した。
# T字路等で複数人が同時に同じマスへ進もうとしても、片方だけが進み、
# 残りは次tickで自分のBFS距離マップに従って別ルート・再試行するだけで
# 詰まりが解消される想定。
STALL_TICKS_PENALTY = -0.05     # 同じマスに留まり続けている場合の追加ペナルティ
STALL_TICKS_THRESHOLD = 3       # 何tick同じ位置に留まったらペナルティを課すか

# 報酬(全エージェント共通)
STEP_PENALTY = -0.02
GOAL_REWARD = 12.0
TIMEOUT_PENALTY = -4.0
DEATH_PENALTY = -8.0
ABILITY_GOOD_FIRE = 0.6          # 未使用の敵に初めて当てた
ABILITY_EMPTY_FIRE = -1.2        # 敵が見えないのに撃った(空撃ち)
ABILITY_WASTED_ON_AFFECTED = -0.8  # すでに炙り出されている敵に撃った(重複)
KILL_REWARD = 3.0
KILL_WHILE_DEBUFFED_BONUS = 1.5  # フラッシュ/リコン状態の敵を倒した追加ボーナス


Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done", "mask2"])


# ---------------------------------------------------------------------------
# touyama_v1 固定チーム定義(他のtouyama_v1学習ファイルと同一)
# ---------------------------------------------------------------------------
TOUYAMA_ROSTER_ORDER = ["Tortlilyan", "いぐるん", "ろびぃな", "夢の街", "えんぺん"]

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
    他のtouyama_v1学習ファイルと同一ロジック。"""
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

print("[touyama_v1] 固定チーム(Attacker/retrieve) 確定ステータス:")
for _name in TOUYAMA_ROSTER_ORDER:
    _s = TOUYAMA_EFFECTIVE_STATS[_name]
    print(
        f"  {_name}: acc={_s['accuracy']:.2f} hs={_s['hs_rate']:.2f} "
        f"dodge={_s['dodge_rate']:.2f} reaction={_s['reaction']:.0f} "
        f"ability={_s['ability']}"
    )


# ---------------------------------------------------------------------------
# マップ読み込み・BFS距離
# ---------------------------------------------------------------------------
def load_grid():
    lines = [l.strip() for l in NEW_MAZE_STR.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = load_grid()
HEIGHT, WIDTH = GRID.shape
WALKABLE = [
    (r, c)
    for r in range(HEIGHT)
    for c in range(WIDTH)
    if GRID[r, c] != 1
]


def bfs_distance_map(goal):
    """goalから各セルへの最短距離マップ(壁越え不可)。"""
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    dist[goal[0], goal[1]] = 0
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist

def line_cells(p1, p2):
    y0, x0 = p1
    y1, x1 = p2
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


def has_los(p1, p2):
    for r, c in line_cells(p1, p2):
        if GRID[r, c] == 1:
            return False
    return True


# ---------------------------------------------------------------------------
# 環境
# ---------------------------------------------------------------------------
class AgentState:
    """retrieve phase候補のアタッカー1体分の軽量な状態表現。
    全員が対称に「スパイクへ向かう」ため、retriever/blockerのような
    役割区分は持たない。game_core.Character非依存の自前実装。"""

    def __init__(self, name, pos, role):
        self.name = name
        self.pos = list(pos)
        self.role = role
        # HUNT(タイガー)はアビリティを持たないため、最初からチャージ0にする。
        self.charge = 0 if role == "HUNT" else 1
        self.hp = MAX_HP
        self.alive = True
        self.moved_this_tick = False
        self.stall_ticks = 0  # 足踏み判定は個体ごとに独立して持つ

        stats = TOUYAMA_EFFECTIVE_STATS[name]
        self.accuracy = stats["accuracy"]
        self.dodge_rate = stats["dodge_rate"]
        self.hs_rate = stats["hs_rate"]
        self.reaction = stats["reaction"] + random.uniform(-10, 10)


ACTION_ABILITY = 5


class RetrieveEnv:
    """touyama_v1ロースターから毎エピソード1〜5人を抽選し、全員が対称に
    「落下スパイクへの最短距離を縮める」ことを学習する環境。
    + 0〜1体の簡易敵スタブ。

    retriever/blockerのような役割分担は廃止した。1マス1キャラの
    衝突は移動解決(シャッフル順で早い者勝ち)で担保されるため、
    全員が同じ目的地(スパイク)へ貪欲に向かっても、通路の奥で
    詰まった側が単に「先に入った方を待つ」だけで自然に解消される
    という想定。"""

    OBS_DIM = 21
    # [0-1]座標 [2-5]壁 [6-9]隣接BFS距離 [10]自己BFS距離
    # [11-14]ロールonehot [15]アビリティ残チャージ
    # [16]視認中敵有無 [17-18]視認中敵相対方向 [19]敵blind [20]敵reveal
    # [21]is_retriever [22]自分がリトリーバーの次の一歩を塞いでいるか
    # [23]リトリーバーの目標までの正規化距離

    def reset(self):
        self.tick = 0

        self.spike_pos = random.choice(WALKABLE)
        self.dist_map = bfs_distance_map(self.spike_pos)

        # 実戦では「キャリアーが死亡した後の生存アタッカー」が対象なので
        # 最大4人程度が典型だが、学習分布を広めに取るため1〜ロースター全員
        # (5人)の範囲でランダムに人数・メンバーを抽選する。
        num_agents = random.randint(1, len(TOUYAMA_ROSTER_ORDER))
        agent_names = random.sample(TOUYAMA_ROSTER_ORDER, num_agents)

        reachable = [p for p in WALKABLE if self.dist_map[p] > 0]
        starts_pool = reachable if reachable else WALKABLE
        start_positions = random.sample(starts_pool, min(num_agents, len(starts_pool)))
        while len(start_positions) < num_agents:
            # 開始候補が足りない極端なケースのフォールバック(重複開始位置を許容)
            start_positions.append(random.choice(starts_pool))

        self.agents = [
            AgentState(name, pos, TOUYAMA_EFFECTIVE_STATS[name]["ability"])
            for name, pos in zip(agent_names, start_positions)
        ]

        # 敵スタブ
        self.enemy_alive = random.random() < ENEMY_SPAWN_PROB
        self.enemy_pos = None
        self.enemy_hp = MAX_HP
        self.enemy_blind = 0
        self.enemy_reveal = 0
        if self.enemy_alive:
            occupied_start = {tuple(a.pos) for a in self.agents} | {tuple(self.spike_pos)}
            candidates = [p for p in WALKABLE if p not in occupied_start]
            self.enemy_pos = list(random.choice(candidates))
            if random.random() < ENEMY_PRE_AFFECTED_PROB:
                if random.random() < 0.5:
                    self.enemy_blind = BLIND_DURATION_TICKS
                else:
                    self.enemy_reveal = REVEAL_DURATION_TICKS

        return self._collect_observations(), self._collect_masks()

    def _occupied_cells(self, exclude=None):
        return {
            tuple(a.pos) for a in self.agents if a is not exclude and a.alive
        }

    def _visible_enemy_for(self, unit):
        if not self.enemy_alive or not unit.alive:
            return False
        return has_los(tuple(unit.pos), tuple(self.enemy_pos))

    # -- 観測 --------------------------------------------------------------
    def _build_obs(self, unit):
        r, c = unit.pos
        wall_up = 1.0 if GRID[r - 1, c] == 1 else 0.0
        wall_down = 1.0 if GRID[r + 1, c] == 1 else 0.0
        wall_left = 1.0 if GRID[r, c - 1] == 1 else 0.0
        wall_right = 1.0 if GRID[r, c + 1] == 1 else 0.0

        occupied = self._occupied_cells(exclude=unit)

        max_dist_scale = float(HEIGHT + WIDTH)
        neighbor_dists = []
        for (dr_, dc_), is_wall in zip(
            CARDINAL, [wall_up, wall_down, wall_left, wall_right]
        ):
            nr, nc = r + dr_, c + dc_
            blocked = is_wall or not (0 <= nr < HEIGHT and 0 <= nc < WIDTH)
            if not blocked and (nr, nc) in occupied:
                blocked = True
            if blocked:
                neighbor_dists.append(1.0)
            else:
                neighbor_dists.append(min(1.0, self.dist_map[nr, nc] / max_dist_scale))

        raw_self_dist = self.dist_map[r, c]
        dist_norm = min(1.0, raw_self_dist / max_dist_scale) if raw_self_dist >= 0 else 1.0
        role_onehot = [1.0 if unit.role == role else 0.0 for role in ROLES]

        visible = self._visible_enemy_for(unit)
        if visible:
            er, ec = self.enemy_pos
            edr = float(np.clip((er - r) / HEIGHT, -1, 1))
            edc = float(np.clip((ec - c) / WIDTH, -1, 1))
            e_present = 1.0
            e_blind = 1.0 if self.enemy_blind > 0 else 0.0
            e_reveal = 1.0 if self.enemy_reveal > 0 else 0.0
        else:
            edr = edc = 0.0
            e_present = e_blind = e_reveal = 0.0

        obs = [
            r / HEIGHT, c / WIDTH,
            wall_up, wall_down, wall_left, wall_right,
            *neighbor_dists,
            dist_norm,
            *role_onehot,
            float(unit.charge),
            e_present, edr, edc, e_blind, e_reveal,
        ]
        obs_arr = np.array(obs, dtype=np.float32)
        assert obs_arr.shape[0] == self.OBS_DIM, (
            f"観測次元がOBS_DIM({self.OBS_DIM})と不一致: {obs_arr.shape[0]}"
        )
        return obs_arr

    def _action_mask_for(self, unit):
        """壁・占有マスへの移動 / チャージ0でのABILITYは禁止する。
        以前は"味方に塞がれたセルへの移動は物理的には選べる"扱いだったが、
        2エージェント構成になったことで衝突を明示的に防ぐ必要があるため、
        占有マスもマスクする(実ゲームのoccupied判定と一致させる)。"""
        mask = [True] * N_ACTIONS
        r, c = unit.pos
        occupied = self._occupied_cells(exclude=unit)
        for i, (dr, dc) in enumerate(CARDINAL):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH) or GRID[nr, nc] == 1:
                mask[i] = False
            elif (nr, nc) in occupied:
                mask[i] = False
        if unit.charge <= 0:
            mask[ACTION_ABILITY] = False
        return np.array(mask, dtype=bool)

    def _collect_observations(self):
        return {a.name: self._build_obs(a) for a in self.agents if a.alive}

    def _collect_masks(self):
        return {a.name: self._action_mask_for(a) for a in self.agents if a.alive}

    # -- ステップ ------------------------------------------------------------
    def step(self, action_dict):
        self.tick += 1
        info = {}

        alive_agents = [a for a in self.agents if a.alive]
        for u in alive_agents:
            u.moved_this_tick = False

        old_dist = {u.name: self.dist_map[tuple(u.pos)] for u in alive_agents}

        # --- 移動の適用(占有マス回避。マスクで既に禁止されているが、
        # 同時移動の衝突を避けるため念のため二重チェックする)。
        # シャッフル順で解決することで「同じマスを複数人が狙った場合、
        # そのtickで先に処理された1人だけが進み、残りはその場に留まる」
        # という早い者勝ちルールになる。これがT字路等の衝突を自然に解消する。 ---
        move_order = list(alive_agents)
        random.shuffle(move_order)
        for u in move_order:
            action = action_dict.get(u.name)
            if action is None or action >= 4:
                continue
            dr, dc = CARDINAL[action]
            nr, nc = u.pos[0] + dr, u.pos[1] + dc
            in_bounds = 0 <= nr < HEIGHT and 0 <= nc < WIDTH
            is_wall = in_bounds and GRID[nr, nc] == 1
            occupied_now = any(
                other is not u and other.alive and tuple(other.pos) == (nr, nc)
                for other in self.agents
            )
            if in_bounds and not is_wall and not occupied_now:
                u.pos = [nr, nc]
                u.moved_this_tick = True

        # --- アビリティ ---
        ability_rewards = {}
        for u in alive_agents:
            if action_dict.get(u.name) == ACTION_ABILITY:
                ability_rewards[u.name] = self._resolve_ability(u)

        # --- 足踏み・接近報酬(全エージェント共通) ---
        rewards = {}
        reached_name = None
        for u in alive_agents:
            if u.moved_this_tick:
                u.stall_ticks = 0
            else:
                u.stall_ticks += 1
            stall_penalty = STALL_TICKS_PENALTY if u.stall_ticks > STALL_TICKS_THRESHOLD else 0.0

            new_dist = self.dist_map[tuple(u.pos)]
            od = old_dist[u.name]
            approach_reward = 0.3 * (od - new_dist) if od >= 0 and new_dist >= 0 else 0.0

            reward = STEP_PENALTY + approach_reward + stall_penalty + ability_rewards.get(u.name, 0.0)

            if tuple(u.pos) == tuple(self.spike_pos):
                reward += GOAL_REWARD
                reached_name = u.name

            rewards[u.name] = reward

        done = reached_name is not None
        if done:
            info["result"] = "reached"
            info["reached_by"] = reached_name

        # --- 戦闘解決(誰かが既にスパイクへ到達していれば戦闘は発生させない) ---
        if not done and self.enemy_alive:
            combat_rewards = self._resolve_combat()
            for name, r in combat_rewards.items():
                rewards[name] = rewards.get(name, 0.0) + r

        # このtickで死亡したエージェントに死亡ペナルティを付与
        for u in alive_agents:
            if not u.alive:
                rewards[u.name] = rewards.get(u.name, 0.0) + DEATH_PENALTY

        if not done and all(not a.alive for a in self.agents):
            done = True
            info["result"] = "died"

        if not done and self.tick >= MAX_TICKS:
            for u in self.agents:
                if u.alive:
                    rewards[u.name] = rewards.get(u.name, 0.0) + TIMEOUT_PENALTY
            done = True
            info["result"] = "timeout"

        # 効果減衰
        self.enemy_blind = max(0, self.enemy_blind - 1)
        self.enemy_reveal = max(0, self.enemy_reveal - 1)

        obs_dict = self._collect_observations() if not done else {}
        mask_dict = self._collect_masks() if not done else {}

        return obs_dict, mask_dict, rewards, done, info

    def _resolve_ability(self, unit):
        if unit.charge <= 0:
            return 0.0
        unit.charge -= 1

        if not self._visible_enemy_for(unit):
            return ABILITY_EMPTY_FIRE

        already_affected = self.enemy_blind > 0 or self.enemy_reveal > 0
        if already_affected:
            return ABILITY_WASTED_ON_AFFECTED

        if unit.role == "FLASH":
            self.enemy_blind = BLIND_DURATION_TICKS
        elif unit.role == "RECON":
            self.enemy_reveal = REVEAL_DURATION_TICKS
        elif unit.role == "SMOKE":
            self.enemy_blind = max(self.enemy_blind, 1)
        # HUNTはここに到達しない(charge=0で常にマスクされているため)。
        return ABILITY_GOOD_FIRE

    def _resolve_combat(self):
        """生存中の全エージェント(最大5体)と敵スタブとの撃ち合いを解決する。"""
        rewards = {}
        units = [a for a in self.agents if a.alive]
        debuffed = self.enemy_blind > 0 or self.enemy_reveal > 0

        visible_units = [u for u in units if has_los(tuple(u.pos), tuple(self.enemy_pos))]
        enemy_target = None
        if visible_units and self.enemy_alive:
            enemy_target = min(
                visible_units,
                key=lambda u: max(abs(u.pos[0] - self.enemy_pos[0]), abs(u.pos[1] - self.enemy_pos[1])),
            )

        # 味方 -> 敵(視認していれば全員が同時に撃つ)
        for u in visible_units:
            accuracy = MOVING_ACCURACY if u.moved_this_tick else u.accuracy
            hit_chance = accuracy * (1.0 - DEFAULT_DODGE)
            if debuffed:
                hit_chance = min(1.0, hit_chance * 1.4)
            if self.enemy_alive and random.random() < hit_chance:
                dmg = HEADSHOT_DAMAGE if random.random() < u.hs_rate else BODY_DAMAGE
                self.enemy_hp -= dmg
                if self.enemy_hp <= 0 and self.enemy_alive:
                    self.enemy_alive = False
                    reward = KILL_REWARD + (KILL_WHILE_DEBUFFED_BONUS if debuffed else 0.0)
                    rewards[u.name] = rewards.get(u.name, 0.0) + reward

        # 敵 -> 味方(視認できている最も近いユニット1体を狙う)
        if enemy_target is not None and self.enemy_alive:
            u = enemy_target
            my_effective_dodge = u.dodge_rate * (REVEALED_DODGE_MULTIPLIER if debuffed else 1.0)
            enemy_hit_chance = DEFAULT_ACCURACY * (1.0 - my_effective_dodge) * MOVING_TARGET_HIT_MULTIPLIER
            if debuffed and self.enemy_blind > 0:
                enemy_hit_chance *= BLIND_ACCURACY_MULTIPLIER
            enemy_hit_chance = max(0.0, min(1.0, enemy_hit_chance))
            if random.random() < enemy_hit_chance:
                dmg = HEADSHOT_DAMAGE if random.random() < DEFAULT_HS_RATE else BODY_DAMAGE
                u.hp -= dmg
                if u.hp <= 0:
                    u.alive = False

        return rewards

# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


def masked_argmax(q_values, mask):
    q = q_values.clone()
    q[~mask] = -1e9
    return int(torch.argmax(q).item())


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------
def _save_checkpoint(policy_net, path, episode, success_rate):
    """learning_attacker_retrieve.py(推論側)が期待するdict形式で保存する
    (train_attacker_carry.py以降と同一方針)。"""
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "obs_dim": RetrieveEnv.OBS_DIM,
            "n_actions": N_ACTIONS,
            "episode": episode,
            "success_rate": success_rate,
            "roster_order": list(TOUYAMA_ROSTER_ORDER),
        },
        path,
    )


def train(
    episodes=EPISODE_COUNT,
    batch_size=128,
    gamma=0.97,
    lr=3e-4,
    buffer_size=50000,
    target_update_every=1000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=int(EPISODE_COUNT * 0.8),
    warmup_steps=2000,
):
    """生存中の全エージェント(1〜5人)は同一の重み共有ネットワークで
    制御し、各tickで全員ぶんの遷移を同じリプレイバッファへ積む
    (パラメータ共有マルチエージェント学習)。"""
    env = RetrieveEnv()
    policy_net = DuelingQNet(RetrieveEnv.OBS_DIM, N_ACTIONS).to(DEVICE)
    target_net = DuelingQNet(RetrieveEnv.OBS_DIM, N_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    replay = deque(maxlen=buffer_size)
    recent_rewards = deque(maxlen=200)
    step_count = 0

    best_success_rate = 0
    best_eval_reward = 0

    zero_obs = np.zeros(RetrieveEnv.OBS_DIM, dtype=np.float32)
    zero_mask = np.zeros(N_ACTIONS, dtype=bool)

    def _select_action(obs, mask, eps):
        if random.random() < eps:
            valid_actions = np.where(mask)[0]
            return int(random.choice(valid_actions))
        state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)
        with torch.no_grad():
            q_values = policy_net(state_t).squeeze(0)
            return masked_argmax(q_values, mask_t)

    def _optimize():
        if len(replay) < batch_size:
            return
        batch = random.sample(replay, batch_size)
        s = torch.from_numpy(np.stack([t.s for t in batch])).float().to(DEVICE)
        a = torch.tensor([t.a for t in batch], device=DEVICE).unsqueeze(1)
        r = torch.tensor([t.r for t in batch], device=DEVICE, dtype=torch.float32).unsqueeze(1)
        s2 = torch.from_numpy(np.stack([t.s2 for t in batch])).float().to(DEVICE)
        d = torch.tensor([t.done for t in batch], device=DEVICE, dtype=torch.float32).unsqueeze(1)
        mask2 = torch.from_numpy(np.stack([t.mask2 for t in batch])).to(DEVICE)

        q_sa = policy_net(s).gather(1, a)
        with torch.no_grad():
            next_q_policy = policy_net(s2)
            next_q_policy_masked = next_q_policy.masked_fill(~mask2, -1e9)
            next_actions = next_q_policy_masked.argmax(dim=1, keepdim=True)
            next_q_target = target_net(s2).gather(1, next_actions)
            y = r + gamma * (1 - d) * next_q_target

        loss = nn.functional.smooth_l1_loss(q_sa, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()

    start_time = time.perf_counter()
    for ep in range(episodes):
        obs_dict, mask_dict = env.reset()
        done = False
        ep_reward = 0.0
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)

        while not done:
            action_dict = {
                key: _select_action(obs, mask_dict[key], eps) for key, obs in obs_dict.items()
            }

            prev_obs_dict = obs_dict
            next_obs_dict, next_mask_dict, rewards, done, info = env.step(action_dict)

            for key, obs in prev_obs_dict.items():
                action = action_dict[key]
                reward = rewards.get(key, 0.0)
                ep_reward += reward

                if key in next_obs_dict:
                    next_obs = next_obs_dict[key]
                    next_mask = next_mask_dict[key]
                    step_done = done
                else:
                    # このtickでエピソード終了、またはこのエージェント自身が
                    # 死亡して脱落した(誰か1人がゴールすればdone=Trueで
                    # 全員揃うが、それ以外の死亡は単独で起こりうる)。
                    next_obs = zero_obs
                    next_mask = zero_mask
                    step_done = True

                replay.append(Transition(obs, action, reward, next_obs, step_done, next_mask))
                step_count += 1

                _optimize()

                if step_count % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            obs_dict, mask_dict = next_obs_dict, next_mask_dict

        recent_rewards.append(ep_reward)
        if ep % 200 == 0:
            eval_reward, success_rate, death_rate = evaluate_greedy(policy_net, episodes=100)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter();
            
            print(f"[EP {ep}/{EPISODE_COUNT}] eps={eps:.3f} "
                f"eval_reward={eval_reward:.3f} success={success_rate:.2%} death={death_rate:.2%} elapse={elapsed_time:.1f}")
            if success_rate > best_success_rate or (success_rate == best_success_rate and eval_reward > best_eval_reward):
                best_success_rate = success_rate
                best_eval_reward = eval_reward
                _save_checkpoint(policy_net, MODEL_SAVE_PATH, ep, success_rate)
                print(f"  -> best model saved (success_rate={success_rate:.2%})")


    _save_checkpoint(policy_net, MODEL_FINAL_PATH, episodes, best_success_rate)
    print("Training complete.")


def evaluate_greedy(policy_net, episodes=100):
    """探索なし(eps=0)でN episode実行し、成功率・死亡率・平均報酬(全エージェント合算)を計測する。"""
    env = RetrieveEnv()
    total_reward = 0.0
    reached = 0
    died = 0
    for _ in range(episodes):
        obs_dict, mask_dict = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            action_dict = {}
            for key, obs in obs_dict.items():
                state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                mask_t = torch.from_numpy(mask_dict[key]).to(DEVICE)
                with torch.no_grad():
                    q_values = policy_net(state_t).squeeze(0)
                    action_dict[key] = masked_argmax(q_values, mask_t)
            obs_dict, mask_dict, rewards, done, info = env.step(action_dict)
            ep_reward += sum(rewards.values())
        total_reward += ep_reward
        if info.get("result") == "reached":
            reached += 1
        elif info.get("result") == "died":
            died += 1
    n = episodes
    return total_reward / n, reached / n, died / n

if __name__ == "__main__":
    train()