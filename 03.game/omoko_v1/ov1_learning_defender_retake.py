"""omoko_v1/learning_defender_retake.py

固定チーム(いぐるん/夢の街/ひつじさん/Tortlilyan/えんぺん)専用の
Defender「retake phase」推論コントローラー(プラント後限定)。

omoko_v1/train_defender_retake.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py への依存はなく、必要なLOS計算・BFS距離マップ・行動マスク・
観測構築はこのファイル内に複製する。game_core からは定数のみを参照する
(ロジックは参照しない)。map_data系のimportも不要(gridはgame_state["grid"]
から都度取得するため、search phaseのような専用マップファイルへの依存がない)。

ステータス(コンボ「ふわんだりぃず」・タイガーパッシブ込みの確定値)は
character_stats.py 側の定義に基づき、run_game.py の既存エンジン
(combo_awakening.py / game_core.Character)が実戦時に自動適用するため、
本ファイルではステータスの再計算は行わない。char.accuracy等の実値を
そのまま利用する。

行動空間は train_defender_retake.py と完全に同一(N_ACTIONS=7):
    0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY, 5=DEFUSE, 6=ABILITY
観測ベクトルも同ファイルの build_observation() と要素・並び順を完全一致
させている(OBS_DIM=37)。ここがずれると学習済み重みと整合しなくなるため、
インデックスコメントを明示して対応関係を追跡できるようにしている。

アビリティ使用時のターゲットは、学習環境(RetakeEnv.apply_ability)が
プラント地点(planted_pos)を中心に効果を計算する設計だったことに合わせ、
常に planted_pos を狙点として実ゲーム側の execute_ai_ability() に渡す。

このチーム(5人)で1つのコントローラーインスタンスを共有する想定
(重み共有Dueling DQN)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    DEFUSE_REQUIRED_TICKS,
    SPIKE_DETONATION_TICKS,
)
from character_stats import CHARACTER_TABLE as STATS_TABLE
from ov1_roster import ROSTER_ORDER
from ov1_train_defender_retake import KNOWN_ENTRY_POINTS_LEFT, KNOWN_ENTRY_POINTS_RIGHT, ENTRY_CORRIDOR_RADIUS
from ov1_train_defender_retake import _facing_from_delta, _facing_towards
from ov1_ultimate_training import (
    ORB_COLLECT_REQUIRED_TICKS,
    orb_context,
    orb_priority,
    ultimate_context,
    ultimate_ready,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
MOVE_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

OBS_DIM = 64  # train_defender_retake.py の build_observation() と要素数を一致させること
              # (role_onehot+facing_onehot[8方向]で45、チーム共有目撃情報4次元+
              # 既知侵入経路3次元の追加で45→52、スモーク被覆フラグ追加で52→53)
N_ACTIONS = 13

SITE_ZONE_RADIUS = 6
ENTRY_READY_RADIUS = 3
DEFUSE_SAFETY_MARGIN_TICKS = 4
ENTRY_SAFETY_MARGIN_TICKS = DEFUSE_SAFETY_MARGIN_TICKS + ENTRY_READY_RADIUS

SIGHTING_STALENESS_CAP = 20  # train_defender_retake.pyのSIGHTING_STALENESS_CAPと同一値
SITE_BOUNDARY_COL_UNSET = None  # game_stateからgridを都度受け取るため列境界は動的に計算する


ACTION_DEFUSE = 5
ACTION_ABILITY = 6
TURN_DIRS = ["N", "S", "E", "W"]
ACTION_TURN_BASE = 7  # 7,8,9,10 = 向き変更のみ(移動なし)。battle_logic.py新契約に合わせ
                       # next_pos=現在地 + {"facing": dir} をMOVEとして返す。
# 観測用。被弾直後の強制向き(battle_logic.py._facing_towards)は斜め8方向を
# 返しうるため、TURN行動(4方向)とは別に観測エンコードは8方向で持つ。
# game_core.FACING_VECTORSの定義順(N,NE,E,SE,S,SW,W,NW)と一致させること。
ALL_FACINGS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_ULTIMATE = 11
ACTION_ORB = 12

ROLE_INDEX = {"フラッシュ": 0, "スモーカー": 1, "シーカー": 2, "タイガー": 3}

DEFAULT_MODEL_PATH = (
    "omoko_v1/data/defender_retake_data/"
    "dqn_defender_retake_best_by_eval.pt"
)


# ---------------------------------------------------------------------------
# Dueling DQN (omoko_v1/train_defender_retake.py と同一構造)
# ---------------------------------------------------------------------------
class DefenderRetakeDuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        # 💡修正: ov1_common_rl.DuelingQNet(学習側)の層名は advantage_head。
        # 名前が異なるだけで構造は同一だったため state_dict のキーが
        # 一致せずロードエラーになっていた。学習側に合わせて改名する。
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions)
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        adv = self.advantage_head(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# 補助関数(LOS / BFS)。abilities_los.py / train_defender_retake.py の複製実装。
#
# 注意: game_state には smokes(煙リスト)が含まれないため、この推論用LOSは
# 壁のみを考慮する(スモークによる遮蔽は考慮しない)。同様に
# ally_ability_active の判定もエネミー側デバフ状態のみで代用し、
# 味方スモーク展開中フラグは常に0として扱う(learning_defender_search.py
# と同じ簡略化方針)。
# ---------------------------------------------------------------------------
def _line_cells(p1, p2):
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


def _has_los(grid, p1, p2):
    for r, c in _line_cells(p1, p2):
        if grid[r, c] == 1:
            return False
    return True


def _smoke_visible_enemy(char, chars, grid, smoke_cells):
    """Return the nearest enemy visible through smoke to a smoke-vision user."""
    if not getattr(char, "sees_through_smoke", False) or not smoke_cells:
        return None
    pos = (int(char.pos[0]), int(char.pos[1]))
    candidates = []
    for enemy in chars:
        if not getattr(enemy, "is_alive", True) or enemy.team == char.team:
            continue
        line = _line_cells(pos, tuple(enemy.pos))
        if len(line) <= 2 or not any(cell in smoke_cells for cell in line):
            continue
        if all(grid[row, col] != 1 for row, col in line):
            candidates.append(enemy)
    return min(candidates, key=lambda e: max(abs(e.pos[0] - pos[0]), abs(e.pos[1] - pos[1]))) if candidates else None


def _bfs_distance_map(grid, goal):
    """goalから各床マスへの最短距離マップ(壁越え不可)。到達不能マスは-1。
    train_defender_retake.py の bfs_distance_map と同一ロジック。"""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _good_directions(dist_map, grid, r, c):
    """train_defender_retake.py の good_directions と同一ロジック。
    BFS距離マップ上で実際に距離を縮められる方向(up/down/left/right)を
    1.0、それ以外を0.0とする4次元フラグ。"""
    good = [0.0, 0.0, 0.0, 0.0]
    height, width = grid.shape
    raw = dist_map[r, c]
    if raw < 0:
        return good
    for i, (dr, dc) in enumerate(CARDINAL_MOVES):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
            nd = dist_map[nr, nc]
            if nd != -1 and nd < raw:
                good[i] = 1.0
    return good


def _bfs_best_direction(dist_map, r0, c0):
    if dist_map is None:
        return 0, 0
    current = dist_map[r0, c0]
    if current < 0:
        return 0, 0
    best = (0, 0)
    best_distance = current
    for dr, dc in CARDINAL_MOVES:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < dist_map.shape[0] and 0 <= nc < dist_map.shape[1]:
            if 0 <= dist_map[nr, nc] < best_distance:
                best = (dr, dc)
                best_distance = dist_map[nr, nc]
    return best


def _planned_route_facing(dist_map, pos):
    r0, c0 = int(pos[0]), int(pos[1])
    first = _bfs_best_direction(dist_map, r0, c0)
    if first == (0, 0):
        return None
    nr, nc = r0 + first[0], c0 + first[1]
    second = _bfs_best_direction(dist_map, nr, nc)
    desired = second if second != (0, 0) and second != first else first
    return _facing_from_delta(desired[0], desired[1], None)


def _forced_combat_facing(char, visible_enemies, enemies):
    if visible_enemies:
        target = min(
            visible_enemies,
            key=lambda enemy: max(
                abs(enemy.pos[0] - char.pos[0]),
                abs(enemy.pos[1] - char.pos[1]),
            ),
        )
        facing = _facing_towards(tuple(char.pos), tuple(target.pos))
        if facing is not None:
            char._combat_facing = facing
            char._combat_facing_staleness = SIGHTING_STALENESS_CAP
        return getattr(char, "_combat_facing", None)
    if any(e.is_alive for e in enemies):
        remaining = getattr(char, "_combat_facing_staleness", 0)
        if remaining > 0:
            char._combat_facing_staleness = remaining - 1
            return getattr(char, "_combat_facing", None)
    return None


def _ability_charge(char):
    """ロールに対応する残チャージ数を取得する。HUNT(アビリティ無し)は常に0。"""
    return {
        "FLASH": getattr(char, "flash_charges", 0),
        "SMOKE": getattr(char, "smoke_charges", 0),
        "RECON": getattr(char, "recon_charges", 0),
    }.get(char.ability_name, 0)


# ---------------------------------------------------------------------------
# チーム共有目撃情報(5人のDefender全体。train_defender_retake.pyの
# _TeamSightingMemoryと同一方針)。誰か一人でも視認していれば共有され、
# 視認が途切れてもSIGHTING_STALENESS_CAP tickの間は保持される。
# ---------------------------------------------------------------------------
class _TeamSightingMemory:
    def __init__(self):
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.last_seen_enemy = None

    def update(self, grid, my_team, chars):
        allies = [c for c in chars if getattr(c, "team", None) == my_team and getattr(c, "is_alive", True)]
        enemies = [c for c in chars if getattr(c, "team", None) != my_team and getattr(c, "is_alive", True)]

        visible = []
        for a in allies:
            for e in enemies:
                if _has_los(grid, tuple(a.pos), tuple(e.pos)) and e not in visible:
                    visible.append(e)

        if visible:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next((e for e in visible if e.name == tracked_name), None)
            if tracked is None:
                tracked = min(
                    visible,
                    key=lambda e: min(
                        max(abs(e.pos[0] - a.pos[0]), abs(e.pos[1] - a.pos[1])) for a in allies
                    ) if allies else 0,
                )
            self.last_seen_enemy = {"pos": tuple(map(int, tracked.pos)), "name": tracked.name, "tick_ago": 0}
        elif self.last_seen_enemy is not None:
            if not any(e.name == self.last_seen_enemy.get("name") for e in enemies):
                self.last_seen_enemy = None
            else:
                self.last_seen_enemy["tick_ago"] += 1
                if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                    self.last_seen_enemy = None


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class Ov1LearningDefenderRetakeController:
    """omoko_v1固定チーム専用、プラント後フェーズ(リテイク)のDefender AI。

    Defenderチーム全員(5人)がこの1インスタンスを共有して呼び出される。

    ステータス(コンボ・タイガーパッシブ込みの確定値)は run_game.py の
    既存エンジンが character_stats.py を経由して自動適用済みの
    char オブジェクトをそのまま利用する(本ファイル側では再計算しない)。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self.model = DefenderRetakeDuelingDQN().to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            if verbose:
                print(f"[Ov1LearningDefenderRetakeController] loaded: {model_path}")
        except Exception as exc:
            print(f"[LOAD ERROR] defender retake(omoko) model '{model_path}' の読込に失敗: {exc}")
        self.model.eval()

        # プラント地点(planted_pos)は1ラウンド中不変のため、
        # BFS距離マップ・site_zoneはラウンド開始後の初回呼び出し時に
        # 1度だけ計算してキャッシュする。
        self._dist_map = None
        self._site_zone = None
        self._dist_map_source = None  # 再計算要否判定用: 直前のplanted_pos

        self.team_sighting = _TeamSightingMemory()
        self._processed_this_tick = set()
        self._active_entry_points = []  # このラウンドのサイト(左/右)に対応する既知侵入経路

        self._debug_log_path = "defender_retake_debug.log"

    # -- ラウンド開始時のリセット -----------------------------------------
    def reset_round(self):
        self._dist_map = None
        self._site_zone = None
        self._dist_map_source = None

        self.team_sighting.reset()
        self._processed_this_tick.clear()
        self._active_entry_points = []

    def _ensure_dist_map(self, grid, planted_pos):
        planted_pos = (int(planted_pos[0]), int(planted_pos[1]))
        if planted_pos == self._dist_map_source and self._dist_map is not None:
            return
        self._dist_map = _bfs_distance_map(grid, planted_pos)
        height, width = grid.shape
        self._site_zone = {
            (r, c)
            for r in range(height)
            for c in range(width)
            if 0 <= self._dist_map[r, c] <= SITE_ZONE_RADIUS
        }
        self._dist_map_source = planted_pos

        # ov1_train_defender_retake.pyのSITE_BOUNDARY_COL(=WIDTH//2)判定と
        # 同一ロジックでサイトを決め、既知侵入経路を確定する。
        site_boundary_col = width // 2
        self._active_entry_points = (
            KNOWN_ENTRY_POINTS_LEFT if planted_pos[1] < site_boundary_col else KNOWN_ENTRY_POINTS_RIGHT
        )

    def _maybe_advance_tick(self, char, grid, chars):
        """同じキャラクターが再び呼ばれたら新しいtickに入ったとみなし、
        チーム共有目撃情報を1回だけ更新する
        (ov1_learning_attacker_guard.pyの_maybe_advance_tickと同一方針)。"""
        if char.name in self._processed_this_tick:
            self._processed_this_tick.clear()
            self.team_sighting.update(grid, char.team, chars)
        self._processed_this_tick.add(char.name)

    def _nearest_visible_entry_point(self, grid, pos):
        """既知侵入経路(_active_entry_points)のうち、posから視認できるものだけを
        候補にし、最も近い1点を返す(視認できるものが無ければNone)。"""
        visible = [p for p in self._active_entry_points if _has_los(grid, pos, p)]
        if not visible:
            return None
        return min(visible, key=lambda p: max(abs(p[0] - pos[0]), abs(p[1] - pos[1])))

    # -- 観測構築 ----------------------------------------------------------
    # train_defender_retake.py の build_observation() と要素・並び順を
    # 完全一致させること。インデックスはコメントで明示する。
    def _build_observation(self, char, game_state, chars, enemies, visible_enemies, detonate_timer):
        grid = game_state["grid"]
        height, width = grid.shape
        planted_pos = game_state["planted_pos"]
        pr, pc = int(planted_pos[0]), int(planted_pos[1])
        r, c = int(char.pos[0]), int(char.pos[1])

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = r / height                                   # [0] 自己座標r
        obs[1] = c / width                                     # [1] 自己座標c
        obs[2] = char.hp / char.max_hp if char.max_hp else 0.0  # [2] 自己HP割合

        good_dir = _good_directions(self._dist_map, grid, r, c)
        obs[3], obs[4], obs[5], obs[6] = good_dir               # [3-6] BFS推奨方向(up,down,left,right)

        raw_dist = self._dist_map[r, c]
        dist_to_plant = min(1.0, raw_dist / (height + width)) if raw_dist >= 0 else 1.0
        obs[7] = dist_to_plant                                  # [7] BFSプラント距離

        obs[8] = 1.0 if (r, c) in self._site_zone else 0.0       # [8] サイトゾーン内フラグ
        obs[9] = 1.0 if max(abs(pr - r), abs(pc - c)) <= 1 else 0.0  # [9] プラント隣接フラグ

        obs[10] = 1.0 if _ability_charge(char) > 0 else 0.0      # [10] 自己アビリティ残チャージ
        obs[11] = char.blind_remaining / BLIND_DURATION_TICKS if BLIND_DURATION_TICKS else 0.0  # [11]
        obs[12] = char.defuse_timer / DEFUSE_REQUIRED_TICKS      # [12] 解除進捗
        obs[13] = detonate_timer / SPIKE_DETONATION_TICKS        # [13] 起爆タイマー割合

        allies = [a for a in chars if a.team == char.team and a.is_alive]
        obs[14] = len(allies) / 5.0                               # [14] 生存味方数
        obs[15] = sum(1 for a in allies if tuple(a.pos) in self._site_zone) / 5.0  # [15] サイトゾーン内味方数
        allies_near_entry = sum(
            1 for a in allies if max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
        )
        obs[16] = allies_near_entry / 5.0                         # [16] エントリー圏内味方数

        others = [a for a in allies if a is not char]
        if others:
            nearest_ally_dist = min(
                max(abs(a.pos[0] - r), abs(a.pos[1] - c)) for a in others
            ) / max(height, width)
        else:
            nearest_ally_dist = 1.0
        obs[17] = nearest_ally_dist                               # [17] 最近接味方距離

        # 味方アビリティ発動中フラグ。smoke情報がgame_stateに無いため、
        # 敵側のデバフ状態(blind/reveal)のみで近似する。
        obs[18] = 1.0 if any(
            e.is_alive and (
                getattr(e, "blind_remaining", 0) > 0 or getattr(e, "reveal_remaining", 0) > 0
            )
            for e in enemies
        ) else 0.0                                                 # [18] 味方アビリティ発動中(近似)

        # 💡追加: 自分が「現在」スモークに覆われているかの直接フラグ。
        # train_defender_retake.py側と同じ理由(古い目撃情報が残っていても、
        # 実際にはスモークで安全なら解除して良いと学習させるため)。
        # game_stateのsmoke_cellsは全スモークの合算セル集合(所有チーム区別なし)
        # のため、team_sighting同様の近似として扱う。
        smoke_cells = game_state.get("smoke_cells") or set()
        obs[19] = 1.0 if (r, c) in smoke_cells else 0.0             # [19] 自己スモーク被覆フラグ

        obs[20] = len(visible_enemies) / 5.0                       # [20] 視認中敵数
        obs[21] = len([e for e in enemies if e.is_alive]) / 5.0    # [21] 生存敵総数

        # [22-33] 視認中の近い敵、最大2体分(6次元 x 2)
        sorted_enemies = sorted(
            visible_enemies,
            key=lambda e: max(abs(e.pos[0] - r), abs(e.pos[1] - c)),
        )
        idx = 22
        for e in sorted_enemies[:2]:
            edx = (e.pos[1] - c) / width
            edy = (e.pos[0] - r) / height
            edist = max(abs(e.pos[0] - r), abs(e.pos[1] - c)) / max(height, width)
            ehp = e.hp / e.max_hp if e.max_hp else 0.0
            eblind = 1.0 if getattr(e, "blind_remaining", 0) > 0 else 0.0
            erevealed = 1.0 if (
                getattr(e, "reveal_remaining", 0) > 0 or getattr(e, "los_revealed", False)
            ) else 0.0
            obs[idx:idx + 6] = [edx, edy, edist, ehp, eblind, erevealed]
            idx += 6
        # 視認中敵が2体未満の残り枠は0.0のまま(np.zerosで初期化済み)

        # [34-37] ロールone-hot(フラッシュ/スモーカー/シーカー/タイガー)
        role_idx = 34 + ROLE_INDEX.get(char.role, 0)
        obs[role_idx] = 1.0

        # [38-45] 自身の向き(8方向)one-hot。train_defender_retake.py の
        # facing_onehot(ALL_FACINGSと同じ並び順)と完全一致させること。
        # 被弾直後はforced_facing_next_tickにより斜め向きになり得るため8方向で持つ。
        if char.facing in ALL_FACINGS:
            obs[38 + ALL_FACINGS.index(char.facing)] = 1.0

        # [46-49] チーム共有の目撃情報(自分が直接視認していなくても、他の味方が
        # 見ていれば共有される)。train_defender_retake.pyのbuild_observationと
        # 同一の埋め方。
        last_seen = self.team_sighting.last_seen_enemy
        if last_seen is not None:
            tr, tc = last_seen["pos"]
            obs[46] = 1.0
            obs[47] = max(-1.0, min(1.0, (tr - r) / max(height, width)))
            obs[48] = max(-1.0, min(1.0, (tc - c) / max(height, width)))
            obs[49] = min(last_seen["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP

        # [50-52] 既知侵入経路のうち視認できる最寄り点への方向。
        nearest_entry = self._nearest_visible_entry_point(grid, (r, c))
        if nearest_entry is not None:
            obs[50] = 1.0
            obs[51] = (nearest_entry[0] - r) / height
            obs[52] = (nearest_entry[1] - c) / width

        available_orbs = {
            tuple(map(int, cell)) for cell in game_state.get("available_orbs", ())
        }
        obs[53:57] = ultimate_context(
            char,
            self_sighting=bool(visible_enemies),
            team_sighting=last_seen is not None,
            tactical=bool(visible_enemies) or last_seen is not None,
        )
        obs[57:64] = orb_context(char, available_orbs, grid, allies)

        return obs

    # -- 行動マスク ---------------------------------------------------------
    # train_defender_retake.py の action_mask() と同一ロジック
    # (危険な場面の行動選択は学習済み方策に任せ、学習側と一致させる)。
    def _action_mask(
        self, char, grid, chars, available_orbs=(),
        under_direct_threat=False, detonate_timer=0.0,
    ):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = {
            tuple(o.pos) for o in chars if o is not char and getattr(o, "is_alive", True)
        }

        for a in range(4):
            dr, dc = MOVE_DELTAS[a]
            nr, nc = r + dr, c + dc
            walkable = (
                0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                and grid[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            mask[a] = walkable
        mask[4] = True  # stay は常に許可

        pr = self._dist_map_source[0] if self._dist_map_source else r
        pc = self._dist_map_source[1] if self._dist_map_source else c
        dist_to_plant = max(abs(pr - r), abs(pc - c))
        mask[ACTION_DEFUSE] = dist_to_plant <= 1

        # Keep the retake moving whenever an unoccupied step shortens the
        # walkable path to the spike. Ability and ultimate actions stay enabled.
        raw_dist = self._dist_map[r, c] if self._dist_map is not None else -1
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
                next_dist = self._dist_map[r + dr, c + dc]
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

        mask[ACTION_ABILITY] = _ability_charge(char) > 0

        # train側と同じく、facingは自動決定する。
        mask[ACTION_TURN_BASE:ACTION_ULTIMATE] = False
        allies = [
            other for other in chars
            if getattr(other, "is_alive", True) and other.team == char.team
        ]
        mask[ACTION_ULTIMATE] = ultimate_ready(char)
        mask[ACTION_ORB] = (
            (r, c) in available_orbs and orb_priority(char, allies)
        )
        if (
            dist_to_plant <= 1
            and detonate_timer <= (
                DEFUSE_REQUIRED_TICKS + ORB_COLLECT_REQUIRED_TICKS
                + DEFUSE_SAFETY_MARGIN_TICKS
            )
        ):
            # Orb collection takes five stationary ticks; preserve that time
            # for an available defuse.
            mask[ACTION_ORB] = False

        return mask

    # -- メイン ----------------------------------------------------------
    def decide_move(self, char, game_state):
        if not char.is_alive:
            return list(char.pos)

        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")

        # このコントローラーはプラント後フェーズ(リテイク)専用。
        # 万一プラント前に呼ばれた場合は安全側としてその場に留まる
        # (上位のフェーズ切替側で search フェーズのコントローラーへ
        # 委譲する想定)。
        if not is_planted or planted_pos is None:
            return list(char.pos)

        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        detonate_timer = float(game_state.get("detonate_timer", 0.0))

        smoke_enemy = _smoke_visible_enemy(
            char, chars, grid, game_state.get("smoke_cells") or set()
        )
        if smoke_enemy is not None:
            return list(char.pos), {"facing": _facing_towards(char.pos, smoke_enemy.pos)}

        self._ensure_dist_map(grid, planted_pos)
        self._maybe_advance_tick(char, grid, chars)

        enemies = [e for e in chars if e.team != char.team]
        visible_enemies = [
            e for e in enemies if e.is_alive and _has_los(grid, tuple(char.pos), tuple(e.pos))
        ]
        smoke_cells = game_state.get("smoke_cells") or set()
        under_direct_threat = any(
            getattr(char, "sees_through_smoke", False)
            or not any(
                cell in smoke_cells
                for cell in _line_cells(tuple(char.pos), tuple(enemy.pos))
            )
            for enemy in visible_enemies
        )

        obs = self._build_observation(char, game_state, chars, enemies, visible_enemies, detonate_timer)
        available_orbs = {
            tuple(map(int, cell)) for cell in game_state.get("available_orbs", ())
        }
        mask = self._action_mask(
            char, grid, chars, available_orbs,
            under_direct_threat=under_direct_threat,
            detonate_timer=detonate_timer,
        )

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(obs_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action_idx = int(torch.argmax(q_values).item())

        forced_facing = _forced_combat_facing(char, visible_enemies, enemies)
        if forced_facing is None and action_idx <= 3:
            dr, dc = MOVE_DELTAS[action_idx]
            first_step = _bfs_best_direction(
                self._dist_map, int(char.pos[0]), int(char.pos[1])
            )
            if (dr, dc) == first_step:
                forced_facing = _planned_route_facing(self._dist_map, char.pos)
            if forced_facing is None:
                forced_facing = _facing_from_delta(
                    dr, dc, getattr(char, "facing", "S")
                )

        if self.verbose:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{char.name},{tuple(char.pos)},planted={tuple(planted_pos)},"
                    f"detonate={detonate_timer:.1f},action={action_idx},"
                    f"Qvals={np.round(q_values.cpu().numpy(), 4).tolist()}\n"
                )

        if action_idx <= 4:
            dr, dc = MOVE_DELTAS[action_idx]
            if forced_facing is not None:
                return [char.pos[0] + dr, char.pos[1] + dc], {"facing": forced_facing}
            return [char.pos[0] + dr, char.pos[1] + dc]

        if action_idx == ACTION_DEFUSE:
            if forced_facing is not None:
                char.facing = forced_facing
            return list(char.pos), "DEFUSE"

        if action_idx == ACTION_ORB:
            return list(char.pos), "COLLECT_ORB"

        if action_idx == ACTION_ULTIMATE:
            facing = forced_facing or getattr(char, "facing", "S")
            if not getattr(char, "facing_forced_this_tick", False):
                char.facing = facing
            payload = {
                "ultimate": str(getattr(char, "ultimate_name", "")).upper(),
                "facing": facing,
            }
            if payload["ultimate"] == "ESCAPE":
                payload["target"] = tuple(map(int, planted_pos))
            return list(char.pos), payload

        if ACTION_TURN_BASE <= action_idx < ACTION_ULTIMATE:
            # battle_logic.py新契約: Defenderは{"facing": ...}を返すとaction_type="MOVE"
            # として扱われ、next_pos=現在地のためその場で向きだけ変わる。
            turn_dir = forced_facing or TURN_DIRS[action_idx - ACTION_TURN_BASE]
            return list(char.pos), {"facing": turn_dir}

        # ACTION_ABILITY: 学習環境(RetakeEnv.apply_ability)がプラント地点
        # 中心に効果を計算する設計だったため、狙点は常にplanted_posとする。
        target_pos = (int(planted_pos[0]), int(planted_pos[1]))
        if forced_facing is not None:
            char.facing = forced_facing
        return list(char.pos), {"ability": char.ability_name, "target": target_pos}
