"""touyama_v1/learning_defender_search_touyama.py

固定チーム(イグルン/夢の街/ろびぃな/Tortlilyan/えんぺん)専用の
Defender「search phase」推論コントローラー(プラント前限定)。

touyama_v1/train_defender_search.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py への依存はなく、必要なLOS計算・行動マスク・観測構築は
このファイル内に複製する。game_core / map_data_search からは定数・
マップ文字列のみを参照する(ロジックは参照しない)。

ステータス(コンボ「ふわんだりぃず」・タイガーパッシブ込みの確定値)は
character_stats_touyama.py 側の定義に基づき、run_game.py の既存エンジン
(combo_awakening.py / game_core.Character)が実戦時に自動適用するため、
本ファイルではステータスの再計算は行わない。char.accuracy等の実値を
そのまま利用する。

touyama_v1/train_defender_search.py と同じ優先順位ツリーを踏襲する:
    1. スパイク確定情報があればそちらへ最優先で寄る
    2. 敵目撃情報があればそちらへ寄る(retake準備)
    3. どちらも無ければ、ラウンド開始時に自チーム内で貪欲割り当てた
       map_data_search.py の7地点(有利ポジション)へ、BFS距離マップに
       基づいて向かい、到着後は静止する

OBS_DIM=36: train_defender_search.py と完全に一致させること。

このチーム(5人)で1つのコントローラーインスタンスを共有する想定
(重み共有Dueling DQN)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
)
from map_data import NEW_MAZE_STR
from map_data_search_touyama import SEARCH_MAZE_STR

from character_stats_touyama import (
    CHARACTER_TABLE as TOUYAMA_STATS_TABLE,
    TOUYAMA_ROSTER_ORDER,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 36  # 31(従来) + 4(BFS距離 + 推奨方向dr,dc + 到着フラグ) + 1(spike_watchフラグ)
ACTION_DIM = 10  # move_idx(0-4) * 2 + use_ability_flag(0/1)

SIGHTING_STALENESS_CAP = 30
ABILITY_RANGE = 8
REACH_RADIUS = 1  # 担当ポジションへ「到着した」とみなすBFS距離

DEFAULT_MODEL_PATH = (
    "touyama_v1/data/defender_search_touyama_data/"
    "dqn_defender_search_touyama_best_by_eval.pt"
)


# ---------------------------------------------------------------------------
# Dueling DQN (touyama_v1/train_defender_search.py と同一構造)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 補助関数(LOS)。abilities_los.py とは独立した複製実装。
#
# 注意: game_state には smokes(煙リスト)が含まれないため、この推論用LOSは
# 壁のみを考慮する(スモークによる遮蔽は考慮しない)。
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


def _bfs_distance_map(grid, goal):
    """指定ゴールから各セルへの最短距離マップ(壁越え不可)。
    train_defender_search.py の bfs_distance_map と同一ロジック。"""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist

def _bfs_best_direction(dist_map, grid, r0, c0):
    """dist_map上で、(r0,c0)から見て最も距離が縮む隣接方向(dr,dc)を返す。
    到達不能・移動不要なら(0,0)を返す。train_defender_search.py の
    bfs_best_direction と同一ロジック。"""
    if dist_map is None:
        return 0, 0
    height, width = grid.shape
    cur = dist_map[r0, c0]
    if cur < 0:
        return 0, 0
    best_dr, best_dc, best_d = 0, 0, cur
    for dr, dc in CARDINAL:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < height and 0 <= nc < width and dist_map[nr, nc] >= 0:
            if dist_map[nr, nc] < best_d:
                best_d = dist_map[nr, nc]
                best_dr, best_dc = dr, dc
    return best_dr, best_dc

def _parse_search_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


def _parse_maze_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


def _find_marker_position(grid, value):
    """train_defender_search.py の _find_marker_position と同一ロジック。
    値が1マスのみである前提(そうでなければ学習時と食い違うため例外にする)。"""
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


def _compute_touyama_fixed_defense_assignment():
    """train_defender_search.py と完全に同一のロジックで固定割当を再現する。

    マップ上の値5,6,7,8,9を TOUYAMA_ROSTER_ORDER のインデックス順に
    そのまま1:1対応させる(5=roster[0], 6=roster[1], ...)。総当たり最適化は
    行わない(学習側で廃止されたのに合わせる)。
    学習時にモデルが学習した「担当地点への距離・方向」はこの固定割当を
    前提とした観測であるため、推論側でも必ず同じ割当を使う必要がある。
    (自己完結ルールにより、学習ファイルをimportせずロジックを複製する)
    """
    search_grid = _parse_search_grid(SEARCH_MAZE_STR)
    return {
        name: _find_marker_position(search_grid, 5 + i)
        for i, name in enumerate(TOUYAMA_ROSTER_ORDER)
    }


_TOUYAMA_FIXED_DEFENSE_ASSIGNMENT = _compute_touyama_fixed_defense_assignment()
_DEFENSE_POSITIONS_CACHE = list(_TOUYAMA_FIXED_DEFENSE_ASSIGNMENT.values())


def _extract_site_positions(grid, max_sites=2):
    """grid内の値2(プラント可能床)セルを、単純な距離クラスタリングで
    サイトごとの代表座標にまとめる。train_defender_search.py と同一ロジック。"""
    cells = list(zip(*np.where(grid == 2)))
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


def _ability_charge(char):
    """ロールに対応する残チャージ数を取得する。HUNT(アビリティ無し)は常に0。"""
    return {
        "FLASH": getattr(char, "flash_charges", 0),
        "SMOKE": getattr(char, "smoke_charges", 0),
        "RECON": getattr(char, "recon_charges", 0),
    }.get(char.ability_name, 0)


# ---------------------------------------------------------------------------
# チーム共有メモリ(スパイク確定情報 / 敵目撃情報)
# ---------------------------------------------------------------------------
class _TeamMemory:
    def __init__(self):
        self.spike_pos = None
        self.spike_held = False  # True: 保持者が視認中(緊急) / False: 地面に落下(待ち伏せ可)
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.spike_pos = None
        self.spike_held = False
        self.last_seen_enemy = None

    def update(self, grid, my_team, chars, spike_ground_pos=None):
        defenders = [c for c in chars if c.team == my_team and c.is_alive]
        enemies = [c for c in chars if c.team != my_team]

        visible_enemies = []
        for d in defenders:
            for e in enemies:
                if not e.is_alive:
                    continue
                if _has_los(grid, tuple(d.pos), tuple(e.pos)):
                    visible_enemies.append(e)

        spike_holder = next(
            (e for e in visible_enemies if getattr(e, "has_spike", False)), None
        )
        if spike_holder is not None:
            self.spike_pos = tuple(spike_holder.pos)
            self.spike_held = True
        elif spike_ground_pos is not None and any(
            _has_los(grid, tuple(d.pos), tuple(spike_ground_pos)) for d in defenders
        ):
            self.spike_pos = tuple(spike_ground_pos)
            self.spike_held = False

        if visible_enemies:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next((e for e in visible_enemies if e.name == tracked_name), None)
            if tracked is None:
                tracked = min(
                    visible_enemies,
                    key=lambda e: min(
                        max(abs(e.pos[0] - d.pos[0]), abs(e.pos[1] - d.pos[1]))
                        for d in defenders
                    ) if defenders else 0,
                )
            self.last_seen_enemy = {
                "pos": tuple(tracked.pos), "name": tracked.name, "tick_ago": 0
            }
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningDefenderSearchTouyamaController:
    """touyama_v1固定チーム専用、プラント前フェーズのDefender AI。

    Defenderチーム全員(5人)がこの1インスタンスを共有して呼び出される
    (learning_defender.LearningDefenderAllAIController と同じ運用形態)。

    ステータス(コンボ・タイガーパッシブ込みの確定値)は run_game.py の
    既存エンジンが character_stats_touyama.py を経由して自動適用済みの
    char オブジェクトをそのまま利用する(本ファイル側では再計算しない)。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self.model = DefenderSearchDuelingDQN().to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            if verbose:
                print(f"[LearningDefenderSearchTouyamaController] loaded: {model_path}")
        except Exception as exc:
            print(f"[LOAD ERROR] defender search(touyama) model '{model_path}' の読込に失敗: {exc}")
        self.model.eval()

        self.team_memory = _TeamMemory()
        self._site_positions_cache = None
        self._processed_this_tick = set()

        # 有利ポジション(7)の割り当て。ラウンド開始時に1度だけ計算する。
        self._defense_positions = list(_DEFENSE_POSITIONS_CACHE)
        self._assigned_positions = {}       # char_name -> (r, c)
        self._assigned_dist_maps = {}       # char_name -> np.ndarray(BFS距離マップ)
        self._prev_defense_bfs_dist = {}    # char_name -> float(デバッグ用に保持)
        self._debug_log_path = "defender_search_touyama_debug.log"

        self._assignment_done = False
        self._grid_cache = None
        self.spike_dist_map = None
        self.sighting_dist_map = None
        self._spike_dist_map_source = None       # 再計算要否判定用: 直前のspike_pos
        self._sighting_dist_map_source = None    # 再計算要否判定用: 直前のlast_seen_enemy["pos"]

    # -- ラウンド開始時のリセット -----------------------------------------
    def reset_round(self):
        self.team_memory.reset()
        self._processed_this_tick.clear()
        self._assigned_positions.clear()
        self._assigned_dist_maps.clear()
        self._prev_defense_bfs_dist.clear()
        self._assignment_done = False
        self.spike_dist_map = None
        self.sighting_dist_map = None
        self._spike_dist_map_source = None
        self._sighting_dist_map_source = None

    def _maybe_advance_tick(self, char, grid, chars, spike_ground_pos):
        """同じキャラクターが再び呼ばれたら新しいtickに入ったとみなし、
        チーム共有メモリを1回だけ更新する。"""
        if char.name in self._processed_this_tick:
            self._processed_this_tick.clear()
            self.team_memory.update(grid, char.team, chars, spike_ground_pos)
        self._processed_this_tick.add(char.name)

    def _update_priority_dist_maps(self, grid):
        """team_memoryのspike_pos/last_seen_enemyが変化した時だけBFSを
        再計算する(全defenderで共有するため、キャラクターごとには呼ばない)。"""
        spike_pos = self.team_memory.spike_pos
        if spike_pos != self._spike_dist_map_source:
            self.spike_dist_map = _bfs_distance_map(grid, spike_pos) if spike_pos is not None else None
            self._spike_dist_map_source = spike_pos

        sighting_pos = (
            self.team_memory.last_seen_enemy["pos"]
            if self.team_memory.last_seen_enemy is not None else None
        )
        if sighting_pos != self._sighting_dist_map_source:
            self.sighting_dist_map = _bfs_distance_map(grid, sighting_pos) if sighting_pos is not None else None
            self._sighting_dist_map_source = sighting_pos

    def _ensure_defense_assignment(self, char, grid, chars):
        """touyama_v1固定チームは、学習時に一意に計算した固定割当
        (_TOUYAMA_FIXED_DEFENSE_ASSIGNMENT)をそのまま使う。スポーン位置・
        担当地点候補が毎ラウンド同一である以上、動的な貪欲割当は不要かつ
        学習時の割当とずれる原因になるため行わない。
        フォールバック: 固定割当が計算できていない場合のみ、旧来の
        名前順貪欲割当に切り替える。"""
        if self._assignment_done:
            return

        teammates = [c for c in chars if c.team == char.team and c.is_alive]

        if _TOUYAMA_FIXED_DEFENSE_ASSIGNMENT:
            for teammate in teammates:
                pos = _TOUYAMA_FIXED_DEFENSE_ASSIGNMENT.get(teammate.name)
                if pos is None:
                    continue
                self._assigned_positions[teammate.name] = pos
                dist_map = _bfs_distance_map(grid, pos)
                self._assigned_dist_maps[teammate.name] = dist_map
                self._prev_defense_bfs_dist[teammate.name] = float(
                    dist_map[int(teammate.pos[0]), int(teammate.pos[1])]
                )
            self._assignment_done = True
            return

        # --- フォールバック: 固定割当が使えない場合のみ旧ロジック ---
        if not self._defense_positions:
            teammates_now = teammates
            fallback = [tuple(c.pos) for c in teammates_now] or [
                (grid.shape[0] // 2, grid.shape[1] // 2)
            ]
            self._defense_positions = fallback

        remaining = list(self._defense_positions)
        for teammate in sorted(teammates, key=lambda c: c.name):
            if remaining:
                pos = min(
                    remaining,
                    key=lambda p: max(abs(p[0] - teammate.pos[0]), abs(p[1] - teammate.pos[1])),
                )
                remaining.remove(pos)
            else:
                pos = self._defense_positions[
                    hash(teammate.name) % len(self._defense_positions)
                ]
            self._assigned_positions[teammate.name] = pos
            dist_map = _bfs_distance_map(grid, pos)
            self._assigned_dist_maps[teammate.name] = dist_map
            self._prev_defense_bfs_dist[teammate.name] = float(
                dist_map[int(teammate.pos[0]), int(teammate.pos[1])]
            )

        self._assignment_done = True

    # -- 観測構築 ----------------------------------------------------------
    def _build_observation(self, char, game_state, site_positions, unit_has_spike_los):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        height, width = grid.shape

        teammates = [c for c in chars if c.team == char.team and c is not char and c.is_alive]
        enemies = [c for c in chars if c.team != char.team]

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = char.pos[0] / height
        obs[1] = char.pos[1] / width
        obs[2] = char.hp / char.max_hp if char.max_hp else 0.0
        obs[3] = 1.0 if getattr(char, "moved_this_tick", False) else 0.0

        ability_index = {"SMOKE": 4, "FLASH": 5, "RECON": 6, "HUNT": 7}.get(char.ability_name, 7)
        obs[ability_index] = 1.0
        obs[8] = 1.0 if _ability_charge(char) > 0 else 0.0

        visible_enemies = [
            e for e in enemies if e.is_alive and _has_los(grid, char.pos, e.pos)
        ]
        obs[9] = 1.0 if visible_enemies else 0.0

        obs[10] = len(teammates) / 4.0
        if teammates:
            nearest_d = min(
                max(abs(t.pos[0] - char.pos[0]), abs(t.pos[1] - char.pos[1])) for t in teammates
            )
            obs[11] = min(nearest_d, height) / height

        obs[12] = 1.0 if any(
            e.is_alive and (
                getattr(e, "blind_remaining", 0) > 0 or getattr(e, "reveal_remaining", 0) > 0
            )
            for e in enemies
        ) else 0.0

        # 味方スモーク展開中フラグ。game_state に smokes が含まれないため
        # 常に0とする(learning_attacker_retrieve.py と同じ簡略化方針)。
        obs[13] = 0.0

        r0, c0 = int(char.pos[0]), int(char.pos[1])

        if self.team_memory.spike_pos is not None:
            obs[14] = 1.0
            best_dr, best_dc = _bfs_best_direction(self.spike_dist_map, grid, r0, c0)
            obs[15] = float(best_dr)
            obs[16] = float(best_dc)

        if self.team_memory.last_seen_enemy is not None:
            ls = self.team_memory.last_seen_enemy
            obs[17] = 1.0
            best_dr, best_dc = _bfs_best_direction(self.sighting_dist_map, grid, r0, c0)
            obs[18] = float(best_dr)
            obs[19] = float(best_dc)
            obs[20] = min(ls["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP

        obs[21] = len(visible_enemies) / 5.0
        if visible_enemies:
            nearest_enemy = min(
                visible_enemies,
                key=lambda e: max(abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])),
            )
            obs[22] = (nearest_enemy.pos[0] - char.pos[0]) / height
            obs[23] = (nearest_enemy.pos[1] - char.pos[1]) / width
            dist = max(
                abs(nearest_enemy.pos[0] - char.pos[0]),
                abs(nearest_enemy.pos[1] - char.pos[1]),
            )
            obs[24] = min(dist, height) / height

        if len(site_positions) >= 1:
            obs[25] = (site_positions[0][0] - char.pos[0]) / height
            obs[26] = (site_positions[0][1] - char.pos[1]) / width
        if len(site_positions) >= 2:
            obs[27] = (site_positions[1][0] - char.pos[0]) / height
            obs[28] = (site_positions[1][1] - char.pos[1]) / width

        # search phaseではdetonate_timerは未使用(プラント前)。
        # ラウンド経過情報を持たないため中立値(0.5)を入れる。
        obs[29] = 0.5
        obs[30] = 0.0

        # --- 担当する有利ポジション(7)へのBFS距離・推奨方向・到着フラグ ---
        # spike/sightingモードの間はこの情報を出さない。到着済みフラグが
        # 「動くな」という学習済みバイアスとして誤って引き継がれ、緊急時の
        # 移動を妨げるのを防ぐため。
        in_position_mode = (
            self.team_memory.spike_pos is None
            and self.team_memory.last_seen_enemy is None
        )
        dist_map = self._assigned_dist_maps.get(char.name) if in_position_mode else None
        if dist_map is not None:
            bfs_dist = dist_map[r0, c0]
            if bfs_dist < 0:
                bfs_dist = height + width

            obs[31] = min(max(bfs_dist, 0), height + width) / (height + width)
            best_dr, best_dc = _bfs_best_direction(dist_map, grid, r0, c0)
            obs[32] = float(best_dr)
            obs[33] = float(best_dc)
            obs[34] = 1.0 if bfs_dist <= REACH_RADIUS else 0.0

        # --- 落下中スパイクにLOSが通っており、待ち伏せすべき状態か ---
        obs[35] = 1.0 if (
            self.team_memory.spike_pos is not None
            and not self.team_memory.spike_held
            and unit_has_spike_los
        ) else 0.0

        return obs, visible_enemies

    # -- 行動マスク ---------------------------------------------------------
    def _action_mask(self, char, grid, chars, lock_movement=False):
        """lock_movement=True の場合、stay以外の移動を禁止する。
        交戦中は静止させ、射撃の当たりやすさを優先する。"""
        mask = np.ones(ACTION_DIM, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = {
            tuple(o.pos) for o in chars if o is not char and getattr(o, "is_alive", True)
        }

        for move_idx, (dr, dc) in enumerate(MOVES):
            if lock_movement and move_idx != 0:
                mask[move_idx * 2] = False
                mask[move_idx * 2 + 1] = False
                continue
            nr, nc = r + dr, c + dc
            walkable = (
                0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                and grid[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            if not walkable:
                mask[move_idx * 2] = False
                mask[move_idx * 2 + 1] = False

        if _ability_charge(char) <= 0 or char.ability_name == "HUNT":
            for move_idx in range(5):
                mask[move_idx * 2 + 1] = False

        return mask

    # -- メイン ----------------------------------------------------------
    def decide_move(self, char, game_state):
        if not char.is_alive:
            return list(char.pos)

        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        is_planted = bool(game_state.get("is_planted", False))

        # このコントローラーはプラント前フェーズ専用。プラント後は
        # 上位のフェーズ切替側で別コントローラーに委譲する想定だが、
        # 万一プラント後にも呼ばれた場合は安全側としてその場に留まる。
        if is_planted:
            return list(char.pos)

        if self._site_positions_cache is None:
            self._site_positions_cache = _extract_site_positions(grid)
            if not self._site_positions_cache:
                self._site_positions_cache = [(grid.shape[0] / 2.0, grid.shape[1] / 2.0)]

        self._ensure_defense_assignment(char, grid, chars)
        self._maybe_advance_tick(char, grid, chars, game_state.get("spike_pos"))
        self._update_priority_dist_maps(grid)

        unit_has_spike_los = (
            self.team_memory.spike_pos is not None
            and not self.team_memory.spike_held
            and _has_los(grid, tuple(char.pos), self.team_memory.spike_pos)
        )
        obs, visible_enemies = self._build_observation(
            char, game_state, self._site_positions_cache, unit_has_spike_los
        )
        mask = self._action_mask(char, grid, chars, lock_movement=bool(visible_enemies))

        if self.verbose:
            mode = (
                "spike" if self.team_memory.spike_pos is not None
                else "sighting" if self.team_memory.last_seen_enemy is not None
                else "position"
            )
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{char.name},{tuple(char.pos)},{mode},"
                    f"spike14={obs[14]:.2f},spike15={obs[15]:.1f},spike16={obs[16]:.1f},"
                    f"sight17={obs[17]:.2f},sight18={obs[18]:.1f},sight19={obs[19]:.1f},"
                    f"pos31={obs[31]:.2f},pos34={obs[34]:.1f},"
                    f"spike_pos={self.team_memory.spike_pos}\n"
                )

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(obs_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action_idx = int(torch.argmax(q_values).item())

        if self.verbose:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                masked_q = q_values.cpu().numpy()
                f.write(f"  Qvals={np.round(masked_q, 4).tolist()} chosen={action_idx}\n")

        move_idx, use_ability_int = divmod(action_idx, 2)
        use_ability = bool(use_ability_int)
        move_offset = MOVES[move_idx]

        # train_defender_search.py と挙動を一致させる: position mode
        # (スパイク情報も敵目撃情報も無い)かつ担当地点未到着の間は、
        # スポーン・担当地点が毎ラウンド固定である以上、ネットワークの
        # 移動判断ではなくBFS最短方向を強制する。学習時のバッファもこの
        # 上書き後の行動で作られているため、推論側もこれに合わせないと
        # 学習内容とズレる。
        in_position_mode = (
            self.team_memory.spike_pos is None
            and self.team_memory.last_seen_enemy is None
        )
        if in_position_mode:
            dist_map = self._assigned_dist_maps.get(char.name)
            if dist_map is not None:
                r0, c0 = int(char.pos[0]), int(char.pos[1])
                cur_dist = dist_map[r0, c0]
                if cur_dist > REACH_RADIUS:
                    move_offset = _bfs_best_direction(dist_map, grid, r0, c0)

        if self.verbose:
            print(
                f"[DEFENDER SEARCH TOUYAMA] {char.name} pos={tuple(char.pos)} "
                f"action={action_idx} move={move_offset} ability={use_ability}"
            )

        next_pos = [char.pos[0] + move_offset[0], char.pos[1] + move_offset[1]]

        if not use_ability:
            return next_pos

        # アビリティ使用: 狙点は「射線内の最近接の敵」を最優先、
        # いなければチーム共有メモリの直近目撃座標を使う。
        target_pos = None
        if visible_enemies:
            nearest = min(
                visible_enemies,
                key=lambda e: max(abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])),
            )
            dist = max(abs(nearest.pos[0] - char.pos[0]), abs(nearest.pos[1] - char.pos[1]))
            if dist <= ABILITY_RANGE:
                target_pos = (int(nearest.pos[0]), int(nearest.pos[1]))
        elif self.team_memory.last_seen_enemy is not None:
            pos = self.team_memory.last_seen_enemy["pos"]
            target_pos = (int(pos[0]), int(pos[1]))

        if target_pos is None:
            # 狙点が定まらない場合はチャージを無駄にしないよう移動のみ行う。
            return next_pos

        return next_pos, {"ability": char.ability_name, "target": target_pos}