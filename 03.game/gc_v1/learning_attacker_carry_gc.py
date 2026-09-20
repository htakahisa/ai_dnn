"""gc_v1/learning_attacker_carry_gc.py

固定チーム(Xdll/SyouTa/Absol/eKo/SugarZ3ro)専用の
Attacker「carry phase」推論コントローラー。

train_attacker_carry.py で学習したDQNモデル(dict形式チェックポイント)をロードし、
スパイク保持者(has_spike==Trueのキャラ)の移動・アビリティ・明示PLANT判断を担当する。
エスコート4人はヒューリスティックで動かす(将来escortモデルに差し替え可能)。

completely self-contained: run_game.py / controllers.py / battle_logic.py /
abilities_los.py は一切importしない。必要なロジックはすべてこのファイル内に
複製する。run_game.py / controllers.py は変更しない。

【重要: target_plant_pos】
実ゲームのrun_game.py init_round()は、ラウンド開始時にgrid==2のマスから
1点だけランダムに選び、それをself.target_plant_posとしてラウンド中固定する。
これはtrain_attacker_carry.pyのCarryEnv.reset()が行っている
「エピソード開始時に1点だけ選んで固定する」設計と完全に一致するため、
このコントローラーはgame_state["target_plant_pos"]をそのまま距離計算に使う
だけでよい(独自に最寄り地点を再計算しない)。

【重要: 優先(代表)地点は本番マップに存在しない】
学習はmap_data_carry_gc.py(grid==5あり)で行うが、本番のmap_data.pyには
grid==5は存在しない。そのため優先地点は「gridの値」ではなく、
チェックポイントに保存された「座標のリスト(priority_cells)」として扱う。
本番マップ上でその座標までのマルチソースBFS距離を計算するだけで、
gridの値そのものには一切依存しない。

【重要: PLANTは明示行動】
train_attacker_carry.py側でPLANTはACTION_DIM上の独立したインデックス
(PLANT_ACTION_INDEX)として学習されている。battle_logic.pyの仕様上、
PLANT以外の行動を選んだ瞬間にchar.plant_timerは即0へリセットされるため、
このコントローラーは学習済みモデルが選んだ行動をそのまま尊重し、
旧バージョンのような「サイトに乗ったら強制PLANT」の上書きは行わない。

観測ベクトル・行動空間はtrain_attacker_carry.pyのbuild_observation() /
decode_action() / build_action_mask()と完全に一致させる必要がある。
ここがズレると学習結果が正しく反映されない。
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from character_stats_gc import (
    CHARACTER_TABLE as GC_STATS_TABLE,
    GC_ROSTER_ORDER,
)
try:
    from .gc_facing import FACING_DIRS, append_facing_onehot
    from .tactical_ability import choose_pre_entry_ability
    from .ultimate_tactics_gc import build_ultimate_action, ultimate_context_features
except ImportError:
    from gc_facing import FACING_DIRS, append_facing_onehot
    from tactical_ability import choose_pre_entry_ability
    from ultimate_tactics_gc import build_ultimate_action, ultimate_context_features

# ---------------------------------------------------------------------------
# 設定(train_attacker_carry.pyと一致させる)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "data/attacker_carry_gc_data/dqn_attacker_carry_gc_best_by_eval.pt"
DEBUG_LOG_PATH = "attacker_carry_gc_debug.log"

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 29
REAL_OBS_DIM = 31  # v3: nearby support and our own plant progress.
TACTICAL_OBS_DIM = 39  # v4: own Macro role/waypoint and actual shooting eligibility.
NAVIGATION_OBS_DIM = 57  # v5: legal neighbor costs, actual move history and deadline.
SCREENING_OBS_DIM = 61  # v6: own-team screen in front of the spike carrier.
FORMATION_OBS_DIM = 65  # v7: persistent designated entry screener and readiness.
ENTRY_SYNC_OBS_DIM = 66  # v8: explicit two-tick entry synchronization state.
ULTIMATE_CONTEXT_OBS_DIM = 70  # v10: ready/combat/objective/urgency cast context.
FACING_HEAD_OBS_DIM = ULTIMATE_CONTEXT_OBS_DIM + len(FACING_DIRS)
FACING_HEAD_VERSION = 1
LEGACY_ACTION_DIM = 11
ACTION_DIM = 12
PLANT_ACTION_INDEX = 10
ULTIMATE_ACTION_INDEX = 11

# 本番map_data.pyには5/6/7は存在しないが、train_attacker_carry.pyとの対称性のため
# 同じ集合定義を維持する(判定は常にFalseになるだけで実害はない)。
SITE_VALUES = frozenset({2, 5})
try:
    from .positioning_gc import preferred_plant_cells, team_plant_target, set_team_plant_target
except ImportError:
    from positioning_gc import preferred_plant_cells, team_plant_target, set_team_plant_target

ABILITY_RANGE = 8
SIGHTING_STALENESS_CAP = 20


# ============================================================================
# ネットワーク(train_attacker_carry.py の AttackerCarryDuelingDQN と同一構造)
# ============================================================================


class AttackerCarryDuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
        # Keep facing factorized from the base action.  The selected action is
        # still an input because moving, planting, and casting call for
        # different view directions in the same observed state.
        self.facing_head = nn.Linear(hidden + action_dim, len(FACING_DIRS))
        self.action_dim = action_dim

    def forward(self, x):
        f = self.feature(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + (a - a.mean(dim=1, keepdim=True))

    def facing_values(self, x, actions):
        features = self.feature(x)
        actions = torch.as_tensor(actions, dtype=torch.long, device=x.device).view(-1)
        action_onehot = torch.nn.functional.one_hot(
            actions, num_classes=self.action_dim
        ).to(dtype=features.dtype)
        return self.facing_head(torch.cat((features, action_onehot), dim=1))


# ============================================================================
# LOS・BFS(abilities_los.py / controllers.py と同等のロジックを複製)
# ============================================================================


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


def _has_los(grid, p1, p2, smoke_cells=None):
    cells = _line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells and len(cells) > 2:
        if any(cell in smoke_cells for cell in cells):
            return False
    return True


def _bfs_distance_map(grid, goal):
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
            if (
                0 <= nr < height
                and 0 <= nc < width
                and grid[nr, nc] != 1
                and dist[nr, nc] == -1
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _bfs_distance_map_multi(grid, sources):
    """複数始点からのマルチソースBFS距離マップ。優先(代表)地点への
    ガイダンス特徴量計算に使う(train_attacker_carry.pyと同一ロジック)。"""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    queue = deque()
    for gr, gc in sources:
        if not (0 <= gr < height and 0 <= gc < width):
            continue
        if grid[gr, gc] == 1:
            continue
        if dist[gr, gc] == -1:
            dist[gr, gc] = 0
            queue.append((gr, gc))
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and grid[nr, nc] != 1
                and dist[nr, nc] == -1
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _bfs_best_direction(dist_map, r0, c0):
    if dist_map is None:
        return 0, 0
    height, width = dist_map.shape
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


def _bfs_next_step(grid, start, goal, occupied, allow_adjacent_goal=True):
    height, width = grid.shape
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return start

    candidate_goals = []
    if grid[goal[0], goal[1]] != 1 and goal not in occupied:
        candidate_goals.append(goal)
    if allow_adjacent_goal or goal in occupied:
        for dr, dc in CARDINAL:
            adj = (goal[0] + dr, goal[1] + dc)
            if (
                0 <= adj[0] < height
                and 0 <= adj[1] < width
                and grid[adj[0], adj[1]] != 1
                and adj not in occupied
            ):
                candidate_goals.append(adj)
    candidate_goals = list(dict.fromkeys(candidate_goals))
    if not candidate_goals:
        return _random_step(grid, start, occupied)

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
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if grid[nxt[0], nxt[1]] == 1 or nxt in occupied:
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if reached is None:
        return _random_step(grid, start, occupied)

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return start
    return step


def _random_step(grid, pos, occupied):
    height, width = grid.shape
    r, c = pos
    valid = [
        (r + dr, c + dc)
        for dr, dc in CARDINAL
        if 0 <= r + dr < height
        and 0 <= c + dc < width
        and grid[r + dr, c + dc] != 1
        and (r + dr, c + dc) not in occupied
    ]
    return random.choice(valid) if valid else pos


# ============================================================================
# コントローラー本体
# ============================================================================


class LearningAttackerCarryGCController:
    """スパイク保持者(has_spike==True)のみDQNで操作する。
    それ以外のgc_v1メンバー(エスコート)はヒューリスティックで動かす。
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        device="auto",
        greedy=True,
        epsilon=0.0,
        debug=False,
    ):
        self.greedy = greedy
        self.epsilon = epsilon
        self.debug = debug

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Carryモデルが見つかりません: {model_path}")

        # train_attacker_carry.pyが生成した信頼済みチェックポイントなので
        # weights_only=False で読み込む(PyTorch 2.6+のデフォルト変更対応)。
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.positioning_version = int(checkpoint.get("positioning_version", 0))
        self.model_episode = checkpoint.get("episode")
        self.model_path = str(model_path)

        ckpt_obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
        ckpt_n_actions = int(checkpoint.get("n_actions", ACTION_DIM))
        expected_obs_dim = (FACING_HEAD_OBS_DIM if self.positioning_version >= 11 else
                            ULTIMATE_CONTEXT_OBS_DIM if self.positioning_version >= 10 else
                            ENTRY_SYNC_OBS_DIM if self.positioning_version >= 8 else
                            FORMATION_OBS_DIM if self.positioning_version >= 7 else
                            SCREENING_OBS_DIM if self.positioning_version >= 6 else
                            NAVIGATION_OBS_DIM if self.positioning_version >= 5 else
                            TACTICAL_OBS_DIM if self.positioning_version >= 4 else
                            REAL_OBS_DIM if self.positioning_version >= 3 else OBS_DIM)
        expected_actions = ACTION_DIM if self.positioning_version >= 9 else LEGACY_ACTION_DIM
        if ckpt_obs_dim != expected_obs_dim or ckpt_n_actions != expected_actions:
            raise ValueError(
                f"チェックポイントの観測/行動空間がこのコントローラーと不一致です: "
                f"obs_dim={ckpt_obs_dim}(期待値{expected_obs_dim}) n_actions={ckpt_n_actions}(期待値{ACTION_DIM})。"
                f"train_attacker_carry.pyのバージョンが古い可能性があります。"
            )

        self.policy_net = AttackerCarryDuelingDQN(
            obs_dim=ckpt_obs_dim, action_dim=ckpt_n_actions
        ).to(self.device)
        incompatible = self.policy_net.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys
                   if not key.startswith("facing_head.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Carry checkpoint keys mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.facing_head_enabled = (
            int(checkpoint.get("facing_head_version", 0)) >= FACING_HEAD_VERSION
            and not incompatible.missing_keys
        )
        self.policy_net.eval()

        # 優先(代表)地点はチェックポイントに座標として保存されている。
        # 本番map_data.pyにはgrid==5が存在しないため、gridの値には一切依存しない。
        self.priority_cells = [
            tuple(map(int, cell)) for cell in (checkpoint.get("priority_cells") or [])
        ]
        self.has_priority_cells = bool(
            checkpoint.get("has_priority_cells", False)
        ) and bool(self.priority_cells)

        # サイト別ウェイポイント(右=6/左=7相当)もチェックポイントに座標群として保存されている。
        # 本番map_data.pyにはgrid==6/7が存在しないため、gridの値には一切依存しない。
        # 未保存(旧チェックポイント)の場合は空dictとなり、常に「通過済み」扱いで従来挙動になる。
        raw_waypoint_cells = checkpoint.get("waypoint_cells") or {}
        self.waypoint_cells = {}
        for site, raw_cells in raw_waypoint_cells.items():
            # 新形式: {"right": [(r,c), ...], "left": [(r,c), ...]}
            # 旧形式: {"right": (r,c), "left": (r,c)}
            # の両方を読めるようにして旧チェックポイントとも互換性を保つ。
            if (
                isinstance(raw_cells, (list, tuple))
                and len(raw_cells) == 2
                and all(isinstance(v, (int, float)) for v in raw_cells)
            ):
                cells = [(int(raw_cells[0]), int(raw_cells[1]))]
            else:
                cells = [(int(cell[0]), int(cell[1])) for cell in (raw_cells or [])]
            if cells:
                self.waypoint_cells[str(site)] = cells

        self._log(
            f"[LOAD] model={model_path} episode={checkpoint.get('episode')} "
            f"success_rate={checkpoint.get('success_rate')} "
            f"has_priority_cells={self.has_priority_cells} priority_cells={self.priority_cells} "
            f"waypoint_cells={self.waypoint_cells}"
        )

        self.game = None

        # gridごとにキャッシュ(通常は試合中一度だけ計算される)
        self._priority_dist_map = None
        self._priority_max_dist = None
        self._waypoint_dist_maps = {}  # site -> dist_map(waypoint_cells由来)

        # ラウンド単位の状態(target_plant_posはラウンド中固定なのでキャッシュしてよい)
        self._sighting = None
        self._plant_cells = (
            None  # このラウンドの target_plant_pos への距離マップキャッシュ用キー
        )
        self._target_dist_map = None
        self._cached_target_pos = None
        self._active_waypoint_site = None
        self._reached_waypoint = True

    # -- run_game.py 側フック(hasattr判定で自動呼び出しされる) -------------
    def set_game(self, game):
        self.game = game
        grid = game.grid
        # IQ creates a new view for every actor, but the immutable grid is shared.
        if getattr(self, "_navigation_grid", None) is grid:
            return
        self._navigation_grid = grid
        self._intent_distance_cache = {}
        self._navigation_history = {}
        self._plant_cells = [
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if int(grid[r, c]) in SITE_VALUES
        ]
        if self.has_priority_cells:
            self._priority_dist_map = _bfs_distance_map_multi(grid, self.priority_cells)
        else:
            self._priority_dist_map = _bfs_distance_map_multi(grid, self._plant_cells)
        finite = self._priority_dist_map[self._priority_dist_map >= 0]
        self._priority_max_dist = (
            int(finite.max()) if finite.size else (grid.shape[0] + grid.shape[1])
        )

        # サイト別ウェイポイント候補群へのマルチソースBFS距離マップ。
        self._waypoint_dist_maps = {
            site: _bfs_distance_map_multi(grid, cells)
            for site, cells in self.waypoint_cells.items()
        }

    def reset_round(self):
        self._navigation_history = {}
        self._sighting = None
        self._target_dist_map = None
        self._cached_target_pos = None
        self._active_waypoint_site = None
        self._reached_waypoint = True

    # -- ログ ---------------------------------------------------------
    def _log(self, message):
        if not self.debug:
            return
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except OSError:
            pass

    # -- ユーティリティ -------------------------------------------------
    def _smoke_cells(self):
        cells = set()
        for smoke in getattr(self.game, "smokes", []):
            cells.update(smoke.get("cells", ()))
        return cells

    def _get_target_dist_map(self, target_pos):
        """target_plant_posはラウンド中固定のはずなので、同じ座標が来る限り
        再計算せずキャッシュを使い回す(実ゲームのinit_round()仕様に依拠)。"""
        target_pos = (int(target_pos[0]), int(target_pos[1]))
        if target_pos != self._cached_target_pos:
            self._target_dist_map = _bfs_distance_map(self.game.grid, target_pos)
            self._cached_target_pos = target_pos
        return self._target_dist_map

    def _update_sighting(self, char, chars, smoke_cells):
        grid = self.game.grid
        visible = [
            other
            for other in chars
            if getattr(other, "is_alive", True)
            and getattr(other, "team", None) != char.team
            and _has_los(grid, char.pos, other.pos, smoke_cells)
        ]
        if visible:
            nearest = min(
                visible,
                key=lambda o: max(
                    abs(o.pos[0] - char.pos[0]), abs(o.pos[1] - char.pos[1])
                ),
            )
            self._sighting = {
                "pos": tuple(nearest.pos),
                "name": nearest.name,
                "tick_ago": 0,
            }
        elif self._sighting is not None:
            self._sighting["tick_ago"] += 1
            if self._sighting["tick_ago"] > SIGHTING_STALENESS_CAP:
                self._sighting = None

    # -- 観測構築(train_attacker_carry.py の build_observation と同一構造) --
    def _build_observation(
        self,
        char,
        chars,
        smoke_cells,
        dist_map,
        elapsed_ticks,
        max_ticks,
        reached_waypoint,
        target_plant_pos=None,
    ):
        modern = self.positioning_version >= 3
        obs_dim = (FACING_HEAD_OBS_DIM if self.positioning_version >= 11 else
                   ULTIMATE_CONTEXT_OBS_DIM if self.positioning_version >= 10 else
                   ENTRY_SYNC_OBS_DIM if self.positioning_version >= 8 else
                   FORMATION_OBS_DIM if self.positioning_version >= 7 else
                   SCREENING_OBS_DIM if self.positioning_version >= 6 else
                   NAVIGATION_OBS_DIM if self.positioning_version >= 5 else
                   TACTICAL_OBS_DIM if self.positioning_version >= 4 else REAL_OBS_DIM if modern else OBS_DIM)
        obs = np.zeros(obs_dim, dtype=np.float32)
        grid = self.game.grid
        height, width = grid.shape
        r0, c0 = int(char.pos[0]), int(char.pos[1])

        obs[0] = char.pos[0] / height
        obs[1] = char.pos[1] / width
        obs[2] = char.hp / char.max_hp if char.max_hp else 0.0
        # move_character() clears moved_this_tick before asking the policy.
        # Training observes the move from the preceding completed tick.
        obs[3] = float(bool(getattr(char, "moved_last_tick",
                                    getattr(char, "moved_this_tick", False))
                            if modern else getattr(char, "moved_this_tick", False)))

        ability_index = {"SMOKE": 4, "FLASH": 5, "RECON": 6, "HUNT": 7}.get(
            char.ability_name
        )
        if ability_index is not None:
            obs[ability_index] = 1.0
        charges = {
            "SMOKE": char.smoke_charges,
            "FLASH": char.flash_charges,
            "RECON": char.recon_charges,
        }.get(char.ability_name, 0)
        obs[8] = 1.0 if charges > 0 else 0.0

        bfs_dist = dist_map[r0, c0] if dist_map is not None else -1
        if bfs_dist < 0:
            bfs_dist = height + width
        obs[9] = min(bfs_dist, height + width) / (height + width)
        best_dr, best_dc = _bfs_best_direction(dist_map, r0, c0)
        obs[10] = float(best_dr)
        obs[11] = float(best_dc)
        obs[12] = 1.0 if int(grid[r0, c0]) in SITE_VALUES else 0.0

        visible_enemies = [
            o
            for o in chars
            if getattr(o, "is_alive", True)
            and o.team != char.team
            and _has_los(grid, char.pos, o.pos, smoke_cells)
        ]
        obs[13] = 1.0 if visible_enemies else 0.0
        obs[14] = len(visible_enemies) / 5.0
        if visible_enemies:
            nearest = min(
                visible_enemies,
                key=lambda o: max(abs(o.pos[0] - r0), abs(o.pos[1] - c0)),
            )
            obs[15] = (nearest.pos[0] - r0) / height
            obs[16] = (nearest.pos[1] - c0) / width
            dist = max(abs(nearest.pos[0] - r0), abs(nearest.pos[1] - c0))
            obs[17] = min(dist, height) / height

        obs[18] = (
            1.0
            if any(
                getattr(o, "is_alive", True)
                and o.team != char.team
                and (
                    getattr(o, "blind_remaining", 0) > 0
                    or getattr(o, "reveal_remaining", 0) > 0
                )
                for o in chars
            )
            else 0.0
        )

        own_smoke_active = any(
            s.get("owner") is not None
            and any(c.name == s.get("owner") and c.team == char.team for c in chars)
            for s in getattr(self.game, "smokes", [])
        )
        obs[19] = 1.0 if own_smoke_active else 0.0

        teammates = [
            o
            for o in chars
            if o is not char and getattr(o, "is_alive", True) and o.team == char.team
        ]
        obs[20] = len(teammates) / 4.0
        if teammates:
            nearest_d = min(
                max(abs(t.pos[0] - r0), abs(t.pos[1] - c0)) for t in teammates
            )
            obs[21] = min(nearest_d, height) / height
        obs[22] = 1.0  # is_carrier
        obs[23] = min(elapsed_ticks, max_ticks) / max_ticks
        obs[24] = (
            sum(
                1 for o in chars if getattr(o, "is_alive", True) and o.team != char.team
            )
            / 5.0
        )

        # --- 優先(代表)地点への誘導特徴量。checkpointのpriority_cells由来の
        # マルチソースBFS(self._priority_dist_map)を参照する。target_plant_pos
        # (dist_map)とは独立した、常時提供される追加ガイダンス。---
        priority_map = self._priority_dist_map
        priority_max = self._priority_max_dist
        if self.positioning_version >= 2 and target_plant_pos is not None:
            priority_map = self._get_target_dist_map(target_plant_pos)
            priority_max = height + width
        p_dist = priority_map[r0, c0] if priority_map is not None else -1
        if p_dist < 0:
            p_dist = priority_max
        obs[25] = min(p_dist, priority_max) / max(1, priority_max)
        p_best_dr, p_best_dc = _bfs_best_direction(priority_map, r0, c0)
        obs[26] = float(p_best_dr)
        obs[27] = float(p_best_dc)

        # サイト別ウェイポイント通過済みフラグ(train_attacker_carry.pyのobs[28]と同一)。
        obs[28] = 1.0 if reached_waypoint else 0.0

        if modern:
            from game_core import PLANT_REQUIRED_TICKS
            obs[29] = sum(max(abs(t.pos[0] - r0), abs(t.pos[1] - c0)) <= 3
                          for t in teammates) / 4.0
            obs[30] = min(1.0, max(0.0, getattr(char, "plant_timer", 0))
                          / max(1, PLANT_REQUIRED_TICKS))

        if self.positioning_version >= 4:
            try:
                from .navigation_intent_gc import intent_features
            except ImportError:
                from navigation_intent_gc import intent_features
            cache = self.__dict__.setdefault("_intent_distance_cache", {})
            obs[REAL_OBS_DIM:TACTICAL_OBS_DIM] = intent_features(self.game, char, chars, cache)
        if self.positioning_version >= 5:
            try:
                from .navigation_intent_gc import navigation_context_features, navigation_intent
            except ImportError:
                from navigation_intent_gc import navigation_context_features, navigation_intent
            history = self.__dict__.setdefault("_navigation_history", {})
            goal = navigation_intent(self.game, char)[0]
            obs[TACTICAL_OBS_DIM:NAVIGATION_OBS_DIM] = navigation_context_features(
                self.game, char, chars, dist_map, history, goal
            )
        if self.positioning_version >= 6:
            try:
                from .navigation_intent_gc import carrier_screening_features
            except ImportError:
                from navigation_intent_gc import carrier_screening_features
            cache = self.__dict__.setdefault("_intent_distance_cache", {})
            obs[NAVIGATION_OBS_DIM:SCREENING_OBS_DIM] = carrier_screening_features(
                self.game, char, chars, cache, escort=False
            )
        if self.positioning_version >= 7:
            try:
                from .navigation_intent_gc import carrier_formation_features
            except ImportError:
                from navigation_intent_gc import carrier_formation_features
            cache = self.__dict__.setdefault("_intent_distance_cache", {})
            obs[SCREENING_OBS_DIM:FORMATION_OBS_DIM] = carrier_formation_features(
                self.game, char, chars, cache, escort=False
            )
        if self.positioning_version >= 8:
            try:
                from .navigation_intent_gc import carrier_entry_sync_feature
            except ImportError:
                from navigation_intent_gc import carrier_entry_sync_feature
            cache = self.__dict__.setdefault("_intent_distance_cache", {})
            obs[FORMATION_OBS_DIM] = carrier_entry_sync_feature(
                self.game, char, chars, cache,
                obs[TACTICAL_OBS_DIM + 14],
            )
        if self.positioning_version >= 10:
            try:
                from .navigation_intent_gc import can_engage, carrier_screening_status
            except ImportError:
                from navigation_intent_gc import can_engage, carrier_screening_status
            cache = self.__dict__.setdefault("_intent_distance_cache", {})
            status = carrier_screening_status(self.game, char, chars, cache)
            final_distance = int(status.get("final_distance", -1))
            obs[ENTRY_SYNC_OBS_DIM:ULTIMATE_CONTEXT_OBS_DIM] = ultimate_context_features(
                char,
                engaged=can_engage(self.game, char, chars),
                objective_window=0 <= final_distance <= 10,
                urgent=(char.max_hp > 0 and char.hp / char.max_hp <= 0.45),
            )
        if self.positioning_version >= 11:
            append_facing_onehot(
                obs[ULTIMATE_CONTEXT_OBS_DIM:FACING_HEAD_OBS_DIM],
                getattr(char, "facing", "N"),
            )
        return obs

    def _build_mask(self, char, chars, on_site):
        grid = self.game.grid
        height, width = grid.shape
        occupied = {
            tuple(o.pos)
            for o in chars
            if o is not char and getattr(o, "is_alive", True)
        }
        action_dim = ACTION_DIM if self.positioning_version >= 9 else LEGACY_ACTION_DIM
        mask = np.ones(action_dim, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        for move_idx, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            walkable = (
                0 <= nr < height
                and 0 <= nc < width
                and grid[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            if not walkable:
                mask[move_idx * 2] = False
                mask[move_idx * 2 + 1] = False

        charges = {
            "SMOKE": char.smoke_charges,
            "FLASH": char.flash_charges,
            "RECON": char.recon_charges,
        }.get(char.ability_name, 0)
        if charges <= 0 or char.ability_name == "HUNT":
            for move_idx in range(5):
                mask[move_idx * 2 + 1] = False

        mask[PLANT_ACTION_INDEX] = bool(on_site)

        if self.positioning_version >= 5:
            # The actual engine spends an ability tick standing still. Legacy
            # move+ability indices could execute either STAY or MOVE depending
            # on target availability. New policies have one explicit ability
            # action (1); moves are 2/4/6/8 and planting remains 10.
            mask[[3, 5, 7, 9]] = False
            mask[1] = mask[1] and self._learned_ability_target(char, chars) is not None

        if self.positioning_version >= 9:
            try:
                from .navigation_intent_gc import navigation_intent
            except ImportError:
                from navigation_intent_gc import navigation_intent
            goal = navigation_intent(self.game, char)[0]
            mask[ULTIMATE_ACTION_INDEX] = build_ultimate_action(
                grid, char, chars, destination=goal
            ) is not None

        return mask

    def _learned_ability_target(self, char, chars):
        visible = [c for c in chars if c.team != char.team and getattr(c, "is_alive", True)
                   and _has_los(self.game.grid, char.pos, c.pos, self._smoke_cells())
                   and max(abs(c.pos[0] - char.pos[0]), abs(c.pos[1] - char.pos[1])) <= ABILITY_RANGE]
        if visible:
            nearest = min(visible, key=lambda c: max(abs(c.pos[0] - char.pos[0]), abs(c.pos[1] - char.pos[1])))
            return tuple(map(int, nearest.pos))
        if self._sighting is not None:
            pos = self._sighting["pos"]
            if max(abs(pos[0] - char.pos[0]), abs(pos[1] - char.pos[1])) <= ABILITY_RANGE:
                return pos
        return None

    def _select_action(self, obs, mask):
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(obs_t).squeeze(0).cpu().numpy()
            q_values = np.where(mask, q_values, -np.inf)
            if not self.greedy and random.random() < self.epsilon:
                valid = np.flatnonzero(mask)
                return int(np.random.choice(valid)) if len(valid) else 0
            return int(np.argmax(q_values))

    def _select_facing(self, obs, action_idx):
        if not getattr(self, "facing_head_enabled", False):
            return None
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            values = self.policy_net.facing_values(
                obs_t, torch.tensor([action_idx], device=self.device)
            ).squeeze(0)
        return FACING_DIRS[int(values.argmax().item())]

    @staticmethod
    def _decode_action(action_idx):
        """train_attacker_carry.py の decode_action と同一。"""
        if int(action_idx) == PLANT_ACTION_INDEX:
            return "PLANT"
        move_idx, use_ability = divmod(int(action_idx), 2)
        return MOVES[move_idx], bool(use_ability)

    # -- エスコート・ヒューリスティック(train_attacker_carry.pyと同一方針) --
    def _escort_ability_action(self, char, visible_enemies):
        charges = {
            "SMOKE": char.smoke_charges,
            "FLASH": char.flash_charges,
            "RECON": char.recon_charges,
        }.get(char.ability_name, 0)
        if charges <= 0 or char.ability_name == "HUNT":
            return None

        if char.ability_name == "SMOKE" and len(visible_enemies) >= 2:
            return ("SMOKE", tuple(map(int, visible_enemies[0].pos)))

        if char.ability_name == "FLASH" and visible_enemies:
            closest = min(
                visible_enemies,
                key=lambda e: max(
                    abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])
                ),
            )
            dist = max(
                abs(closest.pos[0] - char.pos[0]), abs(closest.pos[1] - char.pos[1])
            )
            if dist <= 5:
                return ("FLASH", tuple(map(int, closest.pos)))

        if char.ability_name == "RECON" and not visible_enemies and self._plant_cells:
            return ("RECON", random.choice(self._plant_cells))

        return None

    def _decide_escort_move(self, char, carrier, occupied):
        grid = self.game.grid
        if carrier is None or not getattr(carrier, "is_alive", True):
            return list(char.pos)
        dist = max(abs(carrier.pos[0] - char.pos[0]), abs(carrier.pos[1] - char.pos[1]))
        if random.random() < 0.3:
            nxt = _random_step(grid, tuple(char.pos), occupied)
        elif dist > 5:
            nxt = _bfs_next_step(
                grid,
                tuple(char.pos),
                tuple(carrier.pos),
                occupied,
                allow_adjacent_goal=True,
            )
        else:
            nxt = _random_step(grid, tuple(char.pos), occupied)
        return [int(nxt[0]), int(nxt[1])]

    # -- メインエントリポイント -----------------------------------------
    def decide_move(self, char, game_state):
        chars = game_state.get("chars", [])
        grid = self.game.grid
        r, c = int(char.pos[0]), int(char.pos[1])
        smoke_cells = self._smoke_cells()

        occupied = {
            tuple(o.pos)
            for o in chars
            if o is not char and getattr(o, "is_alive", True)
        }

        # --- キャリア以外(エスコート)はヒューリスティック ---
        if not getattr(char, "has_spike", False):
            carrier = next(
                (
                    o
                    for o in chars
                    if getattr(o, "is_alive", True)
                    and o.team == char.team
                    and getattr(o, "has_spike", False)
                ),
                None,
            )
            visible_enemies = [
                o
                for o in chars
                if getattr(o, "is_alive", True)
                and o.team != char.team
                and _has_los(grid, char.pos, o.pos, smoke_cells)
            ]
            ability = self._escort_ability_action(char, visible_enemies)
            if ability is not None:
                return list(char.pos), {"ability": ability[0], "target": ability[1]}
            next_pos = self._decide_escort_move(char, carrier, occupied)
            return next_pos

        # --- ここからキャリア(DQN) ---
        target_plant_pos = game_state.get("target_plant_pos")
        if self.positioning_version >= 1:
            target_plant_pos = team_plant_target(self.game) or target_plant_pos
        if target_plant_pos is None:
            # 実ゲームでは通常発生しないが、念のためのフォールバック。
            target_plant_pos = self._plant_cells[0] if self._plant_cells else (r, c)
        target_plant_pos = (int(target_plant_pos[0]), int(target_plant_pos[1]))
        if self.positioning_version >= 1:
            site_left = target_plant_pos[1] < grid.shape[1] // 2
            candidates = preferred_plant_cells(grid, [
                p for p in self._plant_cells
                if (p[1] < grid.shape[1] // 2) == site_left
            ])
            if target_plant_pos not in candidates and candidates:
                distances = _bfs_distance_map(grid, (r, c))
                reachable = [p for p in candidates if distances[p] >= 0]
                if reachable:
                    target_plant_pos = min(reachable, key=lambda p: (distances[p], p))
                    set_team_plant_target(self.game, target_plant_pos)

        # ラウンド開始時(_active_waypoint_site未設定)にサイトを判定し、
        # 対応するウェイポイントが存在すればまずそこへの距離マップを使う。
        # train_attacker_carry.py CarryEnv.reset()と同一の判定基準(列がwidth//2未満=左)。
        target_site = "left" if target_plant_pos[1] < grid.shape[1] // 2 else "right"
        if self._active_waypoint_site != target_site:
            width = grid.shape[1]
            self._active_waypoint_site = (
                "left" if target_plant_pos[1] < width // 2 else "right"
            )
            if self._active_waypoint_site in self._waypoint_dist_maps:
                self._reached_waypoint = False
            else:
                self._reached_waypoint = True

        if not self._reached_waypoint:
            waypoint_cells = self.waypoint_cells[self._active_waypoint_site]
            waypoint_dist = min(
                max(abs(r - cell[0]), abs(c - cell[1])) for cell in waypoint_cells
            )
            if waypoint_dist <= 1:
                self._reached_waypoint = True
                # 通過した瞬間だけ切り替える。以降はtarget_plant_pos向けのキャッシュ
                # (_get_target_dist_map)をそのまま使う。
                self._cached_target_pos = None

        if self._reached_waypoint:
            dist_map = self._get_target_dist_map(target_plant_pos)
        else:
            dist_map = self._waypoint_dist_maps[self._active_waypoint_site]

        if self.positioning_version >= 4:
            try:
                from .navigation_intent_gc import navigation_intent
            except ImportError:
                from navigation_intent_gc import navigation_intent
            route_goal, _, _ = navigation_intent(self.game, char)
            if route_goal is not None:
                dist_map = self._get_target_dist_map(route_goal)

        on_site = int(grid[r, c]) in SITE_VALUES

        self._update_sighting(char, chars, smoke_cells)
        pre_entry_ability = choose_pre_entry_ability(
            char,
            chars,
            grid,
            smoke_cells,
            destination=target_plant_pos,
            last_seen=self._sighting["pos"] if self._sighting else None,
            max_range=ABILITY_RANGE,
        )
        if pre_entry_ability is not None and self.positioning_version < 5:
            return list(char.pos), {
                "ability": pre_entry_ability[0],
                "target": pre_entry_ability[1],
            }
        elapsed_ticks = getattr(self.game, "battle_tick", 0)
        max_ticks = (
            getattr(self.game, "round_timer", 100) + elapsed_ticks
        )  # 概算(観測は正規化用途のみ)
        max_ticks = max(max_ticks, 1)

        obs = self._build_observation(
            char,
            chars,
            smoke_cells,
            dist_map,
            elapsed_ticks,
            max_ticks,
            self._reached_waypoint,
            target_plant_pos,
        )
        mask = self._build_mask(char, chars, on_site)
        action_idx = self._select_action(obs, mask)
        facing = self._select_facing(obs, action_idx)
        if facing is not None and not getattr(char, "facing_forced_this_tick", False):
            # Ability/ultimate/plant return before battle_logic's MOVE-facing
            # path, so apply the learned result here as well.  Forced facing
            # from taking damage remains authoritative for that tick.
            char.facing = facing

        if self.positioning_version >= 9 and action_idx == ULTIMATE_ACTION_INDEX:
            ultimate = build_ultimate_action(
                grid, char, chars, destination=route_goal or target_plant_pos
            )
            if ultimate is not None:
                if facing is not None:
                    ultimate = dict(ultimate, facing=facing)
                return list(char.pos), ultimate
        decoded = self._decode_action(action_idx)

        if decoded == "PLANT":
            # PLANTはマスク上、on_site==Trueの時のみ選択され得る。
            # 移動・アビリティ使用は行わない(battle_logic.pyのPLANT分岐と同一仕様)。
            return list(char.pos), "PLANT"

        if self.positioning_version >= 5 and action_idx == 1:
            target = self._learned_ability_target(char, chars)
            payload = {"ability": char.ability_name, "target": target}
            if facing is not None:
                payload["facing"] = facing
            return list(char.pos), payload

        (dr, dc), use_ability = decoded

        if use_ability:
            charges = {
                "SMOKE": char.smoke_charges,
                "FLASH": char.flash_charges,
                "RECON": char.recon_charges,
            }.get(char.ability_name, 0)
            visible_enemies = [
                o
                for o in chars
                if getattr(o, "is_alive", True)
                and o.team != char.team
                and _has_los(grid, char.pos, o.pos, smoke_cells)
            ]
            if charges > 0:
                target_pos = None
                if visible_enemies:
                    nearest = min(
                        visible_enemies,
                        key=lambda o: max(abs(o.pos[0] - r), abs(o.pos[1] - c)),
                    )
                    dist = max(abs(nearest.pos[0] - r), abs(nearest.pos[1] - c))
                    if dist <= ABILITY_RANGE:
                        target_pos = tuple(map(int, nearest.pos))
                elif self._sighting is not None:
                    target_pos = self._sighting["pos"]
                elif self._plant_cells and char.ability_name == "SMOKE":
                    # SMOKE may still use the configured plant-side fallback.
                    # RECON must never be thrown at a blind/random cell: it
                    # needs an actual sighting (handled by the pre-entry gate
                    # above), otherwise the old fallback caused spawn throws.
                    target_pos = random.choice(self._plant_cells)

                if target_pos is not None:
                    payload = {
                        "ability": char.ability_name,
                        "target": target_pos,
                    }
                    if facing is not None:
                        payload["facing"] = facing
                    return list(char.pos), payload

        next_pos = [r + dr, c + dc]
        return next_pos if facing is None else (next_pos, {"facing": facing})
