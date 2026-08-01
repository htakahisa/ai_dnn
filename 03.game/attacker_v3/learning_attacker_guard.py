"""learning_attacker_guard.py

train_attacker_guard.py で学習した Guard Phase 用 Dueling DQN モデルを使い、
スパイク防衛キャラクター（最大5体）の移動・アビリティ判断を行う推論用
コントローラー。

【重要】5体は重み共有】
train_attacker_guard.py 側は生存中の全guardが同一ネットワークの重みを
共有して学習している（パラメータ共有方式）。run_game.py側でも、
プラント後の攻撃側キャラクター全員に「同じ
LearningAttackerGuardController インスタンス」を割り当てる想定。

【配置優先順位ロジックの移植】
train_attacker_guard.py の _build_candidate_tiers / _assign_goals と
完全に同一のロジックをここに移植する：
  1. スパイク設置位置そのもの
  2. 設置位置に隣接する8マス
  3. map_data_guard.py に記載された「6」の座標（学習専用・座標ベース）
他の味方が既に候補を確保している場合は優先順位を1→2→3の順に落とす。
割当の「確保する順番」は、game_state["chars"] 内での登場順
（ゲーム内で安定した順序）を使う。これは学習側の「固定エージェント
index順」に相当する近似であり、厳密に同一である必要はない
（毎tick一貫していれば十分機能する）。

【重要：6は本番マップに存在しない】
map_data_guard.py の grid==6 は学習専用。本番の map_data.py には
存在しないため、carryモデルの優先地点(5)と同じ設計判断で、
6の座標を「座標のリスト」としてチェックポイントに保存し、
推論側はその座標を直接BFSの目的地として使う（gridの値は見ない）。

【緊急対応（解除阻止）】
game_state["defender_defuse_info"] から defuse_timer > 0 の敵を検出し、
見つかった場合は配置優先順位を無視して全guardの目標をその敵の位置に
切り替える（tier=-1の緊急モード。学習側と同一のロジック）。

【観測ベクトルは45次元、train_attacker_guard.py の GuardEnv._get_obs() と
要素の順序・個数を完全に一致させる必要がある】

【既知の制約】
- game_state にはスモークの有無が含まれていないため、視界判定は
  壁のみを考慮したBresenham判定になる（学習環境よりやや楽観的）。
- detonate_timer は game_state からそのまま取得する
  （game_core.py の SPIKE_DETONATION_TICKS=45 と学習側の
  DETONATE_TICKS=45 は一致させてある）。

run_game.py からは他の learning_attacker_*.py 系コントローラーと同様の
インターフェース（decide_move(char, game_state) -> next_pos または
(next_pos, {"ability": ..., "target": (r, c)})）で呼び出される想定。
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from controllers import BaseController


# ---------------------------------------------------------------------------
# 行動定義（train_attacker_guard.py の GuardEnv と同一でなければならない）
# ---------------------------------------------------------------------------
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_ABILITY = range(6)
N_ACTIONS = 6
_MOVE_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_STAY: (0, 0),
}

BLIND_DURATION_TICKS = 3
REVEAL_DURATION_TICKS = 5
ABILITY_TYPES = ("FLASH", "RECON", "SMOKE")
ABILITY_RANGE = 6

DETONATE_TICKS = 45  # game_core.py の SPIKE_DETONATION_TICKS と一致させる
DIST_NORM_MAX = 20.0


# ---------------------------------------------------------------------------
# 汎用ヘルパー（train_attacker_guard.py と同一ロジック）
# ---------------------------------------------------------------------------
def _chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _line_cells(p1, p2):
    y0, x0 = int(p1[0]), int(p1[1])
    y1, x1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
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


def _has_los_walls_only(grid, p1, p2):
    """壁のみを考慮した射線判定。スモークは game_state から参照できないため
    考慮しない（既知の制約。学習環境よりやや楽観的な視界判定になる）。"""
    for r, c in _line_cells(p1, p2):
        if grid[r, c] == 1:
            return False
    return True


def _build_distance_map_from_coords(grid, source_cells):
    height, width = grid.shape
    dist = np.full((height, width), np.inf, dtype=np.float32)
    q = deque()
    for r, c in source_cells:
        if 0 <= r < height and 0 <= c < width and grid[r, c] != 1:
            dist[r, c] = 0.0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def _neighbors8(pos, grid):
    r, c = pos
    height, width = grid.shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                out.append((nr, nc))
    return out


def _build_candidate_tiers(grid, plant_pos, formation_cells):
    """優先順位付きスロット候補を返す: (tier1, tier2, tier3)。
    train_attacker_guard.py と同一ロジック。"""
    r, c = plant_pos
    tier1 = [plant_pos] if grid[r, c] != 1 else []
    tier2 = _neighbors8(plant_pos, grid)
    tier3 = list(formation_cells)
    return tier1, tier2, tier3


def _assign_goals(grid, plant_pos, formation_cells, positions_ordered):
    """近い順の貪欲割当で、各エージェントの誘導目標(goal)とtier(0/1/2)を返す。
    train_attacker_guard.py と同一ロジック。

    positions_ordered: [(agent_key, current_pos), ...] のリスト。
    戻り値: {agent_key: (goal_pos, tier)}  tier: 0=設置位置, 1=隣接8マス, 2=map6
    """
    tier1, tier2, tier3 = _build_candidate_tiers(grid, plant_pos, formation_cells)
    tiers = [tier1, tier2, tier3]
    claimed = set()
    result = {}
    for agent_key, pos in positions_ordered:
        chosen, chosen_tier = None, None
        for tier_idx, tier_cells in enumerate(tiers):
            avail = [s for s in tier_cells if s not in claimed]
            if avail:
                chosen = min(avail, key=lambda s: _chebyshev(pos, s))
                chosen_tier = tier_idx
                break
        if chosen is None:
            chosen, chosen_tier = plant_pos, 0
        claimed.add(chosen)
        result[agent_key] = (chosen, chosen_tier)
    return result


# ---------------------------------------------------------------------------
# Dueling DQN（train_attacker_guard.py と同一アーキテクチャ）
# ---------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
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
        feat = self.feature(x)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class LearningAttackerGuardController(BaseController):
    """Guard Phaseの学習済みモデルで、スパイク防衛キャラクター（最大5体）の
    移動・アビリティ使用を決定する。生存中の全guardは同一インスタンス
    （同一ネットワーク）を共有する想定。
    """

    def __init__(
        self,
        model_path,
        device=None,
        greedy=True,
        epsilon=0.0,
        detonate_ticks=DETONATE_TICKS,
        verbose=False,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.greedy = greedy
        self.epsilon = epsilon
        self.detonate_ticks = detonate_ticks
        self.verbose = verbose

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Guardモデルが見つかりません: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        obs_dim = checkpoint["obs_dim"]
        n_actions = checkpoint.get("n_actions", N_ACTIONS)

        self.policy_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()

        # 学習マップ(map_data_guard.py)上での推奨配置点(6)の座標。
        # 本番マップ(map_data.py)には grid==6 が存在しないため、
        # この座標をそのまま「目的地の集合」として使う。
        self.formation_cells = [tuple(cell) for cell in (checkpoint.get("formation_cells") or [])]

        if self.verbose:
            print(
                f"[LearningAttackerGuardController] モデル読込完了: {model_path} "
                f"(obs_dim={obs_dim}, episode={checkpoint.get('episode')}, "
                f"success_rate={checkpoint.get('success_rate')}, "
                f"formation_cells={self.formation_cells})"
            )

        # goal座標(r, c) -> 壁のみBFS距離マップ のキャッシュ
        self._goal_dist_cache = {}
        # キャラクター名ごとの episode 内状態（移動履歴・停滞カウント）
        self._char_state = {}

    # ------------------------------------------------------------------
    # ラウンド開始時にrun_game.pyから呼ばれる（hasattr判定で自動検出される）
    # ------------------------------------------------------------------
    def reset_round(self):
        self._char_state.clear()
        self._goal_dist_cache.clear()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _get_char_state(self, char):
        return self._char_state.setdefault(
            char.name,
            {"last_delta": (0.0, 0.0), "stuck": 0},
        )

    @staticmethod
    def _is_wall(grid, r, c):
        height, width = grid.shape
        if not (0 <= r < height and 0 <= c < width):
            return True
        return grid[r, c] == 1

    def _get_goal_dist_map(self, grid, goal):
        key = (grid.tobytes(), tuple(goal))
        cached = self._goal_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_from_coords(grid, [tuple(goal)])
            self._goal_dist_cache[key] = cached
        return cached

    def _resolve_plant_pos(self, game_state):
        """スパイク設置位置を取得する。Guardフェーズは設置後を前提とするが、
        念のため未設置時（target_plant_pos）にもフォールバックする。"""
        planted_pos = game_state.get("planted_pos")
        if game_state.get("is_planted") and planted_pos:
            return (int(planted_pos[0]), int(planted_pos[1]))
        target_plant_pos = game_state.get("target_plant_pos")
        if target_plant_pos:
            return (int(target_plant_pos[0]), int(target_plant_pos[1]))
        return None

    def _resolve_active_defuser(self, game_state):
        """解除中（defuse_timer > 0）の敵キャラクターを返す。無ければNone。"""
        defuse_info = game_state.get("defender_defuse_info") or {}
        chars = game_state.get("chars", [])
        for defender_name, (defuse_timer, _required) in defuse_info.items():
            if defuse_timer and defuse_timer > 0:
                for c in chars:
                    if (
                        getattr(c, "name", None) == defender_name
                        and getattr(c, "is_alive", True)
                        and getattr(c, "team", None) == "D"
                    ):
                        return c
        return None

    def _team_guard_order(self, char, game_state):
        """割当優先順を決めるための、生存中の味方(自チーム)の登場順リスト。
        game_state["chars"] 内の順序はゲーム内で安定しているため、
        毎tick一貫した割当優先順として使える。
        """
        chars = game_state.get("chars", [])
        return [
            c for c in chars
            if getattr(c, "is_alive", True) and getattr(c, "team", None) == char.team
        ]

    def _compute_goals(self, char, game_state, grid):
        """全味方guardの誘導目標を計算する。緊急時(誰かが解除中)は
        解除者への接近を、それ以外は配置優先順位に基づく割当を返す。
        戻り値: {char_name: (goal_pos, tier)}  tier: 0/1/2=配置スロット, -1=緊急対応
        """
        team_chars = self._team_guard_order(char, game_state)

        defuser = self._resolve_active_defuser(game_state)
        if defuser is not None:
            defuser_pos = (int(defuser.pos[0]), int(defuser.pos[1]))
            return {c.name: (defuser_pos, -1) for c in team_chars}

        plant_pos = self._resolve_plant_pos(game_state)
        if plant_pos is None:
            # 設置地点が全く取得できない異常系：自分自身の位置に留まる
            return {c.name: ((int(c.pos[0]), int(c.pos[1])), 0) for c in team_chars}

        positions_ordered = [
            (c.name, (int(c.pos[0]), int(c.pos[1]))) for c in team_chars
        ]
        return _assign_goals(grid, plant_pos, self.formation_cells, positions_ordered)

    def _nearest_visible_enemy(self, grid, chars, my_team, from_pos, max_range=None):
        best_char, best_dist = None, None
        for c in chars:
            if not getattr(c, "is_alive", True) or getattr(c, "team", None) == my_team:
                continue
            enemy_pos = (int(c.pos[0]), int(c.pos[1]))
            dist = _chebyshev(from_pos, enemy_pos)
            if max_range is not None and dist > max_range:
                continue
            if not _has_los_walls_only(grid, from_pos, enemy_pos):
                continue
            if best_dist is None or dist < best_dist:
                best_char, best_dist = c, dist
        return best_char, best_dist

    def _team_effect_active(self, chars, my_team):
        """味方の誰かが敵にかけた blind/reveal が現在有効かどうか。"""
        for c in chars:
            if getattr(c, "team", None) == my_team or not getattr(c, "is_alive", True):
                continue
            if getattr(c, "blind_remaining", 0) > 0 or getattr(c, "reveal_remaining", 0) > 0:
                return True
        return False

    # ------------------------------------------------------------------
    # 観測構築（train_attacker_guard.py の GuardEnv._get_obs() と
    # 要素の順序・個数を完全一致させること。全45次元。）
    # ------------------------------------------------------------------
    def _build_obs(self, char, game_state, st, goal, tier):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        r, c = int(char.pos[0]), int(char.pos[1])

        plant_pos = self._resolve_plant_pos(game_state) or (r, c)
        pr, pc = plant_pos
        gr, gc = goal
        urgency = 1.0 if tier == -1 else 0.0

        obs = []
        height, width = grid.shape
        obs.append(r / max(1, height - 1))
        obs.append(c / max(1, width - 1))

        dist_to_plant = _chebyshev((r, c), (pr, pc))
        obs.append(min(1.0, dist_to_plant / DIST_NORM_MAX))
        obs.append(max(-1.0, min(1.0, (pr - r) / DIST_NORM_MAX)))
        obs.append(max(-1.0, min(1.0, (pc - c) / DIST_NORM_MAX)))

        dist_to_goal = _chebyshev((r, c), (gr, gc))
        obs.append(min(1.0, dist_to_goal / DIST_NORM_MAX))
        obs.append(max(-1.0, min(1.0, (gr - r) / DIST_NORM_MAX)))
        obs.append(max(-1.0, min(1.0, (gc - c) / DIST_NORM_MAX)))

        tier_onehot = [0.0, 0.0, 0.0]
        if tier in (0, 1, 2):
            tier_onehot[tier] = 1.0
        obs.extend(tier_onehot)

        obs.append(urgency)

        defuser = self._resolve_active_defuser(game_state)
        if defuser is not None:
            dpos = (int(defuser.pos[0]), int(defuser.pos[1]))
            ddist = _chebyshev((r, c), dpos)
            obs.append(min(1.0, ddist / DIST_NORM_MAX))
            obs.append(max(-1.0, min(1.0, (dpos[0] - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (dpos[1] - c) / DIST_NORM_MAX)))
            has_los_defuser = 1.0 if (
                _has_los_walls_only(grid, (r, c), plant_pos)
                and _has_los_walls_only(grid, (r, c), dpos)
            ) else 0.0
            obs.append(has_los_defuser)
        else:
            obs.extend([1.0, 0.0, 0.0, 0.0])

        obs.append(1.0 if _has_los_walls_only(grid, (r, c), plant_pos) else 0.0)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(grid, nr, nc)
            gdist = _chebyshev((nr, nc), (gr, gc))
            obs.append(1.0 if wall else min(1.0, gdist / DIST_NORM_MAX))

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        enemy_char, enemy_dist = self._nearest_visible_enemy(grid, chars, char.team, (r, c))
        if enemy_char is not None:
            er, ec = int(enemy_char.pos[0]), int(enemy_char.pos[1])
            obs.append(1.0)
            obs.append(max(-1.0, min(1.0, (er - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (ec - c) / DIST_NORM_MAX)))
            obs.append(min(1.0, enemy_dist / DIST_NORM_MAX))
            obs.append(min(1.0, getattr(enemy_char, "blind_remaining", 0) / max(1, BLIND_DURATION_TICKS)))
            obs.append(min(1.0, getattr(enemy_char, "reveal_remaining", 0) / max(1, REVEAL_DURATION_TICKS)))
        else:
            obs.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        total_charges = (
            getattr(char, "flash_charges", 0)
            + getattr(char, "smoke_charges", 0)
            + getattr(char, "recon_charges", 0)
        )
        obs.append(1.0 if total_charges > 0 else 0.0)
        for ability in ABILITY_TYPES:
            obs.append(1.0 if char.ability_name == ability else 0.0)

        obs.append(1.0 if self._team_effect_active(chars, char.team) else 0.0)

        obs.append(st["last_delta"][0])
        obs.append(st["last_delta"][1])
        obs.append(min(1.0, st["stuck"] / 10.0))

        detonate_timer = game_state.get("detonate_timer")
        if detonate_timer is None:
            detonate_timer = self.detonate_ticks
        remaining_ratio = 1.0 - min(1.0, float(detonate_timer) / max(1, self.detonate_ticks))
        obs.append(remaining_ratio)

        obs.append(0.0 if st["last_delta"] == (0.0, 0.0) else 1.0)

        return np.array(obs, dtype=np.float32)

    def _action_mask(self, char, grid):
        r, c = int(char.pos[0]), int(char.pos[1])
        mask = np.ones(N_ACTIONS, dtype=bool)
        for a, (dr, dc) in _MOVE_DELTA.items():
            if a == ACTION_STAY:
                continue
            if self._is_wall(grid, r + dr, c + dc):
                mask[a] = False

        total_charges = (
            getattr(char, "flash_charges", 0)
            + getattr(char, "smoke_charges", 0)
            + getattr(char, "recon_charges", 0)
        )
        if total_charges <= 0:
            mask[ACTION_ABILITY] = False

        return mask

    # ------------------------------------------------------------------
    # コントローラー本体
    # ------------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        st = self._get_char_state(char)

        goals = self._compute_goals(char, game_state, grid)
        goal, tier = goals.get(char.name, ((int(char.pos[0]), int(char.pos[1])), 0))

        obs = self._build_obs(char, game_state, st, goal, tier)
        mask = self._action_mask(char, grid)

        if (not self.greedy) and np.random.random() < self.epsilon:
            action = int(np.random.choice(np.flatnonzero(mask)))
        else:
            with torch.no_grad():
                state_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.policy_net(state_t).squeeze(0).cpu().numpy()
            q = np.where(mask, q, -1e9)
            action = int(np.argmax(q))

        r, c = int(char.pos[0]), int(char.pos[1])

        if action == ACTION_ABILITY:
            enemy_char, _ = self._nearest_visible_enemy(
                grid, chars, char.team, (r, c), max_range=ABILITY_RANGE
            )
            if enemy_char is not None:
                st["last_delta"] = (0.0, 0.0)
                st["stuck"] += 1
                target = (int(enemy_char.pos[0]), int(enemy_char.pos[1]))
                return list(char.pos), {"ability": char.ability_name, "target": target}
            # 射程内に有効な標的がいない場合、実チャージを無駄撃ちしないよう
            # STAY にフォールバックする。
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c]

        dr, dc = _MOVE_DELTA[action]
        nr, nc = r + dr, c + dc

        if action == ACTION_STAY or self._is_wall(grid, nr, nc):
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c]

        st["last_delta"] = (float(dr), float(dc))
        st["stuck"] = 0
        return [nr, nc]