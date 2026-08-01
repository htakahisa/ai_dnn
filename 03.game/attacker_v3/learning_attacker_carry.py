"""learning_attacker_carry.py

train_attacker_carry.py で学習した Carry Phase 用 Dueling DQN モデルを使い、
スパイクキャリアーの移動・PLANT判断を行う推論用コントローラー。

【重要：ナビゲーション目標は game_state["target_plant_pos"] を使う】
学習側は、実ゲームの run_game.py init_round() と同じ選び方
（grid==2のセルからランダムに1点）で、エピソードごとに目標地点を決めて
そこまでの距離を観測・報酬に使っている。
推論側は、この target_plant_pos が既に game_state に含まれているので、
それをそのまま使うだけでよい（run_game.py 側の変更は不要）。
これにより、左右どちらのサイトが今ラウンドの目標かに関わらず、
学習時と同じ形で距離を計算できる。

【重要：優先プラント地点(5)は本番マップに存在しない】
学習は map_data_carry.py（grid==5あり）で行うが、本番の map_data.py には
grid==5 は存在しない。そのため優先地点は「gridの値」ではなく、
チェックポイントに保存された「座標のリスト」として扱う。
本番マップ上でその座標までのBFS距離を計算するだけで、
gridの値そのものには一切依存しない。

このコントローラーが構築する観測ベクトル・行動空間は、
train_attacker_carry.py の CarryEnv._get_obs() / ACTION_* と
完全に一致させる必要がある。ここがズレると学習結果が正しく反映されない。

run_game.py からは他の learning_attacker_*.py 系コントローラーと同様の
インターフェース（decide_move(char, game_state) -> next_pos または
(next_pos, "PLANT")）で呼び出される想定。
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from controllers import BaseController


# ---------------------------------------------------------------------------
# 行動定義（train_attacker_carry.py の CarryEnv と同一でなければならない）
# ---------------------------------------------------------------------------
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_PLANT = range(6)
N_ACTIONS = 6
_MOVE_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_STAY: (0, 0),
}

SITE_CELL_VALUE = 2  # 本番マップに存在する通常のプラント可能セル


# ---------------------------------------------------------------------------
# BFSによる距離マップ（指定した座標群を始点としたマルチソースBFS）
# train_attacker_carry.py の _build_distance_map_from_coords と同一ロジック。
# ---------------------------------------------------------------------------
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


def _cells_with_value(grid, value):
    return list(zip(*np.where(grid == value)))


# ---------------------------------------------------------------------------
# Dueling DQN（train_attacker_carry.py と同一アーキテクチャ）
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


class LearningAttackerCarryController(BaseController):
    """Carry Phaseの学習済みモデルで、スパイクキャリアーの移動・PLANTを決定する。

    キャリアーは常時発動パッシブ「ハンター」を持つ想定のため、
    アビリティ判断は行わない（本コントローラーはcarry役にのみ使用する）。
    """

    def __init__(
        self,
        model_path,
        device=None,
        greedy=True,
        epsilon=0.0,
        max_ticks=90,
        plant_required_ticks=4,
        verbose=False,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.greedy = greedy
        self.epsilon = epsilon
        self.max_ticks = max_ticks
        self.plant_required_ticks = plant_required_ticks
        self.verbose = verbose

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Carryモデルが見つかりません: {model_path}")

        # 自分たちが train_attacker_carry.py で生成した信頼済みチェックポイント
        # なので weights_only=False で読み込む（PyTorch 2.6+ のデフォルト変更対応）。
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        obs_dim = checkpoint["obs_dim"]
        n_actions = checkpoint.get("n_actions", N_ACTIONS)

        self.policy_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()

        # 学習マップ(map_data_carry.py)上での優先地点座標。
        # 本番マップ(map_data.py)には grid==5 が存在しないため、
        # この座標をそのまま「目的地の集合」として使う。
        self.priority_cells = list(checkpoint.get("priority_cells") or [])
        self.has_priority_cells = bool(checkpoint.get("has_priority_cells", False)) and bool(
            self.priority_cells
        )

        if self.verbose:
            print(
                f"[LearningAttackerCarryController] モデル読込完了: {model_path} "
                f"(obs_dim={obs_dim}, episode={checkpoint.get('episode')}, "
                f"success_rate={checkpoint.get('success_rate')}, "
                f"has_priority_cells={self.has_priority_cells}, "
                f"priority_cells={self.priority_cells})"
            )

        # gridのバイト列 -> distance_map のキャッシュ（同一マップなら再計算しない）
        self._target_dist_cache = {}
        self._priority_dist_cache = {}
        self._norm_max_dist_cache = {}
        # キャラクター名ごとの episode 内状態（tick数・移動履歴・停滞カウント）
        self._char_state = {}

    # ------------------------------------------------------------------
    # ラウンド開始時にrun_game.pyから呼ばれる（hasattr判定で自動検出される）
    # ------------------------------------------------------------------
    def reset_round(self):
        self._char_state.clear()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _get_target_dist_map(self, grid, target_pos):
        key = (grid.tobytes(), tuple(target_pos))
        cached = self._target_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_from_coords(grid, [tuple(target_pos)])
            self._target_dist_cache[key] = cached
        return cached

    def _get_norm_max_dist(self, grid):
        """train_attacker_carry.py の norm_max_dist と同じ正規化基準
        （全サイトセルへの最短距離の最大値）を、本番マップ上で計算する。
        target座標によらず固定のスケールにするための基準値。
        """
        key = grid.tobytes()
        cached = self._norm_max_dist_cache.get(key)
        if cached is None:
            site_cells = _cells_with_value(grid, SITE_CELL_VALUE)
            dist_map = _build_distance_map_from_coords(grid, site_cells)
            finite = dist_map[np.isfinite(dist_map)]
            cached = float(finite.max()) if finite.size else 1.0
            self._norm_max_dist_cache[key] = cached
        return cached

    def _get_priority_dist_map(self, grid):
        """優先地点への距離マップ。

        学習時に優先地点ありで学習されたモデル(has_priority_cells=True)の場合、
        本番マップ上で priority_cells の座標までのBFS距離を計算する。
        優先地点なしで学習されたモデルの場合は、通常サイト距離マップと
        同じ値を返す（学習時と同じ分布になるよう合わせる）。
        """
        if not self.has_priority_cells:
            site_cells = _cells_with_value(grid, SITE_CELL_VALUE)
            key = (grid.tobytes(), "fallback")
            cached = self._priority_dist_cache.get(key)
            if cached is None:
                cached = _build_distance_map_from_coords(grid, site_cells)
                self._priority_dist_cache[key] = cached
            return cached

        key = grid.tobytes()
        cached = self._priority_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_from_coords(grid, self.priority_cells)
            self._priority_dist_cache[key] = cached
        return cached

    def _get_char_state(self, char):
        return self._char_state.setdefault(
            char.name,
            {"tick": 0, "last_delta": (0.0, 0.0), "stuck": 0},
        )

    @staticmethod
    def _is_wall(grid, r, c):
        height, width = grid.shape
        if not (0 <= r < height and 0 <= c < width):
            return True
        return grid[r, c] == 1

    @staticmethod
    def _dist_at(dist_map, max_finite, pos, grid):
        r, c = pos
        height, width = grid.shape
        if not (0 <= r < height and 0 <= c < width):
            return max_finite
        d = dist_map[r, c]
        return max_finite if not np.isfinite(d) else float(d)

    def _is_priority_pos(self, pos):
        return self.has_priority_cells and tuple(pos) in set(self.priority_cells)

    def _resolve_target_pos(self, grid, game_state):
        """今ラウンドのナビゲーション目標を決める。

        実ゲームでは run_game.py の init_round() が target_plant_pos を
        ラウンド開始時にランダムに1点選んでおり、これが game_state に
        含まれているので、それをそのまま使う。万一含まれていない場合
        （デフォルト値None等）は、最も近いサイトセルにフォールバックする。
        """
        target = game_state.get("target_plant_pos")
        if target is not None:
            return (int(target[0]), int(target[1]))

        # フォールバック：目標が指定されていない場合は最寄りのサイトへ。
        height, width = grid.shape
        site_cells = _cells_with_value(grid, SITE_CELL_VALUE)
        if not site_cells:
            return (0, 0)
        dist_map = self._get_norm_max_dist  # 未使用だが意図の明示のため参照
        nearest = min(
            site_cells,
            key=lambda cell: abs(cell[0]) + abs(cell[1]),
        )
        return nearest

    def _build_obs(self, char, game_state, st):
        grid = game_state["grid"]
        height, width = grid.shape
        r, c = int(char.pos[0]), int(char.pos[1])

        target_pos = self._resolve_target_pos(grid, game_state)
        target_dist_map = self._get_target_dist_map(grid, target_pos)
        norm_max_dist = self._get_norm_max_dist(grid)

        def target_distance(pos):
            return self._dist_at(target_dist_map, norm_max_dist, pos, grid)

        obs = []
        obs.append(r / max(1, height - 1))
        obs.append(c / max(1, width - 1))
        obs.append(min(1.0, target_distance((r, c)) / norm_max_dist))

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(grid, nr, nc)
            obs.append(1.0 if wall else 0.0)
            obs.append(1.0 if wall else min(1.0, target_distance((nr, nc)) / norm_max_dist))

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        plant_timer = getattr(char, "plant_timer", 0)
        obs.append(1.0 if grid[r, c] == SITE_CELL_VALUE else 0.0)
        obs.append(1.0 if plant_timer > 0 else 0.0)
        obs.append(plant_timer / max(1, self.plant_required_ticks))
        obs.append(1.0 - min(1.0, st["tick"] / max(1, self.max_ticks)))
        obs.append(st["last_delta"][0])
        obs.append(st["last_delta"][1])
        obs.append(min(1.0, st["stuck"] / 10.0))

        # --- 優先プラント地点関連の特徴量（座標ベース、targetとは独立）---
        priority_dist_map = self._get_priority_dist_map(grid)
        finite_p = priority_dist_map[np.isfinite(priority_dist_map)]
        max_finite_p = float(finite_p.max()) if finite_p.size else 1.0

        def priority_distance(pos):
            return self._dist_at(priority_dist_map, max_finite_p, pos, grid)

        obs.append(min(1.0, priority_distance((r, c)) / max_finite_p))
        obs.append(1.0 if self._is_priority_pos((r, c)) else 0.0)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(grid, nr, nc)
            obs.append(1.0 if wall else min(1.0, priority_distance((nr, nc)) / max_finite_p))

        return np.array(obs, dtype=np.float32)

    def _action_mask(self, char, grid):
        r, c = int(char.pos[0]), int(char.pos[1])
        mask = np.ones(N_ACTIONS, dtype=bool)
        for a, (dr, dc) in _MOVE_DELTA.items():
            if a == ACTION_STAY:
                continue
            if self._is_wall(grid, r + dr, c + dc):
                mask[a] = False
        return mask

    # ------------------------------------------------------------------
    # コントローラー本体
    # ------------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        st = self._get_char_state(char)
        st["tick"] += 1

        obs = self._build_obs(char, game_state, st)
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

        if action == ACTION_PLANT:
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            # battle_logic.py の move_character が現在座標(r, c)を見て
            # grid[r, c] == 2 かどうかで判定するため、next_posは現在地でよい。
            return [r, c], "PLANT"

        dr, dc = _MOVE_DELTA[action]
        nr, nc = r + dr, c + dc

        if action == ACTION_STAY or self._is_wall(grid, nr, nc):
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c]

        st["last_delta"] = (float(dr), float(dc))
        st["stuck"] = 0
        return [nr, nc]