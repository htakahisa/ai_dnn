"""gc_v1/learning_attacker_escort_gc.py

固定チーム(Xdll/Syouta/Absol/eKo/SugarZ3ro)専用の
Attacker Carry Phase「escort(護衛)」推論コントローラー。

train_attacker_escort.py(gc_v1版)で学習した、Escort役4体用の
Dueling DQNモデル(dict形式チェックポイント)を使い、各escortキャラクターの
移動・アビリティ判断を行う。

【4体は重み共有】
train_attacker_escort.py側は4体のescortが同一ネットワークの重みを
共有して学習している(パラメータ共有方式)。そのため、run_game.py側でも
carry以外の4キャラクターすべてに「同じLearningAttackerEscortGCController
インスタンス」を割り当てる想定である(インスタンスは1つ、decide_moveは
キャラクターごとに呼ばれる)。

completely self-contained: run_game.py / controllers.py / battle_logic.py /
abilities_los.py は一切importしない。必要なロジックはすべてこのファイル内に
複製する。run_game.py / controllers.py は変更しない。

【観測ベクトルはtrain_attacker_escort.py(gc_v1版)のEscortEnv._get_obs()と
完全に一致させる必要がある(全41次元)。ここがズレると学習結果が正しく
反映されない。汎用版(旧learning_attacker_escort.py)からの変更点は、
ABILITY_TYPESに"HUNT"(タイガー役)が加わったことによる
アビリティonehotの3種→4種化(OBS_DIM: 36→41)のみ。】

【本番環境との差異・既知の制約】
1. キャリアーの「進むべき方向」予測
   学習環境では、キャラクター同士の衝突を考慮しない固定BFS経路
   (壁のみを障害物とした経路)をキャリアーの行動基準にしていた。
   推論側もこれに合わせ、味方・敵の位置を無視した「壁のみのBFS勾配」で
   キャリアーの理想進行方向を毎tick再計算する。これにより
   「自分がその理想進行方向のマスに立っているかどうか」を
   ブロック中フラグとして使える。

2. スモークによる射線遮蔽は考慮できない
   battle_logic.pyのmove_characterが渡すgame_stateには、現在有効な
   スモークの情報が含まれていないため、視界判定は壁のみを考慮した
   Bresenham判定になる(学習環境よりやや楽観的)。

3. アビリティの発動判定
   char.ability_name / char.flash_charges / char.smoke_charges /
   char.recon_chargesなど、実際のCharacterオブジェクトが持つ値を
   そのまま使う。HUNT(タイガー)役はgame_core.pyの仕様上これらが
   全て0で初期化されるため、total_charges<=0判定で自動的に
   アビリティ行動がマスクされる(追加分岐は不要)。射程内に有効な
   標的がいない場合は、実際のアビリティチャージを無駄撃ちしないよう
   STAYにフォールバックする。

run_game.pyからは他のlearning_attacker_*.py系コントローラーと同様の
インターフェース(decide_move(char, game_state) -> next_pos または
(next_pos, {"ability": ..., "target": (r, c)}))で呼び出される想定。
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from character_stats_gc import (
    CHARACTER_TABLE as GC_STATS_TABLE,
    GC_ROSTER_ORDER,
)

# ---------------------------------------------------------------------------
# 行動定義(train_attacker_escort.py の EscortEnv と同一でなければならない)
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
# HUNT(タイガー役)を含む4種。HUNTはアビリティ行動を持たないため
# total_charges<=0判定で自動的にマスクされる(game_core.pyの仕様上、
# タイガー役はflash/smoke/recon_chargesが全て0で初期化されるため)。
ABILITY_TYPES = ("FLASH", "RECON", "SMOKE", "HUNT")
ABILITY_RANGE = 6

DIST_BAND_MIN = 2
DIST_BAND_MAX = 7
DIST_NORM_MAX = 15.0

OBS_DIM = 41  # train_attacker_escort.py(gc_v1版) EscortEnv._obs_dim() と一致


# ---------------------------------------------------------------------------
# 汎用ヘルパー(train_attacker_escort.py と同一ロジック)
# ---------------------------------------------------------------------------
def _chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _line_cells(p1, p2):
    """Bresenham法で2点間のセル列を返す(abilities_los.py と同一ロジック)。"""
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
    """壁のみを考慮した射線判定。スモークはgame_stateから参照できないため
    考慮しない(既知の制約。学習環境よりやや楽観的な視界判定になる)。"""
    for r, c in _line_cells(p1, p2):
        if grid[r, c] == 1:
            return False
    return True


def _build_distance_map_walls_only(grid, source_cells):
    """指定座標群を始点とした、壁のみを障害物としたマルチソースBFS距離マップ。
    キャラクター同士の占有は考慮しない(学習環境の固定経路と同じ前提)。
    """
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


# ---------------------------------------------------------------------------
# Dueling DQN(train_attacker_escort.py と同一アーキテクチャ)
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


class LearningAttackerEscortGCController:
    """Escort Phase(gc_v1固定チーム版)の学習済みモデルで、護衛
    キャラクター4体の移動・アビリティ使用を決定する。4体は同一インスタンス
    (同一ネットワーク)を共有する想定。
    """

    def __init__(
        self,
        model_path,
        device=None,
        greedy=True,
        epsilon=0.0,
        max_ticks=90,
        verbose=False,
    ):
        self.device = device or torch.device("cpu")
        self.greedy = greedy
        self.epsilon = epsilon
        self.max_ticks = max_ticks
        self.verbose = verbose

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Escortモデルが見つかりません: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
        n_actions = int(checkpoint.get("n_actions", N_ACTIONS))

        if obs_dim != OBS_DIM or n_actions != N_ACTIONS:
            raise ValueError(
                f"チェックポイントの観測/行動空間がこのコントローラーと不一致です: "
                f"obs_dim={obs_dim}(期待値{OBS_DIM}) n_actions={n_actions}(期待値{N_ACTIONS})。"
                f"train_attacker_escort.pyのバージョンが古い可能性があります。"
            )

        self.policy_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()

        if self.verbose:
            print(
                f"[LearningAttackerEscortGCController] モデル読込完了: {model_path} "
                f"(obs_dim={obs_dim}, episode={checkpoint.get('episode')}, "
                f"success_rate={checkpoint.get('success_rate')}, "
                f"roster_order={checkpoint.get('roster_order')}, "
                f"spike_holder_default={checkpoint.get('spike_holder_default')})"
            )

        # goal座標(r, c) -> 壁のみBFS距離マップ のキャッシュ
        self._goal_dist_cache = {}
        # carry_pos(r, c) -> 壁のみBFS距離マップ のキャッシュ
        # (escort自身からキャリアーまでの距離・方向をBFSベースで測るため)
        self._carry_dist_cache = {}
        # キャラクター名ごとのラウンド内状態(tick数・移動履歴・停滞カウント)
        self._char_state = {}

    # ------------------------------------------------------------------
    # ラウンド開始時にrun_game.pyから呼ばれる(hasattr判定で自動検出される)
    # ------------------------------------------------------------------
    def reset_round(self):
        self._char_state.clear()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
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

    def _get_goal_dist_map(self, grid, goal):
        key = (grid.tobytes(), tuple(goal))
        cached = self._goal_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_walls_only(grid, [tuple(goal)])
            self._goal_dist_cache[key] = cached
        return cached

    def _get_carry_dist_map(self, grid, carry_pos):
        """carry_posを起点とした壁のみBFS距離マップ(キャッシュ付き)。
        escort自身からキャリアーまでの距離・方向はチェビシェフ距離ではなく
        こちらを使う。"""
        key = (grid.tobytes(), tuple(carry_pos))
        cached = self._carry_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_walls_only(grid, [tuple(carry_pos)])
            self._carry_dist_cache[key] = cached
        return cached

    def _resolve_carry_and_goal(self, char, game_state):
        """護衛対象(キャリアー)の位置と、その目的地(goal)を決める。

        優先順位:
          1. 生存している味方でスパイクを持っている者 -> その位置。
             goalはis_plantedならplanted_pos、そうでなければtarget_plant_pos。
          2. 誰もスパイクを持っていない場合(設置済み) -> planted_posを
             疑似的なキャリアー位置として扱う(サイト周辺の護衛に切り替わる)。
          3. スパイクが地面に落ちている場合 -> spike_posを疑似的な
             キャリアー位置として扱う(回収を待つ形で近くに集まる)。
          4. どれも取得できない場合 -> target_plant_pos、それも無ければ
             自分自身の位置(実質、何もしない)。
        """
        chars = game_state.get("chars", [])
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")
        target_plant_pos = game_state.get("target_plant_pos")
        spike_pos = game_state.get("spike_pos")

        carrier = next(
            (
                c for c in chars
                if getattr(c, "is_alive", True)
                and getattr(c, "team", None) == char.team
                and getattr(c, "has_spike", False)
            ),
            None,
        )

        if carrier is not None:
            carry_pos = tuple(int(v) for v in carrier.pos)
            goal = tuple(planted_pos) if is_planted and planted_pos else (
                tuple(target_plant_pos) if target_plant_pos else carry_pos
            )
            return carry_pos, goal

        if is_planted and planted_pos:
            pos = tuple(int(v) for v in planted_pos)
            return pos, pos

        if spike_pos:
            pos = tuple(int(v) for v in spike_pos)
            return pos, pos

        if target_plant_pos:
            pos = tuple(int(v) for v in target_plant_pos)
            return pos, pos

        pos = tuple(int(v) for v in char.pos)
        return pos, pos

    def _predict_carry_next_step(self, grid, carry_pos, goal):
        """キャリアーの理想進行方向(他キャラクターの占有を無視した、
        壁のみのBFS勾配)を予測する。他エージェントを避けないため、
        「今このマスに立っていたらキャリアーの進路を塞いでいる」
        という判定にそのまま使える。
        """
        if carry_pos == goal:
            return carry_pos

        dist_map = self._get_goal_dist_map(grid, goal)
        height, width = grid.shape
        r, c = carry_pos
        best_cell = carry_pos
        best_dist = dist_map[r, c] if 0 <= r < height and 0 <= c < width else np.inf

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if grid[nr, nc] == 1:
                continue
            d = dist_map[nr, nc]
            if np.isfinite(d) and d < best_dist:
                best_dist = d
                best_cell = (nr, nc)

        return best_cell

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
        """味方の誰かが敵にかけたblind/revealが現在有効かどうか。
        スモークの有無はgame_stateから取得できないため考慮しない
        (既知の制約)。
        """
        for c in chars:
            if getattr(c, "team", None) == my_team or not getattr(c, "is_alive", True):
                continue
            if getattr(c, "blind_remaining", 0) > 0 or getattr(c, "reveal_remaining", 0) > 0:
                return True
        return False

    # ------------------------------------------------------------------
    # 観測構築(train_attacker_escort.py(gc_v1版)の
    # EscortEnv._get_obs()と要素の順序・個数を完全一致させること。全41次元。)
    # ------------------------------------------------------------------
    def _build_obs(self, char, game_state, st):
        grid = game_state["grid"]
        height, width = grid.shape
        chars = game_state.get("chars", [])
        r, c = int(char.pos[0]), int(char.pos[1])

        carry_pos, goal = self._resolve_carry_and_goal(char, game_state)
        cr, cc = carry_pos
        next_step = self._predict_carry_next_step(grid, carry_pos, goal)
        carry_dist_map = self._get_carry_dist_map(grid, carry_pos)

        obs = []
        obs.append(r / max(1, height - 1))
        obs.append(c / max(1, width - 1))

        # escort自身からキャリアーまでの距離・方向はBFS実距離ベース
        # (チェビシェフ距離は壁を無視するため、曲がった通路で
        # 実際の経路と逆方向を指してしまうことがある)
        raw_dist = carry_dist_map[r, c]
        dist_to_carry = raw_dist if np.isfinite(raw_dist) else DIST_NORM_MAX
        obs.append(min(1.0, dist_to_carry / DIST_NORM_MAX))

        # 方向成分も座標の単純差分ではなく、BFS距離を最も縮める方向を使う
        best_dr, best_dc, best_d = 0, 0, dist_to_carry
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not self._is_wall(grid, nr, nc):
                nd = carry_dist_map[nr, nc]
                if np.isfinite(nd) and nd < best_d:
                    best_d = nd
                    best_dr, best_dc = dr, dc
        obs.append(float(best_dr))
        obs.append(float(best_dc))

        # キャリアーの進行方向(次の理想セルへの差分)
        obs.append(float(np.sign(next_step[0] - cr)))
        obs.append(float(np.sign(next_step[1] - cc)))

        # 壁フラグ(4方向)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        # 各方向に動いた場合のキャリアーまでのBFS距離勾配
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(grid, nr, nc)
            if wall:
                obs.append(1.0)
            else:
                gdist = carry_dist_map[nr, nc]
                gdist = gdist if np.isfinite(gdist) else DIST_NORM_MAX
                obs.append(min(1.0, gdist / DIST_NORM_MAX))

        # 斜め壁フラグ
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        # 隣接4方向に生存中の味方escortがいるか(train_attacker_escort.pyと一致させる)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            occupied_by_ally = any(
                getattr(oc, "is_alive", True)
                and getattr(oc, "team", None) == char.team
                and oc is not char
                and not getattr(oc, "has_spike", False)
                and (int(oc.pos[0]), int(oc.pos[1])) == (nr, nc)
                for oc in chars
            )
            obs.append(1.0 if occupied_by_ally else 0.0)

        # 距離帯の逸脱量
        if dist_to_carry < DIST_BAND_MIN:
            band_dev = (dist_to_carry - DIST_BAND_MIN) / DIST_NORM_MAX
        elif dist_to_carry > DIST_BAND_MAX:
            band_dev = (dist_to_carry - DIST_BAND_MAX) / DIST_NORM_MAX
        else:
            band_dev = 0.0
        obs.append(band_dev)

        # 最寄りの視認可能な敵
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

        # 自分のアビリティ状態(実際のCharacterオブジェクトの値をそのまま使う)。
        # HUNT(タイガー)役はflash/smoke/recon_chargesが常に0なので、
        # total_charges<=0となり自然に「未使用フラグ=0」相当の扱いになる
        # (train_attacker_escort.py側でescort_ability_used初期値をTrue相当に
        # している挙動と一致する)。
        total_charges = (
            getattr(char, "flash_charges", 0)
            + getattr(char, "smoke_charges", 0)
            + getattr(char, "recon_charges", 0)
        )
        obs.append(1.0 if total_charges > 0 else 0.0)
        for ability in ABILITY_TYPES:
            obs.append(1.0 if char.ability_name == ability else 0.0)

        # チーム状況：誰かの効果(blind/reveal)が現在有効か
        obs.append(1.0 if self._team_effect_active(chars, char.team) else 0.0)

        obs.append(st["last_delta"][0])
        obs.append(st["last_delta"][1])
        obs.append(min(1.0, st["stuck"] / 10.0))
        obs.append(1.0 - min(1.0, st["tick"] / max(1, self.max_ticks)))
        # 自分が現在、キャリアーの理想進行先セルに立っているか(＝塞いでいるか)
        obs.append(1.0 if (r, c) == next_step and (r, c) != (cr, cc) else 0.0)

        obs_arr = np.array(obs, dtype=np.float32)
        assert obs_arr.shape[0] == OBS_DIM, (
            f"観測次元がOBS_DIM({OBS_DIM})と不一致: {obs_arr.shape[0]}。"
            f"train_attacker_escort.pyとのズレを確認してください。"
        )
        return obs_arr

    def _action_mask(self, char, grid, chars):
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
            # HUNT(タイガー)役もここで自動的にマスクされる(常にtotal_charges==0のため)。
            mask[ACTION_ABILITY] = False
        else:
            # チャージがあっても、射程内に有効な標的がいなければABILITYは
            # マスクする。学習済みネットワークがABILITYを選び続けて
            # 実質STAYのままブロックし続けるのを防ぐ。
            enemy_char, _ = self._nearest_visible_enemy(
                grid, chars, char.team, (r, c), max_range=ABILITY_RANGE
            )
            if enemy_char is None:
                mask[ACTION_ABILITY] = False

        return mask

    # ------------------------------------------------------------------
    # コントローラー本体
    # ------------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        st = self._get_char_state(char)
        st["tick"] += 1

        obs = self._build_obs(char, game_state, st)
        mask = self._action_mask(char, grid, chars)

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
            # STAYにフォールバックする。
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