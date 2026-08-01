"""learning_defender_search.py

Defender「search phase」用の推論コントローラー(プラント前限定)。
train_defender_search.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py への依存はなく、必要なLOS計算・行動マスク・観測構築は
このファイル内に複製する。game_core からは定数のみ参照する(ロジックは
参照しない)。

LearningDefenderAllAIController と同様、Defenderチーム全体で1つの
コントローラーインスタンスを共有する想定(重み共有Dueling DQN)。
decide_move はキャラクターごとに毎tick呼び出されるため、チーム共有メモリ
(team_memory)は「同じキャラクターが再度呼ばれたら次のtickに入った」と
みなして更新するタイミングを内部で管理する。

decide_move の戻り値は以下のいずれか:
    - next_pos (list[int, int])                        : 通常移動
    - (next_pos, {"ability": name, "target": (r, c)})   : アビリティ使用

このモデルはプラント前フェーズ専用。is_planted=True の場合は
このコントローラーの責務外なので、その場に留まる安全策のみ行う
(実運用では MultiRoleAttackerController 同様、上位のフェーズ切替側で
別コントローラーに委譲することを想定)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 31
ACTION_DIM = 10  # move_idx(0-4) * 2 + use_ability_flag(0/1)

SIGHTING_STALENESS_CAP = 30
ABILITY_RANGE = 8

DEFAULT_MODEL_PATH = "data/defender_search_data/dqn_defender_search_best.pt"


# ---------------------------------------------------------------------------
# Dueling DQN (train_defender_search.py と同一構造)
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
# 壁のみを考慮する(スモークによる遮蔽は考慮しない)。これは
# learning_attacker_retrieve.py の _has_los と同じ簡略化方針。
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
        self.last_seen_enemy = None  # {"pos": (r, c), "tick_ago": int}

    def reset(self):
        self.spike_pos = None
        self.last_seen_enemy = None

    def update(self, grid, my_team, chars):
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

        if visible_enemies:
            self.last_seen_enemy = {"pos": tuple(visible_enemies[0].pos), "tick_ago": 0}
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningDefenderSearchController:
    """プラント前フェーズ専用のDefender AI。

    Defenderチーム全員(最大5人)がこの1インスタンスを共有して呼び出される
    (learning_defender.LearningDefenderAllAIController と同じ運用形態)。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self.model = DefenderSearchDuelingDQN().to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            if verbose:
                print(f"[LearningDefenderSearchController] loaded: {model_path}")
        except Exception as exc:
            print(f"[LOAD ERROR] defender search model '{model_path}' の読込に失敗: {exc}")
        self.model.eval()

        self.team_memory = _TeamMemory()
        self._site_positions_cache = None
        self._processed_this_tick = set()

    # -- ラウンド開始時のリセット -----------------------------------------
    def reset_round(self):
        self.team_memory.reset()
        self._processed_this_tick.clear()
        # site_positions はマップ形状に依存するだけなのでラウンドをまたいで
        # キャッシュを保持して構わない(クリアしない)。

    def _maybe_advance_tick(self, char, grid, chars):
        """同じキャラクターが再び呼ばれたら新しいtickに入ったとみなし、
        チーム共有メモリを1回だけ更新する。"""
        if char.name in self._processed_this_tick:
            self._processed_this_tick.clear()
            self.team_memory.update(grid, char.team, chars)
        self._processed_this_tick.add(char.name)

    # -- 観測構築 ----------------------------------------------------------
    def _build_observation(self, char, game_state, site_positions):
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

        # 味方スモーク展開中か。game_state には smokes が含まれないため、
        # 代替として「自チームの誰かが直近でSMOKEチャージを消費済みか」を
        # 判定できないので、ここでは常に0とする(壁LOSのみの簡略化と同様、
        # 学習時のsmoke重複防止は事後の重み付けとして反映済みという前提)。
        obs[13] = 0.0

        if self.team_memory.spike_pos is not None:
            sp = self.team_memory.spike_pos
            obs[14] = 1.0
            obs[15] = (sp[0] - char.pos[0]) / height
            obs[16] = (sp[1] - char.pos[1]) / width

        if self.team_memory.last_seen_enemy is not None:
            ls = self.team_memory.last_seen_enemy
            obs[17] = 1.0
            obs[18] = (ls["pos"][0] - char.pos[0]) / height
            obs[19] = (ls["pos"][1] - char.pos[1]) / width
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

        round_timer = game_state.get("detonate_timer")
        # search phaseではdetonate_timerは未使用(プラント前)。
        # ここではラウンド経過情報を持たないため中立値(0.5)を入れる。
        obs[29] = 0.5
        obs[30] = 0.0

        return obs, visible_enemies

    # -- 行動マスク ---------------------------------------------------------
    def _action_mask(self, char, grid, chars):
        mask = np.ones(ACTION_DIM, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = {
            tuple(o.pos) for o in chars if o is not char and getattr(o, "is_alive", True)
        }

        for move_idx, (dr, dc) in enumerate(MOVES):
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

        self._maybe_advance_tick(char, grid, chars)

        obs, visible_enemies = self._build_observation(char, game_state, self._site_positions_cache)
        mask = self._action_mask(char, grid, chars)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(obs_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action_idx = int(torch.argmax(q_values).item())

        move_idx, ability_flag = divmod(action_idx, 2)
        use_ability = bool(ability_flag)
        move_offset = MOVES[move_idx]

        if self.verbose:
            print(
                f"[DEFENDER SEARCH] {char.name} pos={tuple(char.pos)} "
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