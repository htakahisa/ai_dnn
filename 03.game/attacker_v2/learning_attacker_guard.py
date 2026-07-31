# learning_attacker_guard.py
"""
train_attacker_guard.py で学習したモデルを使い、実ゲーム(run_game.py)内で
guardフェーズ(プラント後)のアタッカー全員を動かす推論用コントローラー。

MultiRoleAttackerController から `if game_state.get("is_planted"): self.guard_controller.decide_move(...)`
という形で、生存アタッカー1体ごとに毎tick呼ばれる想定。全員が同一の共有モデル
(パラメータ共有)を使うため、内部で味方全体の状態を都度観測に組み込む。

💡簡略化している点:
- 射撃はロジック側(process_battle)が自動で解決するため、このコントローラーは
  「移動」と「アビリティ発動(方向のみ)」だけを決定する。
- 学習環境(GuardMultiEnv)では detonate_timer を観測に含めていたが、
  現行の run_game.py の game_state にはこの値が渡ってきていない。
  暫定的に DETONATE_TICKS 固定値へフォールバックするが、精度を上げたい場合は
  run_game.py 側で game_state に "detonate_timer": self.detonate_timer を追加すること。
- 敵(Defender)は視認できた中から「解除中(busy)の相手」を最優先、
  次に「最も近い相手」を選んで観測に使う(学習環境は単体のbotのみを想定していたため)。
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from controllers import BaseController

# ===========================================================================
# 学習時(train_attacker_guard.py)と一致させる定数
# ===========================================================================
MAX_GUARDS = 3
NUM_PERIMETER_SLOTS = 8
NUM_ENTRANCE_SLOTS = 4
NUM_TARGET_SLOTS = NUM_PERIMETER_SLOTS + NUM_ENTRANCE_SLOTS
OBS_DIM = 93
N_ACTIONS = 5
ABILITY_TYPES = ["flash", "smoke", "recon"]

GUARD_MIN_DIST = 2
GUARD_MAX_DIST = 5
GUARD_DEFUSE_REQUIRED = 6
DETONATE_TICKS = 45  # 💡 run_game.py の VisualFPSBattle.detonate_timer 初期値と一致させる

ENTRANCE_MIN_DIST = 2
ENTRANCE_MAX_DIST = 5
ENTRANCE_MIN_SEPARATION = 3

PERIMETER_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# ===========================================================================
# 共通ヘルパー(train_attacker_guard.py と同じロジックを複製)
# ===========================================================================
def bfs_distances(target, grid):
    height, width = grid.shape
    dist = np.full((height, width), np.inf)
    tr, tc = target
    dist[tr, tc] = 0
    q = deque([(tr, tc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def split_site_components(cells):
    cells_set = set(cells)
    visited = set()
    components = []
    for cell in cells:
        if cell in visited:
            continue
        comp = []
        queue = deque([cell])
        visited.add(cell)
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        components.append(comp)
    return components


def multi_source_bfs(site_cells, grid):
    height, width = grid.shape
    dist = np.full((height, width), np.inf)
    q = deque()
    for (r, c) in site_cells:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def find_corner_cells(grid):
    height, width = grid.shape
    corners = []
    for r in range(height):
        for c in range(width):
            if grid[r, c] == 1:
                continue
            north = grid[r - 1, c] == 1 if r - 1 >= 0 else True
            south = grid[r + 1, c] == 1 if r + 1 < height else True
            west = grid[r, c - 1] == 1 if c - 1 >= 0 else True
            east = grid[r, c + 1] == 1 if c + 1 < width else True
            pairs = [(north, west), (north, east), (south, west), (south, east)]
            if any(a and b for a, b in pairs):
                corners.append((r, c))
    return corners


def select_entrance_cells(grid, dist_map, corner_cells, min_dist, max_dist, num_points, min_separation):
    candidates = []
    for (r, c) in corner_cells:
        d = dist_map[r, c]
        if np.isfinite(d) and min_dist <= d <= max_dist:
            candidates.append((r, c))
    # 💡学習時はエピソード毎にshuffleしていたが、推論時は同じマップなら常に同じ並びに
    # なるよう固定順(座標順)にする。学習時と完全一致はしないが、対象集合自体は同一。
    candidates.sort()

    selected = []
    for pos in candidates:
        if all(max(abs(pos[0] - s[0]), abs(pos[1] - s[1])) >= min_separation for s in selected):
            selected.append(pos)
        if len(selected) >= num_points:
            break
    return selected


# ===========================================================================
# Dueling DQN構造(train_attacker_guard.py と同一)
# ===========================================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, action_dim)

    def forward(self, x):
        h = self.shared(x)
        value = self.value_head(h)
        advantage = self.advantage_head(h)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


def masked_argmax(q_values, mask):
    if mask.sum() == 0:
        return int(np.argmax(q_values))
    masked = np.where(mask > 0, q_values, -np.inf)
    return int(np.argmax(masked))


def masked_softmax_action(q_values, mask, temperature=0.5):
    if mask.sum() == 0:
        masked = q_values
    else:
        masked = np.where(mask > 0, q_values, -np.inf)
    probs = np.exp((masked - np.max(masked)) / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ===========================================================================
# 推論用コントローラー
# ===========================================================================
class LearningAttackerGuardController(BaseController):
    def __init__(self, model_path, greedy=False, device="cpu", temperature=0.5):
        super().__init__()
        self.device = torch.device(device)
        self.model = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.greedy = greedy
        self.temperature = temperature

        # ---- マップ依存の事前計算結果(初回呼び出し時にキャッシュ) ----
        self._cache_grid_id = None
        self.site_components = None
        self.site_cell_sets = None
        self.site_dist_maps = None
        self.entrance_candidates = None
        self.corner_cells = None

        # ---- ラウンド内で保持する状態(reset_roundでクリア) ----
        self.last_action = {}          # char.name -> 直前のaction(int)
        self.pending_move_target = {}  # 使わないが、他コントローラーとのインターフェース統一のため保持

    def reset_round(self):
        self.last_action.clear()

    # -----------------------------------------------------------------
    # マップ依存データの初期化(グリッドが変わらない限り一度だけ計算)
    # -----------------------------------------------------------------
    def _ensure_map_cache(self, grid):
        grid_id = id(grid)
        if self._cache_grid_id == grid_id:
            return
        self._cache_grid_id = grid_id

        self.corner_cells = find_corner_cells(grid)
        plant_cells = list(zip(*np.where(grid == 2)))
        self.site_components = split_site_components(plant_cells) if plant_cells else []
        self.site_cell_sets = [set(comp) for comp in self.site_components]
        self.site_dist_maps = [multi_source_bfs(comp, grid) for comp in self.site_components]
        self.entrance_candidates = [
            select_entrance_cells(
                grid, dmap, self.corner_cells,
                ENTRANCE_MIN_DIST, ENTRANCE_MAX_DIST, NUM_ENTRANCE_SLOTS, ENTRANCE_MIN_SEPARATION,
            )
            for dmap in self.site_dist_maps
        ]

    def _nearest_site_index(self, planted_pos):
        if not self.site_dist_maps:
            return None
        dists = [dmap[planted_pos[0], planted_pos[1]] for dmap in self.site_dist_maps]
        return int(np.argmin(dists))

    # -----------------------------------------------------------------
    # ターゲットセル(周囲8マス + 通路入口)の算出
    # -----------------------------------------------------------------
    def _build_target_positions(self, grid, planted_pos, site_index):
        target_positions = [None] * NUM_TARGET_SLOTS
        for i, (dr, dc) in enumerate(PERIMETER_OFFSETS):
            r, c = planted_pos[0] + dr, planted_pos[1] + dc
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1:
                target_positions[i] = (r, c)

        entrances = self.entrance_candidates[site_index] if site_index is not None else []
        for j in range(NUM_ENTRANCE_SLOTS):
            slot = NUM_PERIMETER_SLOTS + j
            if j < len(entrances):
                target_positions[slot] = entrances[j]
        return target_positions

    # -----------------------------------------------------------------
    # 敵(Defender)の選定: 視認できる中から「解除中」優先、次点「最近接」
    # -----------------------------------------------------------------
    def _select_enemy(self, self_pos, grid, chars, defender_defuse_info):
        visible = [
            d for d in chars
            if d.team == "D" and d.is_alive
            and self.has_line_of_sight(self_pos, tuple(d.pos), grid)
        ]
        if not visible:
            return None

        def defuse_progress(d):
            info = defender_defuse_info.get(d.name)
            return info[0] if info else 0

        visible.sort(
            key=lambda d: (
                -defuse_progress(d),  # 解除タイマーが進んでいる相手を優先
                max(abs(d.pos[0] - self_pos[0]), abs(d.pos[1] - self_pos[1])),  # 次に近い相手
            )
        )
        return visible[0]

    # -----------------------------------------------------------------
    # 観測構築(GuardMultiEnv._get_obs_for と同じ並びで再現)
    # -----------------------------------------------------------------
    def _build_observation(self, char, game_state, target_positions, planted_pos, site_index):
        grid = game_state["grid"]
        chars = game_state["chars"]
        height, width = grid.shape
        pr, pc = char.pos

        base = [pr / (height - 1), pc / (width - 1)]
        walls = [
            0.0 if 0 <= pr + dr < height and 0 <= pc + dc < width and grid[pr + dr, pc + dc] != 1 else 1.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

        last_onehot = [0.0] * N_ACTIONS
        prev_action = self.last_action.get(char.name)
        if prev_action is not None:
            last_onehot[prev_action] = 1.0

        site_max_dist = max(height, width) * 2
        if site_index is not None:
            d_site = self.site_dist_maps[site_index][pr, pc]
        else:
            d_site = np.inf
        dist_to_site = [d_site / site_max_dist if np.isfinite(d_site) else 1.0]

        chebyshev_to_plant = max(abs(pr - planted_pos[0]), abs(pc - planted_pos[1]))
        in_range_flag = [1.0 if GUARD_MIN_DIST <= chebyshev_to_plant <= GUARD_MAX_DIST else 0.0]

        teammates = [
            c for c in chars
            if c.team == "A" and c.is_alive and c.name != char.name
        ]
        num_alive = 1 + len(teammates)
        teammate_count_norm = [min(num_alive, MAX_GUARDS) / MAX_GUARDS]

        teammates_sorted = sorted(
            teammates,
            key=lambda c: max(abs(c.pos[0] - pr), abs(c.pos[1] - pc))
        )
        teammate_feats = []
        max_dist = max(height, width)
        for slot in range(MAX_GUARDS - 1):
            if slot < len(teammates_sorted):
                t = teammates_sorted[slot]
                tr, tc = t.pos
                d = max(abs(tr - pr), abs(tc - pc))
                teammate_feats += [1.0, (tr - pr) / height, (tc - pc) / width, d / max_dist]
            else:
                teammate_feats += [0.0, 0.0, 0.0, 0.0]

        # ---- ターゲットセルへの視線(自分 + 味方) ----
        target_feats = []
        for t in range(NUM_TARGET_SLOTS):
            pos = target_positions[t]
            if pos is None:
                target_feats += [0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            tr, tc = pos
            self_los = 1.0 if self.has_line_of_sight((pr, pc), pos, grid) else 0.0
            others_los_count = sum(
                1 for tm in teammates
                if self.has_line_of_sight(tuple(tm.pos), pos, grid)
            )
            others_norm = others_los_count / max(1, MAX_GUARDS - 1)
            target_feats += [1.0, (tr - pr) / height, (tc - pc) / width, self_los, others_norm]

        own_ability_type = char.ability_type if char.ability_type in ABILITY_TYPES else "flash"
        ability_onehot = [1.0 if own_ability_type == t else 0.0 for t in ABILITY_TYPES]
        charge_map = {
            "flash": char.flash_charges,
            "smoke": char.smoke_charges,
            "recon": char.recon_charges,
        }
        ability_charge = [1.0 if charge_map.get(own_ability_type, 0) > 0 else 0.0]

        defender_defuse_info = game_state.get("defender_defuse_info", {})
        enemy_char = self._select_enemy((pr, pc), grid, chars, defender_defuse_info)
        enemy = [0.0, 0.0, 0.0]
        bot_busy = [0.0]
        defuse_progress = [0.0]
        bot_blind_flag = [0.0]
        if enemy_char is not None:
            er, ec = enemy_char.pos
            enemy = [1.0, (er - pr) / height, (ec - pc) / width]
            info = defender_defuse_info.get(enemy_char.name)
            timer = info[0] if info else 0
            bot_busy = [1.0 if timer > 0 else 0.0]
            defuse_progress = [timer / GUARD_DEFUSE_REQUIRED]
            bot_blind_flag = [1.0 if getattr(enemy_char, "blind_remaining", 0) > 0 else 0.0]
        enemy_info = enemy + bot_busy + defuse_progress + bot_blind_flag

        detonate_timer = game_state.get("detonate_timer", DETONATE_TICKS)
        detonate_ratio = [detonate_timer / DETONATE_TICKS]

        obs = (
            base + walls + last_onehot + dist_to_site + in_range_flag + teammate_count_norm +
            teammate_feats + target_feats + ability_onehot + ability_charge + enemy_info + detonate_ratio
        )
        return np.array(obs, dtype=np.float32)

    # -----------------------------------------------------------------
    # アクションマスク
    # -----------------------------------------------------------------
    def _build_mask(self, char, grid):
        pr, pc = char.pos
        height, width = grid.shape
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            nr, nc = pr + dr, pc + dc
            mask[a] = 1.0 if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 else 0.0

        own_ability_type = char.ability_type if char.ability_type in ABILITY_TYPES else None
        charge_map = {
            "flash": char.flash_charges,
            "smoke": char.smoke_charges,
            "recon": char.recon_charges,
        }
        mask[4] = 1.0 if own_ability_type and charge_map.get(own_ability_type, 0) > 0 else 0.0
        return mask

    # -----------------------------------------------------------------
    # アビリティの狙う方向(実際の発射経路はabilities.py側が計算する)
    # -----------------------------------------------------------------
    def _aim_direction(self, char, game_state, grid):
        pr, pc = char.pos
        defender_defuse_info = game_state.get("defender_defuse_info", {})
        enemy_char = self._select_enemy((pr, pc), grid, game_state["chars"], defender_defuse_info)
        if enemy_char is not None:
            return tuple(enemy_char.pos)

        prev_action = self.last_action.get(char.name)
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if prev_action in moves:
            dr, dc = moves[prev_action]
            return (pr + dr, pc + dc)

        planted_pos = game_state.get("planted_pos")
        if planted_pos is not None:
            return tuple(planted_pos)
        return (pr, pc + 1)

    # -----------------------------------------------------------------
    # メインエントリポイント
    # -----------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        planted_pos = game_state.get("planted_pos")
        if planted_pos is None:
            # 💡プラント前にguardが呼ばれることは想定外だが、念のためフォールバック
            return self.get_next_pos_random(char.pos, grid), "MOVE"

        self._ensure_map_cache(grid)
        site_index = self._nearest_site_index(planted_pos)
        target_positions = self._build_target_positions(grid, planted_pos, site_index)

        obs = self._build_observation(char, game_state, target_positions, planted_pos, site_index)
        mask = self._build_mask(char, grid)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(obs_t).squeeze(0).cpu().numpy()

        if self.greedy:
            action = masked_argmax(q_values, mask)
        else:
            action = masked_softmax_action(q_values, mask, temperature=self.temperature)

        self.last_action[char.name] = action

        if action == 4:
            aim_cell = self._aim_direction(char, game_state, grid)   # 💡修正: このクラス自身のメソッドを使用
            ability_name = char.ability_type.upper()
            return char.pos, {"ability": ability_name, "target": tuple(aim_cell)}

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        r, c = char.pos
        next_pos = [r + dr, c + dc]
        return next_pos, "MOVE"