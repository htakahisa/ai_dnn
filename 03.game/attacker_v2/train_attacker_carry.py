# train_attacker_carry.py
"""
スパイクキャリアーAI(carryフェーズ専用)の学習スクリプト。
外部の学習系モジュール(train_defender_combined.py等)には依存せず、
必要なクラスはすべてこのファイル内に複製する。
"""

import sys

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# このファイルの1階層上をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR

# 💡変更: アビリティの物理パラメータ(速度・射程tick・持続時間・範囲)は
from game_core import (
    WINNING_ROUNDS,
    TICK_TIME,
    MAX_HP,
    BODY_DAMAGE,
    HEADSHOT_DAMAGE,
    SHOOT_INTERVAL_TICKS,
    SIDE_PANEL_WIDTH,
    PLANT_REQUIRED_TICKS,
    DEFUSE_REQUIRED_TICKS,
    SMOKE_DURATION_TICKS,
    MOVING_ACCURACY,
    MOVING_TARGET_HIT_MULTIPLIER,
    BLIND_DURATION_TICKS,
    FLASH_BURST_DURATION_TICKS,
    BLIND_ACCURACY_MULTIPLIER,
    FLASH_SPEED_CELLS_PER_TICK,
    FLASH_MAX_FLIGHT_TICKS,
    RECON_SPEED_CELLS_PER_TICK,
    REVEAL_DURATION_TICKS,
    REVEALED_DODGE_MULTIPLIER,
    RECON_REVEAL_SIZE,
    COMBO_DISPLAY_TICKS,
    COMBO_BANNER_HEIGHT,
    ROUND_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
    RECON_BURST_DISPLAY_TICKS,
    SMOKE_WARNING_TICKS,
    ROUND_TRANSITION_TICKS,
    ABILITY_TYPES,
    FLASH_BLIND_TICKS,
    SMOKE_RADIUS,
    RECON_RADIUS,
)

OBS_DIM = 26
N_ACTIONS = 6   # 0:上 1:下 2:左 3:右 4:設置 5:アビリティ使用
NUM_EPISODES = 9000
SAVE_INTERVAL = 100
ABILITY_TYPES = ["flash", "smoke", "recon"]

# 💡変更: 命中判定に使う独自の距離しきい値(FLASH_RANGE等)は廃止。
# 実際の投射物飛行+LOS判定で成否が決まるため不要になった。
ENEMY_SPAWN_PROB = 0.5        # 敵が出現するエピソードの割合
SPOTTED_PENALTY = -5.0        # 敵と視認しあっている(かつ敵が怯んでいない)間、毎tick与えるペナルティ
ABILITY_USE_COST = -0.2       # アビリティ使用の基礎コスト(空振り防止)
RECON_SUCCESS_BONUS = 2.0     # リコンが実際に敵を検知できた場合の追加報酬
FLASH_SUCCESS_BONUS = 8.0     # フラッシュが実際に敵に成立した場合の追加報酬

# 💡追加: リコンが敵を検知できなくても、曲がり角付近を確認できていれば
# 「そこは安全だと分かった」という情報価値として小さい報酬を与える。
# CORNER_CHECK_RADIUS より遠い場所(見通しの良い通路等)への空振りは対象外。
RECON_EMPTY_INFO_BONUS = 0.0  # 0で無効
CORNER_CHECK_RADIUS = 2

# 💡追加: flashが外れた場合の救済ボーナス(排他的、いずれか1つのみ加算)
FLASH_REACTIVE_THROW_BONUS = 5.0   # 敵視認中に反応して投げた場合(命中しなくても反応自体は正しい)
FLASH_EMPTY_INFO_BONUS = 0.0  # 0で無効      # 曲がり角付近を狙っていた場合(reconより情報価値が低いため小さめ)
FLASH_OPEN_THROW_MIN_PATH = 3      # この距離以上飛べば「開けた場所へ投げた」とみなす
FLASH_OPEN_THROW_BONUS = 0.0  # 0で無効      # 開けた通路へ適切に投げた場合(壁際1マスでの無駄撃ちと区別)

# 💡追加: サイトに近づいた状態でサイト方面へアビリティを使った場合のボーナス。
# 「敵の有無に関わらず、突入前に予防的に使っておく」行動を評価するための項目で、
# 命中判定(FLASH_SUCCESS_BONUS等)とは独立に加算される。
SITE_APPROACH_BONUS = 1.0
SITE_APPROACH_DIST_THRESHOLD = 5

EVAL_EPISODES = 100  # 💡変更: 30→100。敵出現確率50%のブレをならしやすくする

# ===========================================================================
# BFS距離マップ
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


# 💡追加: grid==2のセル群を隣接関係(4方向)の連結成分ごとに分割する。
# 左サイト/右サイトのように離れた場所に複数のサイトが存在する場合、
# それぞれ別サイトとして区別するために必要。
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

# 💡追加: 曲がり角(L字の角)を検出する。敵の待ち伏せ位置候補として使う。
# 「壁が直交する2方向にある」セルを角とみなす(直線の壁沿いや開けた部屋は対象外)。
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
            perpendicular_pairs = [(north, west), (north, east), (south, west), (south, east)]
            if any(a and b for a, b in perpendicular_pairs):
                corners.append((r, c))
    return corners


# 💡追加: 2点間のBresenham線分上のセル一覧を返す(controllers.pyのhas_line_of_sightから流用・拡張)
def line_cells(p1, p2):
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


# 💡追加: 壁・スモークを考慮した視線判定
# 💡追加: 壁・スモークを考慮した視線判定
def has_line_of_sight(p1, p2, grid, smoke_cells=None):
    cells = line_cells(p1, p2)
    for (r, c) in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells:
        for cell in cells:
            if cell in smoke_cells:
                return False
    return True


# 💡追加: abilities.py の _projectile_path と同じロジック(壁 or マップ端まで伸びる経路)。
# AbilityMixinのインスタンスメソッドに依存せずに使えるよう、モジュールレベル関数として複製する。
def compute_projectile_path(start, aimed_cell, grid):
    height, width = grid.shape
    sr, sc = start
    ar, ac = aimed_cell
    dr, dc = ar - sr, ac - sc
    if dr == 0 and dc == 0:
        return [start]
    scale = max(height, width) * 3
    far = (sr + dr * scale, sc + dc * scale)
    raw = line_cells(start, far)
    path = [start]
    for rr, cc in raw[1:]:
        if not (0 <= rr < height and 0 <= cc < width):
            break
        if grid[rr, cc] == 1:
            break
        path.append((rr, cc))
    return path

# 💡追加: 元々はCarryOnlyEnvのメソッドだったが、learning_attacker_carry.py側でも
# 同じロジックが必要になったため、モジュールレベル関数として共通化する。
def multi_source_bfs(site_cells, grid):
    """サイト内の全マスを起点とした同時BFS。
    dist[r,c] = 最寄りサイトマスまでの距離
    label[r,c] = そのマスから見た最寄りサイトマスの座標(観測の目標点として利用)"""
    height, width = grid.shape
    dist = np.full((height, width), np.inf)
    label = [[None] * width for _ in range(height)]
    q = deque()
    for (r, c) in site_cells:
        dist[r, c] = 0
        label[r][c] = (r, c)
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    label[nr][nc] = label[r][c]
                    q.append((nr, nc))
    return dist, label


# ===========================================================================
# リプレイバッファ(mask / next_mask対応版)
# ===========================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.append((obs, action, reward, next_obs, done, mask, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, mask, next_mask = zip(*batch)
        return (np.array(obs), np.array(action), np.array(reward),
                np.array(next_obs), np.array(done), np.array(mask), np.array(next_mask))


# ===========================================================================
# Dueling DQN構造
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


# ===========================================================================
# carry専用環境
# ===========================================================================
class CarryOnlyEnv:
    def __init__(self, fixed_grid, plant_candidates):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.plant_candidates = [tuple(p) for p in plant_candidates]
        self.max_steps = 150

        self.site_components = split_site_components(self.plant_candidates)
        self.site_maps = [multi_source_bfs(comp, self.grid) for comp in self.site_components]
        self.site_cell_sets = [set(comp) for comp in self.site_components]

        # 💡追加: 敵の待ち伏せ候補(曲がり角セル)を1回だけ計算しておく
        self.corner_cells = find_corner_cells(self.grid)

    def _is_walkable(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1

    def _random_walkable(self):
        while True:
            p = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if self._is_walkable(*p):
                return p

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.last_action = None
        self.player_pos = self._random_walkable()

        # 💡変更: ラウンドごとにどちらのサイトを狙うかをランダムに選び、そのサイト専用の
        # dist_map/label_mapを使う(実ゲームのtarget_plant_posランダム割当を再現するため)
        self.current_site_index = random.randrange(len(self.site_components))
        self.dist_map, self.label_map = self.site_maps[self.current_site_index]
        self.site_cells = self.site_cell_sets[self.current_site_index]

        self.prev_dist = self.dist_map[self.player_pos[0]][self.player_pos[1]]

        self.own_ability_type = random.choice(ABILITY_TYPES)
        self.own_ability_charge = 1.0
        self.last_ability_result = None  # {"type": str, "success": bool} を使用時に格納
        self.last_site_approach_triggered = False  # 💡追加: サイト方面ボーナスが成立したか

        # 敵(defender)を確率的にスポーンさせる。存在する場合は曲がり角に固定配置(待ち伏せ)。
        self.bot_present = random.random() < ENEMY_SPAWN_PROB
        if self.bot_present:
            self.bot_pos = random.choice(self.corner_cells) if self.corner_cells else self._random_walkable()
        else:
            self.bot_pos = None
        self.bot_blind_remaining = 0
        self.recon_reveal_remaining = 0
        self.smoke_cells = set()
        self.smoke_remaining = 0

        # 💡追加: 飛翔中の投射物(1round1個までなので同時に1つしか存在しない)
        self.pending_flash = None   # {"path": [...], "progress": int, "ticks_alive": int}
        self.pending_recon = None   # {"path": [...], "progress": int}

        self.pos_history = deque(maxlen=6)
        return self._get_obs(), {}

    def _get_obs(self):
        pr, pc = self.player_pos
        gr, gc = self.label_map[pr][pc]

        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]

        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        # 💡変更: action=5(アビリティ使用)を追加したため6次元に拡張
        last_onehot = [0.0] * 6
        if self.last_action is not None:
            last_onehot[self.last_action] = 1.0

        max_dist = max(self.height, self.width) * 2
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr][nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)

        ability_onehot = [1.0 if self.own_ability_type == t else 0.0 for t in ABILITY_TYPES]
        ability_charge = [self.own_ability_charge]

        # 💡変更: blind_flagは「敵が怯んでいるか」を表す(フラッシュは敵にかけるため)
        enemy_blinded_flag = [1.0 if self.bot_blind_remaining > 0 else 0.0]

        # 💡変更: 敵情報はLOS成立時、またはリコン察知中はLOS抜きで見える
        enemy = [0.0, 0.0, 0.0]
        if self.bot_present:
            los = has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells)
            if los or self.recon_reveal_remaining > 0:
                br, bc = self.bot_pos
                enemy = [1.0, (br - pr) / self.height, (bc - pc) / self.width]

        return np.array(
            base + walls + last_onehot + dists + ability_onehot + ability_charge + enemy_blinded_flag + enemy,
            dtype=np.float32
        )

    def get_action_mask(self):
        """移動: 壁でなければ1。設置(4): サイト内なら1。アビリティ(5): チャージが残っていれば1。"""
        pr, pc = self.player_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            mask[a] = 1.0 if self._is_walkable(pr + dr, pc + dc) else 0.0
        mask[4] = 1.0 if (pr, pc) in self.site_cells else 0.0
        mask[5] = 1.0 if self.own_ability_charge > 0 else 0.0  # 💡追加
        return mask

    def step(self, action):
        self.current_step += 1
        self.last_action = action
        pr, pc = self.player_pos

        if action == 4:
            if (pr, pc) in self.site_cells:
                reward, terminated = 200.0, True
            else:
                reward, terminated = -1.0, False
        elif action == 5:
            reward = self._use_ability()
            terminated = False
        else:
            reward, terminated = self._step_move(action)

        # 💡追加: 飛翔中の投射物(前tick以前に投げたものも含む)を1tick分進める。
        # このtickのactionが移動や設置であっても、投射物は独立して飛び続ける(実ゲームと同じ)。
        reward += self._advance_projectiles()

        # 毎tick共通の視認判定。敵が存在し、怯んでおらず、LOS(壁・スモーク考慮)が
        # 通っている間はペナルティのみ与える(生死判定はしない、終了もしない)
        if self.bot_present and self.bot_blind_remaining <= 0:
            if has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells):
                reward += SPOTTED_PENALTY

        # 各種タイマーの減衰
        if self.bot_blind_remaining > 0:
            self.bot_blind_remaining -= 1
        if self.recon_reveal_remaining > 0:
            self.recon_reveal_remaining -= 1
        if self.smoke_remaining > 0:
            self.smoke_remaining -= 1
            if self.smoke_remaining == 0:
                self.smoke_cells = set()

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}


    def _get_aim_direction(self):
        """狙う方向(target_cell相当)を決める。
        優先度: 1) 敵が見えている/リコン察知中ならその方向 2) 直前の移動方向
        3) どちらもなければBFS勾配上の最善方向(目標に近づく向き)。
        戻り値: (狙うマス, 敵視認による反応的な使用かどうか)"""
        pr, pc = self.player_pos

        if self.bot_present:
            visible = has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells)
            if visible or self.recon_reveal_remaining > 0:
                return self.bot_pos, True  # 💡変更: is_reactive=True を追加

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if self.last_action in moves:
            dr, dc = moves[self.last_action]
            return (pr + dr, pc + dc), False  # 💡変更: is_reactive=False を追加

        # フォールバック: BFS距離が一番小さくなる方向(＝目標に近づく方向)
        best_dir, best_dist = None, self.dist_map[pr, pc]
        for dr, dc in moves.values():
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr, nc]
                if np.isfinite(d) and d < best_dist:
                    best_dist, best_dir = d, (dr, dc)
        if best_dir is None:
            best_dir = (0, 1)
        return (pr + best_dir[0], pc + best_dir[1]), False  # 💡変更: is_reactive=False を追加

    def _use_ability(self):
        """アビリティ発動処理。abilities.pyと同じく、フラッシュ/リコンは投射物として飛ばし、
        着弾は_advance_projectiles()側で多tickかけて解決する(即座には成否が決まらない)。
        スモークのみ実ゲームと同じく即座にtarget_cellへ設置する。"""
        if self.own_ability_charge <= 0:
            return -1.0  # 通常はmaskで防ぐが、念のためのフォールバック

        self.own_ability_charge = 0.0
        ability_type = self.own_ability_type
        aimed_cell, is_reactive = self._get_aim_direction()  # 💡変更: タプルで受け取る
        path = compute_projectile_path(self.player_pos, aimed_cell, self.grid)

        # 💡暫定でfalseを入れておき、着弾時(または即時判定できるsmoke)に更新する
        self.last_ability_result = {"type": ability_type, "success": False}

        reward = ABILITY_USE_COST

        # 💡追加: サイトに近い状態で、サイト方面(経路がサイトのセルに到達)へ使用した場合のボーナス。
        # 敵の有無・命中とは無関係に、「予防的にサイト方面を確認/制圧しておく」行動を評価する。
        self.last_site_approach_triggered = False  # 💡追加
        player_dist_to_site = self.dist_map[self.player_pos[0], self.player_pos[1]]
        if np.isfinite(player_dist_to_site) and player_dist_to_site <= SITE_APPROACH_DIST_THRESHOLD:
            if any(cell in self.site_cells for cell in path):
                reward += SITE_APPROACH_BONUS
                self.last_site_approach_triggered = True  # 💡追加

        if ability_type == "flash":
            self.pending_flash = {"path": path, "progress": 0, "ticks_alive": 0, "is_reactive": is_reactive}
        elif ability_type == "recon":
            self.pending_recon = {"path": path, "progress": 0}
        elif ability_type == "smoke":
            # 💡実ゲームの_place_smokeと同じ: target_cell周辺SMOKE_RADIUS範囲に即設置。
            # ここではaimed_cellの代わりに、届く最遠マス(path[-1])を設置地点とする。
            tr, tc = path[-1]
            self.smoke_cells = {
                (rr, cc)
                for rr in range(tr - SMOKE_RADIUS, tr + SMOKE_RADIUS + 1)
                for cc in range(tc - SMOKE_RADIUS, tc + SMOKE_RADIUS + 1)
                if self._is_walkable(rr, cc)
            }
            self.smoke_remaining = SMOKE_DURATION_TICKS
            # 💡成功判定: 実際にプレイヤー→敵のLOSを遮断できていれば成功とみなす
            if self.bot_present and not has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells):
                self.last_ability_result["success"] = True

        return reward

    def _advance_projectiles(self):
        """飛翔中のflash/recon投射物を1tick分進める。actionの種類に関わらず毎step呼ぶ。
        abilities.py の _advance_ability_effects と同じ速度・射程tick・着弾判定ロジック。"""
        reward = 0.0

        if self.pending_flash is not None:
            p = self.pending_flash
            p["ticks_alive"] += 1
            next_progress = p["progress"] + FLASH_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge or p["ticks_alive"] >= FLASH_MAX_FLIGHT_TICKS:
                impact = p["path"][p["progress"]]
                success = False
                if self.bot_present and has_line_of_sight(impact, self.bot_pos, self.grid, self.smoke_cells):
                    self.bot_blind_remaining = max(self.bot_blind_remaining, BLIND_DURATION_TICKS)
                    success = True
                    reward += FLASH_SUCCESS_BONUS

                if not success:
                    ir, ic = impact
                    if p["is_reactive"]:
                        reward += FLASH_REACTIVE_THROW_BONUS
                    elif self.corner_cells and min(
                        max(abs(ir - cr), abs(ic - cc)) for (cr, cc) in self.corner_cells
                    ) <= CORNER_CHECK_RADIUS:
                        reward += FLASH_EMPTY_INFO_BONUS
                    elif len(p["path"]) - 1 >= FLASH_OPEN_THROW_MIN_PATH:
                        reward += FLASH_OPEN_THROW_BONUS

                self.last_ability_result = {"type": "flash", "success": success}
                self.pending_flash = None

        if self.pending_recon is not None:
            p = self.pending_recon
            next_progress = p["progress"] + RECON_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge:
                impact = p["path"][p["progress"]]
                ir, ic = impact
                radius = RECON_REVEAL_SIZE // 2
                success = False
                if self.bot_present:
                    br, bc = self.bot_pos
                    if abs(br - ir) <= radius and abs(bc - ic) <= radius:
                        self.recon_reveal_remaining = max(self.recon_reveal_remaining, REVEAL_DURATION_TICKS)
                        success = True
                        reward += RECON_SUCCESS_BONUS

                # 💡追加: 検知できなくても、曲がり角の近くを確認できていれば
                # 「そこは安全と確認できた」小さな情報価値を報酬として与える。
                # (曲がり角から遠い場所への空振りは対象外にして、闇雲な連打を抑制する)
                if not success and self.corner_cells:
                    nearest_corner_dist = min(
                        max(abs(ir - cr), abs(ic - cc)) for (cr, cc) in self.corner_cells
                    )
                    if nearest_corner_dist <= CORNER_CHECK_RADIUS:
                        reward += RECON_EMPTY_INFO_BONUS

                self.last_ability_result = {"type": "recon", "success": success}
                self.pending_recon = None

        return reward


    def _step_move(self, action):
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        r, c = self.player_pos
        nr, nc = r + moves[action][0], c + moves[action][1]

        if not self._is_walkable(nr, nc):
            return -1.5, False

        self.player_pos = (nr, nc)
        new_dist = self.dist_map[nr][nc]  # 💡最寄りサイトマスまでの距離(マルチソース)
        shaping = (self.prev_dist - new_dist) * 0.5
        reward = -1.0 + shaping

        if np.isfinite(new_dist) and new_dist <= 3:
            reward += 1.5

        pos_tuple = (nr, nc)
        near_goal = np.isfinite(new_dist) and new_dist <= 3
        if not near_goal and pos_tuple in self.pos_history and new_dist >= self.prev_dist:
            reward -= 2.0
        self.pos_history.append(pos_tuple)

        self.prev_dist = new_dist
        return reward, False

# ===========================================================================
# マスクを考慮した行動選択ヘルパー
# ===========================================================================
def masked_argmax(q_values, mask):
    """maskが0の行動を-infにしてargmaxを取る(全滅マスク対策として全0なら素のargmax)"""
    if mask.sum() == 0:
        return int(np.argmax(q_values))
    masked = np.where(mask > 0, q_values, -np.inf)
    return int(np.argmax(masked))


def masked_random_action(mask):
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return random.randint(0, N_ACTIONS - 1)
    return int(random.choice(valid))


def masked_softmax_action(q_values, mask, temperature=0.5):
    if mask.sum() == 0:
        masked = q_values
    else:
        masked = np.where(mask > 0, q_values, -np.inf)
    probs = np.exp((masked - np.max(masked)) / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ===========================================================================
# 学習ループ
# ===========================================================================
def train():
    writer = SummaryWriter(log_dir="logs")

    EVAL_BEST_SAVE_DIR = "data/attacker_carry_data/"
    SAVE_DIR = "data_temp/attacker_carry_data/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = CarryOnlyEnv(fixed_grid, plant_candidates)

    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9985
    lr = 0.0005
    IMPROVEMENT_MARGIN = 5.0

    device = torch.device("cpu")
    q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=30000)

    epsilon = epsilon_start
    best_eval_reward = -float('inf')
    # 💡追加: 成功率・到達tick数トラッキング用
    success_window = deque(maxlen=100)
    ticks_window = deque(maxlen=100)  # 成功時のtickのみ格納

    # 💡追加: アビリティ使用状況トラッキング用(種別ごと)
    # assigned: そのラウンドでその種別が割り当てられた回数(分母)
    # used: 割り当てられた回のうち実際に使用した割合
    # success: 使用した回のうち成立(命中/意味があった)した割合
    ability_assigned_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    ability_used_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    ability_success_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}

    # 💡追加: サイト方面ボーナスの成立率トラッキング用。
    # 分母は「実際にアビリティを使用したエピソード」のみ(使わなかった回は対象外)。
    site_approach_window = deque(maxlen=100)

    print(f"学習を開始します。デバイス: {device} | 入力次元: {OBS_DIM}")
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        mask = env.get_action_mask()
        episode_reward = 0.0
        losses = []

        while True:
            if random.random() < epsilon:
                action = masked_random_action(mask)
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_t).squeeze(0).cpu().numpy()
                action = masked_argmax(q_values, mask)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_mask = env.get_action_mask()

            replay_buffer.push(obs, action, reward, next_obs, terminated, mask, next_mask)
            obs, mask = next_obs, next_mask
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                b_obs, b_act, b_rew, b_nobs, b_term, b_mask, b_next_mask = replay_buffer.sample(batch_size)
                b_obs_t = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act_t = torch.tensor(b_act, dtype=torch.long, device=device).unsqueeze(1)
                b_rew_t = torch.tensor(b_rew, dtype=torch.float32, device=device).unsqueeze(1)
                b_nobs_t = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_term_t = torch.tensor(b_term, dtype=torch.float32, device=device).unsqueeze(1)
                b_next_mask_t = torch.tensor(b_next_mask, dtype=torch.float32, device=device)

                current_q = q_net(b_obs_t).gather(1, b_act_t)
                with torch.no_grad():
                    next_q_online = q_net(b_nobs_t)
                    # 💡マスク: 次状態で不可能な行動をDouble DQNのargmax候補から除外
                    masked_next_q = next_q_online.masked_fill(b_next_mask_t == 0, -1e9)
                    next_actions = masked_next_q.argmax(dim=1, keepdim=True)
                    max_next_q = target_net(b_nobs_t).gather(1, next_actions)
                    target_q = b_rew_t + (1.0 - b_term_t) * gamma * max_next_q

                loss = nn.SmoothL1Loss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(loss.item())

            if terminated or truncated:
                if episode % 10 == 0:
                    target_net.load_state_dict(q_net.state_dict())
                break

        # 💡追加: このenvではterminated=Trueは設置成功時のみ発生する
        # (交戦などの失敗要因が未実装のため、truncated=タイムアウト失敗と等価)
        success = terminated
        success_window.append(1.0 if success else 0.0)
        if success:
            ticks_window.append(env.current_step)

        # 💡追加: このエピソードで割り当てられていたアビリティ種別の分母を必ず増やす
        assigned_type = env.own_ability_type
        ability_assigned_window[assigned_type].append(1.0)
        for t in ABILITY_TYPES:
            if t != assigned_type:
                ability_assigned_window[t].append(0.0)

        used_result = env.last_ability_result  # None または {"type":..., "success":...}
        for t in ABILITY_TYPES:
            used_flag = 1.0 if (used_result is not None and used_result["type"] == t) else 0.0
            ability_used_window[t].append(used_flag)
        if used_result is not None:
            ability_success_window[used_result["type"]].append(1.0 if used_result["success"] else 0.0)
            # 💡追加: 使用した回のみを分母に、サイト方面ボーナスが成立したかを記録
            site_approach_window.append(1.0 if env.last_site_approach_triggered else 0.0)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Success_Rate", np.mean(success_window), episode)
        if ticks_window:
            writer.add_scalar("Train/Ticks_To_Plant", np.mean(ticks_window), episode)

        # 💡追加: アビリティ種別ごとの使用率・成功率をtensorboardへ
        for t in ABILITY_TYPES:
            writer.add_scalar(f"Train/Ability_UsageRate_{t}", np.mean(ability_used_window[t]), episode)
            if ability_success_window[t]:
                writer.add_scalar(f"Train/Ability_SuccessRate_{t}", np.mean(ability_success_window[t]), episode)

        # 💡追加: サイト方面ボーナスの成立率(使用した回のうち何%がサイト方面への使用だったか)
        if site_approach_window:
            writer.add_scalar("Train/Ability_SiteApproachRate", np.mean(site_approach_window), episode)

        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            # 💡追加: アビリティ使用状況の簡易サマリーを表示
            # 💡追加: アビリティ使用状況の簡易サマリーを表示
            ability_summary = " | ".join(
                f"{t}: use={np.mean(ability_used_window[t]):.0%}"
                f" succ={(np.mean(ability_success_window[t]) if ability_success_window[t] else float('nan')):.0%}"
                for t in ABILITY_TYPES
            )
            print(f"   [Ability] {ability_summary}")
            # 💡追加: サイト方面ボーナスの成立率もコンソールに表示
            site_approach_rate = np.mean(site_approach_window) if site_approach_window else float('nan')
            print(f"   [Ability SiteApproach] rate={site_approach_rate:.0%} (n_used={len(site_approach_window)})")

           

            eval_rewards = []
            eval_success_count = 0
            eval_success_ticks = []

            eval_ability_used = {t: 0 for t in ABILITY_TYPES}
            eval_ability_success = {t: 0 for t in ABILITY_TYPES}
            eval_site_approach_count = 0  # 💡追加

            for _ in range(EVAL_EPISODES):
                eval_obs, _ = env.reset()
                eval_mask = env.get_action_mask()
                eval_reward = 0.0
                while True:
                    with torch.no_grad():
                        q_values = q_net(torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    a = masked_softmax_action(q_values, eval_mask, temperature=0.5)
                    eval_obs, r, term, trunc, _ = env.step(a)
                    eval_mask = env.get_action_mask()
                    eval_reward += r
                    if term or trunc:
                        if term:
                            eval_success_count += 1
                            eval_success_ticks.append(env.current_step)
                        break
                eval_rewards.append(eval_reward)

                # 💡追加: このエピソードのアビリティ使用結果を反映
                if env.last_ability_result is not None:
                    t = env.last_ability_result["type"]
                    eval_ability_used[t] += 1
                    if env.last_ability_result["success"]:
                        eval_ability_success[t] += 1
                    if env.last_site_approach_triggered:  # 💡追加
                        eval_site_approach_count += 1

            mean_eval = np.mean(eval_rewards)
            eval_success_rate = eval_success_count / EVAL_EPISODES
            writer.add_scalar("Eval/Carry_Reward", mean_eval, episode)
            writer.add_scalar("Eval/Success_Rate", eval_success_rate, episode)
            if eval_success_ticks:
                writer.add_scalar("Eval/Mean_Ticks_To_Plant", np.mean(eval_success_ticks), episode)

            # 💡追加: eval時のアビリティ種別ごとの使用回数・成功率をtensorboardへ
            for t in ABILITY_TYPES:
                writer.add_scalar(f"Eval/Ability_UsedCount_{t}", eval_ability_used[t], episode)
                if eval_ability_used[t] > 0:
                    writer.add_scalar(f"Eval/Ability_SuccessRate_{t}", eval_ability_success[t] / eval_ability_used[t], episode)

            # 💡追加: eval時のサイト方面ボーナス成立率(使用した回のうち何%か)
            total_used = sum(eval_ability_used.values())
            if total_used > 0:
                writer.add_scalar("Eval/Ability_SiteApproachRate", eval_site_approach_count / total_used, episode)

            print(f"   [Eval] mean={mean_eval:.2f} success_rate={eval_success_rate:.2%} "
                  f"avg_ticks={np.mean(eval_success_ticks) if eval_success_ticks else float('nan'):.1f} n={EVAL_EPISODES}")
            # 💡追加: eval時のサイト方面ボーナス成立率もコンソールに表示
            eval_site_rate = (eval_site_approach_count / total_used) if total_used > 0 else float('nan')
            print(f"   [Eval SiteApproach] rate={eval_site_rate:.0%} (used={total_used})")
            print(f"   [Eval Ability] " + " | ".join(
                f"{t}: used={eval_ability_used[t]} succ={eval_ability_success[t]}/{eval_ability_used[t] if eval_ability_used[t] else 0}"
                for t in ABILITY_TYPES
            ))

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(EVAL_BEST_SAVE_DIR, "dqn_attacker_carry_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_carry_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_carry_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_carry_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()