"""touyama_v2/common_rl.py

touyama_v2配下の train_*.py 群で重複していた共通ロジックを集約したファイル。
- BFS距離マップ・LOS判定・BFS経路移動などのマップ非依存ユーティリティ
  (grid/height/widthは呼び出し側から明示的に渡す。モジュールグローバルな
  GRID等には依存しない)
- Dueling DQN・ReplayBuffer・行動選択・Double DQN TD誤差計算などの学習基盤

【自己完結ルールとの関係】
touyama_v2配下のtrain_*.py同士でのimportは今回のリファクタリングにより
許容する。ただし run_game.py / controllers.py / battle_logic.py /
abilities_los.py など feature モジュール本体は引き続き一切importしない。
このファイル自体もそれらに依存しない完全自己完結の実装。

【注意】select_actionの探索時ランダム選択は本ファイルではrandom.choiceに
統一している(元ファイルによってはnp.random.choiceを使用していたものが
あり、乱数消費順が変わる。学習ロジック・報酬設計への影響はない)。
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")

CARDINAL_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


# ============================================================================
# マップ読み込み
# ============================================================================

def parse_grid(maze_str):
    """マップ文字列(改行区切りの数字グリッド)をnp.int32配列に変換する。"""
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


# ============================================================================
# 幾何・LOS
# ============================================================================

def line_cells(p1, p2):
    """Bresenham法による2点間のセル列。"""
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


def has_los(grid, p1, p2, smoke_cells=None):
    """壁・スモークを考慮した射線判定。smoke_cellsは省略可(壁判定のみ)。"""
    cells = line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells and len(cells) > 2:
        if any(cell in smoke_cells for cell in cells):
            return False
    return True


# ============================================================================
# BFS距離マップ・方向
# ============================================================================

def bfs_distance_map(grid, goal):
    """goalから各床マスへの最短距離マップ(壁越え不可)。到達不能マスは-1。"""
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


def bfs_best_direction(dist_map, r0, c0):
    """dist_map上で、(r0,c0)から見て最も距離が縮む隣接方向(dr,dc)を返す。
    到達不能・移動不要なら(0,0)を返す。"""
    height, width = dist_map.shape
    cur = dist_map[r0, c0]
    if cur < 0:
        return 0, 0
    best_dr, best_dc, best_d = 0, 0, cur
    for dr, dc in CARDINAL_MOVES:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < height and 0 <= nc < width and dist_map[nr, nc] >= 0:
            if dist_map[nr, nc] < best_d:
                best_d = dist_map[nr, nc]
                best_dr, best_dc = dr, dc
    return best_dr, best_dc


def good_directions(dist_map, r, c):
    """上下左右のうち、BFS距離マップ上で距離を実際に縮められる方向を1.0、
    そうでない方向を0.0とする4次元フラグ(行動ID 0=UP,1=DOWN,2=LEFT,3=RIGHT)。"""
    good = [0.0, 0.0, 0.0, 0.0]
    height, width = dist_map.shape
    raw = dist_map[r, c]
    if raw < 0:
        return good
    for i, (dr, dc) in enumerate(CARDINAL_MOVES):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            nd = dist_map[nr, nc]
            if nd != -1 and nd < raw:
                good[i] = 1.0
    return good


# ============================================================================
# 移動(占有マス回避)
# ============================================================================

def random_step(grid, pos, occupied):
    """壁・占有マスを避けたランダム1マス移動。候補が無ければその場に留まる。"""
    height, width = grid.shape
    r, c = pos
    valid = [
        (r + dr, c + dc) for dr, dc in CARDINAL_MOVES
        if 0 <= r + dr < height and 0 <= c + dc < width
        and grid[r + dr, c + dc] != 1 and (r + dr, c + dc) not in occupied
    ]
    return random.choice(valid) if valid else tuple(pos)


def bfs_next_step(grid, start, goal, occupied, allow_adjacent_goal=True):
    """経路探索BFS(占有マス回避)。controllers.BaseController.move_towards_target
    と同等のロジック。到達不能ならrandom_stepにフォールバックする。"""
    height, width = grid.shape
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return start

    candidate_goals = []
    if grid[goal[0], goal[1]] != 1 and goal not in occupied:
        candidate_goals.append(goal)
    if allow_adjacent_goal or goal in occupied:
        for dr, dc in CARDINAL_MOVES:
            adj = (goal[0] + dr, goal[1] + dc)
            if (
                0 <= adj[0] < height and 0 <= adj[1] < width
                and grid[adj[0], adj[1]] != 1 and adj not in occupied
            ):
                candidate_goals.append(adj)
    candidate_goals = list(dict.fromkeys(candidate_goals))
    if not candidate_goals:
        return random_step(grid, start, occupied)

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
        for dr, dc in CARDINAL_MOVES:
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
        return random_step(grid, start, occupied)

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return start
    return step


def resolve_spawn_collision(grid, pos, occupied):
    """スポーン候補が重複/壁だった場合に、BFSで最寄りの空きマスへ逃がす。"""
    height, width = grid.shape
    if pos not in occupied and grid[pos[0], pos[1]] != 1:
        return pos
    visited = {pos}
    queue = deque([pos])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            if grid[nr, nc] == 1:
                continue
            if (nr, nc) not in occupied:
                return (nr, nc)
            queue.append((nr, nc))
    return pos


# ============================================================================
# Dueling DQN
# ============================================================================

class DuelingQNet(nn.Module):
    """全train_*.pyで共通の層構成(hidden//2で統一。元ファイルの一部は
    リテラル値(64/32)を使っていたが、いずれもhidden//2と一致する値だった
    ため統一しても挙動は変わらない)。"""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions)
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def soft_update(target_net, net, tau=0.005):
    for t_param, param in zip(target_net.parameters(), net.parameters()):
        t_param.data.copy_(tau * param.data + (1.0 - tau) * t_param.data)


# ============================================================================
# ReplayBuffer(namedtupleクラスを外から渡す汎用版。フィールド名・順序は
# 各train_*.py側のTransition定義に委ねる)
# ============================================================================

class ReplayBuffer:
    def __init__(self, transition_cls, capacity):
        self.transition_cls = transition_cls
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(self.transition_cls(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return self.transition_cls(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 行動選択
# ============================================================================

def select_action(net, state, mask, epsilon, device=DEVICE, fallback_action=0):
    """epsilon-greedy + マスク付き行動選択。マスクが全てFalseの場合は
    fallback_actionを返す(元ファイルでは主に4=STAYが該当)。"""
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) == 0:
        return fallback_action
    if random.random() < epsilon:
        return int(random.choice(valid_indices))
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = net(state_t).squeeze(0).cpu().numpy()
    q_values = np.where(mask, q_values, -np.inf)
    return int(np.argmax(q_values))


# ============================================================================
# Double DQN TD誤差計算・最適化ステップ
# ============================================================================

def compute_double_dqn_loss(
    policy_net, target_net, state, action, reward, next_state, done, next_mask,
    gamma, device=DEVICE,
):
    """全train_*.pyで共通のDouble DQN損失計算(smooth L1)。
    引数はnumpy配列(またはリスト)を想定し、内部でtensor化する。"""
    states = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.array(action), dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(np.array(done), dtype=torch.float32, device=device)
    next_masks = torch.as_tensor(np.array(next_mask), dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        next_q_policy = next_q_policy.masked_fill(~next_masks, -float("inf"))
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states).gather(1, next_actions).squeeze(1)
        next_q_target = torch.nan_to_num(next_q_target, neginf=0.0)
        target = rewards + gamma * next_q_target * (1.0 - dones)

    return F.smooth_l1_loss(q_values, target)


def optimize_double_dqn_step(
    policy_net, target_net, optimizer, state, action, reward, next_state, done, next_mask,
    gamma, device=DEVICE, max_grad_norm=10.0,
):
    """compute_double_dqn_loss + backward + clip + step までを1回で行う。
    呼び出し側(各train_*.py)はbuffer.sample()で得たbatchの各フィールドを
    そのままここに渡すだけでよい。"""
    loss = compute_double_dqn_loss(
        policy_net, target_net, state, action, reward, next_state, done, next_mask,
        gamma, device,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return float(loss.item())