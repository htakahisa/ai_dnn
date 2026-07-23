# learning_attacker_multi.py
from collections import deque
from pathlib import Path
import random
import numpy as np
import torch

from controllers import BaseController
from train_attacker_multi import (
    DuelingQNetwork,
    EnemyMemoryTracker,
    FixedEscortController,
    K_ENEMIES,
    ENEMY_MEMORY_TICKS,
    OBS_DIM,
    N_ACTIONS,
    DEFUSE_REQUIRED,
    ESCORT_OFFSET_MAX,
)


class LearningAttackerMultiController(BaseController):
    """【AIモデル適用・複数敵対応版】アタッカー操作クラス。
    キャリアー本人はDQNで判断し、護衛役はFixedEscortControllerに委譲する。"""

    def __init__(self, model_path="dqn_attacker_multi_final.pt", obs_dim=OBS_DIM, n_actions=N_ACTIONS, greedy=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path_obj = Path(model_path)
        full_path = model_path_obj if model_path_obj.is_absolute() else Path(__file__).resolve().parent / model_path_obj

        self.model = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(str(full_path), map_location=self.device))
        self.model.eval()

        self.greedy = greedy
        self.last_actions = {}
        self.pos_history = {}
        self.cached_target_pos = {}
        self.cached_dist_maps = {}

        self.escort_ctrl = FixedEscortController()
        self.enemy_memory_per_char = {}  # char.name -> EnemyMemoryTracker

    def reset_round(self):
        self.last_actions.clear()
        self.pos_history.clear()
        self.cached_target_pos.clear()
        self.cached_dist_maps.clear()
        self.enemy_memory_per_char.clear()

    def _get_memory(self, char_name):
        if char_name not in self.enemy_memory_per_char:
            self.enemy_memory_per_char[char_name] = EnemyMemoryTracker(
                memory_ticks=ENEMY_MEMORY_TICKS, k_enemies=K_ENEMIES
            )
        return self.enemy_memory_per_char[char_name]

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        spike_pos = game_state["spike_pos"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        target_plant_pos = game_state.get("target_plant_pos")
        chars = game_state["chars"]
        r, c = char.pos

        # --- 局面判定 ---
        if is_planted and planted_pos:
            target_pos = tuple(planted_pos)
            carrying, planted_flag = False, True
        elif char.has_spike and target_plant_pos:
            target_pos = tuple(target_plant_pos)
            carrying, planted_flag = True, False
        elif spike_pos is not None:
            target_pos = tuple(spike_pos)
            carrying, planted_flag = False, False
        else:
            # 誰かがスパイクを持っている(自分ではない) → 護衛ロジックへ
            holder = next((ch for ch in chars if ch.is_alive and ch.has_spike), None)
            if holder is not None:
                return self._decide_escort_move(char, holder, grid, chars, target_plant_pos)
            return self.get_next_pos_random(char.pos, grid), "MOVE"

        if self.cached_target_pos.get(char.name) != target_pos:
            self.cached_target_pos[char.name] = target_pos
            self.cached_dist_maps[char.name] = self._compute_bfs_map(target_pos, grid)

        obs = self._make_observation(char, target_pos, grid, chars, carrying, planted_flag, game_state)

        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0).cpu().numpy()

        # 💡action=4(PLANT/DEFUSE)無効化の条件:
        #    guard中(planted_flag): 常に有効
        #    carry中: サイト内(grid==2)にいなければ無効
        #    それ以外(retrieve): 常に無効
        q_values = q_values.copy()
        if not planted_flag:
            if carrying and grid[r, c] == 2:
                pass  # サイト内のcarry中はaction=4を有効のままにする
            else:
                q_values[4] = -np.inf

        if self.greedy:
            action = int(np.argmax(q_values))
        else:
            probs = np.exp((q_values - np.max(q_values)) / 0.5)
            probs = probs / probs.sum()
            action = np.random.choice(len(probs), p=probs)

        self.last_actions[char.name] = action
        self.pos_history.setdefault(char.name, deque(maxlen=7)).append(tuple(char.pos))

        if action == 4:
            if carrying:
                return char.pos, "PLANT"
            return char.pos, "MOVE"

        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = [r + moves[action][0], c + moves[action][1]]
        height, width = grid.shape
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width and grid[next_pos[0], next_pos[1]] != 1:
            return next_pos, "MOVE"
        else:
            holder = next((ch for ch in chars if ch.is_alive and ch.has_spike), None)
            if holder is not None:
                return self._decide_escort_move(char, holder, grid, chars, target_plant_pos)  # 💡target_plant_posを渡す
            return self.get_next_pos_random(char.pos, grid), "MOVE"

    def _decide_escort_move(self, char, holder, grid, chars, target_plant_pos):
        # 💡変更: front/back(前衛/後衛)の区別を廃止。carry中は敵と接敵しない前提のため、
        #    holder以外の護衛は全員「前衛的にプラントサイト方向をカバーする」扱いに統一する。
        escorts = [ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != holder.name and not ch.has_spike]
        escorts.sort(key=lambda ch: max(abs(ch.pos[0] - holder.pos[0]), abs(ch.pos[1] - holder.pos[1])))

        escort_rank = next((i for i, ch in enumerate(escorts) if ch.name == char.name), None)

        target_key = f"{char.name}_escort_target"
        if self.cached_target_pos.get(target_key) != tuple(target_plant_pos):
            self.cached_target_pos[target_key] = tuple(target_plant_pos)
            self.cached_dist_maps[target_key] = self._compute_bfs_map(tuple(target_plant_pos), grid)
        dist_map = self.cached_dist_maps[target_key]

        # 💡団子防止: 護衛ごとにオフセット距離を少しずつずらす(近い人ほど短め、遠い人ほど長め)
        offset_steps = ESCORT_OFFSET_MAX if escort_rank is None else max(2, ESCORT_OFFSET_MAX - escort_rank * 3)
        target = self.escort_ctrl.compute_target(tuple(char.pos), dist_map, grid, "front", steps_override=offset_steps)

        occupied = [tuple(holder.pos)] + [tuple(e.pos) for e in escorts if e.name != char.name]
        next_pos = self.escort_ctrl.next_move(tuple(char.pos), target, grid, occupied)

        self.last_actions[char.name] = None
        self.pos_history.setdefault(char.name, deque(maxlen=7)).append(tuple(char.pos))
        return next_pos, "MOVE"

    def _make_observation(self, char, target_pos, grid, chars, carrying, planted_flag, game_state):
        pr, pc = char.pos
        tr, tc = target_pos
        height, width = grid.shape

        base = [pr / (height - 1), pc / (width - 1), tr / (height - 1), tc / (width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_act = self.last_actions.get(char.name, None)
        last_onehot = [0.0] * 5
        if last_act is not None:
            last_onehot[last_act] = 1.0

        max_dist = max(height, width) * 2
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc, grid):
                d = self.cached_dist_maps[char.name][nr, nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)

        phase_flags = [1.0 if carrying else 0.0, 1.0 if planted_flag else 0.0]

        # --- 敵情報: guard中(planted_flag)のみ意味を持つ。見えた場合だけ記憶を更新 ---
        visible_ids = set()
        if planted_flag:
            memory = self._get_memory(char.name)
            observer_positions = [tuple(char.pos)]
            other_att = next(
                (ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != char.name and not ch.has_spike),
                None
            )
            if other_att is not None:
                observer_positions.append(tuple(other_att.pos))
            defenders = [(d.name, tuple(d.pos), d.is_alive) for d in chars if d.team == "D"]
            visible_ids = memory.update(observer_positions, defenders, grid)
            enemy_feats = memory.build_features((pr, pc), height, width, visible_ids)
        else:
            enemy_feats = [0.0] * (K_ENEMIES * 5)

        # --- 味方 ---
        ally = [0.0, 0.0, 0.0]
        if planted_flag:
            other = next(
                (ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != char.name and not ch.has_spike),
                None
            )
            if other is not None:
                ally = [1.0, (other.pos[0] - pr) / height, (other.pos[1] - pc) / width]
        elif carrying:   # 💡追加: carry中も護衛(自分以外のattacker)の位置を渡す
            escorts = [ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != char.name and not ch.has_spike]
            if escorts:
                nearest = min(escorts, key=lambda ch: max(abs(ch.pos[0] - pr), abs(ch.pos[1] - pc)))
                ally = [1.0, (nearest.pos[0] - pr) / height, (nearest.pos[1] - pc) / width]

        # --- 解除進行フラグ: run_game.py の game_state から受け取る想定
        #     (未実装の間は常に [0.0, 0.0]。run_game.py 側の対応が必要) ---
        defuse_info = [0.0, 0.0]
        defuse_progress_map = game_state.get("defender_defuse_info")  # {defender_name: (timer, required)}
        if planted_flag and defuse_progress_map:
            max_progress = 0.0
            any_defusing = False
            for _, (timer, required) in defuse_progress_map.items():
                if timer > 0:
                    any_defusing = True
                    max_progress = max(max_progress, timer / required)
            if any_defusing:
                defuse_info = [1.0, min(max_progress, 1.0)]

        return np.array(base + walls + last_onehot + dists + phase_flags
                         + enemy_feats + ally + defuse_info, dtype=np.float32)

    def _is_walkable(self, r, c, grid):
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1

    def _compute_bfs_map(self, target, grid):
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