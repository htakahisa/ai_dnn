# learning_attacker.py
from collections import deque
from pathlib import Path
import random
import numpy as np
import torch

from controllers import BaseController
from train_attacker_combined import DuelingQNetwork


class LearningAttackerController(BaseController):
    """【AIモデル適用】アタッカー側の操作クラス（1モデルで retrieve/carry/guard 全局面をカバー）"""

    def __init__(self, model_path="dqn_attacker_combined_best.pt", obs_dim=25, n_actions=5, greedy=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path_obj = Path(model_path)
        full_path = model_path_obj if model_path_obj.is_absolute() else Path(__file__).resolve().parent / model_path_obj

        self.model = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(str(full_path), map_location=self.device))
        self.model.eval()

        self.last_actions = {}
        self.pos_history = {}
        self.cached_target_pos = {}
        self.cached_dist_maps = {}
        
        self.greedy = greedy


    def _select_action_by_iq(self, q_values, char):
        """IQ100以上では従来どおりの行動選択を保ち、100未満だけ判断ミスを加える。"""
        q_values = np.asarray(q_values, dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(q_values))
        if len(valid) == 0:
            return 0

        # まず、IQ実装前と同じ方法で通常行動を選ぶ。
        if self.greedy:
            baseline_action = int(valid[np.argmax(q_values[valid])])
        else:
            valid_q = q_values[valid]
            probs = np.exp((valid_q - np.max(valid_q)) / 0.5)
            probs = probs / probs.sum()
            baseline_action = int(np.random.choice(valid, p=probs))

        iq = float(getattr(char, "effective_iq", getattr(char, "iq", 100.0)))
        decision_accuracy = max(0.0, min(1.0, iq / 100.0))

        # IQ100までは「従来行動を維持できる確率」として扱う。
        if iq <= 100.0:
            if random.random() < decision_accuracy or len(valid) == 1:
                return baseline_action
        else:
            # 100超過分は小さな「判断修正率」へ還元する。
            # 例: IQ130なら15%の確率で、Softmaxが外した行動をargmaxへ修正する。
            best_action = int(valid[np.argmax(q_values[valid])])
            overcap_correction = min(0.20, max(0.0, (iq - 100.0) / 200.0))
            if baseline_action != best_action and random.random() < overcap_correction:
                return best_action
            return baseline_action

        # IQ100未満で判断に失敗した時だけ、通常選択とは別の有効行動を選ぶ。
        alternatives = valid[valid != baseline_action]
        if len(alternatives) == 0:
            return baseline_action

        alternative_q = q_values[alternatives]
        probs = np.exp((alternative_q - np.max(alternative_q)) / 0.5)
        probs = probs / probs.sum()
        return int(np.random.choice(alternatives, p=probs))

    def reset_round(self):
        self.last_actions.clear()
        self.pos_history.clear()
        self.cached_target_pos.clear()
        self.cached_dist_maps.clear()

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
            # スパイクが落ちている → 拾いに行く
            target_pos = tuple(spike_pos)
            carrying, planted_flag = False, False
        else:
            # スパイクが無く、誰かが持っている → 距離に応じて追従/分散する
            holder = next((ch for ch in chars if ch.is_alive and ch.has_spike), None)
            if holder:
                dist_to_holder = max(abs(holder.pos[0] - r), abs(holder.pos[1] - c))
                if random.random() < 0.3:
                    return self.get_next_pos_random(char.pos, grid), "MOVE"
                if dist_to_holder > 5:
                    return self.move_towards_target(char.pos, holder.pos, grid), "MOVE"
                else:
                    return self.get_next_pos_random(char.pos, grid), "MOVE"
            return self.get_next_pos_random(char.pos, grid), "MOVE"

        if self.cached_target_pos.get(char.name) != target_pos:
            self.cached_target_pos[char.name] = target_pos
            self.cached_dist_maps[char.name] = self._compute_bfs_map(target_pos, grid)

        # 💡追加: 設置サイトに実際に到達したら、モデルの判断を待たず強制的にPLANTを選ぶ
        # (defenderのDEFUSE強制ロジックと同じ考え方。これがないと確率的サンプリングで
        #  action=4以外が選ばれて、着いたのに離れてしまうことがある)
        if carrying and tuple(char.pos) == target_pos:
            self.last_actions[char.name] = 4
            self.pos_history.setdefault(char.name, deque(maxlen=7)).append(tuple(char.pos))
            return char.pos, "PLANT"

        obs = self._make_observation(char, target_pos, grid, chars, carrying, planted_flag)

        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0).cpu().numpy()

        # 💡追加: guard中(planted_flag)以外は、action=4(設置/見張り)をそもそも選択肢から除外する。
        # carrying中に未到達でaction=4が選ばれると、run_game.py側で「on_siteでないPLANT」として
        # 何もせずreturnされてしまい、そのtickが完全に無駄になる(移動できずに足止めされる)ため。
        # retrieveフェーズでも同様に無意味な停止を防ぐ。
        if not planted_flag:
            q_values = q_values.copy()
            q_values[4] = -np.inf

        # 実効IQ100なら常に最適手、80なら約80%で最適手を選ぶ。
        # greedy=TrueでもIQの影響は残し、キャラクター差を維持する。
        action = self._select_action_by_iq(q_values, char)

        self.last_actions[char.name] = action
        self.pos_history.setdefault(char.name, deque(maxlen=7)).append(tuple(char.pos))

        if action == 4:
            if carrying:
                return char.pos, "PLANT"
            return char.pos, "MOVE"  # guard中の見張り(停止)

        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = [r + moves[action][0], c + moves[action][1]]
        height, width = grid.shape
        if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width and grid[next_pos[0], next_pos[1]] != 1:
            return next_pos, "MOVE"
        return char.pos, "MOVE"

    def _make_observation(self, char, target_pos, grid, chars, carrying, planted_flag):
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

        enemy = [0.0, 0.0, 0.0]
        if planted_flag:
            for d in chars:
                if d.is_alive and d.team == "D" and self.has_line_of_sight(char.pos, d.pos, grid):
                    edr = (d.pos[0] - pr) / height
                    edc = (d.pos[1] - pc) / width
                    enemy = [1.0, edr, edc]
                    break

        #  味方attacker(guard中)の位置
        teammate = [0.0, 0.0, 0.0]
        if planted_flag:
            other = next(
                (ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != char.name and not ch.has_spike),
                None
            )
            if other is not None:
                teammate = [1.0, (other.pos[0] - pr) / height, (other.pos[1] - pc) / width]

        return np.array(base + walls + last_onehot + dists + phase_flags + enemy + teammate, dtype=np.float32)

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