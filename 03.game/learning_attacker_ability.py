# learning_attacker_ability.py
# 💡このファイルはtrain_attacker_ability.py以外の他ファイル(train_attacker_multi, learning_attacker_multi等)に依存しない。
from collections import deque
from pathlib import Path
import numpy as np
import torch


from controllers import BaseController
from train_attacker_ability import (
    DuelingQNetwork,
    EnemyMemoryTracker,
    ABILITY_TYPES,
    OBS_DIM,
    N_ACTIONS,
    K_ENEMIES,
    ENEMY_MEMORY_TICKS,
    DEFUSE_REQUIRED,
    ESCORT_OFFSET_MAX,
)


class _EscortHelper:
    """gradient_walkはFixedEscortController固有のロジックなのでここに残すが、
    move_towards_target自体はBaseControllerのものをそのまま使う(重複実装しない)。"""

    def __init__(self, offset_max=ESCORT_OFFSET_MAX):
        self.offset_max = offset_max
        self._base = BaseController()

    def _gradient_walk(self, start, dist_map, grid, steps, seek_smaller, min_dist_from_goal=1):
        height, width = grid.shape
        pos = tuple(start)
        for _ in range(steps):
            r, c = pos
            if seek_smaller and dist_map[r, c] <= min_dist_from_goal:
                break
            best = None
            best_val = dist_map[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width) or grid[nr, nc] == 1:
                    continue
                val = dist_map[nr, nc]
                if not np.isfinite(val):
                    continue
                if seek_smaller and val < min_dist_from_goal:
                    continue
                if seek_smaller and val < best_val:
                    best_val, best = val, (nr, nc)
                elif not seek_smaller and val > best_val:
                    best_val, best = val, (nr, nc)
            if best is None:
                break
            pos = best
        return pos

    def compute_target(self, escort_pos, dist_map, grid, role, steps_override=None, min_dist_from_goal=1):
        seek_smaller = (role == "front")
        steps = steps_override if steps_override is not None else self.offset_max
        return self._gradient_walk(escort_pos, dist_map, grid, steps, seek_smaller, min_dist_from_goal=min_dist_from_goal)

    def next_move(self, escort_pos, target, grid, occupied_positions):
        blocked_grid = grid.copy()
        for pos in occupied_positions:
            pr, pc = pos
            if tuple(pos) != tuple(escort_pos):
                blocked_grid[pr, pc] = 1
        next_pos = self._base.move_towards_target(escort_pos, target, blocked_grid)
        if tuple(next_pos) == tuple(escort_pos):
            next_pos = self._base.move_towards_target(escort_pos, target, grid)
        return next_pos

# =====================================================================
# 🎮 ability対応アタッカーコントローラー（他コントローラーを継承しない独立実装）
# =====================================================================
class LearningAttackerAbilityController(BaseController):
    """ability対応版。共通基盤のBaseControllerを継承する(controllers.pyは全コントローラー共有の基盤)。"""

    def __init__(self, model_path="attacker_ability_data/dqn_attacker_ability_final.pt", greedy=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path_obj = Path(model_path)
        full_path = model_path_obj if model_path_obj.is_absolute() else Path(__file__).resolve().parent / model_path_obj

        self.model = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(self.device)
        self.model.load_state_dict(torch.load(str(full_path), map_location=self.device))
        self.model.eval()

        self.greedy = greedy
        self.last_actions = {}
        self.pos_history = {}
        self.cached_target_pos = {}
        self.cached_dist_maps = {}
        self.cached_choke_points = {}

        self.escort_helper = _EscortHelper()
        self.enemy_memory_per_char = {}

    def reset_round(self):
        self.last_actions.clear()
        self.pos_history.clear()
        self.cached_target_pos.clear()
        self.cached_dist_maps.clear()
        self.cached_choke_points.clear()
        self.enemy_memory_per_char.clear()

    # -------------------------------------------------------------
    # 汎用ヘルパー(BaseControllerから複製)
    # -------------------------------------------------------------
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

    def _get_memory(self, char_name):
        if char_name not in self.enemy_memory_per_char:
            self.enemy_memory_per_char[char_name] = EnemyMemoryTracker(
                memory_ticks=ENEMY_MEMORY_TICKS, k_enemies=K_ENEMIES
            )
        return self.enemy_memory_per_char[char_name]

    def _charge_for(self, char):
        return {
            "flash": char.flash_charges,
            "smoke": char.smoke_charges,
            "recon": char.recon_charges,
        }.get(char.ability_type, 0)

    def _compute_choke_points(self, target_pos, grid):
        dist_map = self.cached_dist_maps.get(target_pos)
        if dist_map is None:
            return []
        height, width = grid.shape
        candidates = []
        for r in range(height):
            for c in range(width):
                if grid[r, c] == 1:
                    continue
                d = dist_map[r, c]
                if not np.isfinite(d) or d < 2 or d > 10:
                    continue
                n = r - 1 >= 0 and grid[r - 1, c] != 1
                s = r + 1 < height and grid[r + 1, c] != 1
                w = c - 1 >= 0 and grid[r, c - 1] != 1
                e = c + 1 < width and grid[r, c + 1] != 1
                vertical_corridor = n and s and not w and not e
                horizontal_corridor = w and e and not n and not s
                if vertical_corridor or horizontal_corridor:
                    candidates.append((d, (r, c)))
        candidates.sort(key=lambda x: x[0])
        return [pos for _, pos in candidates]

    # -------------------------------------------------------------
    # 護衛(スパイクを持っていないアタッカー)の移動
    # -------------------------------------------------------------
    def _decide_escort_move(self, char, holder, grid, chars, target_plant_pos):
        escorts = [ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != holder.name and not ch.has_spike]
        escorts.sort(key=lambda ch: max(abs(ch.pos[0] - holder.pos[0]), abs(ch.pos[1] - holder.pos[1])))
        escort_rank = next((i for i, ch in enumerate(escorts) if ch.name == char.name), None)

        target_key = f"{char.name}_escort_target"
        if self.cached_target_pos.get(target_key) != tuple(target_plant_pos):
            self.cached_target_pos[target_key] = tuple(target_plant_pos)
            self.cached_dist_maps[target_key] = self._compute_bfs_map(tuple(target_plant_pos), grid)
        dist_map = self.cached_dist_maps[target_key]

        offset_steps = ESCORT_OFFSET_MAX if escort_rank is None else max(2, ESCORT_OFFSET_MAX - escort_rank * 3)
        target = self.escort_helper.compute_target(tuple(char.pos), dist_map, grid, "front", steps_override=offset_steps)

        occupied = [tuple(holder.pos)] + [tuple(e.pos) for e in escorts if e.name != char.name]
        next_pos = self.escort_helper.next_move(tuple(char.pos), target, grid, occupied)

        self.last_actions[char.name] = None
        self.pos_history.setdefault(char.name, deque(maxlen=7)).append(tuple(char.pos))
        return next_pos, "MOVE"

    # -------------------------------------------------------------
    # 観測構築
    # -------------------------------------------------------------
    def _make_observation(self, char, target_pos, grid, chars, carrying, planted_flag, game_state):
        pr, pc = char.pos
        tr, tc = target_pos
        height, width = grid.shape

        base = [pr / (height - 1), pc / (width - 1), tr / (height - 1), tc / (width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_act = self.last_actions.get(char.name, None)
        last_onehot = [0.0] * N_ACTIONS
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

        # 💡変更: carry中はallyを使わない
        ally = [0.0, 0.0, 0.0]
        if planted_flag:
            other = next(
                (ch for ch in chars if ch.is_alive and ch.team == "A" and ch.name != char.name and not ch.has_spike),
                None
            )
            if other is not None:
                ally = [1.0, (other.pos[0] - pr) / height, (other.pos[1] - pc) / width]

        defuse_info = [0.0, 0.0]
        defuse_progress_map = game_state.get("defender_defuse_info")
        if planted_flag and defuse_progress_map:
            max_progress = 0.0
            any_defusing = False
            for _, (timer, required) in defuse_progress_map.items():
                if timer > 0:
                    any_defusing = True
                    max_progress = max(max_progress, timer / required)
            if any_defusing:
                defuse_info = [1.0, min(max_progress, 1.0)]

        ability_type = char.ability_type if char.ability_type in ABILITY_TYPES else "none"
        ability_onehot = [0.0] * 4
        ability_onehot[ABILITY_TYPES.index(ability_type)] = 1.0
        charges = [1.0 if self._charge_for(char) > 0 else 0.0]
        self_blind = [1.0 if char.blind_remaining > 0 else 0.0]

        return np.array(base + walls + last_onehot + dists + phase_flags
                         + enemy_feats + ally + defuse_info
                         + ability_onehot + charges + self_blind, dtype=np.float32)

    # -------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        spike_pos = game_state["spike_pos"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        target_plant_pos = game_state.get("target_plant_pos")
        chars = game_state["chars"]
        r, c = char.pos

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
            holder = next((ch for ch in chars if ch.is_alive and ch.has_spike), None)
            if holder is not None:
                return self._decide_escort_move(char, holder, grid, chars, target_plant_pos)
            return self.get_next_pos_random(char.pos, grid), "MOVE"

        if self.cached_target_pos.get(char.name) != target_pos:
            self.cached_target_pos[char.name] = target_pos
            self.cached_dist_maps[char.name] = self._compute_bfs_map(target_pos, grid)

        obs = self._make_observation(char, target_pos, grid, chars, carrying, planted_flag, game_state)

        #if char.has_spike:
        #    print(f"{char.name} pos={char.pos} last_action_stored={self.last_actions.get(char.name)} obs_last_onehot={obs[8:14]}")

        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0).cpu().numpy()

        q_values = q_values.copy()
        if not planted_flag:
            if carrying and grid[r, c] == 2:
                pass
            else:
                q_values[4] = -np.inf
            q_values[5] = -np.inf
        else:
            if self._charge_for(char) <= 0:
                q_values[5] = -np.inf

        if self.greedy:
            action = int(np.argmax(q_values))
        else:
            probs = np.exp((q_values - np.max(q_values)) / 0.5)
            probs = probs / probs.sum()
            action = int(np.random.choice(len(probs), p=probs))

        self.last_actions[char.name] = action
        self.pos_history.setdefault(char.name, deque(maxlen=7)).append(tuple(char.pos))

        if action == 5:
            memory = self._get_memory(char.name)
            if memory.memory:
                nearest = min(
                    memory.memory.items(),
                    key=lambda kv: max(
                        abs(kv[1]["pos"][0] - char.pos[0]),
                        abs(kv[1]["pos"][1] - char.pos[1]),
                    ),
                )
                target_cell = nearest[1]["pos"]
            else:
                if target_pos not in self.cached_choke_points:
                    self.cached_choke_points[target_pos] = self._compute_choke_points(target_pos, grid)
                choke_points = self.cached_choke_points[target_pos]
                target_cell = choke_points[0] if choke_points else target_pos
            return target_cell, "ABILITY"

        if action == 4:
            if carrying:
                return char.pos, "PLANT"
            return char.pos, "MOVE"

    
        #if char.has_spike:
        #    print(f"{char.name} PRE-FALLBACK r={r} c={c} char.pos={char.pos}")


        holder = next((ch for ch in chars if ch.is_alive and ch.has_spike), None)
        if holder is not None:
            return self._decide_escort_move(char, holder, grid, chars, target_plant_pos)
        return self.get_next_pos_random(char.pos, grid), "MOVE"