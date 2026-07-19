# controllers.py
import random
from collections import deque
import numpy as np

class BaseController:
    """すべての操作クラスの基底となるクラス"""
    def decide_move(self, char, game_state):
        raise NotImplementedError

    def get_next_pos_random(self, pos, grid):
        """共通で使えるランダム移動ロジック"""
        r, c = pos
        height, width = grid.shape
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid = [
            (r + dr, c + dc) for dr, dc in moves 
            if 0 <= r + dr < height and 0 <= c + dc < width and grid[r + dr, c + dc] != 1
        ]
        return list(random.choice(valid)) if valid else pos

    def move_towards_target(self, pos, target, grid):
        """BFS（幅優先探索）を用いて、壁を迂回する本当の最短ルートで1マス進む"""
        start = tuple(pos)
        goal = tuple(target)
        
        if start == goal:
            return list(pos)

        height, width = grid.shape
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # BFSのためのキューと探索済み(親ノード)の記録
        queue = deque([start])
        parent = {start: None}
        
        found = False
        while queue:
            curr = queue.popleft()
            if curr == goal:
                found = True
                break
                
            r, c = curr
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    # 壁(1)でなければ進める
                    if grid[nr, nc] != 1 and (nr, nc) not in parent:
                        parent[(nr, nc)] = curr
                        queue.append((nr, nc))
        
        # ゴールまでのルートが見つかった場合、スタートから最初の1歩を逆算する
        if found:
            curr = goal
            while parent[curr] != start:
                curr = parent[curr]
            return list(curr)
            
        # 万が一、完全に孤立したエリアなどで経路がない場合はランダム移動にフォールバック
        return self.get_next_pos_random(pos, grid)


class DefaultAttackerController(BaseController):
    """アタッカー側の標準ロジック"""
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        spike_pos = game_state["spike_pos"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        target_plant_pos = game_state.get("target_plant_pos") # 【追加】今ラウンドの目標プラント位置
        chars = game_state["chars"]
        r, c = char.pos

        # ---------------------------------------------------------------------
        # 2. プラント後の防衛ロジック
        # ---------------------------------------------------------------------
        if is_planted and planted_pos:
            dist_to_spike = max(abs(planted_pos[0] - r), abs(planted_pos[1] - c))
            if dist_to_spike <= 3:
                return self.get_next_pos_random(char.pos, grid)
            else:
                return self.move_towards_target(char.pos, planted_pos, grid)

        # ---------------------------------------------------------------------
        # プラント前の通常ロジック
        # ---------------------------------------------------------------------
        # スパイク持ちは指定されたランダムなプラントサイトを目指す
        if char.has_spike:
            if target_plant_pos:
                # 【変更】一番近い場所ではなく、ゲーム開始時に選ばれた本命サイトへ進む
                return self.move_towards_target(char.pos, target_plant_pos, grid)
            
            # フォールバック用（指定がなければ従来通り一番近い場所へ）
            plants = list(zip(*np.where(grid == 2)))
            if plants:
                target = min(plants, key=lambda p: abs(p[0] - r) + abs(p[1] - c))
                return self.move_towards_target(char.pos, target, grid)
            return self.get_next_pos_random(char.pos, grid)

        # 1. スパイクを持っていないアタッカーは護衛または回収
        if spike_pos:
            return self.move_towards_target(char.pos, spike_pos, grid)
        
        # スパイクを誰かが持っている場合、そのキャラを「ついていく」
        holder = next((c for c in chars if c.is_alive and c.has_spike), None)
        if holder:
            dist_to_holder = max(abs(holder.pos[0] - r), abs(holder.pos[1] - c))
            if random.random() < 0.3:
                return self.get_next_pos_random(char.pos, grid)
            
            if dist_to_holder > 5:
                return self.move_towards_target(char.pos, holder.pos, grid)
            else:
                return self.get_next_pos_random(char.pos, grid)

        return self.get_next_pos_random(char.pos, grid)


class DefaultDefenderController(BaseController):
    """ディフェンダー側の標準ロジック"""
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        r, c = char.pos

        # 3. プラントされた場合の挙動
        if is_planted and planted_pos:
            dist_to_spike = max(abs(planted_pos[0] - r), abs(planted_pos[1] - c))
            if dist_to_spike <= 1:
                return char.pos
            else:
                return self.move_towards_target(char.pos, planted_pos, grid)

        # プラント前は通常のランダム索敵
        return self.get_next_pos_random(char.pos, grid)