# controllers.py
import random
from collections import deque
import numpy as np

class BaseController:
    """すべての操作クラスの基底となるクラス"""
    def decide_move(self, char, game_state):
        raise NotImplementedError
        
    @staticmethod
    def has_line_of_sight(p1, p2, grid):
        x0, y0, x1, y1 = p1[1], p1[0], p2[1], p2[0]
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        curr_x, curr_y = x0, y0
        while True:
            if grid[curr_y, curr_x] == 1:
                return False
            if curr_x == x1 and curr_y == y1:
                return True
            e2 = 2 * err
            if e2 >= dy: err += dy; curr_x += sx
            if e2 <= dx: err += dx; curr_y += sy




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
        target_plant_pos = game_state.get("target_plant_pos")
        chars = game_state["chars"]
        r, c = char.pos

        # ================================================================
        # 【新】アビリティチェック（移動前に実行）
        # ================================================================
        ability_action = self._decide_ability(char, game_state)
        if ability_action:
            return char.pos, ability_action

        # ================================================================
        # 移動ロジック（常に最適移動）
        # ================================================================
        
        # 1. プラント後の防衛ロジック
        if is_planted and planted_pos:
            dist_to_spike = max(abs(planted_pos[0] - r), abs(planted_pos[1] - c))
            if dist_to_spike <= 3:
                return self.get_next_pos_random(char.pos, grid)
            else:
                optimal = self.move_towards_target(char.pos, planted_pos, grid)
                return optimal

        # 2. プラント前の通常ロジック
        # 【最優先】スパイク持ちの挙動
        if char.has_spike:
            if target_plant_pos:
                if list(char.pos) == list(target_plant_pos):
                    return char.pos  
                optimal = self.move_towards_target(char.pos, target_plant_pos, grid)
                return optimal
            
            plants = list(zip(*np.where(grid == 2)))
            if plants:
                target = plants[0] 
                if list(char.pos) == list(target):
                    return char.pos
                optimal = self.move_towards_target(char.pos, target, grid)
                return optimal
            return self.get_next_pos_random(char.pos, grid)

        # 現在誰かがスパイクを持っているか確認
        holder = next((c for c in chars if c.is_alive and c.has_spike), None)

        # スパイクが落ちており、生存している味方の誰もスパイクを持っていない場合は回収に行く
        if holder is None and spike_pos is not None:
            if list(char.pos) == list(spike_pos):
                return char.pos
            optimal = self.move_towards_target(char.pos, spike_pos, grid)
            return optimal
        
        # すでに他の味方がスパイクを持っている場合、そのキャラを護衛（追従）する
        if holder:
            dist_to_holder = max(abs(holder.pos[0] - r), abs(holder.pos[1] - c))
            if random.random() < 0.3:
                return self.get_next_pos_random(char.pos, grid)
            
            if dist_to_holder > 5:
                optimal = self.move_towards_target(char.pos, holder.pos, grid)
                return optimal
            else:
                return self.get_next_pos_random(char.pos, grid)

        return self.get_next_pos_random(char.pos, grid)

    def _decide_ability(self, char, game_state):
        """
        アビリティを使うべきかチェック。
        
        戻り値：
        - None（使わない）
        - {"ability": "SMOKE"|"FLASH"|"RECON", "target": (r, c)} 形式
        """
        chars = game_state["chars"]
        grid = game_state["grid"]
        
        # 射線が通ってる敵を取得
        visible_enemies = [
            enemy for enemy in chars
            if enemy.is_alive
            and enemy.team != char.team
            and self.has_line_of_sight(char.pos, enemy.pos, grid)
        ]
        
        # ===== 1. リコン（索敵）=====
        # ルール：敵が見えない かつ リコンチャージがある
        if char.recon_charges > 0 and len(visible_enemies) == 0:
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            dr, dc = random.choice(directions)
            target_r = char.pos[0] + dr * 5
            target_c = char.pos[1] + dc * 5
            target_r = max(0, min(grid.shape[0] - 1, target_r))
            target_c = max(0, min(grid.shape[1] - 1, target_c))
            return {
                "ability": "RECON",
                "target": (target_r, target_c)
            }
        
        # ===== 2. スモーク（牽制・移動援護）=====
        # ルール：敵が2人以上見える
        if char.smoke_charges > 0 and len(visible_enemies) >= 2:
            enemy_pos = visible_enemies[0].pos
            return {
                "ability": "SMOKE",
                "target": tuple(enemy_pos)
            }
        
        # ===== 3. フラッシュ（敵閃光）=====
        # ルール：敵が1人以上見え、かつ5マス以内に接近
        if char.flash_charges > 0 and len(visible_enemies) > 0:
            closest_enemy = min(
                visible_enemies,
                key=lambda e: max(
                    abs(e.pos[0] - char.pos[0]), 
                    abs(e.pos[1] - char.pos[1])
                )
            )
            dist = max(
                abs(closest_enemy.pos[0] - char.pos[0]), 
                abs(closest_enemy.pos[1] - char.pos[1])
            )
            if dist <= 5:
                return {
                    "ability": "FLASH",
                    "target": tuple(closest_enemy.pos)
                }
        
        return None


class DefaultDefenderController(BaseController):
    """ディフェンダー側の標準ロジック"""
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        r, c = char.pos

        # プラントされた場合の挙動
        if is_planted and planted_pos:
            dist_to_spike = max(abs(planted_pos[0] - r), abs(planted_pos[1] - c))
            if dist_to_spike <= 1:
                return char.pos
            else:
                optimal = self.move_towards_target(char.pos, planted_pos, grid)
                return optimal

        # プラント前は通常のランダム索敵
        return self.get_next_pos_random(char.pos, grid)


class UserInputController(BaseController):
    """人間の入力（マウスクリック）によって動かすコントローラー。
    選択したキャラクターをクリックした地点へBFS最短経路で移動させる。"""

    def __init__(self):
        super().__init__()
        self.selected_char = None   # 現在選択中のキャラ名
        self.targets = {}           # キャラ名 -> 目的地座標(タプル)

    def reset_round(self):
        """ラウンド開始時に選択状態・目的地をリセットする"""
        self.selected_char = None
        self.targets.clear()

    def handle_click(self, r, c, grid, chars, my_team):
        """
        キャンバスクリック時に呼び出す。
        r, c: クリックされたマス座標
        grid: マップグリッド
        chars: 全キャラクターのリスト
        my_team: このコントローラーが担当するチーム（"A" or "D"）
        """
        # 壁をクリックしたら選択解除
        if grid[r, c] == 1:
            self.selected_char = None
            return

        # 自チームの生存キャラがそのマスにいればそれを選択する（移動中でも選択可能）
        clicked_char = next(
            (ch for ch in chars if ch.is_alive and ch.team == my_team and tuple(ch.pos) == (r, c)),
            None
        )
        if clicked_char is not None:
            self.selected_char = clicked_char.name
            return

        # 選択中のキャラがいれば、そのマスを新しい目的地として設定する
        if self.selected_char is not None:
            self.targets[self.selected_char] = (r, c)
            self.selected_char = None  # 指示を出したら選択解除

    def decide_move(self, char, game_state):
        grid = game_state["grid"]

        # 💡追加: defender操作時、プラント後にスパイク隣接エリアにいれば自動的に解除を試みる
        # (attackerの自動PLANTと同様、クリック操作だけでは解除の意思表示ができないため)
        is_planted = game_state.get("is_planted")
        planted_pos = game_state.get("planted_pos")
        if char.team == "D" and is_planted and planted_pos:
            dist = max(abs(char.pos[0] - planted_pos[0]), abs(char.pos[1] - planted_pos[1]))
            if dist <= 1:
                return char.pos, "DEFUSE"
                
        target = self.targets.get(char.name)
        if target is None:
            return char.pos  # 目的地未設定ならその場に留まる

        if tuple(char.pos) == target:
            del self.targets[char.name]
            return char.pos  # 到着したら目的地をクリア

        return self.move_towards_target(char.pos, target, grid)