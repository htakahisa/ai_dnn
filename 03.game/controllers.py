# controllers.py
import random
from collections import deque

import numpy as np


class BaseController:
    """すべての操作クラスの基底となるクラス。"""

    CARDINAL_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def decide_move(self, char, game_state):
        raise NotImplementedError

    @staticmethod
    def has_line_of_sight(p1, p2, grid):
        """Bresenham法で壁による射線遮断を判定する。"""
        x0, y0, x1, y1 = int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0])
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        curr_x, curr_y = x0, y0

        while True:
            if grid[curr_y, curr_x] == 1:
                return False
            if curr_x == x1 and curr_y == y1:
                return True

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                curr_x += sx
            if e2 <= dx:
                err += dx
                curr_y += sy

    @staticmethod
    def _alive_occupied_positions(chars, moving_char=None):
        """
        生存キャラクターが占有している座標を返す。

        moving_char自身の現在地はブロック対象から除外する。
        """
        occupied = set()

        for other in chars or []:
            if other is moving_char:
                continue
            if not getattr(other, "is_alive", True):
                continue

            pos = getattr(other, "pos", None)
            if pos is None or len(pos) != 2:
                continue

            occupied.add((int(pos[0]), int(pos[1])))

        return occupied

    @staticmethod
    def _in_bounds(pos, grid):
        r, c = int(pos[0]), int(pos[1])
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

    @classmethod
    def _is_walkable(cls, pos, grid, blocked):
        r, c = int(pos[0]), int(pos[1])
        return (
            cls._in_bounds((r, c), grid)
            and grid[r, c] != 1
            and (r, c) not in blocked
        )

    def get_next_pos_random(self, pos, grid, chars=None, moving_char=None):
        """
        上下左右から、壁・マップ外・生存キャラクター占有マスを除外して
        ランダムに1マス移動する。

        動ける場所がない場合はその場に留まる。
        """
        r, c = int(pos[0]), int(pos[1])
        blocked = self._alive_occupied_positions(chars, moving_char)

        valid = []
        for dr, dc in self.CARDINAL_MOVES:
            candidate = (r + dr, c + dc)
            if self._is_walkable(candidate, grid, blocked):
                valid.append(candidate)

        return list(random.choice(valid)) if valid else [r, c]

    def _candidate_goals(self, goal, grid, blocked, allow_adjacent_goal):
        """
        BFSで到達対象にするゴール候補を返す。

        goalが空いていればgoalそのものを使う。
        goalが他キャラクターに占有されている場合、または
        allow_adjacent_goal=Trueの場合は、goalの上下左右にある
        到達可能マスも候補にする。
        """
        goal = (int(goal[0]), int(goal[1]))
        candidates = []

        if self._is_walkable(goal, grid, blocked):
            candidates.append(goal)

        if allow_adjacent_goal or goal in blocked:
            for dr, dc in self.CARDINAL_MOVES:
                adjacent = (goal[0] + dr, goal[1] + dc)
                if self._is_walkable(adjacent, grid, blocked):
                    candidates.append(adjacent)

        # 重複を順序を保って除去する
        return list(dict.fromkeys(candidates))

    def move_towards_target(
        self,
        pos,
        target,
        grid,
        chars=None,
        moving_char=None,
        allow_adjacent_goal=False,
    ):
        """
        BFSで壁と生存キャラクターを避けながら、目標へ1マス進む。

        targetが味方などに占有されている場合は、targetそのものへ
        進もうとせず、到達可能な隣接マスをゴール候補にする。

        allow_adjacent_goal=Trueの場合は、targetが空いていても
        target周辺の隣接マスへの到達を許可する。護衛・追従向け。
        """
        start = (int(pos[0]), int(pos[1]))
        goal = (int(target[0]), int(target[1]))

        blocked = self._alive_occupied_positions(chars, moving_char)
        blocked.discard(start)

        if start == goal:
            return [start[0], start[1]]

        candidate_goals = self._candidate_goals(
            goal=goal,
            grid=grid,
            blocked=blocked,
            allow_adjacent_goal=allow_adjacent_goal,
        )

        if not candidate_goals:
            return self.get_next_pos_random(
                pos=start,
                grid=grid,
                chars=chars,
                moving_char=moving_char,
            )

        candidate_goal_set = set(candidate_goals)
        queue = deque([start])
        parent = {start: None}
        reached_goal = None

        while queue:
            current = queue.popleft()

            if current in candidate_goal_set:
                reached_goal = current
                break

            r, c = current
            for dr, dc in self.CARDINAL_MOVES:
                nxt = (r + dr, c + dc)

                if nxt in parent:
                    continue
                if not self._is_walkable(nxt, grid, blocked):
                    continue

                parent[nxt] = current
                queue.append(nxt)

        if reached_goal is None:
            return self.get_next_pos_random(
                pos=start,
                grid=grid,
                chars=chars,
                moving_char=moving_char,
            )

        # startの直後の1マスまで親をたどる
        step = reached_goal
        while parent[step] is not None and parent[step] != start:
            step = parent[step]

        if parent[step] is None:
            return [start[0], start[1]]

        return [int(step[0]), int(step[1])]

    def shortest_path_distance(self, pos, target, grid):
        """壁を考慮した上下左右の最短距離。到達不能なら無限大。"""
        start = (int(pos[0]), int(pos[1]))
        goal = (int(target[0]), int(target[1]))
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            (r, c), distance = queue.popleft()
            for dr, dc in self.CARDINAL_MOVES:
                nxt = (r + dr, c + dc)
                if nxt in visited or not self._in_bounds(nxt, grid):
                    continue
                if grid[nxt[0], nxt[1]] == 1:
                    continue
                if nxt == goal:
                    return distance + 1
                visited.add(nxt)
                queue.append((nxt, distance + 1))
        return float("inf")


class DefaultAttackerController(BaseController):
    """アタッカー側の標準ロジック。"""

    def __init__(self):
        self.spike_retriever_name = None

    def reset_round(self):
        self.spike_retriever_name = None

    def _choose_spike_retriever(self, chars, spike_pos, grid):
        alive_attackers = [
            other for other in chars
            if getattr(other, "is_alive", True)
            and getattr(other, "team", None) == "A"
        ]
        current = next(
            (other for other in alive_attackers
             if getattr(other, "name", None) == self.spike_retriever_name),
            None,
        )
        if current is not None:
            return current
        if not alive_attackers:
            self.spike_retriever_name = None
            return None
        retriever = min(
            alive_attackers,
            key=lambda other: (
                self.shortest_path_distance(other.pos, spike_pos, grid),
                str(getattr(other, "name", "")),
            ),
        )
        self.spike_retriever_name = getattr(retriever, "name", None)
        return retriever

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        spike_pos = game_state.get("spike_pos")
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")
        target_plant_pos = game_state.get("target_plant_pos")
        chars = game_state.get("chars", [])
        r, c = int(char.pos[0]), int(char.pos[1])

        # アビリティ判定は移動より先に行う
        ability_action = self._decide_ability(char, game_state)
        if ability_action:
            return list(char.pos), ability_action

        # 1. プラント後：スパイク周辺を防衛
        if is_planted and planted_pos:
            dist_to_spike = max(
                abs(int(planted_pos[0]) - r),
                abs(int(planted_pos[1]) - c),
            )

            if dist_to_spike <= 3:
                return self.get_next_pos_random(
                    char.pos,
                    grid,
                    chars=chars,
                    moving_char=char,
                )

            return self.move_towards_target(
                char.pos,
                planted_pos,
                grid,
                chars=chars,
                moving_char=char,
                allow_adjacent_goal=True,
            )

        # 2. スパイク所持者は、サイト属性のマスなら場所を問わず即設置する。
        if getattr(char, "has_spike", False):
            self.spike_retriever_name = None
            if grid[r, c] == 2:
                return list(char.pos), "PLANT"

            if target_plant_pos is not None:
                return self.move_towards_target(
                    char.pos,
                    target_plant_pos,
                    grid,
                    chars=chars,
                    moving_char=char,
                )

            plant_cells = list(zip(*np.where(grid == 2)))
            if plant_cells:
                target = min(
                    plant_cells,
                    key=lambda cell: self.shortest_path_distance(char.pos, cell, grid),
                )
                return self.move_towards_target(
                    char.pos,
                    target,
                    grid,
                    chars=chars,
                    moving_char=char,
                )

            return self.get_next_pos_random(
                char.pos, grid, chars=chars, moving_char=char
            )

        holder = next(
            (
                other
                for other in chars
                if getattr(other, "is_alive", True)
                and getattr(other, "team", None) == char.team
                and getattr(other, "has_spike", False)
            ),
            None,
        )

        # 3. 落下スパイクは最短の1人だけが直接回収し、他は回収役を護衛する。
        if holder is None and spike_pos is not None:
            retriever = self._choose_spike_retriever(chars, spike_pos, grid)
            if retriever is char:
                if list(map(int, char.pos)) == list(map(int, spike_pos)):
                    return list(char.pos)
                return self.move_towards_target(
                    char.pos, spike_pos, grid, chars=chars, moving_char=char
                )

            if retriever is not None:
                return self.move_towards_target(
                    char.pos, retriever.pos, grid,
                    chars=chars, moving_char=char, allow_adjacent_goal=True,
                )

        # 4. スパイク所持者を護衛
        if holder is not None and holder is not char:
            dist_to_holder = max(
                abs(int(holder.pos[0]) - r),
                abs(int(holder.pos[1]) - c),
            )

            if random.random() < 0.3:
                return self.get_next_pos_random(
                    char.pos,
                    grid,
                    chars=chars,
                    moving_char=char,
                )

            if dist_to_holder > 5:
                # holder本人のマスは占有されているので隣接マスへ向かう
                return self.move_towards_target(
                    char.pos,
                    holder.pos,
                    grid,
                    chars=chars,
                    moving_char=char,
                    allow_adjacent_goal=True,
                )

            return self.get_next_pos_random(
                char.pos,
                grid,
                chars=chars,
                moving_char=char,
            )

        return self.get_next_pos_random(
            char.pos,
            grid,
            chars=chars,
            moving_char=char,
        )

    def _decide_ability(self, char, game_state):
        """
        アビリティを使うべきか判定する。

        戻り値:
        - None
        - {"ability": "SMOKE"|"FLASH"|"RECON", "target": (r, c)}
        """
        chars = game_state.get("chars", [])
        grid = game_state["grid"]
        target_plant_pos = game_state.get("target_plant_pos")
        planted_pos = game_state.get("planted_pos")

        visible_enemies = [
            enemy
            for enemy in chars
            if getattr(enemy, "is_alive", True)
            and getattr(enemy, "team", None) != char.team
            and self.has_line_of_sight(char.pos, enemy.pos, grid)
        ]

        # 1. リコン
        site_pos = planted_pos if planted_pos is not None else target_plant_pos
        if (
            getattr(char, "recon_charges", 0) > 0
            and not visible_enemies
            and site_pos is not None
        ):
            dist_to_site = max(
                abs(int(site_pos[0]) - int(char.pos[0])),
                abs(int(site_pos[1]) - int(char.pos[1])),
            )
            recon_trigger_distance = 10

            if dist_to_site <= recon_trigger_distance:
                return {
                    "ability": "RECON",
                    "target": (int(site_pos[0]), int(site_pos[1])),
                }

        # 2. スモーク
        if getattr(char, "smoke_charges", 0) > 0 and len(visible_enemies) >= 2:
            enemy_pos = visible_enemies[0].pos
            return {
                "ability": "SMOKE",
                "target": (int(enemy_pos[0]), int(enemy_pos[1])),
            }

        # 3. フラッシュ
        if getattr(char, "flash_charges", 0) > 0 and visible_enemies:
            closest_enemy = min(
                visible_enemies,
                key=lambda enemy: max(
                    abs(int(enemy.pos[0]) - int(char.pos[0])),
                    abs(int(enemy.pos[1]) - int(char.pos[1])),
                ),
            )
            distance = max(
                abs(int(closest_enemy.pos[0]) - int(char.pos[0])),
                abs(int(closest_enemy.pos[1]) - int(char.pos[1])),
            )

            if distance <= 5:
                return {
                    "ability": "FLASH",
                    "target": (
                        int(closest_enemy.pos[0]),
                        int(closest_enemy.pos[1]),
                    ),
                }

        return None


class DefaultDefenderController(BaseController):
    """ディフェンダー側の標準ロジック。"""

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")
        chars = game_state.get("chars", [])
        r, c = int(char.pos[0]), int(char.pos[1])

        # 移動や解除より先に、使用価値があるアビリティを判定する。
        ability_action = self._decide_ability(char, game_state)
        if ability_action is not None:
            return list(char.pos), ability_action

        if is_planted and planted_pos is not None:
            dist_to_spike = max(
                abs(int(planted_pos[0]) - r),
                abs(int(planted_pos[1]) - c),
            )

            if dist_to_spike <= 1:
                return list(char.pos), "DEFUSE"

            # スパイク本体のマスではなく解除可能な隣接マスへ進む
            return self.move_towards_target(
                char.pos,
                planted_pos,
                grid,
                chars=chars,
                moving_char=char,
                allow_adjacent_goal=True,
            )

        return self.get_next_pos_random(
            char.pos,
            grid,
            chars=chars,
            moving_char=char,
        )

    def _decide_ability(self, char, game_state):
        """ディフェンダー側のアビリティ使用判断。"""
        chars = game_state.get("chars", [])
        grid = game_state["grid"]
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")
        target_plant_pos = game_state.get("target_plant_pos")

        visible_enemies = [
            enemy
            for enemy in chars
            if getattr(enemy, "is_alive", True)
            and getattr(enemy, "team", None) != char.team
            and self.has_line_of_sight(char.pos, enemy.pos, grid)
        ]

        # 1. RECON: 敵が見えていないとき、攻防の中心となるサイトを索敵する。
        site_pos = planted_pos if is_planted and planted_pos is not None else target_plant_pos
        if (
            getattr(char, "recon_charges", 0) > 0
            and not visible_enemies
            and site_pos is not None
        ):
            distance_to_site = max(
                abs(int(site_pos[0]) - int(char.pos[0])),
                abs(int(site_pos[1]) - int(char.pos[1])),
            )
            if distance_to_site <= 10:
                return {
                    "ability": "RECON",
                    "target": (int(site_pos[0]), int(site_pos[1])),
                }

        # 2. SMOKE: 複数の敵が射線内にいるとき、最も近い敵付近を遮断する。
        if getattr(char, "smoke_charges", 0) > 0 and len(visible_enemies) >= 2:
            closest_enemy = min(
                visible_enemies,
                key=lambda enemy: max(
                    abs(int(enemy.pos[0]) - int(char.pos[0])),
                    abs(int(enemy.pos[1]) - int(char.pos[1])),
                ),
            )
            return {
                "ability": "SMOKE",
                "target": (
                    int(closest_enemy.pos[0]),
                    int(closest_enemy.pos[1]),
                ),
            }

        # 3. FLASH: 近距離に視認中の敵がいるときに使用する。
        if getattr(char, "flash_charges", 0) > 0 and visible_enemies:
            closest_enemy = min(
                visible_enemies,
                key=lambda enemy: max(
                    abs(int(enemy.pos[0]) - int(char.pos[0])),
                    abs(int(enemy.pos[1]) - int(char.pos[1])),
                ),
            )
            distance = max(
                abs(int(closest_enemy.pos[0]) - int(char.pos[0])),
                abs(int(closest_enemy.pos[1]) - int(char.pos[1])),
            )
            if distance <= 5:
                return {
                    "ability": "FLASH",
                    "target": (
                        int(closest_enemy.pos[0]),
                        int(closest_enemy.pos[1]),
                    ),
                }

        return None


class UserInputController(BaseController):
    """
    人間の入力によって動かすコントローラー。

    選択したキャラクターを、クリックした地点へ
    BFS最短経路で1マスずつ移動させる。
    """

    def __init__(self):
        super().__init__()
        self.selected_char = None
        self.targets = {}

    def reset_round(self):
        """ラウンド開始時に選択状態・目的地をリセットする。"""
        self.selected_char = None
        self.targets.clear()

    def handle_click(self, r, c, grid, chars, my_team):
        """
        r, c:
            クリックされたマス座標
        grid:
            マップグリッド
        chars:
            全キャラクター
        my_team:
            操作対象チーム
        """
        r, c = int(r), int(c)

        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            self.selected_char = None
            return

        if grid[r, c] == 1:
            self.selected_char = None
            return

        clicked_char = next(
            (
                char
                for char in chars
                if getattr(char, "is_alive", True)
                and getattr(char, "team", None) == my_team
                and tuple(map(int, char.pos)) == (r, c)
            ),
            None,
        )

        if clicked_char is not None:
            self.selected_char = clicked_char.name
            return

        if self.selected_char is not None:
            self.targets[self.selected_char] = (r, c)
            self.selected_char = None

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")

        # 手動D操作時も、隣接していれば解除を継続する
        if char.team == "D" and is_planted and planted_pos is not None:
            distance = max(
                abs(int(char.pos[0]) - int(planted_pos[0])),
                abs(int(char.pos[1]) - int(planted_pos[1])),
            )
            if distance <= 1:
                return list(char.pos), "DEFUSE"

        target = self.targets.get(char.name)
        if target is None:
            return list(char.pos)

        if tuple(map(int, char.pos)) == tuple(map(int, target)):
            self.targets.pop(char.name, None)
            return list(char.pos)

        next_pos = self.move_towards_target(
            char.pos,
            target,
            grid,
            chars=chars,
            moving_char=char,
        )

        # 他キャラクターにより一時的に経路が塞がれても、
        # 目的地は消さず次Tickに再試行する。
        return next_pos
