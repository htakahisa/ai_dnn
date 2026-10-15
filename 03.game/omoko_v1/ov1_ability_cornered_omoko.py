"""おもこ固有の個人能力「追い詰められたおもこ」。"""

import random

# 持続時間（tick）。99なら99tick、0なら無効。各ラウンド1回発動する。
OMOKO_CORNERED_DURATION_TICKS = 99
OMOKO_CORNERED_NAME = "追い詰められたおもこ"

# 一時的な動作確認用。通常条件へ戻すときはFalseにする。
# Trueでは味方の人数に関係なく、setupを含むラウンド開始後の指定tickで発動。
# おもこが死亡済み、敵が全滅済みの場合は発動しない。
DEBUG_OMOKO_CORNERED_ENABLED = False
DEBUG_OMOKO_CORNERED_START_TICK = 65

_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_STEPS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
          (1, 0), (1, -1), (0, -1), (-1, -1))
_PRIORITIES = ((3, 4, 5), (2, 6), (1, 7), (0,))


class CorneredOmokoMixin:
    def _omoko_cornered_active(self, char, *, battle_tick=None):
        if char.name != "おもこ" or not char.is_alive:
            return False
        if not any(c.is_alive and c.team != char.team for c in self.chars):
            char.omoko_cornered_remaining = 0
            return False
        if not char.omoko_cornered_triggered:
            if OMOKO_CORNERED_DURATION_TICKS <= 0:
                return False
            if DEBUG_OMOKO_CORNERED_ENABLED:
                setup = getattr(self, "defender_setup_phase", None)
                if getattr(setup, "active", False):
                    return False
                # 移動時点ではprocess_battleによるtick加算より前。
                live_tick = self.battle_tick + 1 if battle_tick is None else battle_tick
                elapsed = getattr(setup, "ticks_elapsed", 0) + live_tick
                if elapsed < DEBUG_OMOKO_CORNERED_START_TICK:
                    return False
            elif any(c.is_alive and c.team == char.team and c is not char
                     for c in self.chars):
                return False
            char.omoko_cornered_triggered = True
            char.omoko_cornered_remaining = int(OMOKO_CORNERED_DURATION_TICKS)
        return char.omoko_cornered_remaining > 0

    def _omoko_cornered_movement_blocked(self, char):
        return (
            getattr(char, "movement_disabled_remaining", 0) > 0
            or getattr(char, "fate_loom_remaining", 0) > 0
            or self._ramp_blocks_movement(char)
            or any(p.get("owner") == char.name
                   for p in getattr(self, "escape_portals", []))
        )

    def _prepare_omoko_cornered_movement(self, char):
        """通常のAI行動を置き換え、全員の移動後まで瞬間移動を待つ。"""
        self._trigger_cell_effects(char)
        char.omoko_cornered_blocked_this_tick = self._omoko_cornered_movement_blocked(char)
        if char.movement_disabled_remaining > 0:
            char.movement_disabled_remaining -= 1
        char.moved_last_tick = bool(char.moved_this_tick)
        char.moved_this_tick = False
        char.stopped_after_move_this_tick = False
        char.was_in_smoke_before_move = tuple(char.pos) in self._smoke_cells()
        char.entered_smoke_this_tick = False
        char.exited_smoke_this_tick = False
        char.collecting_orb_this_tick = False
        char.orb_collect_timer = 0
        char.collecting_orb_pos = None
        char.is_collecting_orb = False
        char.plant_timer = 0
        char.is_planting = False
        char.defuse_timer = 0
        if self.active_defuser_name == char.name:
            self.active_defuser_name = None

    def _omoko_cornered_destinations(self, char, enemy):
        facing = _DIRECTIONS.index(enemy.facing)
        old_pos = tuple(char.pos)
        for offsets in _PRIORITIES:
            cells = []
            for offset in offsets:
                dr, dc = _STEPS[(facing + offset) % 8]
                pos = (enemy.pos[0] + dr, enemy.pos[1] + dc)
                if (0 <= pos[0] < self.height and 0 <= pos[1] < self.width
                        and self.grid[pos] != 1
                        and not self._is_position_occupied(char, pos, old_pos)):
                    cells.append(pos)
            if cells:
                return cells
        return []

    def _finish_omoko_cornered_combat(self):
        """能力で敵を全滅させたDefenderを、射撃終了後に解除位置へ移す。"""
        for char in self.chars:
            if (char.name != "おもこ" or not char.is_alive
                    or not getattr(char, "omoko_cornered_this_tick", False)
                    or any(c.is_alive and c.team != char.team for c in self.chars)):
                continue
            # 残り時間があっても終了し、次tickから通常の解除判断を許可する。
            char.omoko_cornered_remaining = 0
            char.omoko_cornered_this_tick = False
            char.omoko_cornered_target = None
            if (char.team != "D" or not self.is_planted or self.planted_pos is None
                    or self._omoko_cornered_movement_blocked(char)):
                continue
            old_pos = tuple(char.pos)
            cells = []
            for dr, dc in _STEPS:
                pos = (self.planted_pos[0] + dr, self.planted_pos[1] + dc)
                if (0 <= pos[0] < self.height and 0 <= pos[1] < self.width
                        and self.grid[pos] != 1
                        and not self._is_position_occupied(char, pos, old_pos)):
                    cells.append(pos)
            if not cells:
                continue
            destination = old_pos if old_pos in cells else random.choice(cells)
            self._update_occupancy_after_move(old_pos, destination)
            char.pos = list(destination)
            char.facing = self._facing_towards(destination, self.planted_pos)
            char.moved_this_tick = char.moved_this_tick or destination != old_pos
            if destination != old_pos:
                self._trigger_cell_effects(char, during_battle=True)
            self._finalize_movement_transition_state(char)

    def _advance_omoko_cornered(self):
        """現在の敵の向きで移動先を決め、通常射撃へ標的を渡す。"""
        for char in self.chars:
            char.omoko_cornered_target = None
            char.omoko_cornered_this_tick = False
            if not self._omoko_cornered_active(char, battle_tick=self.battle_tick):
                continue
            char.omoko_cornered_this_tick = True
            char.omoko_cornered_remaining -= 1
            blocked = char.omoko_cornered_blocked_this_tick
            char.omoko_cornered_blocked_this_tick = False
            if blocked or self._omoko_cornered_movement_blocked(char):
                continue
            enemies = [c for c in self.chars if c.is_alive and c.team != char.team]
            if len(enemies) > 1:
                enemies = [c for c in enemies if c.name != char.omoko_cornered_last_target]
            candidates = [(enemy, self._omoko_cornered_destinations(char, enemy))
                          for enemy in enemies]
            candidates = [(enemy, cells) for enemy, cells in candidates if cells]
            if not candidates:
                continue
            enemy, cells = random.choice(candidates)
            destination = random.choice(cells)
            old_pos = tuple(char.pos)
            self._update_occupancy_after_move(old_pos, destination)
            char.pos = list(destination)
            char.facing = self._facing_towards(destination, enemy.pos)
            char.moved_this_tick = destination != old_pos
            char.omoko_cornered_target = enemy
            self._trigger_cell_effects(char, during_battle=True)
            self._finalize_movement_transition_state(char)
