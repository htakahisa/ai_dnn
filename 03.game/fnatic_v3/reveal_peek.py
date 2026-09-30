"""Short defensive peeks at currently revealed enemies using shot geometry."""

from collections import deque

from abilities_los import AbilityLosMixin
from fnatic_v1_rules import distance, pos
from .retake import URGENT_RETAKE_TICKS


PEEK_RADIUS = 3


class FnaticRevealPeek:
    def result(self, ctrl, char, state):
        if char.team != 'D' or state.get('defender_setup_active'):
            return None
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        if state.get('is_planted') and getattr(owner, 'detonate_timer', state.get('detonate_timer', 55)) < URGENT_RETAKE_TICKS:
            return None
        grid = state['grid']
        enemies = [c for c in state.get('chars', ()) if c.team != char.team and c.is_alive]
        revealed = [c for c in enemies if self._revealed(c)]
        if not revealed:
            return None
        known = [c for c in enemies if self._revealed(c) or ctrl._los(pos(char), pos(c), grid,
                 smoke=not getattr(char, 'sees_through_smoke', False))]
        occupied = {pos(getattr(c, 'real_character', c)) for c in state.get('chars', ())
                    if c.team == char.team and c.is_alive and c.name != char.name}
        occupied.update(pos(c) for c in known)

        def shootable(cell, enemy):
            ignore_smoke = (getattr(char, 'sees_through_smoke', False)
                            or getattr(enemy, 'reveal_remaining', 0) > 0)
            if not ctrl._los(cell, pos(enemy), grid, smoke=not ignore_smoke):
                return False
            line = AbilityLosMixin._line_cells(None, cell, pos(enemy))
            return not (set(line[1:-1]) & occupied)

        start = pos(char)
        current = [c for c in known if shootable(start, c)]
        if current:
            nearest = min(current, key=lambda c: (distance(start, pos(c)), c.name))
            return ctrl._result(char, start, pos(nearest))
        # Explore walking steps, not a square radius: walls and bodies can
        # make a geometrically nearby firing cell require a much longer path.
        queue = deque([(start, 0, start)])
        visited = {start}
        choices = []
        best_length = None
        while queue:
            cell, length, first = queue.popleft()
            if best_length is not None and length > best_length:
                break
            targets = [c for c in revealed if shootable(cell, c)]
            if targets:
                nearest = min(targets, key=lambda c: (distance(cell, pos(c)), c.name))
                choices.append((-len(targets), distance(cell, pos(nearest)), cell, first, nearest))
                best_length = length
                continue
            if length == PEEK_RADIUS:
                continue
            for dr, dc in ctrl.CARDINAL_MOVES:
                nxt = (cell[0] + dr, cell[1] + dc)
                if (nxt in visited or nxt in occupied
                        or not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1])
                        or grid[nxt] == 1):
                    continue
                visited.add(nxt)
                queue.append((nxt, length + 1, nxt if length == 0 else first))
        if not choices:
            return None
        _, _, _, step, enemy = min(choices, key=lambda item: item[:4])
        return ctrl._result(char, step, pos(enemy))

    @staticmethod
    def _revealed(enemy):
        return getattr(enemy, 'reveal_remaining', 0) > 0 or getattr(enemy, 'los_revealed', False)
