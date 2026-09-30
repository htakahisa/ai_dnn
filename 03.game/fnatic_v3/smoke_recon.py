"""Wall-only firing positions for recon responses to opposing smoke."""

from collections import deque

from fnatic_v1_rules import pos
from game_core import REVEAL_DURATION_TICKS


class FnaticSmokeRecon:
    MOVE_LIMIT = 7

    def __init__(self):
        self.smokes = {}
        self.completed = {}
        self.pending = {}
        self.last_cast = {}

    def normal_cast_ready(self, char, tick):
        last = self.last_cast.get(char.name)
        return last is None or tick - last >= REVEAL_DURATION_TICKS + 3

    def result(self, controller, char, state):
        if (not char.is_alive or char.ability_name != 'RECON'
                or char.recon_charges <= 0):
            return None
        owner = getattr(controller.game, 'real_game', controller.game)
        records = getattr(owner, 'smokes', state.get('smokes', ()))
        active = {id(s): s for s in records if s.get('remaining_ticks', 0) > 0}
        # Retain references while active so object IDs cannot be reused for a
        # new smoke between this player's decisions.
        for name, done in self.completed.items():
            done.intersection_update(active)
        self.smokes = active
        tick = int(getattr(owner, 'battle_tick', 0))
        pending = self.pending.pop(char.name, None)
        if pending is not None and char.recon_charges < pending[1]:
            if pending[0] in active:
                self.completed.setdefault(char.name, set()).add(pending[0])
            self.last_cast[char.name] = pending[2]

        chars = getattr(owner, 'chars', state.get('chars', ()))
        grid = state['grid']
        targets = []
        for key, smoke in active.items():
            team = smoke.get('team')
            if team is None:
                # Legacy records can be identified only if the owner's name
                # is unambiguous, including owners who have already died.
                teams = {c.team for c in chars if c.name == smoke.get('owner')}
                team = next(iter(teams)) if len(teams) == 1 else None
            if team is None or team == char.team or key in self.completed.get(char.name, ()):
                continue
            cells = [tuple(map(int, p)) for p in smoke.get('cells', ())]
            cells = [p for p in cells if 0 <= p[0] < grid.shape[0]
                     and 0 <= p[1] < grid.shape[1] and grid[p] != 1]
            center = smoke.get('center')
            center = tuple(center) if center is not None else None
            cells.sort(key=lambda p: (p != center, p))
            if cells:
                targets.append((key, cells))
        if not targets:
            return None

        start = pos(char)
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            cell, length = queue.popleft()
            for key, cells in targets:
                aim = next((p for p in cells if p != cell
                            and controller._los(cell, p, grid, smoke=False)), None)
                if aim is None:
                    continue
                if cell == start:
                    path_builder = getattr(owner, '_projectile_path', None)
                    if path_builder is not None and len(path_builder(start, aim)) <= 1:
                        continue
                    action = controller._recon_at_smoke(char, aim, state, key)
                    if action is not None:
                        self.pending[char.name] = (key, char.recon_charges, tick)
                        # Share confirmed-use cooldown with ordinary attacker
                        # utility so it does not immediately spend the reserve.
                        if hasattr(controller, 'utility_pending'):
                            controller.utility_pending[char.name] = (char.recon_charges, tick)
                    return action
                occupied = {pos(c) for c in state.get('chars', ())
                            if c.is_alive and c is not char and pos(c) != start}
                nxt, _, _ = controller._route(start, [cell], grid)
                if nxt in occupied:
                    nxt, _, _ = controller._route(start, [cell], grid, occupied)
                return controller._result(char, nxt, aim)
            if length >= self.MOVE_LIMIT:
                continue
            for dr, dc in controller.CARDINAL_MOVES:
                nxt = (cell[0] + dr, cell[1] + dc)
                if (nxt not in seen and 0 <= nxt[0] < grid.shape[0]
                        and 0 <= nxt[1] < grid.shape[1] and grid[nxt] != 1):
                    seen.add(nxt)
                    queue.append((nxt, length + 1))
        return None
