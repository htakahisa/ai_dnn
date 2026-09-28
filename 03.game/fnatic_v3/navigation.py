"""Shared ally avoidance without changing stationary ability actions."""

from fnatic_v1_rules import FnaticRulesController, pos
from map_data_defender_setup import is_setup_position_allowed

from .positions import distances


class FnaticNavigation:
    def __init__(self):
        self.context = None
        self.yields = {}
        self.previous = {}

    def begin(self, char, state):
        self.context = {'char': char, 'state': state, 'plan': None}

    def record(self, ctrl, start, goals, blocked, result):
        ctx = self.context
        if ctx is None or tuple(start) != pos(ctx['char']):
            return
        allies = {pos(getattr(c, 'real_character', c)) for c in ctx['state'].get('chars', ())
                  if c.team == ctx['char'].team and c.is_alive and c.name != ctx['char'].name}
        goals = tuple(tuple(p) for p in goals)
        nxt, _, goal = result
        if goal is not None:
            old = ctx['plan']
            blocker = nxt if nxt in allies else None
            if old is not None and old['goals'] == goals:
                blocker = blocker or old['blocked']
            ctx['plan'] = {'goals': goals, 'blocked': blocker}
            if tuple(nxt) == tuple(start) and goal == tuple(start):
                ctx['plan'] = None
        elif set(blocked) & allies:
            # Some callers avoid every body on the entire route. Identify a
            # real obstruction, including one farther down a narrow corridor.
            grid = ctx['state']['grid']
            cursor = tuple(start)
            for _ in range(grid.size):
                step, _, endpoint = FnaticRulesController._route(
                    ctrl, cursor, goals, grid, set(blocked) - allies,
                    setup=bool(ctx['state'].get('defender_setup_active')))
                if endpoint is None or step == cursor:
                    break
                if step in allies:
                    ctx['plan'] = {'goals': goals, 'blocked': step}
                    break
                cursor = step
                if cursor == endpoint:
                    break

    @staticmethod
    def _clock(ctrl, state):
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        setup = bool(state.get('defender_setup_active'))
        phase = getattr(owner, 'defender_setup_phase', None)
        tick = getattr(phase, 'ticks_elapsed', 0) if setup else getattr(owner, 'battle_tick', 0)
        return setup, int(tick)

    def _yield_step(self, ctrl, char, request, grid, occupied, setup):
        requester, goals, gate, _, _ = request
        path = {pos(requester)}
        cursor = pos(requester)
        for _ in range(grid.size):
            nxt, _, end = ctrl._route(cursor, goals, grid, setup=setup)
            if end is None or nxt == cursor:
                break
            path.add(nxt)
            cursor = nxt
            if cursor == end:
                break
        options = [(char.pos[0] + dr, char.pos[1] + dc) for dr, dc in ctrl.CARDINAL_MOVES]
        options = [p for p in options if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]
                   and grid[p] != 1 and p not in occupied
                   and (not setup or is_setup_position_allowed(*p))]
        off_path = [p for p in options if p not in path]
        if off_path:
            return min(off_path, key=lambda p: (p == self.previous.get(char.name), p))
        away = distances(pos(requester), grid)
        forward = [p for p in options if away.get(p, 0) > away.get(pos(char), 0)]
        return min(forward, key=lambda p: (p == self.previous.get(char.name), p)) if forward else None

    def finish(self, ctrl, char, state, result):
        ctx = self.context
        self.context = None
        plan = ctx['plan'] if ctx is not None else None
        if isinstance(result, tuple):
            destination, action = result[0], result[1] if len(result) > 1 else None
        else:
            destination, action = result, None
        protected = ((isinstance(action, str) and action != 'MOVE')
                     or (isinstance(action, dict) and ('ability' in action or 'ultimate' in action))
                     or getattr(char, 'plant_timer', 0) > 0 or getattr(char, 'defuse_timer', 0) > 0)
        if protected:
            return result
        grid = state['grid']
        setup, tick = self._clock(ctrl, state)
        actual = {c.name: getattr(c, 'real_character', c) for c in state.get('chars', ())
                  if c.team == char.team and c.is_alive}
        occupied = {pos(c) for name, c in actual.items() if name != char.name}
        enemies = [c for c in state.get('chars', ()) if c.team != char.team and c.is_alive]
        occupied.update(pos(c) for c in enemies)
        start, nxt = pos(char), tuple(map(int, destination))
        fighting = any(ctrl._los(start, pos(c), grid) or getattr(c, 'reveal_remaining', 0) > 0
                       for c in enemies) if not setup else False
        if getattr(getattr(ctrl, 'retake', None), 'preparing', False):
            # A teammate may need a side step to reach a throwing position.
            fighting = False
        for name, request in list(self.yields.items()):
            requester, goals, gate, expires, phase = request
            if requester.name == char.name and (plan is None or plan['goals'] != goals):
                # A shared guard-slot exchange or another tactical decision
                # can end the original movement before its old endpoint.
                del self.yields[name]
                continue
            current = actual.get(requester.name)
            receiver = actual.get(name)
            gate_distance = ctrl._route(gate, goals, grid, setup=setup)[1]
            # Without a side pocket, yielding means walking ahead. Keep
            # clearing the lane rather than returning as soon as the mover
            # enters the first vacated cell and forcing it to retreat.
            if receiver is not None and current is not None:
                receiver_distance = ctrl._route(pos(receiver), goals, grid, setup=setup)[1]
                mover_distance = ctrl._route(pos(current), goals, grid, setup=setup)[1]
                if receiver_distance < gate_distance and mover_distance >= receiver_distance:
                    gate, gate_distance = pos(receiver), receiver_distance
            if (current is None or receiver is None or phase != setup or tick >= expires or pos(current) == gate
                    or ctrl._route(pos(current), goals, grid, setup=setup)[1] < gate_distance):
                del self.yields[name]
            else:
                self.yields[name] = (current, goals, gate, expires, phase)
        request = self.yields.get(char.name)
        if request is not None and not fighting:
            gate = request[2]
            if nxt == start or nxt == gate:
                # Keep the vacated cell open until the requester passes it.
                if start != gate:
                    return ctrl._result(char, start)
                yielded = self._yield_step(ctrl, char, request, grid, occupied, setup)
                if yielded is not None:
                    self.previous[char.name] = start
                    return ctrl._result(char, yielded)
        blocked = plan['blocked'] if plan else None
        collision = nxt in occupied or (nxt == start and blocked is not None)
        if not collision or (nxt == start and fighting):
            if nxt != start:
                self.previous[char.name] = start
            return result
        if setup and getattr(getattr(ctrl, 'engineer', None), 'target', None) == start:
            return result
        goals = plan['goals'] if plan else (nxt,)
        blocker = next((c for name, c in actual.items() if name != char.name
                        and pos(c) == (blocked if blocked is not None else nxt)), None)
        if blocker is not None:
            self.yields.setdefault(blocker.name, (actual.get(char.name, char), goals, pos(blocker), tick + 8, setup))
        # Formation waiting can still request a clear lane. It must not make
        # a carrier overtake the entry solely because a route was blocked.
        formation = getattr(ctrl, 'formation', None)
        waiting = char.name in getattr(formation, 'navigation_waiting', ())
        waiting |= char.name in getattr(formation, 'passage_waits', {})
        if nxt == start and waiting:
            return result
        detour, length, endpoint = ctrl._route(start, goals, grid, occupied, setup=setup)
        previous = self.previous.get(char.name)
        if endpoint is not None and detour == previous:
            alternative, other_length, other_end = ctrl._route(start, goals, grid,
                                                               occupied | {previous}, setup=setup)
            if other_end is not None and other_length <= length + 2:
                detour = alternative
        if endpoint is not None and detour not in occupied and detour != start:
            self.previous[char.name] = start
            payload = action if isinstance(action, dict) else None
            return (list(detour), payload) if payload is not None else ctrl._result(char, detour)
        if blocker is not None:
            # Continue along the open part of a lane while the blocker yields;
            # a distant occupied endpoint must not hold the entire team back.
            direct, _, end = ctrl._route(start, goals, grid, setup=setup)
            if end is not None and direct not in occupied and direct != start:
                self.previous[char.name] = start
                return ctrl._result(char, direct)
        return ctrl._result(char, start) if nxt in occupied else result
