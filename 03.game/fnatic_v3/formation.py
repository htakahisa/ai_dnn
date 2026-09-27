"""Pre-plant formation rules and editable Fnatic role priorities."""

from fnatic_v1_rules import distance, pos

from .positions import distances


FRONT_PRIORITY = ('Derke', 'Chronicle', 'Leo', 'Boaster', 'Alfajer')
MID_PRIORITY = ('Alfajer', 'Boaster')
FRONT_SELECTION_RADIUS = 7  # Grid distance: max(row delta, column delta).
MAIN_GROUP_RADIUS = 3      # Walking distance around the carrier.

# Independent Fnatic settings, initially based on GC MID_FORWARD_CONTROL.
MID_CONTROL_POSITIONS = (
    (11, 16), (11, 17), (11, 21), (11, 22), (11, 23),
    (12, 16), (12, 17), (12, 18), (12, 21), (12, 22), (12, 23), (12, 24),
    (13, 16), (13, 17), (13, 18), (13, 23), (13, 24), (13, 25),
)
MID_WATCH_POSITION = (6, 20)


class FnaticFormation:
    def __init__(self):
        self.mid_name = None
        self.mid_target = None
        self.front_name = None

    def assign_mid(self, allies, holder):
        available = {c.name for c in allies if c is not holder and c.name != holder.name}
        if self.mid_name not in available:
            self.mid_name = next((name for name in MID_PRIORITY if name in available), None)
            self.mid_target = None

    def mid_result(self, ctrl, char, grid, blocked, visible):
        if self.mid_target is None:
            goals = [p for p in MID_CONTROL_POSITIONS
                     if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]
                     and grid[p] != 1]
            _, _, self.mid_target = ctrl._route(pos(char), goals, grid)
        nxt, _, _ = ctrl._route(pos(char), [self.mid_target] if self.mid_target else [],
                               grid, blocked)
        return ctrl._result(char, nxt, pos(visible[0]) if visible else MID_WATCH_POSITION)

    def result(self, ctrl, char, holder, allies, grid, blocked, visible, destination=None):
        """Keep one nearby front player, the main group, and one separate Mid."""
        # Team formation uses our own teammates' shared positions. Enemy aim
        # and observations still use the perception view supplied by the game.
        allies = [getattr(c, 'real_character', c) for c in allies]
        holder = next(c for c in allies if c.name == holder.name)
        blocked = {pos(c) for c in allies if c.name != char.name}
        target = destination if destination is not None else ctrl.target
        self.assign_mid(allies, holder)
        if char.name == self.mid_name:
            return self.mid_result(ctrl, char, grid, blocked, visible)
        if target is None:
            return ctrl._result(char, pos(char))

        anchor = pos(holder)
        carrier_lengths = distances(anchor, grid)
        target_lengths = distances(target, grid)
        remaining = target_lengths.get(anchor, float('inf'))
        main = [c for c in allies if c.name != self.mid_name]
        nearby = {c.name: c for c in main if c.name != holder.name
                  and distance(pos(c), anchor) <= FRONT_SELECTION_RADIUS
                  and pos(c) in carrier_lengths}
        self.front_name = next((name for name in FRONT_PRIORITY if name in nearby), None)
        front = nearby.get(self.front_name)
        aim = pos(visible[0]) if visible else target

        # Front goals are 1-3 walking steps closer to the selected plant point.
        forward = [p for p, length in carrier_lengths.items()
                   if 1 <= length <= MAIN_GROUP_RADIUS and p != target
                   and target_lengths.get(p, float('inf')) < remaining]
        if remaining <= 2:
            # Once at site, allow the entry player to continue beyond the plant
            # instead of standing on the carrier's reserved planting cell.
            dr, dc = target[0] - anchor[0], target[1] - anchor[1]
            beyond = [p for p, length in carrier_lengths.items()
                      if 1 <= length <= MAIN_GROUP_RADIUS and p != target
                      and (p[0] - anchor[0]) * dr + (p[1] - anchor[1]) * dc > dr * dr + dc * dc]
            if beyond:
                forward = beyond
            carrier_next, _, _ = ctrl._route(anchor, [target], grid)
            # At a site edge the front player may need to step through the
            # plant and hold beside it; never park on the carrier's next cell.
            forward = [p for p in forward if p != carrier_next]
        forward = list(dict.fromkeys(forward))

        if char.name == holder.name:
            # A leading teammate can occupy a later cell in a narrow lane.
            # Plan the walking route, then check this tick's next cell, instead
            # of requiring the entire path to be empty before advancing.
            nxt, _, _ = ctrl._route(anchor, [target], grid)
            if nxt in blocked:
                nxt, _, _ = ctrl._route(anchor, [target], grid, blocked)
            # The nearby group must keep up; the Mid player is independent.
            if any(carrier_lengths.get(pos(c), float('inf')) > MAIN_GROUP_RADIUS
                   for c in main if c.name != holder.name):
                nxt = anchor
            if front is not None and remaining > 2:
                # Keep a full step of separation after the carrier's move.
                if (target_lengths.get(pos(front), float('inf'))
                        >= target_lengths.get(nxt, remaining)):
                    nxt = anchor
            return ctrl._result(char, nxt, aim)

        # No teammate may sit on the selected planting cell.
        reserved = blocked | {target}
        if char.name == self.front_name:
            goals = [p for p in forward if p not in reserved]
            if goals:
                # Stay around two steps ahead, rather than racing to the site.
                desired = max(1, remaining - 2)
                score = lambda p: abs(target_lengths.get(p, remaining) - desired)
                reachable = distances(pos(char), grid)
                goal = min((p for p in goals if p in reachable),
                           key=lambda p: (score(p), reachable[p], p), default=None)
                # The front player can cross the plant cell to clear the site;
                # it never holds that cell or issues PLANT itself.
                nxt, _, _ = ctrl._route(pos(char), [goal] if goal else [], grid, blocked)
                return ctrl._result(char, nxt, aim)

        # All remaining players are the main group, with no independent lurk.
        carrier_next, _, _ = ctrl._route(anchor, [target], grid)
        excluded = {anchor, target, carrier_next}
        if front is not None:
            excluded.update(forward)
        goals = [p for p, length in carrier_lengths.items()
                 if 1 <= length <= 2 and p not in excluded
                 and p not in blocked]
        if not goals:
            # A narrow lane may have no rear/side slot; keep a reachable nearby
            # cell while still leaving the carrier's next cell and plant free.
            goals = [p for p, length in carrier_lengths.items()
                     if 1 <= length <= MAIN_GROUP_RADIUS
                     and p not in {anchor, target, carrier_next} and p not in blocked]
        # Follow through a lane occupied farther ahead without taking a long
        # detour away from the main group. Only the next step must be vacant.
        nxt, _, _ = ctrl._route(pos(char), goals, grid, {target})
        if nxt in blocked:
            alternative, _, _ = ctrl._route(pos(char), goals, grid, reserved)
            if carrier_lengths.get(alternative, float('inf')) < carrier_lengths.get(pos(char), float('inf')):
                nxt = alternative
            else:
                nxt = pos(char)
        if (carrier_lengths.get(pos(char), float('inf')) <= MAIN_GROUP_RADIUS
                and carrier_lengths.get(nxt, float('inf')) > MAIN_GROUP_RADIUS):
            nxt = pos(char)
        if visible and nxt == pos(char):
            ability = ctrl._ability(char, pos(visible[0]), 'FLASH')
            if ability:
                return ability
        return ctrl._result(char, nxt, aim)
