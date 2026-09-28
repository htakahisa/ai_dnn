"""Pre-plant formation rules and editable Fnatic role priorities."""

from itertools import permutations

from fnatic_v1_rules import distance, pos

from .positions import distances

FRONT_PRIORITY = ("Derke", "Chronicle", "Leo", "Boaster", "Alfajer")
MID_PRIORITY = ("Alfajer", "Boaster")
FRONT_SELECTION_RADIUS = 7  # Grid distance: max(row delta, column delta).
MAIN_GROUP_RADIUS = 3  # Walking distance around the carrier.
REGROUP_RADIUS = 12  # Only these nearby main-team members can hold the carrier.
UNSCREENED_PLANT_RADIUS = 7
UNSCREENED_TIME_LIMIT = 30
SITE_ENTRY_RADIUS = 8
DERKE_ENTRY_DEPTH = 5  # Walkable site cells around the plant, beyond the approach.

# Independent Fnatic settings, initially based on GC MID_FORWARD_CONTROL.
MID_CONTROL_POSITIONS = (
    (11, 16),
    (11, 17),
    (11, 21),
    (11, 22),
    (11, 23),
    (12, 16),
    (12, 17),
    (12, 18),
    (12, 21),
    (12, 22),
    (12, 23),
    (12, 24),
    (13, 16),
    (13, 17),
    (13, 18),
    (13, 23),
    (13, 24),
    (13, 25),
)
MID_WATCH_POSITION = (6, 20)


class FnaticFormation:
    def __init__(self):
        self.mid_name = None
        self.mid_target = None
        self.front_name = None
        self.front_names = ()
        self.previous_frontline = set()
        self.exposed_after_losses = False
        self.loss_context = None
        self.entry_context = None
        self.entry_targets = {}
        self.passage_waits = {}
        self.mid_exclusions = set()
        self.detached_names = set()
        self.rush_context = None
        self.rush_goal = None
        self.navigation_waiting = set()

    def _commit_derke(
        self, ctrl, holder, main, grid, target, remaining, carrier_lengths
    ):
        """Once the entry starts, Derke clears the site without a moving leash."""
        context = (holder.name, target)
        derke = next(
            (c for c in main if c.name == "Derke" and c.name != holder.name), None
        )
        if (
            context != self.rush_context
            or derke is None
            or not getattr(holder, "has_spike", False)
        ):
            self.rush_context = context
            self.rush_goal = None
        if (
            derke is None
            or not getattr(holder, "has_spike", False)
            or self.rush_goal is not None
        ):
            return
        lengths = distances(target, grid)
        if (
            distance(pos(derke), pos(holder)) > FRONT_SELECTION_RADIUS
            or pos(derke) not in carrier_lengths
            or min(remaining, lengths.get(pos(derke), float("inf"))) > SITE_ENTRY_RADIUS
        ):
            return
        approach = [pos(derke)]
        while approach[-1] != target:
            nxt, _, goal = ctrl._route(approach[-1], [target], grid)
            if goal is None or nxt == approach[-1]:
                return
            approach.append(nxt)
        previous = approach[-2] if len(approach) > 1 else pos(holder)
        dr, dc = target[0] - previous[0], target[1] - previous[1]
        candidates = [
            p
            for p, length in lengths.items()
            if 1 <= length <= DERKE_ENTRY_DEPTH and p not in approach
        ]
        if not candidates:
            return
        site = [p for p in candidates if grid[p] == 2]
        if site:
            candidates = site
        occupied = {pos(c) for c in main if c.name != "Derke"}
        free = [p for p in candidates if p not in occupied]
        self.rush_goal = min(
            free or candidates,
            key=lambda p: (
                -((p[0] - target[0]) * dr + (p[1] - target[1]) * dc),
                -lengths[p],
                p,
            ),
        )
        self.passage_waits.pop("Derke", None)

    def _entry_goals(self, ctrl, holder, grid, target, main):
        """Give entries fixed site destinations instead of a moving leash."""
        entry_names = self.front_names + tuple(
            c.name
            for c in main
            if c.name != holder.name and c.name not in self.front_names
        )
        context = (holder.name, target, entry_names, self.rush_goal)
        if context != self.entry_context:
            self.entry_context = context
            self.entry_targets = {}
            route = [pos(holder)]
            while route[-1] != target:
                nxt, _, goal = ctrl._route(route[-1], [target], grid)
                if goal is None or nxt == route[-1]:
                    break
                route.append(nxt)
            previous = route[-2] if len(route) > 1 else target
            dr, dc = target[0] - previous[0], target[1] - previous[1]
            target_lengths = distances(target, grid)
            candidates = [
                p
                for p, length in target_lengths.items()
                if 1 <= length <= 3 and p not in route and p != self.rush_goal
            ]
            candidates.sort(
                key=lambda p: (
                    grid[p] != 2,
                    target_lengths[p],
                    -((p[0] - target[0]) * dr + (p[1] - target[1]) * dc),
                    p,
                )
            )
            if self.rush_goal is not None and "Derke" in entry_names:
                self.entry_targets["Derke"] = self.rush_goal
                entry_names = tuple(name for name in entry_names if name != "Derke")
            if len(candidates) < len(entry_names):
                candidates += [p for p in route[-4:-1] if p not in candidates]
            candidates = candidates[:8]
            front = {c.name: c for c in main}
            lengths = {name: distances(pos(front[name]), grid) for name in entry_names}
            choices = list(permutations(candidates, len(entry_names)))
            if choices:
                chosen = min(
                    choices,
                    key=lambda cells: (
                        sum(candidates.index(p) for p in cells),
                        sum(
                            lengths[name].get(p, float("inf"))
                            for name, p in zip(entry_names, cells)
                        ),
                        cells,
                    ),
                )
                self.entry_targets.update(zip(entry_names, chosen))
        return self.entry_targets

    def _clear_passage(self, ctrl, char, holder, allies, grid, target, visible, result):
        """Let a teammate pass when a formation slot seals a narrow lane."""
        if char.name == self.mid_name or char.name in self.detached_names:
            return result
        anchor = pos(holder)
        remaining = distances(target, grid)
        carrier_lengths = distances(anchor, grid)
        main = [
            c
            for c in allies
            if c.name != self.mid_name and c.name not in self.detached_names
        ]
        blocked = {pos(c) for c in allies if c.name != char.name}
        aim = pos(visible[0]) if visible else target
        waiting = self.passage_waits.get(char.name)
        if waiting:
            other_name, endpoint, vacated, expires = waiting
            other = next((c for c in main if c.name == other_name), None)
            lengths = distances(endpoint, grid)
            if (
                int(getattr(ctrl.game, "battle_tick", 0)) < expires
                and other is not None
                and (
                    char.name != holder.name
                    or carrier_lengths.get(pos(other), float("inf")) <= REGROUP_RADIUS
                )
                and pos(other) not in {endpoint, vacated}
                and other_name not in self.passage_waits
                and lengths.get(pos(other), float("inf")) >= lengths.get(vacated, 0)
            ):
                result = ctrl._result(char, pos(char), aim)
            else:
                del self.passage_waits[char.name]
        # When someone yielded for our passage, finish that passage even if
        # their sidestep moved the normal formation slot back onto our cell.
        for requester, (recipient, endpoint, _, expires) in sorted(
            self.passage_waits.items()
        ):
            if (
                recipient != char.name
                or requester not in {c.name for c in main}
                or int(getattr(ctrl.game, "battle_tick", 0)) >= expires
            ):
                continue
            nxt, direct_length, _ = ctrl._route(pos(char), [endpoint], grid)
            if nxt in blocked:
                alternative, length, _ = ctrl._route(
                    pos(char), [endpoint], grid, blocked
                )
                nxt = alternative if length <= direct_length + 2 else pos(char)
            if nxt != pos(char) and carrier_lengths.get(nxt, float("inf")) <= max(
                FRONT_SELECTION_RADIUS,
                carrier_lengths.get(pos(char), FRONT_SELECTION_RADIUS),
            ):
                return ctrl._result(char, nxt, aim)
        if tuple(result[0]) != pos(char):
            return result
        order = {
            name: index for index, name in enumerate((*self.front_names, holder.name))
        }
        for other in sorted(main, key=lambda c: order.get(c.name, len(order))):
            if other.name == char.name:
                continue
            if (
                char.name == holder.name
                and carrier_lengths.get(pos(other), float("inf")) > REGROUP_RADIUS
            ):
                continue
            # A teammate waiting to yield cannot use the space we vacate.
            # Keep moving instead of creating a circular passage queue.
            if other.name in self.passage_waits:
                continue
            if other.name == holder.name:
                goals = [target]
            elif other.name == "Derke" and self.rush_goal is not None:
                goals = [self.rush_goal]
            elif (
                other.name in self.entry_targets
                and remaining[anchor] <= SITE_ENTRY_RADIUS
            ):
                goals = [self.entry_targets[other.name]]
            elif other.name in self.front_names:
                desired = max(1, remaining[anchor] - 2)
                goals = [
                    p
                    for p, length in carrier_lengths.items()
                    if 1 <= length <= MAIN_GROUP_RADIUS
                    and p != target
                    and remaining[p] < remaining[anchor]
                ]
                if goals:
                    best = min(abs(remaining[p] - desired) for p in goals)
                    goals = [p for p in goals if abs(remaining[p] - desired) == best]
            else:
                goals = [
                    p
                    for p, length in carrier_lengths.items()
                    if 1 <= length <= MAIN_GROUP_RADIUS and p != target
                ]
            other_blocked = {pos(c) for c in allies if c.name != other.name}
            _, occupied_length, _ = ctrl._route(pos(other), goals, grid, other_blocked)
            # Clear a queue cooperatively, including allies that block the
            # exit needed by another teammate farther back.
            _, clear_length, endpoint = ctrl._route(pos(other), goals, grid)
            if endpoint is None or occupied_length <= clear_length + 2:
                continue
            path = [pos(other)]
            while path[-1] != endpoint:
                nxt, _, _ = ctrl._route(path[-1], [endpoint], grid)
                path.append(nxt)
            if pos(char) not in path[1:]:
                continue
            options = [
                p
                for p, length in distances(pos(char), grid).items()
                if length == 1
                and p not in blocked
                and carrier_lengths.get(p, float("inf"))
                <= max(FRONT_SELECTION_RADIUS, carrier_lengths[pos(char)])
            ]
            off_path = [p for p in options if p not in path and p != target]
            if off_path:
                nxt = min(off_path, key=lambda p: (remaining[p], p))
                self.passage_waits[char.name] = (
                    other.name,
                    endpoint,
                    pos(char),
                    int(getattr(ctrl.game, "battle_tick", 0)) + 4,
                )
                return ctrl._result(char, nxt, aim)
            # At a one-cell doorway, proceed until there is room to yield.
            forward = [p for p in options if remaining[p] < remaining[pos(char)]]
            if forward:
                return ctrl._result(
                    char, min(forward, key=lambda p: (remaining[p], p)), aim
                )
        return result

    def assign_mid(self, allies, holder):
        available = {
            c.name
            for c in allies
            if c is not holder
            and c.name != holder.name
            and c.name not in self.mid_exclusions
        }
        if self.mid_name not in available:
            self.mid_name = next(
                (name for name in MID_PRIORITY if name in available), None
            )
            self.mid_target = None

    def mid_result(self, ctrl, char, grid, blocked, visible):
        if self.mid_target is None:
            goals = [
                p
                for p in MID_CONTROL_POSITIONS
                if 0 <= p[0] < grid.shape[0]
                and 0 <= p[1] < grid.shape[1]
                and grid[p] != 1
                and p != getattr(getattr(ctrl, "engineer", None), "mid_target", None)
            ]
            _, _, self.mid_target = ctrl._route(pos(char), goals, grid)
        nxt, _, _ = ctrl._route(
            pos(char), [self.mid_target] if self.mid_target else [], grid, blocked
        )
        return ctrl._result(
            char, nxt, pos(visible[0]) if visible else MID_WATCH_POSITION
        )

    def _loss_response(
        self,
        ctrl,
        char,
        holder,
        allies,
        main,
        grid,
        blocked,
        target,
        target_lengths,
        remaining,
        visible,
        round_timer,
    ):
        """React to loss of actual front players, not ordinary opening order."""
        context = (holder.name, target)
        if context != self.loss_context:
            self.loss_context = context
            self.previous_frontline = set()
            self.exposed_after_losses = False
        if not getattr(holder, "has_spike", False):
            return None
        alive = {c.name for c in allies}
        ahead = {
            c.name
            for c in main
            if c.name != holder.name
            and target_lengths.get(pos(c), float("inf")) < remaining
        }
        if (
            not ahead
            and self.previous_frontline
            and not self.previous_frontline.intersection(alive)
        ):
            self.exposed_after_losses = True
        if ahead:
            self.previous_frontline = ahead
            self.exposed_after_losses = False
        elif self.previous_frontline.intersection(alive):
            # A living front player moved back; that is not a casualty.
            self.previous_frontline = set()
        aim = pos(visible[0]) if visible else target
        if not self.exposed_after_losses:
            return None
        if (
            remaining > UNSCREENED_PLANT_RADIUS
            and round_timer > UNSCREENED_TIME_LIMIT
            and ctrl._rotate_target(holder, grid)
        ):
            # A new site has a new front. Do not reuse entry destinations,
            # yield tickets, or the casualty history from the abandoned site.
            self.loss_context = (holder.name, ctrl.target)
            self.previous_frontline = set()
            self.exposed_after_losses = False
            self.front_name = None
            self.front_names = ()
            self.entry_context = None
            self.entry_targets = {}
            self.passage_waits = {}
            return None
        # Continue the original entry when close or late. If the opposite site
        # has no reachable mapped plant, keep advancing instead of deadlocking.
        if char.name == holder.name:
            nxt, _, _ = ctrl._route(pos(char), [target], grid)
            if nxt in blocked:
                nxt, _, _ = ctrl._route(pos(char), [target], grid, blocked)
            return ctrl._result(char, nxt, aim)
        return None

    def result(
        self,
        ctrl,
        char,
        holder,
        allies,
        grid,
        blocked,
        visible,
        destination=None,
        round_timer=None,
        transit=False,
    ):
        if round_timer is None:
            owner = getattr(ctrl.game, "real_game", ctrl.game)
            round_timer = getattr(owner, "round_timer", 100)
        self.navigation_waiting.discard(char.name)
        result = self._formation_result(
            ctrl, char, holder, allies, grid, blocked, visible, destination, round_timer, transit
        )
        if destination is not None or ctrl.target is None:
            return result
        allies = [getattr(c, "real_character", c) for c in allies]
        holder = next(c for c in allies if c.name == holder.name)
        if (
            char.name == "Derke"
            and self.rush_goal is not None
            and pos(char) != self.rush_goal
        ):
            # Old passage tickets must not send the committed entry back to
            # a short formation slot while the team is following him in.
            self.passage_waits.pop("Derke", None)
            return result
        return self._clear_passage(
            ctrl, char, holder, allies, grid, ctrl.target, visible, result
        )

    def _formation_result(
        self,
        ctrl,
        char,
        holder,
        allies,
        grid,
        blocked,
        visible,
        destination=None,
        round_timer=100,
        transit=False,
    ):
        """Keep the nearby front pair, main group, and one separate Mid."""
        # Team formation uses our own teammates' shared positions. Enemy aim
        # and observations still use the perception view supplied by the game.
        allies = [getattr(c, "real_character", c) for c in allies]
        holder = next(c for c in allies if c.name == holder.name)
        blocked = {pos(c) for c in allies if c.name != char.name}
        target = destination if destination is not None else ctrl.target
        self.assign_mid(allies, holder)
        if target is None:
            return ctrl._result(char, pos(char))

        anchor = pos(holder)
        carrier_lengths = distances(anchor, grid)
        target_lengths = distances(target, grid)
        remaining = target_lengths.get(anchor, float("inf"))
        main = [
            c
            for c in allies
            if c.name != self.mid_name and c.name not in self.detached_names
        ]
        loss_target = ctrl.target if transit else target
        loss_lengths = distances(loss_target, grid) if transit else target_lengths
        loss_result = self._loss_response(
            ctrl,
            char,
            holder,
            allies,
            main,
            grid,
            blocked,
            loss_target,
            loss_lengths,
            loss_lengths.get(anchor, float("inf")),
            visible,
            round_timer,
        )
        if loss_result is not None:
            return loss_result
        if destination is None and target != ctrl.target:
            target = ctrl.target
            target_lengths = distances(target, grid)
            remaining = target_lengths.get(anchor, float("inf"))
        if char.name == self.mid_name:
            return self.mid_result(ctrl, char, grid, blocked, visible)
        if not transit:
            self._commit_derke(ctrl, holder, main, grid, target, remaining, carrier_lengths)
        nearby = {
            c.name: c
            for c in main
            if c.name != holder.name
            and distance(pos(c), anchor) <= FRONT_SELECTION_RADIUS
            and pos(c) in carrier_lengths
        }
        self.front_names = tuple(name for name in FRONT_PRIORITY if name in nearby)[:2]
        if self.rush_goal is not None and "Derke" not in self.front_names:
            derke = next(c for c in main if c.name == "Derke")
            nearby["Derke"] = derke
            self.front_names = ("Derke", *self.front_names[:1])
        self.front_name = self.front_names[0] if self.front_names else None
        front = nearby.get(self.front_name)
        aim = pos(visible[0]) if visible else target
        if remaining <= SITE_ENTRY_RADIUS and not transit:
            self._entry_goals(ctrl, holder, grid, target, main)

        if char.name == "Derke" and self.rush_goal is not None:
            entry_blocked = blocked | {pos(c) for c in visible if c.is_alive}
            nxt, _, _ = ctrl._route(pos(char), [self.rush_goal], grid)
            if nxt in entry_blocked:
                nxt, _, _ = ctrl._route(
                    pos(char), [self.rush_goal], grid, entry_blocked
                )
            return ctrl._result(char, nxt, aim)

        # Front goals are 1-3 walking steps closer to the selected plant point.
        forward = [
            p
            for p, length in carrier_lengths.items()
            if 1 <= length <= MAIN_GROUP_RADIUS
            and p != target
            and target_lengths.get(p, float("inf")) < remaining
        ]
        if remaining <= 2:
            # Once at site, allow the entry player to continue beyond the plant
            # instead of standing on the carrier's reserved planting cell.
            dr, dc = target[0] - anchor[0], target[1] - anchor[1]
            beyond = [
                p
                for p, length in carrier_lengths.items()
                if 1 <= length <= MAIN_GROUP_RADIUS
                and p != target
                and (p[0] - anchor[0]) * dr + (p[1] - anchor[1]) * dc
                > dr * dr + dc * dc
            ]
            if beyond:
                forward = beyond
            carrier_next, _, _ = ctrl._route(anchor, [target], grid)
            # At a site edge the front player may need to step through the
            # plant and hold beside it; never park on the carrier's next cell.
            forward = [p for p in forward if p != carrier_next]
            if not forward:
                forward = [
                    p
                    for p, length in carrier_lengths.items()
                    if 1 <= length <= FRONT_SELECTION_RADIUS
                    and target_lengths.get(p) == 1
                    and p != carrier_next
                ]
        forward = list(dict.fromkeys(forward))

        if char.name == holder.name:
            # A leading teammate can occupy a later cell in a narrow lane.
            # Plan the walking route, then check this tick's next cell, instead
            # of requiring the entire path to be empty before advancing.
            nxt, _, _ = ctrl._route(anchor, [target], grid)
            if nxt in blocked:
                nxt, _, _ = ctrl._route(anchor, [target], grid, blocked)
                if target_lengths.get(nxt, float("inf")) >= remaining:
                    nxt = anchor
            # Wait only for main-team members within twelve walking steps.
            # On entry, follow teammates ahead rather than waiting for them
            # to return; nearby lagging main-team members still need to join.
            site_entered = any(
                c.name != holder.name
                and grid[pos(c)] == 2
                and (c.pos[1] < grid.shape[1] // 2) == (target[1] < grid.shape[1] // 2)
                and target_lengths.get(pos(c), float("inf")) < remaining
                for c in main
            )
            entry_started = (
                remaining <= SITE_ENTRY_RADIUS
                or self.rush_goal is not None
                or site_entered
            )
            if any(
                carrier_lengths.get(pos(c), float("inf")) <= REGROUP_RADIUS
                and (
                    not entry_started
                    or target_lengths.get(pos(c), float("inf")) >= remaining
                )
                and carrier_lengths.get(pos(c), float("inf"))
                > (
                    FRONT_SELECTION_RADIUS
                    if entry_started or c.name in self.front_names
                    else MAIN_GROUP_RADIUS
                )
                for c in main
                if c.name != holder.name
                and not (c.name == "Derke" and self.rush_goal is not None)
            ):
                nxt = anchor
                self.navigation_waiting.add(char.name)
            close_front = tuple(
                name
                for name in self.front_names
                if carrier_lengths.get(pos(nearby[name]), float("inf"))
                <= REGROUP_RADIUS
            )
            screened_site = remaining <= SITE_ENTRY_RADIUS and any(
                grid[pos(nearby[name])] == 2
                or pos(nearby[name]) == self.entry_targets.get(name)
                for name in close_front
            )
            if close_front and remaining > 2 and not screened_site:
                # Keep a full step of separation after the carrier's move.
                if all(
                    target_lengths.get(pos(nearby[name]), float("inf"))
                    >= target_lengths.get(nxt, remaining)
                    for name in close_front
                ):
                    nxt = anchor
                    self.navigation_waiting.add(char.name)
            return ctrl._result(char, nxt, aim)

        # No teammate may sit on the selected planting cell.
        reserved = blocked | {target}
        if char.name in self.entry_targets and remaining <= SITE_ENTRY_RADIUS:
            goal = self.entry_targets[char.name]
            # Keep following a corridor while an ally occupies a later cell.
            # Only the next step must be empty; a distant blocker must not
            # freeze every teammate still approaching the doorway.
            nxt, direct_length, _ = ctrl._route(pos(char), [goal], grid)
            if nxt in blocked:
                alternative, length, _ = ctrl._route(pos(char), [goal], grid, blocked)
                nxt = alternative if length <= direct_length + 2 else pos(char)
            if nxt == pos(char) and pos(char) != goal:
                for other in main:
                    cell = pos(other)
                    if (
                        other.name == char.name
                        or self.entry_targets.get(other.name) != cell
                    ):
                        continue
                    if (
                        ctrl._route(pos(char), [goal], grid, blocked - {cell})[2]
                        is not None
                    ):
                        self.entry_targets[other.name] = goal
                        self.entry_targets[char.name] = cell
                        nxt, _, _ = ctrl._route(pos(char), [cell], grid, blocked)
                        break
            if (
                carrier_lengths.get(pos(char), float("inf")) <= FRONT_SELECTION_RADIUS
                and carrier_lengths.get(nxt, float("inf")) > FRONT_SELECTION_RADIUS
            ):
                nxt = pos(char)
            return ctrl._result(char, nxt, aim)
        if char.name in self.front_names:
            goals = [p for p in forward if p not in reserved]
            if goals:
                # Stay around two steps ahead, rather than racing to the site.
                offset = (
                    3 - self.front_names.index(char.name)
                    if len(self.front_names) > 1
                    else 2
                )
                desired = max(1, remaining - offset)
                score = lambda p: abs(target_lengths.get(p, remaining) - desired)
                reachable = distances(pos(char), grid, blocked)
                # Let the front pair pass a bottleneck. If we block the other
                # front player's route to its slot (or off the plant cell),
                # advance one more step rather than holding the passage.
                for name in self.front_names:
                    if name == char.name:
                        continue
                    other = nearby[name]
                    other_blocked = {pos(c) for c in allies if c.name != name}
                    other_offset = 3 - self.front_names.index(name)
                    other_desired = max(1, remaining - other_offset)
                    other_goals = [
                        p
                        for p in forward
                        if p not in other_blocked and p not in {target, pos(other)}
                    ]
                    if not other_goals and remaining <= 3:
                        other_goals = [
                            p
                            for p in carrier_lengths
                            if target_lengths.get(p) == 1
                            and p not in other_blocked
                            and p not in {target, pos(other)}
                            and carrier_lengths[p] <= FRONT_SELECTION_RADIUS
                        ]
                    if not other_goals:
                        # The only forward endpoint may be our current cell.
                        # Move aside so the other entry can take it over.
                        other_goals = [pos(char)]
                    best = min(
                        abs(target_lengths.get(p, remaining) - other_desired)
                        for p in other_goals
                    )
                    other_goals = [
                        p
                        for p in other_goals
                        if abs(target_lengths.get(p, remaining) - other_desired) == best
                    ]
                    _, blocked_length, _ = ctrl._route(
                        pos(other), other_goals, grid, other_blocked
                    )
                    _, clear_length, clear_goal = ctrl._route(
                        pos(other), other_goals, grid, other_blocked - {pos(char)}
                    )
                    if clear_goal is None or blocked_length <= clear_length + 1:
                        continue
                    if remaining <= 2:
                        dr, dc = target[0] - anchor[0], target[1] - anchor[1]
                        progress = (
                            lambda p: (p[0] - anchor[0]) * dr + (p[1] - anchor[1]) * dc
                        )
                    else:
                        progress = lambda p: -target_lengths.get(p, float("inf"))
                    deeper = [
                        p
                        for p in goals
                        if p in reachable and progress(p) > progress(pos(char))
                    ]
                    if not deeper:
                        deeper = [p for p in goals if p in reachable and p != pos(char)]
                    if not deeper and remaining <= 3:
                        deeper = [
                            p
                            for p in carrier_lengths
                            if target_lengths.get(p) == 1
                            and p in reachable
                            and p not in reserved
                            and p != pos(char)
                            and carrier_lengths[p] <= FRONT_SELECTION_RADIUS
                        ]
                    if not deeper and remaining <= 3:
                        deeper = [
                            p
                            for p in carrier_lengths
                            if p in reachable
                            and p not in reserved
                            and p != pos(char)
                            and carrier_lengths[p] <= FRONT_SELECTION_RADIUS
                            and distance(p, pos(char)) == 1
                            and target_lengths.get(p, float("inf")) <= remaining + 2
                        ]
                    if deeper:
                        goals = deeper
                        break
                goal = min(
                    (p for p in goals if p in reachable),
                    key=lambda p: (score(p), reachable[p], p),
                    default=None,
                )
                # The front player can cross the plant cell to clear the site;
                # it never holds that cell or issues PLANT itself.
                nxt, route_length, _ = ctrl._route(
                    pos(char), [goal] if goal else [], grid, blocked
                )
                goal_lengths = distances(goal, grid) if goal is not None else {}
                if route_length > goal_lengths.get(pos(char), float("inf")) + 2:
                    nxt, _, _ = ctrl._route(pos(char), [goal] if goal else [], grid)
                    if nxt in blocked:
                        nxt = pos(char)
                limit = (
                    FRONT_SELECTION_RADIUS
                    if remaining <= 3
                    else max(
                        MAIN_GROUP_RADIUS + 1,
                        carrier_lengths.get(pos(char), MAIN_GROUP_RADIUS),
                    )
                )
                if carrier_lengths.get(nxt, float("inf")) > limit:
                    nxt, _, _ = ctrl._route(pos(char), [goal] if goal else [], grid)
                    if nxt in blocked or carrier_lengths.get(nxt, float("inf")) > limit:
                        nxt = pos(char)
                return ctrl._result(char, nxt, aim)

        # All remaining players are the main group, with no independent lurk.
        carrier_next, _, _ = ctrl._route(anchor, [target], grid)
        excluded = {anchor, target, carrier_next}
        if front is not None:
            excluded.update(forward)
        goals = [
            p
            for p, length in carrier_lengths.items()
            if 1 <= length <= 2 and p not in excluded and p not in blocked
        ]
        if not goals:
            # A narrow lane may have no rear/side slot; keep a reachable nearby
            # cell while still leaving the carrier's next cell and plant free.
            goals = [
                p
                for p, length in carrier_lengths.items()
                if 1 <= length <= MAIN_GROUP_RADIUS
                and p not in {anchor, target, carrier_next}
                and p not in blocked
            ]
        # Follow through a lane occupied farther ahead without taking a long
        # detour away from the main group. Only the next step must be vacant.
        nxt, direct_length, _ = ctrl._route(pos(char), goals, grid, {target})
        if nxt in blocked:
            alternative, alternative_length, _ = ctrl._route(
                pos(char), goals, grid, reserved
            )
            if (
                carrier_lengths.get(alternative, float("inf"))
                < carrier_lengths.get(pos(char), float("inf"))
                or alternative_length <= direct_length + 2
            ):
                nxt = alternative
            else:
                nxt = pos(char)
        if (
            carrier_lengths.get(pos(char), float("inf")) <= MAIN_GROUP_RADIUS
            and carrier_lengths.get(nxt, float("inf")) > MAIN_GROUP_RADIUS
        ):
            nxt = pos(char)
        return ctrl._result(char, nxt, aim)
