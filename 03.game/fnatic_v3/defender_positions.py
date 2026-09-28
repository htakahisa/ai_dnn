"""Fnatic's editable pre-plant defensive positions, shared across setup/live."""

import random

import numpy as np

from fnatic_v1_rules import pos
from map_data_defender_setup import is_setup_position_allowed

from .map_data_defender_fnatic import DEFENDER_POSITION_STR, POSITION_WEIGHTS
from .positions import distances, parse_grid, region


class FnaticDefenderPositions:
    def __init__(self, map_text=None, weights=None):
        text = DEFENDER_POSITION_STR if map_text is None else map_text
        self.grid = parse_grid(text) if text.strip() else None
        self.weights = dict(POSITION_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(range(3, 10)) or any(w <= 0 for w in self.weights.values()):
            raise ValueError('Fnatic defender weights must be positive for each marker 3 through 9')
        self.candidates = {
            marker: tuple(tuple(map(int, p)) for p in zip(*np.where(self.grid == marker)))
            if self.grid is not None else ()
            for marker in range(3, 10)
        }
        self.reset_round()

    def reset_round(self):
        self.targets = {}
        self.setup_blocked = None
        self.drop_region = None
        self.saved_targets = None
        self.reserved = set()
        self.trap_regions = set()
        self.coverage_context = None
        self.region_limits = None
        self.engineer_region = None
        self.preferred_engineer_region = None

    def cover_traps(self, traps, allies, grid):
        live = {c.name for c in allies if c.is_alive}
        covered = {region(trap['pos'], grid) for trap in traps
                   if trap.get('team') == 'D' and trap.get('owner') == 'Alfajer'
                   and 'Alfajer' in live}
        if covered != self.trap_regions:
            self.trap_regions = covered
            self.coverage_context = None
            self.region_limits = None
            self.engineer_region = None

    def _coverage(self, actual, legal, grid, setup):
        if not self.trap_regions or self.drop_region is not None:
            self.region_limits = None
            self.coverage_context = None
            return
        cells = set().union(*legal.values())
        capacities = {label: sum(region(p, grid) == label for p in cells)
                      for label in ('A', 'MID', 'B')}
        labels = [label for label, count in capacities.items() if count]
        context = (frozenset(self.trap_regions), tuple(c.name for c in actual), setup, tuple(labels),
                   self.preferred_engineer_region)
        if self.coverage_context == context:
            return
        self.coverage_context = context
        limits = dict.fromkeys(labels, 0)
        engineer = next((c for c in actual if c.name == 'Alfajer'), None)
        self.engineer_region = None
        if engineer is not None:
            covered = [p for p in legal[engineer.name] if region(p, grid) in self.trap_regions]
            preferred = [p for p in legal[engineer.name] if region(p, grid) == self.preferred_engineer_region]
            covered = preferred or covered
            if covered:
                old = self.targets.get(engineer.name)
                lengths = distances(pos(engineer), grid)
                closest = old if old in covered else min(covered, key=lambda p: (
                    lengths.get(p, float('inf')), p))
                self.engineer_region = region(closest, grid)
                limits[self.engineer_region] = 1
        while sum(limits.values()) < len(actual):
            available = [label for label in labels if limits[label] < capacities[label]]
            if not available:
                break
            # Cover the untrapped routes first; with five players and one
            # trapped region this gives one there and two on each other route.
            scores = {label: (1 if label in self.trap_regions else 2) / (limits[label] + 1)
                      for label in available}
            score = max(scores.values())
            label = random.choice([label for label in available if scores[label] == score])
            limits[label] += 1
        self.region_limits = limits

    def focus_drop(self, dropped, grid):
        if dropped is None:
            if self.drop_region is not None:
                self.targets = self.saved_targets or {}
                self.saved_targets = None
                self.drop_region = None
        elif self.drop_region is None:
            # Share the first report until pickup, rather than rerolling each
            # player's regional orders due to IQ noise in the spike position.
            self.saved_targets = dict(self.targets)
            self.targets = {}
            self.drop_region = region(dropped, grid)

    def validate(self, grid):
        # A blank template is inactive and preserves the existing controller,
        # including its use on custom simulator maps.
        if not any(self.candidates.values()):
            return
        if self.grid.shape != grid.shape:
            raise ValueError('Fnatic defender position map must match game map dimensions')
        if not np.array_equal(self.grid == 1, grid == 1):
            raise ValueError('Fnatic defender position map walls must match the game map')

    def goal(self, char, allies, grid, setup=False):
        if not any(self.candidates.values()):
            return None
        blocked = ()
        if setup:
            if self.setup_blocked is None:
                self.setup_blocked = {tuple(map(int, p)) for p in zip(*np.where(grid != 1))
                                      if not is_setup_position_allowed(*p)}
            blocked = self.setup_blocked
        # Assign the whole living team together, so decision order and IQ
        # coordinate noise do not change the chosen distribution or slots.
        actual = sorted((getattr(c, 'real_character', c) for c in allies if c.is_alive),
                        key=lambda c: c.name)
        all_cells = {p for cells in self.candidates.values() for p in cells}
        all_cells -= self.reserved
        if self.drop_region is not None:
            all_cells = {p for p in all_cells if region(p, grid) == self.drop_region}
        legal = {c.name: (all_cells & distances(pos(c), grid, blocked).keys()) - set(blocked)
                 for c in actual}
        if self.preferred_engineer_region is not None and self.drop_region is None and 'Alfajer' in legal:
            preferred = {p for p in legal['Alfajer'] if region(p, grid) == self.preferred_engineer_region}
            if preferred:
                legal['Alfajer'] = preferred
                cell = self.targets.get('Alfajer')
                if cell not in preferred:
                    cell = self._pick(preferred)
                self.targets = {name: p for name, p in self.targets.items() if name == 'Alfajer' or p != cell}
                self.targets['Alfajer'] = cell
        self._coverage(actual, legal, grid, setup)
        self.targets = {name: p for name, p in self.targets.items()
                        if name in legal and p in legal[name]}
        counts = dict.fromkeys(('A', 'MID', 'B'), 0)
        if self.region_limits is not None:
            for c in sorted(actual, key=lambda c: (c.name != 'Alfajer', c.name)):
                cell = self.targets.get(c.name)
                if cell is None:
                    continue
                label = region(cell, grid)
                if (counts[label] >= self.region_limits.get(label, 0)
                        or (c.name == 'Alfajer' and self.engineer_region is not None
                            and label != self.engineer_region)):
                    del self.targets[c.name]
                else:
                    counts[label] += 1
        used = set(self.targets.values())
        unassigned = [c for c in actual if c.name not in self.targets]
        # All players share the same preferences; avoid consistently giving
        # the frequent groups to the first names in alphabetical order.
        random.shuffle(unassigned)
        if self.region_limits is not None:
            unassigned.sort(key=lambda c: c.name != 'Alfajer')
        for c in unassigned:
            available = legal[c.name] - used
            if self.region_limits is not None:
                balanced = {p for p in available
                            if counts[region(p, grid)] < self.region_limits.get(region(p, grid), 0)}
                if c.name == 'Alfajer' and self.engineer_region is not None:
                    own = {p for p in balanced if region(p, grid) == self.engineer_region}
                    balanced = own or balanced
                available = balanced or available
            groups = {marker: [p for p in cells if p in legal[c.name] and p not in used]
                      for marker, cells in self.candidates.items()}
            groups = {marker: [p for p in cells if p in available] for marker, cells in groups.items()}
            groups = {marker: cells for marker, cells in groups.items() if cells}
            if not groups:
                continue
            markers = list(groups)
            marker = random.choices(markers, weights=[self.weights[m] for m in markers], k=1)[0]
            cell = random.choice(groups[marker])
            self.targets[c.name] = cell
            used.add(cell)
            counts[region(cell, grid)] += 1
        return self.targets.get(char.name)

    def _pick(self, available):
        groups = {marker: [p for p in cells if p in available] for marker, cells in self.candidates.items()}
        groups = {marker: cells for marker, cells in groups.items() if cells}
        markers = list(groups)
        marker = random.choices(markers, weights=[self.weights[m] for m in markers], k=1)[0]
        return random.choice(groups[marker])
