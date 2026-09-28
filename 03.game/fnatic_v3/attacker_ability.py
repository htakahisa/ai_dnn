"""Mapped lamp placement and regrouping for Fnatic's engineer."""

import random

import numpy as np

from fnatic_v1_rules import pos
from .positions import distances, parse_grid
from .formation import MID_CONTROL_POSITIONS, MID_WATCH_POSITION


class FnaticEngineerRoute:
    def __init__(self, map_text=None):
        if map_text is None:
            from .map_data_attacker_ability_fnatic import ENGINEER_LAMP_STR
            map_text = ENGINEER_LAMP_STR
        self.map_grid = parse_grid(map_text) if map_text.strip() else None
        self.candidates = ()
        if self.map_grid is not None:
            self.candidates = tuple(tuple(map(int, p)) for p in zip(*np.where(self.map_grid == 3)))
        self.reset_round()

    def reset_round(self):
        self.completed = set()
        self.target = None
        self.pending = None
        self.mid_target = None

    def validate(self, grid):
        if self.map_grid is None:
            return
        if self.map_grid.shape != grid.shape:
            raise ValueError('Fnatic ability map must match game map dimensions')
        if not np.array_equal(self.map_grid == 1, grid == 1):
            raise ValueError('Fnatic ability map walls must match the game map')

    def available(self, char, grid, traps):
        # Confirm a cast from the game's actual charge consumption. A request
        # rejected by the game must not count as a completed placement.
        if self.pending is not None:
            cell, charges = self.pending
            if int(getattr(char, 'ramp_charges', 0)) < charges:
                self.completed.add(cell)
                self.target = None
                self.pending = None
        occupied = {tuple(trap['pos']) for trap in traps if trap['team'] == char.team}
        lengths = distances(pos(char), grid)
        return tuple(p for p in self.candidates
                     if p not in self.completed and p not in occupied and p in lengths)

    def has_work(self, char, grid, traps):
        available = self.available(char, grid, traps)
        return (getattr(char, 'ability_name', '') == 'RAMP'
                and int(getattr(char, 'ramp_charges', 0)) > 0 and bool(available))

    def result(self, ctrl, char, grid, blocked, visible, traps, setup=False):
        available = self.available(char, grid, traps)
        if setup:
            available = tuple(p for p in available
                              if ctrl._route(pos(char), [p], grid, setup=True)[2] is not None)
        if self.target not in available:
            self.target = random.choice(available) if available else None
            self.pending = None
        aim = pos(visible[0]) if visible else self.target
        if self.target is None or (setup and self.target == pos(char)):
            return ctrl._result(char, pos(char), aim)
        if self.target == pos(char):
            self.pending = (self.target, int(char.ramp_charges))
            return list(char.pos), {'ability': 'RAMP'}
        nxt, _, _ = ctrl._route(pos(char), [self.target], grid, setup=setup)
        if nxt in blocked:
            nxt, _, _ = ctrl._route(pos(char), [self.target], grid, blocked, setup=setup)
        return ctrl._result(char, nxt, aim)

    def mid_result(self, ctrl, char, grid, blocked, visible):
        # The regular Mid player keeps its own slot. Alfajer uses a separate
        # destination so two independent players cannot block each other.
        reserved = ctrl.formation.mid_target
        goals = [p for p in MID_CONTROL_POSITIONS
                 if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]
                 and grid[p] != 1 and p not in blocked and p != reserved]
        if self.mid_target not in goals:
            _, _, self.mid_target = ctrl._route(pos(char), goals, grid)
        nxt, _, _ = ctrl._route(pos(char), [self.mid_target] if self.mid_target else [], grid)
        if nxt in blocked:
            nxt, _, _ = ctrl._route(pos(char), [self.mid_target], grid, blocked)
        return ctrl._result(char, nxt, pos(visible[0]) if visible else MID_WATCH_POSITION)
