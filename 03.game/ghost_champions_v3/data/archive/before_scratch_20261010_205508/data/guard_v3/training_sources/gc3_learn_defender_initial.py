"""Sample soft initial deployment regions; retain the v2 live defense."""
from pathlib import Path
import math
import numpy as np
from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController
from ghost_champions_v2.geometry import route
from ghost_champions_v2.rl.unit_observation import action_capabilities
from gc_v1.gc_facing import facing_towards
from map_data_defender_setup import get_setup_mask
from . import gc3_map_defender_initial as candidate_map
from .gc3_spatial_policy import SpatialCandidatePolicy, DEFENDER_ARRIVAL_RADIUS

DEFAULT_INITIAL_CHECKPOINT = None


class GhostChampionsV3DefenderController(GhostChampionsV2DefenderController):
    def __init__(self, *, initial_policy=None, initial_checkpoint=DEFAULT_INITIAL_CHECKPOINT, **kwargs):
        self.initial_policy = initial_policy if initial_policy is not None else (
            SpatialCandidatePolicy.load(initial_checkpoint) if initial_checkpoint else SpatialCandidatePolicy())
        self.initial_checkpoint = Path(initial_checkpoint).resolve() if initial_checkpoint else None
        self.initial_goals, self.initial_aims, self.initial_records = {}, {}, {}
        self.initial_map_key = None
        super().__init__(**kwargs)

    def reset_round(self):
        self.initial_goals, self.initial_aims, self.initial_records = {}, {}, {}
        self.initial_map_key = None
        super().reset_round()

    def decide_move(self, char, state):
        if not state.get("defender_setup_active"):
            return super().decide_move(char, state)
        caps = action_capabilities(char, state)
        if not char.is_alive or not caps["can_act"]:
            return list(char.pos), "HOLD"
        if self.initial_map_key != candidate_map.DEFENDER_SETUP_POSITION_MAP_GC_STR:
            self.initial_goals, self.initial_aims, self.initial_records = {}, {}, {}
            self.initial_map_key = candidate_map.DEFENDER_SETUP_POSITION_MAP_GC_STR
        name = str(char.name)
        setup_grid = np.array(state["grid"], copy=True)
        setup_grid[np.asarray(get_setup_mask()) != 0] = 1
        occupied = {tuple(c.pos) for c in state.get("chars", ())
                    if c.team == char.team and c.is_alive and str(c.name) != name}
        if name not in self.initial_goals:
            point, position_record = self.initial_policy.sample(char, state, self.initial_map_key,
                kind="position", phase="defender", legal_grid=setup_grid,
                blocked=occupied | set(self.initial_goals.values()),
                max_steps=max(0, int(state.get("defender_setup_ticks_remaining", 20))-1))
            aim, watch_record = self.initial_policy.sample(char, state, self.initial_map_key,
                kind="watch", phase="defender", origin=point)
            self.initial_goals[name], self.initial_aims[name] = point, aim
            self.initial_records[name] = [position_record, watch_record]
        goal, aim = self.initial_goals[name], self.initial_aims[name]
        self.macro.phase = "setup"
        self.macro.goals = dict(self.initial_goals)
        self.macro.allocation_source = "v3_sampled_soft_policy"
        goals = [tuple(map(int, p)) for p in np.argwhere(setup_grid != 1)
                 if math.dist(p, goal) <= DEFENDER_ARRIVAL_RADIUS]
        path = route(setup_grid, tuple(char.pos), goals, occupied)
        if caps["can_move"] and len(path) > 1:
            return list(path[1]), "MOVE", {"move_step_limit": 1}
        if caps["can_turn"]:
            return list(char.pos), {"facing": facing_towards(char.pos, aim) or getattr(char, "facing", "S")}
        return list(char.pos), "HOLD"

    def defender_snapshot(self):
        snapshot = super().defender_snapshot()
        snapshot["initial_v3"] = dict(goals=dict(self.initial_goals), aims=dict(self.initial_aims),
                                     checkpoint=str(self.initial_checkpoint) if self.initial_checkpoint else None,
                                     selection="sampled", arrival_radius=DEFENDER_ARRIVAL_RADIUS)
        return snapshot
