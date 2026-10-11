"""Analysis labels from fixed learned plant execution, using public inputs."""
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_learn_attacker_analysis import Route, branch_sequence, route_entry_positions


class FixedPlantAnalysisCollector(TouyamaV3AttackerPlantController):
    @property
    def mode(self):
        return self.route_mode

    def __init__(self, scenario, analysis, plant, *, seed, exploration, exploration_scope,
                 exploration_probe_steps=0, exploration_probe_ticks=0, exploration_round_start_only=False):
        # Route exploration is separate from plant action exploration. The
        # latter remains disabled: these labels measure the adopted policy.
        super().__init__(scenario, analysis, plant, seed=seed, training=False)
        self.planning_rng = self.rng
        self.planning_exploration = exploration
        self.planning_exploration_scope = exploration_scope
        self.planning_probe_steps = exploration_probe_steps
        self.planning_probe_ticks = exploration_probe_ticks
        self.planning_round_start_only = exploration_round_start_only

    def reset_round(self):
        super().reset_round()
        self.frames = []
        self.frame_key = None
        self.current_observation = None

    def select_route(self, snapshot, observation):
        self.current_observation = observation.copy()
        return super().select_route(snapshot, observation)

    def prepare_team_tick(self):
        super().prepare_team_tick()
        if (not self.plans or self.snapshot.is_planted or self.snapshot.phase == 'setup'
                or self.cache == self.frame_key):
            return
        self.frame_key = self.cache
        snap = self.snapshot
        holder = next((a for a in snap.allies if a.alive and a.has_spike), None)
        index = self.cursors.get(holder.slot, 0) if holder else 0
        cells = self.route.cells[index:]
        remaining = Route(self.route.site, cells, branch_sequence(self.scenario, cells))
        belief = self.analysis.analyze(self.analysis_encoder, snap, self.current_observation, [remaining])
        self.frames.append(dict(tick=snap.tick, observation=self.current_observation.copy(),
            route_features=self.analysis_encoder.route_features(snap, remaining, self.route_mode, belief['placement']),
            decision_id=self.route_planner.decision_id, site=self.route.site,
            phase=self.attack_plan.phase, alive=sum(a.alive for a in snap.allies)))
        outside,inside=route_entry_positions(self.scenario,remaining)
        self.frames[-1].update(entry_outside=outside,entry_inside=inside,
                              carrier_slot=holder.slot if holder else -1)
