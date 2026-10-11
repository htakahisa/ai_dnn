"""Public-only attacker execution: analysis routes, learned local plant actions."""
from collections import Counter, deque

import numpy as np
import torch
from pathlib import Path

from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.actions import build_masks, to_game_action, MOVE_STEPS
from grid_paths import distance_map
from touyama_v3.tv3_attacker_route_planner import AdaptiveAttackPlanner
from touyama_v3.tv3_attacker_combat import AttackerCombatCoach
from touyama_v3.tv3_model_extensions import ExtensionContext,load_extensions
from touyama_v3.tv3_attacker_entry_coordination import EntryCoordinator
from touyama_v3.tv3_attacker_site_entry import (
    PlantSiteAssembly, traffic_distances, site_entry_features, ENTRY_FEATURES as SITE_ENTRY_FEATURES,
    teacher_distances,
    SITE_SWITCH_MARGIN, SITE_SWITCH_TRAVEL_COST,
)
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, Route, branch_sequence
from touyama_v3.tv3_learn_attacker_plant import (
    PlantEncoder, PLANT_VERSION, MOVEMENTS, ROUTE_CORRIDOR_CELLS, ROUTE_RETREAT_CELLS,
    MAX_EDGE_TRAVERSALS, EDGE_TRAVERSAL_WINDOW,
    load_frozen_analysis, load_plant,
)


class TouyamaV3AttackerPlantController:
    handles_team_perception = True

    def __init__(self, scenario, analysis, policy, *, seed=42, training=False,opponent=None):
        self.scenario, self.analysis, self.policy = scenario, analysis, policy
        self.root_policy=policy
        self.sensor = FrcPerceptionBuilder("A")
        self.analysis_encoder = AttackerEncoder(scenario, getattr(analysis, 'execution_version', 3))
        self.encoder = PlantEncoder(scenario, getattr(policy, "execution_version", PLANT_VERSION))
        self.encoder.experimental_flash_teacher=getattr(policy,'experimental_flash_teacher',False)
        self.rng = np.random.default_rng(seed)
        self.training = training
        self.custom_opponent=opponent or getattr(policy,'opponent',None)
        self.custom_pipeline=load_extensions(getattr(policy,'custom_extension_contract',None),
            role='attacker_plant',opponent=self.custom_opponent)
        self.epsilon = self.teacher_probability = 0.
        self.previous_rounds = []
        self.game = None
        self.reset_round()

    def set_game(self, game):
        self.game = getattr(game, "_real", game)

    @classmethod
    def from_best(cls, opponent, *, best_dir=None, analysis_dir=None, scenario=None):
        from touyama_v3.tv3_scenario import Scenario
        scenario = scenario or Scenario()
        best_dir = Path(best_dir) if best_dir else Path(__file__).resolve().parent / "data" / "best"
        analysis, signature, _ = load_frozen_analysis(analysis_dir or best_dir, opponent, scenario)
        policy, _ = load_plant(best_dir / opponent / "attacker_plant_best.pt", scenario, opponent, signature)
        return cls(scenario, analysis, policy,opponent=opponent)

    def reset_round(self):
        if hasattr(self.root_policy,'common_policy'):
            self.policy=self.root_policy.common_policy
            self.encoder=PlantEncoder(self.scenario,self.policy.execution_version)
            self.common_encoder=self.encoder
            self.custom_encoder=PlantEncoder(self.scenario,self.root_policy.execution_version)
        self.sensor.reset()
        self.analysis_encoder.reset()
        self.cache = self.snapshot = self.route = self.route_mode = None
        self.cursors, self.inputs, self.actions, self.plans, self.decisions = {}, {}, {}, {}, {}
        self.previous_inputs = {}
        self.pending_public_flashes=[]
        self.edges = Counter()
        self.edge_history = deque()
        self.endpoints = {}
        self.retriever = None
        self.replans = 0
        self.route_planner = AdaptiveAttackPlanner(site_switch_margin=SITE_SWITCH_MARGIN,
                                                   switch_travel_cost=SITE_SWITCH_TRAVEL_COST)
        self.entry_coordinator = EntryCoordinator(self.scenario)
        self.site_assembly = PlantSiteAssembly(self.scenario,
            follow_route_entry=getattr(self.analysis, 'execution_version', 3) >= 4,
            bind_changed_entry_only=getattr(self.analysis, 'execution_version', 3) >= 4)
        self.site_entry_state = None
        self.attack_plan = None
        self.encoder.combat_coach = type(self.encoder.combat_coach)(self.scenario)
        self.encoder.support_coach.reset()
        self.encoder.empty_evidence=type(self.encoder.empty_evidence)()
        if self.encoder.execution_version >= 12:
            from touyama_v3.tv3_learn_attacker_corner_utility import FriendlyFlashTracker
            self.encoder.flash_tracker=FriendlyFlashTracker(self.scenario)
        self.events = []
        if hasattr(self.root_policy,'common_policy'):
            self.custom_encoder.flash_tracker=self.encoder.flash_tracker
            self.custom_encoder.empty_evidence=self.encoder.empty_evidence
            self.custom_encoder.support_coach=self.encoder.support_coach
            if type(self.custom_encoder.combat_coach)==type(self.encoder.combat_coach):
                self.custom_encoder.combat_coach=self.encoder.combat_coach

    def select_weight_policy(self,site):
        if hasattr(self.root_policy,'common_policy'):
            custom=site==self.root_policy.weight_scope
            self.policy=self.root_policy if custom else self.root_policy.common_policy
            self.encoder=self.custom_encoder if custom else self.common_encoder

    @property
    def training_current_policy(self):
        return self.training and (not hasattr(self.root_policy,'common_policy') or self.policy is self.root_policy)

    def select_route(self, snapshot, observation):
        self.route_planner.allow_deadline_route_reserve=getattr(getattr(self,'encoder',None),'execution_version',0)>=18
        holder = next((a for a in getattr(snapshot, 'allies', ()) if a.alive and a.has_spike), None)
        previous = self.previous_inputs.get(holder.name) if holder else None
        coordinated_wait = (holder and self.site_entry_state and self.site_entry_state['wait']
                            and holder.slot in self.site_entry_state['ready'])
        combat_wait = (previous is not None and previous.combat is not None
                       and previous.combat.reason in ('assemble', 'stop_shoot', 'cover_retreat'))
        if coordinated_wait or combat_wait:
            self.route_planner.progress_tick = snapshot.tick
        planning_options = {} if not hasattr(self, 'planning_rng') else dict(
            rng=self.planning_rng, exploration=self.planning_exploration,
            exploration_scope=self.planning_exploration_scope,
            exploration_probe_steps=self.planning_probe_steps,
            exploration_probe_ticks=self.planning_probe_ticks,
            exploration_round_start_only=self.planning_round_start_only)
        plan = self.route_planner.update(snapshot, observation, self.analysis_encoder, self.analysis,
                                         self.route, self.route_mode, **planning_options)
        if plan is not None:
            invalid = {slot for slot,(outside,inside) in self.route_planner.flank_entries.items()
                       if outside == inside or self.scenario.grid[outside] == 2
                       or self.scenario.grid[inside] != 2
                       or abs(outside[0]-inside[0])+abs(outside[1]-inside[1]) != 1}
            for slot in invalid:
                self.route_planner.flank_entries.pop(slot)
                plan.scout_goals.pop(slot,None)
            if invalid and self.route_planner.events:
                self.route_planner.events[-1]['flank_entries'] = dict(self.route_planner.flank_entries)
                self.route_planner.events[-1]['scout_goals'] = dict(plan.scout_goals)
        self.attack_plan = plan
        if plan is None:
            return False
        if not plan.changed:
            return True
        if self.route is not None:
            self.replans += 1
        self.route, self.route_mode = plan.route, plan.mode
        self.cursors, self.endpoints = {}, {}
        self.edges.clear()
        self.edge_history.clear()
        self.events.append({"type": "route", "tick": snapshot.tick, "site": self.route.site,
                            "key": self.route.key, "branches": self.route.branches,
                            "phase": plan.phase, "reason": plan.reason})
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        free = list(self.scenario.sites[self.route.site])
        if holder:
            self.endpoints[holder.slot] = self.route.cells[-1]
            free.remove(self.route.cells[-1])
        for a in sorted((a for a in snapshot.allies if a.alive and not a.has_spike), key=lambda a: a.slot):
            if not free:
                break
            goal = min(free, key=lambda p: (distance_map(self.scenario.grid, p)[self.route.cells[-1]], p))
            self.endpoints[a.slot] = goal
            free.remove(goal)
        return True

    def goal_for(self, snapshot, ally):
        goal=self._assigned_goal_for(snapshot,ally)
        if self.encoder.execution_version>=17:
            from touyama_v3.tv3_attacker_entry_yield import yielding_entry_goal
            holder=next((a for a in snapshot.allies if a.alive and a.has_spike),None)
            carrier_inputs=self.inputs.get(holder.name) if holder else None
            if carrier_inputs is not None:
                replacement=yielding_entry_goal(self.scenario,snapshot,ally,goal,carrier_inputs.goal,self.route.site)
                if replacement is not None:
                    self.endpoints[ally.slot]=replacement
                    self.events.append(dict(type='entry_yield',tick=snapshot.tick,slot=ally.slot,
                        position=ally.position,old_goal=goal,new_goal=replacement))
                    return replacement
        return goal

    def _assigned_goal_for(self, snapshot, ally):
        self.cursors.setdefault(ally.slot, 0)
        if ally.slot == self.retriever:
            return snapshot.spike_dropped
        if self.site_entry_state and ally.slot in self.site_entry_state['goals']:
            return self.site_entry_state['goals'][ally.slot]
        if ally.slot in self.attack_plan.scout_goals:
            return self.attack_plan.scout_goals[ally.slot]
        path = self.route.cells
        cursor = self.cursors.get(ally.slot, 0)
        if ally.position in path[cursor:]:
            cursor = path.index(ally.position, cursor)
        self.cursors[ally.slot] = cursor
        if cursor >= len(path) - 5:
            return path[-1] if ally.has_spike else self.endpoints.get(ally.slot, path[-1])
        return path[min(cursor + 4, len(path) - 1)]

    def prepare_team_tick(self):
        setup = self.game.defender_setup_phase.active
        key = (self.game.current_round, "setup" if setup else "live",
               self.game.defender_setup_phase.ticks_remaining if setup else self.game.battle_tick,
               bool(self.game.is_planted))
        if self.cache == key:
            return
        snapshot = self.sensor.build(self.game)
        if self.encoder.execution_version >= 12:
            allies={a.slot:a for a in snapshot.allies}
            remaining=[]
            for tick,slot,origin,aim,charges in self.pending_public_flashes:
                if snapshot.tick<=tick:
                    remaining.append((tick,slot,origin,aim,charges))
                elif slot in allies and allies[slot].charges<charges:
                    self.encoder.flash_tracker.record(tick,origin,aim)
            self.pending_public_flashes=remaining
        self.snapshot, self.cache = snapshot, key
        self.previous_inputs = self.inputs
        self.inputs, self.actions, self.plans = {}, {}, {}
        if setup or snapshot.is_planted:
            return
        observation = self.analysis_encoder.observe(snapshot, self.previous_rounds)
        if not self.select_route(snapshot, observation):
            return
        self.select_weight_policy(self.route.site)
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        if holder is None and snapshot.spike_dropped:
            alive = [a for a in snapshot.allies if a.alive]
            if not any(a.slot == self.retriever for a in alive):
                field = distance_map(self.scenario.grid, snapshot.spike_dropped)
                self.retriever = min(alive, key=lambda a: (field[a.position], a.slot)).slot if alive else None
        else:
            self.retriever = None
        cursor = self.cursors.get(holder.slot, 0) if holder else 0
        if holder and holder.position in self.route.cells[cursor:]:
            cursor = self.route.cells.index(holder.position, cursor)
            self.cursors[holder.slot] = cursor
        suffix = self.route.cells[cursor:]
        remaining = Route(self.route.site, suffix, branch_sequence(self.scenario, suffix))
        analysis = self.analysis.analyze(self.analysis_encoder, snapshot, observation, [remaining])
        masks = build_masks(snapshot)
        coordination = self.entry_coordinator.observe(snapshot, self.route, self.route_planner.flank_entries)
        self.site_entry_state = self.site_assembly.observe(snapshot, self.route)
        if (self.encoder.execution_version>=16 and holder and self.site_entry_state
                and self.site_entry_state['wait'] and holder.slot in self.site_entry_state['ready']
                and not holder.blind):
            from game_core import PLANT_REQUIRED_TICKS
            from touyama_v3.tv3_learn_attacker_corner_utility import confirmed_flash_step_window,flash_window_allows_completion
            inside=self.site_assembly.entries.get(holder.slot)
            if inside is not None:
                tactical=self.encoder.empty_evidence.tactical_snapshot(snapshot,temporary_body_cover=True)
                window=confirmed_flash_step_window(self.scenario,tactical,holder,inside,
                    self.analysis_encoder.history.tracks,self.encoder.flash_tracker)
                travel=distance_map(self.scenario.grid,inside)[holder.position]
                if flash_window_allows_completion(window,travel,PLANT_REQUIRED_TICKS):
                    self.site_assembly.launched=True
                    self.site_entry_state=self.site_assembly.observe(snapshot,self.route)
        if self.site_entry_state is not None:
            coordination = self.site_entry_state
        reserved = set()
        names = {a.slot: a.name for a in snapshot.allies}
        for ally in sorted((a for a in snapshot.allies if a.alive), key=lambda a: (not a.has_spike, a.slot != self.retriever, a.slot)):
            goal = self.goal_for(snapshot, ally)
            from touyama_v3.tv3_learn_attacker_fire_support import public_threats
            threats=public_threats(snapshot,self.analysis_encoder.history.tracks)
            carrier_inputs=self.inputs.get(holder.name) if holder else None
            carrier_goal=carrier_inputs.goal if carrier_inputs else None
            custom_context=ExtensionContext('attacker_plant',self.custom_opponent,self.route.site,snapshot,ally,goal,threats=threats,carrier_goal=carrier_goal)
            goal=self.custom_pipeline.adjust_goal(custom_context,goal)
            inputs = self.encoder.encode(snapshot, ally, goal, self.route, analysis,
                self.analysis_encoder.history.tracks, masks, self.cursors[ally.slot], self.route_mode, coordination)
            custom_context=ExtensionContext('attacker_plant',self.custom_opponent,self.route.site,snapshot,ally,goal,threats=threats,carrier_goal=carrier_goal)
            inputs=self.custom_pipeline.adjust_inputs(custom_context,inputs)
            self._restrict_movement(ally, inputs, reserved)
            inputs.traffic = traffic_distances(snapshot, ally, goal, reserved)
            inputs.observation[-SITE_ENTRY_FEATURES-2:-2] = site_entry_features(
                snapshot, ally, goal, inputs.traffic, coordination)
            inputs.observation[-2] = float(inputs.mask[-2])
            inputs.observation[-1] = float(self._has_progress_move(ally,inputs))
            if not inputs.mask[inputs.teacher]:
                candidates = np.flatnonzero(inputs.mask[:40])
                def rank(i):
                    dr, dc = MOVE_STEPS.get(MOVEMENTS[i // 8], (0, 0))
                    p = ally.position[0] + dr, ally.position[1] + dc
                    field = teacher_distances(inputs.distances, inputs.traffic, ally.position)
                    if self.encoder.execution_version >= 11 and inputs.support_distances is not None:
                        field = inputs.support_distances
                    d = field[p]
                    if self.encoder.execution_version>=19:
                        return d if d >= 0 else 9999,int(inputs.actions[i].facing!=inputs.combat.facing),i
                    return d if d >= 0 else 9999, i
                if self.encoder.execution_version>=19 and inputs.combat.reason=='cover_retreat':
                    from touyama_v3.tv3_learn_attacker_masked_teacher import masked_retreat_teacher
                    tactical=self.encoder.empty_evidence.tactical_snapshot(snapshot,temporary_body_cover=True)
                    inputs.teacher=masked_retreat_teacher(self.scenario,tactical,ally,inputs,
                        self.analysis_encoder.history.tracks)
                else:
                    inputs.teacher = int(min(candidates, key=rank))
            actor_training=self.training_current_policy
            if actor_training and self.rng.random() < self.teacher_probability:
                action_index = inputs.teacher
            elif actor_training and self.rng.random() < self.epsilon:
                action_index = int(self.rng.choice(np.flatnonzero(inputs.mask)))
            else:
                with torch.no_grad():
                    values = self.policy(torch.tensor(inputs.observation).unsqueeze(0))[0].numpy()
                action_index = int(np.where(inputs.mask, values, -np.inf).argmax())
            action = inputs.actions[action_index]
            self.inputs[ally.name] = inputs
            self.plans[ally.name] = (action_index, inputs, ally)
            self.actions[ally.name] = to_game_action(snapshot, ally.slot, action, names)
            dr, dc = MOVE_STEPS.get(action.kind, (0, 0))
            reserved.add((ally.position[0] + dr, ally.position[1] + dc))

    def _restrict_movement(self, ally, inputs, reserved):
        self._expire_edges(self.snapshot.tick)
        inputs.engine_movement_mask = inputs.mask[:40].copy()
        inputs.movement_debug = {}
        path = self.route.cells
        cursor = self.cursors[ally.slot]
        allowed_path = path[max(0, cursor - ROUTE_RETREAT_CELLS):]
        entry_state = getattr(self, 'site_entry_state', None)
        entry_goal = entry_state and entry_state['goals'].get(ally.slot)
        if entry_goal is not None:
            from touyama_v3.tv3_learn_attacker_analysis import shortest_path
            blocked_sites = set(self.scenario.sites['R' if self.route.site == 'L' else 'L'])
            if not self.site_entry_state['launched']:
                blocked_sites |= set(self.scenario.sites[self.route.site])
            allowed_path = shortest_path(self.scenario, ally.position, (entry_goal,), blocked_sites) or (ally.position,)
        elif ally.slot in self.attack_plan.scout_goals:
            from touyama_v3.tv3_learn_attacker_analysis import shortest_path
            goal = self.attack_plan.scout_goals[ally.slot]
            blocked_sites = set(self.scenario.sites["L"]) | set(self.scenario.sites["R"]) if ally.slot in self.route_planner.flank_entries and self.scenario.grid[goal] != 2 else set()
            allowed_path = shortest_path(self.scenario, ally.position, (goal,), blocked_sites) or (ally.position,)
        corridor = np.minimum.reduce([distance_map(self.scenario.grid, p) for p in allowed_path])
        recovering = ally.slot == self.retriever
        for i in np.flatnonzero(inputs.mask[:40]):
            kind = MOVEMENTS[i // 8]
            if kind == "STAY":
                continue
            dr, dc = MOVE_STEPS[kind]
            destination = ally.position[0] + dr, ally.position[1] + dc
            if destination in reserved:
                inputs.mask[i] = False
                inputs.movement_debug.setdefault(kind, []).append('team_reservation')
            elif not recovering:
                if getattr(inputs, 'diagnostic_allow_detour', False):
                    continue  # Experimental teacher only; never set by production encoders.
                edge = tuple(sorted((ally.position, destination)))
                public_contact = bool(self.encoder.combat_coach.contacts(self.snapshot, ally, ally.position))
                retreat = inputs.combat is not None and inputs.combat.reason == "cover_retreat"
                progress_escape=(self.encoder.execution_version==14 and destination in self.route.cells
                    and inputs.traffic[ally.position]>=0
                    and 0<=inputs.traffic[destination]<inputs.traffic[ally.position])
                if (self.encoder.execution_version>=16 and destination in self.route.cells
                        and inputs.traffic[ally.position]>=0
                        and 0<=inputs.traffic[destination]<inputs.traffic[ally.position]
                        and self.edges[(ally.slot,edge)]>=MAX_EDGE_TRAVERSALS):
                    from touyama_v3.tv3_learn_attacker_corner_utility import confirmed_flash_covers_step
                    tactical=self.encoder.empty_evidence.tactical_snapshot(self.snapshot,temporary_body_cover=True)
                    progress_escape=confirmed_flash_covers_step(self.scenario,tactical,ally,destination,
                        self.analysis_encoder.history.tracks,self.encoder.flash_tracker)
                if (self.edges[(ally.slot, edge)] >= MAX_EDGE_TRAVERSALS
                        and not (ally.blind or public_contact or retreat or progress_escape)):
                    inputs.mask[i] = False
                    inputs.movement_debug.setdefault(kind, []).append('recent_edge_limit')
                if not (ally.blind or public_contact or retreat) and corridor[ally.position] <= ROUTE_CORRIDOR_CELLS and corridor[destination] > ROUTE_CORRIDOR_CELLS:
                    inputs.mask[i] = False
                    inputs.movement_debug.setdefault(kind, []).append('route_corridor')

    @staticmethod
    def _has_progress_move(ally, inputs):
        traffic = getattr(inputs, 'traffic', None)
        field = teacher_distances(inputs.distances, traffic, ally.position)
        old = field[ally.position]
        return any(inputs.mask[i] and 0 <= field[(ally.position[0]+MOVE_STEPS[MOVEMENTS[i//8]][0],
                       ally.position[1]+MOVE_STEPS[MOVEMENTS[i//8]][1])] < old for i in range(8,40))

    def _expire_edges(self, tick):
        while self.edge_history and tick-self.edge_history[0][0] >= EDGE_TRAVERSAL_WINDOW:
            _,key = self.edge_history.popleft()
            self.edges[key] -= 1
            if self.edges[key] <= 0:
                del self.edges[key]

    def decide_move(self, char, state):
        if self.game.is_planted:
            return list(char.pos), {"facing": char.facing}
        self.prepare_team_tick()
        name = str(getattr(char, "base_name", char.name))
        if name in self.plans:
            self.decisions[name] = self.plans[name]
            selected,inputs,before=self.plans[name]
            action=inputs.actions[selected]
            if self.encoder.execution_version>=12 and before.ability_name=='FLASH' and action.kind=='ABILITY' and action.target is not None:
                pending=(self.snapshot.tick,before.slot,before.position,action.target,before.charges)
                if pending not in self.pending_public_flashes:
                    self.pending_public_flashes.append(pending)
        return self.actions.get(name, (list(char.pos), {"facing": char.facing}))

    def observe_executed_moves(self, chars):
        by_name = {str(getattr(c, "base_name", c.name)): c for c in chars}
        for name, (selected, inputs, before) in self.decisions.items():
            action=inputs.actions[selected]
            if (self.encoder.execution_version >= 12 and before.ability_name=='FLASH'
                    and action.kind=='ABILITY' and action.target is not None
                    and getattr(by_name[name],'flash_charges',before.charges)<before.charges):
                self.encoder.flash_tracker.record(self.snapshot.tick,before.position,action.target)
            after = tuple(map(int, by_name[name].pos))
            if after != before.position:
                edge = tuple(sorted((before.position, after)))
                self.edges[(before.slot, edge)] += 1
                self.edge_history.append((self.snapshot.tick,(before.slot,edge)))
