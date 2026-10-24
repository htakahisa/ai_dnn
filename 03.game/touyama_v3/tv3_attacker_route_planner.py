"""Shared public-observation, receding-horizon attack planning."""
from dataclasses import dataclass

from game_core import PLANT_REQUIRED_TICKS

# Normal planning settings. Every route is provisional until planting starts.
REASSESS_INTERVAL = 3
STALL_TICKS = 6
WAYPOINT_STEPS = 4
ENTRY_STEPS = 6
SWITCH_MARGIN = .05
STALL_STEP_PENALTY = .3
SCOUT_INFORMATION_WEIGHT = .2
EXPLORATION_SCORE_WINDOW = .15
FLANK_MAX_EXTRA_STEPS = 10
FLANK_PLAYERS = 2


def planner_schema():
    return {"version": 2, "group_until_entry": True, "interval": REASSESS_INTERVAL, "stall_ticks": STALL_TICKS,
            "waypoint_steps": WAYPOINT_STEPS, "entry_steps": ENTRY_STEPS,
            "switch_margin": SWITCH_MARGIN, "information_weight": SCOUT_INFORMATION_WEIGHT,
            "stall_step_penalty": STALL_STEP_PENALTY,
            "exploration_window": EXPLORATION_SCORE_WINDOW,
            "flank_extra_steps": FLANK_MAX_EXTRA_STEPS, "flank_players": FLANK_PLAYERS,
            "labels": "chosen_route_then_public_adaptive_continuation",
            "commit": "plant_progress_only", "replan_limit": None}


@dataclass
class AttackPlan:
    route: object
    mode: str
    phase: str
    changed: bool
    reason: str
    waypoint: tuple
    scout_goals: dict


class AdaptiveAttackPlanner:
    def __init__(self, *, site_switch_margin=None, switch_travel_cost=0.):
        # Optional plant execution settings; default analysis collection stays unchanged.
        self.site_switch_margin = site_switch_margin
        self.switch_travel_cost = switch_travel_cost
        self.last_tick = -REASSESS_INTERVAL
        self.last_signal = None
        self.last_position = None
        self.progress_tick = 0
        self.plan = None
        self.events = []
        self.decision_id = 0
        self.enable_flanks = True
        self.flank_entries = {}
        self.flank_site = None
        self.exploration_probe = None
        self.reassess_new_contacts = False  # Diagnostic until separately validated.
        self.decision_debug = None
        self.allow_deadline_route_reserve=False  # Versioned plant executor only; analysis collection unchanged.

    def update(self, snapshot, observation, encoder, model, route=None, mode=None, *, rng=None, exploration=0.,
               exploration_scope="score_window", exploration_probe_steps=0, exploration_probe_ticks=0,
               exploration_round_start_only=False, progress_tick_override=None, stall_ticks=STALL_TICKS):
        if exploration_scope not in ("score_window", "all_routes"):
            raise ValueError("Unknown route exploration scope")
        from touyama_v3.tv3_learn_attacker_analysis import Route, candidate_routes, branch_sequence, TIME_MARGIN_TICKS, route_entry_positions
        holder = next((a for a in snapshot.allies if a.alive and a.has_spike), None)
        origin = holder.position if holder else snapshot.spike_dropped
        if origin is None:
            return None if self.plan is None else AttackPlan(route or self.plan.route, mode or self.plan.mode,
                "scout", False, "spike_location_unknown", self.plan.waypoint, self.plan.scout_goals)
        if progress_tick_override is not None:
            self.last_position, self.progress_tick = origin, progress_tick_override
        elif origin != self.last_position:
            self.last_position, self.progress_tick = origin, snapshot.tick
        signal = (tuple((s.enemy_id, s.position, s.source) for s in snapshot.sightings),
                  tuple((a.slot, a.alive, round(a.hp, 1), a.has_spike) for a in snapshot.allies),
                  tuple(e.alive for e in snapshot.enemies), frozenset(snapshot.visible_cells), snapshot.spike_dropped)
        safety_signal = signal[:3] + (signal[-1],)
        probe_active = False
        if self.exploration_probe is not None:
            probe_cells, started, probe_signal = self.exploration_probe
            advanced = probe_cells.index(origin) if origin in probe_cells else exploration_probe_steps
            probe_active = (safety_signal == probe_signal and advanced < exploration_probe_steps
                            and snapshot.tick-started < exploration_probe_ticks)
            if not probe_active:
                self.exploration_probe = None
        # Once planting starts, completing it takes priority over route changes.
        if holder and holder.plant_progress > 0 and route is not None:
            support = dict(self.plan.scout_goals) if self.plan else {}
            for a in snapshot.allies:
                if a.slot in self.flank_entries:
                    outside, inside = self.flank_entries[a.slot]
                    if encoder.scenario.grid[a.position] == 2:
                        support.pop(a.slot, None)
                    else:
                        support[a.slot] = inside if a.position == outside else outside
            self.plan = AttackPlan(route, mode, "plant", False, "plant_commit", origin, support)
            return self.plan
        previous_contacts={i:p for i,p,_ in self.last_signal[0]} if self.last_signal is not None else {}
        fresh_contact=any(i not in previous_contacts or previous_contacts[i]!=p for i,p,_ in signal[0])
        urgent = self.last_signal is not None and signal[1:3] != self.last_signal[1:3]
        urgent = urgent or self.reassess_new_contacts and fresh_contact
        stalled = snapshot.tick - self.progress_tick >= stall_ticks
        due = snapshot.tick - self.last_tick >= REASSESS_INTERVAL
        if route is not None and not urgent and not due:
            return AttackPlan(route, mode, self.plan.phase, False, "observe", self.plan.waypoint, self.plan.scout_goals)
        remaining = None
        if route is not None and origin in route.cells:
            cells = route.cells[route.cells.index(origin):]
            if len(cells) - 1 + PLANT_REQUIRED_TICKS <= snapshot.round_timer:
                remaining = Route(route.site, cells, branch_sequence(encoder.scenario, cells))
        candidates = candidate_routes(encoder.scenario, origin, snapshot.round_timer)
        if remaining and all(r.key != remaining.key for r in candidates):
            candidates.append(remaining)
        if not candidates and self.allow_deadline_route_reserve:
            candidates=candidate_routes(encoder.scenario,origin,snapshot.round_timer+TIME_MARGIN_TICKS)
        analysis = model.analyze(encoder, snapshot, observation, candidates)
        ranked = analysis["candidates"]
        if not ranked:
            return None
        # Scout until contact or an approach has actually been observed empty.
        known = bool(encoder.history.tracks) or not any(e.alive for e in snapshot.enemies)
        phase = "approach" if known else "scout"
        for candidate in ranked:
            candidate["planning_score"] = candidate["score"] + (SCOUT_INFORMATION_WEIGHT * candidate.get("information_gain", 0.) if phase == "scout" else 0.)
            if stalled and remaining and len(remaining.cells) > 1 and len(candidate["route"].cells) > 1:
                if candidate["route"].cells[1] == remaining.cells[1]:
                    candidate["planning_score"] -= STALL_STEP_PENALTY
        ranked.sort(key=lambda c: c["planning_score"], reverse=True)
        chosen = ranked[0]
        reference = next((c for c in ranked if remaining and c["route"].key == remaining.key and c["mode"] == mode), None)
        probe_active = probe_active and reference is not None
        changed_info = self.last_signal is None or signal != self.last_signal
        reason = "initial_scout" if route is None else "stalled" if stalled else "public_state_changed" if changed_info else "periodic_review"
        # Exploration is a short-horizon choice at observed decision points,
        # never a random site commitment for the entire round.
        explored = False
        options = []
        eligible_point = route is None if exploration_round_start_only else (route is None or changed_info or stalled)
        if not probe_active and rng is not None and exploration and eligible_point and rng.random() < exploration:
            explored = True
            if exploration_scope == "all_routes":
                # Sample geometry first so duplicate preserve/supported modes do
                # not increase a route's probability. Low predicted scores must
                # not prevent collection of the evidence needed to correct them.
                groups = {}
                for candidate in ranked:
                    groups.setdefault(candidate['route'].key, []).append(candidate)
                keys = sorted(groups)
                options = groups[keys[int(rng.integers(len(keys)))]]
            else:
                options = [c for c in ranked if c["planning_score"] >= ranked[0]["planning_score"] - EXPLORATION_SCORE_WINDOW]
            chosen = options[int(rng.integers(len(options)))]
        sampled_key = chosen['route'].key if explored else None
        if probe_active:
            chosen = reference
            reason = "exploration_probe"
        unrestricted_probe = explored and exploration_scope == "all_routes"
        def choice_row(candidate):
            outside,inside=route_entry_positions(encoder.scenario,candidate['route'])
            return dict(site=candidate['route'].site,key=candidate['route'].key,
                        mode=candidate['mode'],score=candidate['planning_score'],
                        entry_outside=outside,entry_inside=inside,
                        raw_score=candidate['score'],plant_probability=candidate.get('plant_probability'))
        proposed=choice_row(chosen)
        blocked_by=[]
        if reference and not unrestricted_probe and not urgent and not stalled and chosen["planning_score"] <= reference["planning_score"] + SWITCH_MARGIN:
            if chosen['route'].key!=reference['route'].key or chosen['mode']!=reference['mode']:
                blocked_by.append('route_hysteresis')
            chosen = reference
        site_gate=None
        if self.site_switch_margin is not None and not unrestricted_probe and route is not None and chosen['route'].site != route.site:
            same_site = next((c for c in ranked if c['route'].site == route.site), None)
            if same_site is not None:
                extra_steps = max(0, len(chosen['route'].cells)-len(same_site['route'].cells))
                switching_cost = self.site_switch_margin + self.switch_travel_cost*extra_steps
                site_gate=dict(proposed=choice_row(chosen),same_site=choice_row(same_site),
                    extra_steps=extra_steps,cost=switching_cost,
                    blocked=chosen['planning_score']<=same_site['planning_score']+switching_cost)
                if chosen['planning_score'] <= same_site['planning_score'] + switching_cost:
                    blocked_by.append('site_switch_cost')
                    chosen = same_site
        self.decision_debug=dict(tick=snapshot.tick,proposed=proposed,chosen=choice_row(chosen),
            fresh_public_contact=fresh_contact,urgent=urgent,blocked_by=blocked_by,site_gate=site_gate,
            stalled=stalled,progress_tick=self.progress_tick,
            best_by_site={side:choice_row(next(c for c in ranked if c['route'].site==side))
                          for side in ('L','R') if any(c['route'].site==side for c in ranked)},
            candidate_entries=sorted({(c['route'].site,*route_entry_positions(encoder.scenario,c['route']))
                                      for c in ranked}))
        changed = route is None or remaining is None or chosen["route"].key != remaining.key or chosen["mode"] != mode
        selected_route = chosen["route"] if changed else route
        selected_mode = chosen["mode"]
        if unrestricted_probe and exploration_probe_steps and exploration_probe_ticks:
            self.exploration_probe = (chosen['route'].cells, snapshot.tick, safety_signal)
        suffix = chosen["route"].cells
        if len(suffix) <= ENTRY_STEPS:
            phase = "entry"
        elif not known and chosen.get("observed_empty_fraction", 0.) >= .5:
            phase = "approach"
        waypoint = suffix[min(WAYPOINT_STEPS, len(suffix)-1)]
        scouts = {}
        rejected_entries = []
        if self.enable_flanks and (phase == "entry" or self.flank_entries) and any(e.alive for e in snapshot.enemies):
            def entry(r):
                index = next((i for i, p in enumerate(r.cells) if encoder.scenario.grid[p] == 2), len(r.cells)-1)
                return r.cells[max(0, index-1)], r.cells[index]
            if self.flank_site != selected_route.site:
                self.flank_entries = {}
                self.flank_site = selected_route.site
            self.flank_entries = {a.slot: self.flank_entries[a.slot] for a in snapshot.allies
                                  if a.alive and not a.has_spike and a.slot in self.flank_entries}
            primary = entry(selected_route)
            checked_entries = getattr(encoder, 'execution_version', 3) >= 4
            rejected_entries = []
            if checked_entries:
                from grid_paths import distance_map
                approach_board = encoder.scenario.grid.copy()
                approach_board[approach_board == 2] = 1
                valid_flanks = {}
                for a in snapshot.allies:
                    if a.slot not in self.flank_entries:
                        continue
                    outside, inside = self.flank_entries[a.slot]
                    distance = distance_map(approach_board, outside)[a.position]
                    reason = ('now_primary_entry' if outside == primary[0] else
                              'outside_path_unreachable' if distance < 0 else
                              'entry_deadline' if distance + 1 + PLANT_REQUIRED_TICKS + TIME_MARGIN_TICKS > snapshot.round_timer else None)
                    if reason:
                        rejected_entries.append(dict(slot=a.slot, outside=outside, reason=reason))
                    else:
                        valid_flanks[a.slot] = (outside, inside)
                self.flank_entries = valid_flanks
            alternate = next((c for c in ranked if c["route"].site == selected_route.site
                and entry(c["route"])[0] != primary[0]
                and len(c["route"].cells) <= len(suffix) + FLANK_MAX_EXTRA_STEPS
                and c["planning_score"] >= chosen["planning_score"] - EXPLORATION_SCORE_WINDOW), None)
            if not self.flank_entries and alternate:
                outside, inside = entry(alternate["route"])
                from grid_paths import distance_map
                distance = distance_map(approach_board if checked_entries else encoder.scenario.grid, outside)
                eligible = []
                for a in snapshot.allies:
                    if not a.alive or a.has_spike:
                        continue
                    reason = ('outside_path_unreachable' if distance[a.position] < 0 else
                              'entry_deadline' if distance[a.position] + 1 + PLANT_REQUIRED_TICKS + TIME_MARGIN_TICKS > snapshot.round_timer else None)
                    if checked_entries and reason:
                        rejected_entries.append(dict(slot=a.slot, outside=outside, reason=reason))
                    else:
                        eligible.append(a)
                players = sorted(eligible,
                                 key=lambda a: (distance[a.position] if distance[a.position] >= 0 else 9999, a.slot))
                self.flank_entries = {a.slot: (outside, inside) for a in players[:FLANK_PLAYERS]}
            for a in snapshot.allies:
                if a.slot in self.flank_entries:
                    outside, inside = self.flank_entries[a.slot]
                    if encoder.scenario.grid[a.position] == 2:
                        scouts.pop(a.slot, None)
                    else:
                        scouts[a.slot] = inside if a.position == outside else outside
        self.last_tick, self.last_signal = snapshot.tick, signal
        self.plan = AttackPlan(selected_route, selected_mode, phase, changed, reason, waypoint, scouts)
        self.decision_id += 1
        self.events.append({"tick": snapshot.tick, "decision": self.decision_id, "phase": phase,
                            "reason": reason, "changed": changed, "site": selected_route.site,
                            "key": selected_route.key, "waypoint": waypoint,
                            "scout_goals": scouts, "risk": chosen.get("entry_pressure"),
                            "flank_entries": dict(self.flank_entries),
                            "information_gain": chosen.get("information_gain"),
                            "score": chosen["planning_score"],
                            "branches": selected_route.branches,
                            "entry": next((selected_route.cells[max(0, i-1)] for i, p in enumerate(selected_route.cells)
                                           if encoder.scenario.grid[p] == 2), None),
                            "exploration_scope": exploration_scope, "exploration_applied": explored,
                            "exploration_requested_key": sampled_key,
                            "exploration_retained": bool(explored and chosen['route'].key == sampled_key),
                            "exploration_probe_active": bool(probe_active),
                            "candidate_routes": len({c['route'].key for c in ranked}),
                            "exploration_eligible_routes": (len({c['route'].key for c in ranked})
                                if explored and exploration_scope == "all_routes"
                                else len({c['route'].key for c in options}))})
        if getattr(encoder, 'execution_version', 3) >= 4:
            self.events[-1]['rejected_entry_assignments'] = rejected_entries
            outside,inside=route_entry_positions(encoder.scenario,chosen['route'])
            self.events[-1].update(public_contact_count=len(snapshot.sightings),
                entry_boundary=dict(outside=outside,inside=inside),
                predicted_plant_probability=chosen.get('plant_probability'),
                predicted_hp_damage=chosen.get('expected_hp_damage'),
                prediction_target='selected_site_adaptive_completion')
        return self.plan
