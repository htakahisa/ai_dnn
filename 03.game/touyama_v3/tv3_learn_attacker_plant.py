"""Opponent-specific shared-player plant DQN, driven by frozen route analysis."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import io
import json
import math
from game_core import BLIND_DURATION_TICKS

import numpy as np
import torch
from torch import nn
from grid_lines import WALL_LOS_VERSION

from frc_v1 import FACING
from frc_v1.actions import FrcAction, KINDS, MOVE_STEPS, validate_action, target_required
from touyama_v3.tv3_defender_policy import PolicyEncoder, OBS_DIM as BASE_OBS_DIM, ACTION_DIM, MOVEMENTS
from touyama_v3.tv3_observer import facing
from touyama_v3.tv3_learn_public_hazards import hazard_schema
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_checkpoint import load_checkpoint
from touyama_v3.tv3_attacker_entry_utility import predicted_entry_utility, utility_schema, pending_flash_impact, flash_entry_step
from touyama_v3.tv3_attacker_route_planner import planner_schema
from touyama_v3.tv3_attacker_combat import AttackerCombatCoach, effective_utility, combat_schema, COMBAT_FEATURES, plant_ready, LOW_HP_FRACTION
from touyama_v3.tv3_attacker_entry_coordination import ENTRY_FEATURES, coordination_schema, entry_features
from touyama_v3.tv3_attacker_site_entry import (
    ENTRY_FEATURES as SITE_ENTRY_FEATURES, site_entry_schema, traffic_distances, site_entry_features,
    teacher_distances,
)

from touyama_v3.tv3_learn_attacker_fire_support import (
    FIRE_SUPPORT_FEATURES, fire_support_schema, fire_support_features,
    public_threats, defensive_advice, smoke_screen, PublicPlantCombatCoach,
)

from touyama_v3.tv3_learn_attacker_support import (
    SUPPORT_FEATURES, support_schema, support_targets, support_features, support_advice, SupportCoach,
)

from touyama_v3.tv3_learn_attacker_corner_utility import (
    CORNER_UTILITY_FEATURES,FRIENDLY_FLASH_FEATURES,FriendlyFlashTracker,
    corner_utility_schema,fill_corner_targets,corner_utility_features,
    confirmed_flash_covers_step,confirmed_flash_step_window,flash_window_allows_completion,
)

from touyama_v3.tv3_learn_attacker_recon_cover import (
    RECON_COVER_FEATURES,recon_cover_schema,recon_cover_plan,recon_cover_advice,recon_cover_features,
    incoming_clear,
)
from touyama_v3.tv3_attacker_threat_evidence import PublicEmptyEvidence, empty_evidence_schema

PLANT_VERSION = 19  # New teacher-label validation; saved models keep their original execution contract.
PLANT_ACTION = ACTION_DIM - 2
ORB_ACTION = ACTION_DIM - 1
REGIONS = 14
LEGACY_OBS_DIM = BASE_OBS_DIM + 5 * (REGIONS + 1) + 24 + COMBAT_FEATURES + ENTRY_FEATURES + SITE_ENTRY_FEATURES + 2
V10_OBS_DIM = LEGACY_OBS_DIM + FIRE_SUPPORT_FEATURES
V11_OBS_DIM = V10_OBS_DIM + SUPPORT_FEATURES
V12_OBS_DIM = V11_OBS_DIM + CORNER_UTILITY_FEATURES + FRIENDLY_FLASH_FEATURES
OBS_DIM = V12_OBS_DIM + RECON_COVER_FEATURES


def observation_dim(version):
    return {9: LEGACY_OBS_DIM, 10: V10_OBS_DIM, 11: V11_OBS_DIM, 12: V12_OBS_DIM, 13: OBS_DIM, 14: OBS_DIM, 15: OBS_DIM, 16: OBS_DIM, 17: OBS_DIM, 18: OBS_DIM, 19: OBS_DIM}[version]
CARRIER_FEATURE_INDEX = BASE_OBS_DIM + 5 * (REGIONS + 1)
GAMMA = .98
ROUTE_CORRIDOR_CELLS = 2
ROUTE_RETREAT_CELLS = 3
MAX_EDGE_TRAVERSALS = 2
EDGE_TRAVERSAL_WINDOW = 8


class PlantDQN(nn.Module):
    def __init__(self, execution_version=PLANT_VERSION):
        super().__init__()
        if execution_version not in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
            raise ValueError("Unsupported plant execution version")
        self.execution_version = execution_version
        dimension = observation_dim(execution_version)
        self.net = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Linear(128, ACTION_DIM))

    def forward(self, observation):
        return self.net(observation)


def policy_schema(scenario, execution_version=PLANT_VERSION, *, custom_extensions=None,weight_routing=None):
    if execution_version not in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
        raise ValueError("Unsupported plant execution version")
    schema = {"version": execution_version, "board": scenario.grid.tolist(), "scenario": scenario.signature, "branches": scenario.branches,
        "obs_dim": observation_dim(execution_version), "action_dim": ACTION_DIM, "gamma": GAMMA,
        "route_corridor": ROUTE_CORRIDOR_CELLS, "retreat": ROUTE_RETREAT_CELLS,
        "edge_limit": MAX_EDGE_TRAVERSALS, "edge_window": EDGE_TRAVERSAL_WINDOW, "planner": planner_schema(),
        "sensor": "frc_public_team_v1", "scope": "preplant_only",
        "site_choice": "public_online_analysis_no_site_balancing", "execution": "bounded_rally_mission_plant_v9" if execution_version == 9 else "public_target_support_plant_v10",
        "action_selection": "learned_no_forced_forward_or_plant",
        "combat": combat_schema(),
        "entry_utility": utility_schema(), "entry_coordination": coordination_schema(),
        "site_entry": site_entry_schema(),
        "plant_legal_progress_features": 2, "wall_los": WALL_LOS_VERSION, "public_hazards": hazard_schema()}
    if execution_version >= 10:
        schema["fire_support"] = fire_support_schema()
    if execution_version >= 11:
        schema["execution"] = "public_carrier_support_plant_v11"
        schema["carrier_support"] = support_schema()
    if execution_version >= 12:
        schema['execution']='public_corner_utility_plant_v12'
        schema['corner_utility']=corner_utility_schema()
        schema['friendly_flash_window']=dict(version=1,features=FRIENDLY_FLASH_FEATURES,
            evidence='own_charge_consumption_and_public_burst',freshness_ticks=5,
            duration=BLIND_DURATION_TICKS,selection='learned_actions',teacher_override=False)
    if execution_version >= 13:
        schema['execution']='public_recon_wall_cover_plant_v13'
        schema['recon_cover']=recon_cover_schema()
    if execution_version==14:
        schema['execution']='public_route_progress_escape_plant_v14'
        schema['recent_edge_progress_exception']='engine_legal_on_selected_route_decreasing_goal_traffic_v1'
    if execution_version>=15:
        schema['execution']='public_confirmed_empty_threats_plant_v15'
        schema['empty_evidence']=empty_evidence_schema()
    if execution_version>=16:
        schema['execution']='public_temporary_body_cover_plant_v16'
        schema['temporary_body_cover']=dict(version=1,incoming='potential_after_mobile_ally_leaves',
            persistent='movement_disabled_and_not_warping',outgoing='current_actual_bodies',
            teacher='potential_threat_enables_utility_without_unmarked_forced_hold',selection='learned_actions')
        schema['flash_progress_release']=dict(version=1,edge='selected_route_decreasing_goal_traffic',
            evidence='all_known_incoming_lines_covered_by_confirmed_owned_flash',
            preserves='engine_mask_team_reservation_route_corridor',selection='learned_actions')
        schema['flash_step_teacher']=dict(version=1,next_step='goal_progress_on_selected_route',
            evidence='confirmed_owned_flash_covers_all_known_incoming_lines',
            eligibility='not_blind_not_low_hp',selection='teacher_examples_only')
        schema['flash_rally_release']=dict(version=1,eligibility='carrier_ready_known_entry_lines_covered',
            duration='entry_distance_plus_required_plant_ticks_plus_one',selection='release_goal_wait_not_force_action')
    if execution_version>=17:
        from touyama_v3.tv3_attacker_entry_yield import entry_yield_schema
        schema['execution']='public_entry_yield_plant_v17'
        schema['entry_yield']=entry_yield_schema()
    if execution_version>=18:
        schema['execution']='public_deadline_route_candidates_plant_v18'
        schema['deadline_route_reserve']=dict(version=1,trigger='normal_candidates_and_remaining_route_empty',
            relaxation='remove_time_margin_only_keep_travel_plus_plant_feasibility',selection='learned_actions')
    if execution_version>=19:
        schema['execution']='public_masked_retreat_teacher_plant_v19'
        schema['masked_retreat_teacher']=dict(version=1,trigger='retreat_teacher_removed_by_coordination',
            order=('public_incoming_lines','unreturnable_lines','distance_to_retreat_step','facing'),
            selection='teacher_examples_only_legal_actions')
        schema['masked_teacher_facing']='preserve_combat_advice_facing_on_equal_distance'
    if custom_extensions is not None:
        schema['custom_extensions']=custom_extensions
    if weight_routing is not None:
        schema['weight_routing']=weight_routing
    return schema


def attach_common_policy(model,payload,scenario,opponent,analysis_hash,site):
    """Pin actual common weights without registering them in the custom optimizer."""
    if site not in ('L','R') or not isinstance(payload,bytes):
        raise ValueError('Site-specific weights require a pinned common checkpoint')
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if (state['opponent']!=opponent or state['analysis_hash']!=analysis_hash
            or state['schema']!=policy_schema(scenario,state['schema']['version'])):
        raise ValueError('Common model schema/opponent/analysis mismatch; nested custom models are not common')
    common=PlantDQN(state['schema']['version'])
    common.load_state_dict(state['model'])
    common.opponent=opponent
    common.eval()
    common.requires_grad_(False)
    object.__setattr__(model,'common_policy',common)
    model.common_policy_payload=payload
    model.weight_scope=site
    model.weight_routing=dict(site=site,common_sha256=hashlib.sha256(payload).hexdigest(),
        transition_boundary='end_custom_episode_on_scope_exit')
    return dict(model.weight_routing)


def routing_checkpoint_fields(model):
    payload=getattr(model,'common_policy_payload',None)
    return {} if payload is None else dict(common_policy_payload=payload)


def restore_policy_routing(model,state,scenario):
    from touyama_v3.tv3_model_extensions import extension_weight_scope
    scope=extension_weight_scope(state['schema'].get('custom_extensions'))
    routing=state['schema'].get('weight_routing')
    if scope in ('L','R'):
        actual=attach_common_policy(model,state.get('common_policy_payload'),scenario,
            state['opponent'],state['analysis_hash'],scope)
        if actual!=routing:
            raise ValueError('Pinned common weight hash/routing mismatch')
    elif routing is not None or 'common_policy_payload' in state:
        raise ValueError('Unexpected common weight routing')
    return routing


def load_frozen_analysis(directory, opponent, scenario):
    path = Path(directory) / opponent / "attacker_analysis_best.pt"
    payload = path.read_bytes()
    try:
        model, _, state = load_analysis(io.BytesIO(payload), scenario, opponent)
    except ValueError as error:
        raise ValueError(f"Frozen attacker analysis is incompatible: opponent={opponent}, file={path}. "
                         f"{error}") from error
    model.requires_grad_(False)
    return model, hashlib.sha256(payload).hexdigest(), state


def load_plant(path, scenario, opponent=None, analysis_hash=None):
    state = load_checkpoint(path)
    version = state["schema"].get("version")
    if version not in (9,10,11,12,13,14,15,16,17,18,19):
        raise ValueError("Plant model map/action/schema mismatch")
    from touyama_v3.tv3_model_extensions import load_extensions
    contract=state['schema'].get('custom_extensions')
    pipeline=load_extensions(contract,role='attacker_plant',opponent=state.get('opponent'))
    if version not in (9,10,11,12,13,14,15,16,17,18,19) or json.dumps(state["schema"], sort_keys=True) != json.dumps(policy_schema(scenario, version,custom_extensions=pipeline.contract,weight_routing=state['schema'].get('weight_routing')), sort_keys=True):
        raise ValueError("Plant model map/action/schema mismatch")
    if opponent is not None and state["opponent"] != opponent:
        raise ValueError("Plant model opponent mismatch")
    if analysis_hash is not None and state["analysis_hash"] != analysis_hash:
        raise ValueError("Frozen analysis changed since plant training")
    model = PlantDQN(version)
    model.opponent=state['opponent']
    model.custom_extension_contract=pipeline.contract
    model.load_state_dict(state["model"])
    restore_policy_routing(model,state,scenario)
    model.eval()
    return model, state


@dataclass
class PlantInputs:
    observation: np.ndarray
    mask: np.ndarray
    actions: tuple
    teacher: int
    distances: np.ndarray
    goal: tuple
    combat: object = None
    traffic: object = None
    support: object = None
    support_distances: object = None


def predicted_targets(scenario, snapshot, ally, analysis):
    """Map landmarks weighted by regional beliefs, never hidden enemy cells."""
    seen = {s.position for s in snapshot.sightings}
    visible = set(snapshot.visible_cells)
    ranked = []
    for region in scenario.names:
        mass = sum(row[region] for row in analysis["placement"].values())
        cells = scenario.branches[region]
        candidates = [p for p in cells if p not in visible or p in seen]
        if mass < .35 or not candidates:
            continue
        pos = min(candidates, key=lambda p: (not scenario.clear(ally.position, p),
                        abs(p[0] - ally.position[0]) + abs(p[1] - ally.position[1]), p))
        distance = abs(pos[0] - ally.position[0]) + abs(pos[1] - ally.position[1])
        ranked.append((mass / (1 + distance / 12), pos, mass))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [(p, mass) for _, p, mass in ranked[:3]]


class PlantEncoder:
    def __init__(self, scenario, execution_version=PLANT_VERSION):
        if execution_version not in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
            raise ValueError("Unsupported plant execution version")
        self.execution_version = execution_version
        if len(scenario.names) != REGIONS:
            raise ValueError("Plant encoder needs the current 14-region map")
        self.scenario = scenario
        self.base = PolicyEncoder(scenario)
        self.combat_coach = (AttackerCombatCoach if execution_version == 9 else PublicPlantCombatCoach)(scenario)
        self.support_coach = SupportCoach()
        self.flash_tracker = FriendlyFlashTracker(scenario)
        self.experimental_flash_teacher=False  # Disabled after 36-round regression; diagnostics may opt in.
        self.empty_evidence=PublicEmptyEvidence()

    def encode(self, snapshot, ally, goal, route, analysis, tracks, masks, cursor, route_mode, coordination=None):
        site = np.asarray([float(route.site == "L"), float(route.site == "R")], np.float32)
        base = self.base.encode(snapshot, ally, goal, site, tracks, masks)
        traffic = traffic_distances(snapshot, ally, goal)
        routes = teacher_distances(base.distances, traffic, ally.position)
        targets = predicted_targets(self.scenario, snapshot, ally, analysis)
        entry_plan = predicted_entry_utility(self.scenario, snapshot, ally, analysis, route, masks)
        if self.execution_version>=15:
            snapshot=self.empty_evidence.tactical_snapshot(snapshot,temporary_body_cover=self.execution_version>=16)
        seen = self.combat_coach.contacts(snapshot, ally, ally.position)
        direction = ally.facing if ally.forced_facing else facing(ally.position,
            min(seen, key=lambda p: max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1]))) if seen else targets[0][0] if targets else goal)
        movement = np.flatnonzero(base.mask[:40])
        def initial_score(i):
            dr, dc = MOVE_STEPS.get(base.actions[i].kind, (0, 0))
            p = ally.position[0]+dr, ally.position[1]+dc
            d = routes[p]
            return d if d >= 0 else 9999, int(base.actions[i].facing != direction), i
        initial = int(min(movement, key=initial_score))
        dr, dc = MOVE_STEPS.get(base.actions[initial].kind, (0, 0))
        destination = ally.position[0]+dr, ally.position[1]+dc
        hint = self.combat_coach.preaim_hint(snapshot, ally, destination, goal, analysis)
        advice = self.combat_coach.advise(snapshot, ally, destination, hint, tracks)
        # Plant training needs examples of winning a supported 3v2/5v3 fight,
        # not an unconditional retreat merely because two contacts are visible.
        if (self.execution_version == 9 and advice.reason == 'cover_retreat' and advice.contacts >= 2
                and advice.supporters >= advice.contacts and not ally.blind
                and ally.hp > ally.max_hp * LOW_HP_FRACTION):
            advice = replace(advice, position=ally.position, reason='stop_shoot')
        utility_snapshot = snapshot
        screen = None
        if self.execution_version >= 10:
            advice = defensive_advice(self.scenario, snapshot, ally, advice, tracks, masks)
            utility_snapshot = replace(snapshot, sightings=public_threats(snapshot, tracks))
            if ally.ability_name == 'SMOKE':
                screen = smoke_screen(self.scenario, snapshot, ally, utility_snapshot.sightings,
                                      masks.target[ally.slot, 0].reshape(self.scenario.grid.shape))
        recon_plan = recon_cover_plan(self.scenario,snapshot,ally,tracks,masks,destination) if self.execution_version>=13 else None
        support_plan = None
        support_values = []
        if self.execution_version >= 11:
            support_threats = support_targets(self.scenario, snapshot, route, tracks)
            support_plan = self.support_coach.plan(self.scenario, snapshot, ally, goal, route, support_threats, advice) if recon_plan is None else None
            support_values = support_features(self.scenario, snapshot, ally, support_threats, support_plan).tolist()
            advice = support_advice(self.scenario, snapshot, ally, advice, support_threats, support_plan)
        if self.execution_version >= 12 and self.experimental_flash_teacher and not ally.blind and ally.hp>ally.max_hp*LOW_HP_FRACTION:
            potential=[s for s in public_threats(snapshot,tracks) if self.scenario.clear(ally.position,s.position)
                       or self.scenario.clear(destination,s.position)]
            if potential and all(self.flash_tracker.suppressed(snapshot,s.position,tracks) for s in potential):
                advice=replace(advice,position=destination,reason='advance',contacts=0,future_contacts=0)
        if self.execution_version >= 13:
            advice = recon_cover_advice(ally,advice,recon_plan)
        if self.execution_version>=16:
            potential=sum(incoming_clear(self.scenario,snapshot,ally,destination,t)
                          for t in public_threats(snapshot,tracks))
            advice=replace(advice,future_contacts=max(advice.future_contacts,potential))
            if (destination in route.cells and traffic[ally.position]>=0
                    and 0<=traffic[destination]<traffic[ally.position]
                    and ally.hp>ally.max_hp*LOW_HP_FRACTION
                    and confirmed_flash_covers_step(self.scenario,snapshot,ally,destination,tracks,self.flash_tracker)):
                advice=replace(advice,position=destination,reason='advance',contacts=0,future_contacts=0)
            if (ally.has_spike and self.scenario.grid[ally.position]==2
                    and ally.hp>ally.max_hp*LOW_HP_FRACTION):
                from game_core import PLANT_REQUIRED_TICKS
                window=confirmed_flash_step_window(self.scenario,snapshot,ally,ally.position,tracks,self.flash_tracker)
                if flash_window_allows_completion(window,0,PLANT_REQUIRED_TICKS-max(0,ally.plant_progress)):
                    advice=replace(advice,position=ally.position,reason='advance',contacts=0,future_contacts=0)
        direction = advice.facing
        actions, legal = list(base.actions), base.mask.copy()
        # Replace defender-specific terminal actions with attacker PLANT.
        actions[PLANT_ACTION] = FrcAction("PLANT", direction)
        legal[PLANT_ACTION] = (masks.kind[ally.slot, KINDS.index("PLANT")]
                              and self.scenario.site_of(ally.position) == route.site)
        actions[ORB_ACTION] = FrcAction("COLLECT_ORB", direction)
        known = sorted({*seen, *(p for p, tick, _ in tracks.values() if snapshot.tick-tick <= 15)},
                       key=lambda p: (not self.scenario.clear(ally.position, p),
                                      max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1])), p))
        coordinates = (known[:3] + [None] * max(0, 3 - len(known)))
        coordinates += [targets[i][0] if i < len(targets) else None for i in range(2)]
        if entry_plan:
            coordinates[3] = entry_plan[0]["target"]
        if self.execution_version >= 10:
            known = sorted({s.position for s in utility_snapshot.sightings},
                           key=lambda p: (max(abs(p[0]-ally.position[0]), abs(p[1]-ally.position[1])), p))
            coordinates[:3] = known[:3] + [None] * max(0, 3-len(known))
            if screen is not None:
                coordinates[3] = screen
                entry_plan = None
        coordinates += [route.cells[-1], snapshot.spike_dropped, ally.position]
        if self.execution_version >= 12:
            coordinates=fill_corner_targets(self.scenario,snapshot,ally,route,tracks,masks,direction,coordinates)
        allies = sorted(snapshot.allies, key=lambda a: (a.position, a.ability_name))
        for kind_index, kind in enumerate(("ABILITY", "ULTIMATE")):
            for index, pos in enumerate(coordinates):
                action_index = 40 + kind_index * 8 + index
                if kind == "ABILITY" and ally.ability_name == "DANCE":
                    recipient = allies[index] if index < 5 else None
                    action = FrcAction(kind, direction, ally_slot=recipient.slot if recipient else None)
                else:
                    no_target = kind == "ULTIMATE" and not target_required(ally.slot, kind, masks)
                    action = FrcAction(kind, direction, None if no_target else pos)
                actions[action_index] = action
                try:
                    validate_action(snapshot, masks, ally.slot, action)
                    cast = {"ability": ally.ability_name, "target": pos}
                    predicted_cast = entry_plan is not None and pos == entry_plan[0]["target"]
                    legal[action_index] = (not (kind == "ABILITY" and ally.ability_name in ("FLASH", "RECON", "SMOKE", "ASH"))
                                           or predicted_cast or effective_utility(self.scenario, utility_snapshot, ally, cast))
                except (ValueError, TypeError, IndexError):
                    legal[action_index] = False
        extra = [float(ally.has_spike), ally.plant_progress / 4, float(route.site == "R"),
                 cursor / max(1, len(route.cells) - 1), len(route.cells) / 100,
                 float(route_mode == "supported"), analysis["placement_entropy"] / math.log(REGIONS + 1),
                 float(snapshot.spike_dropped is not None), goal[0] / 26, goal[1] / 44]
        extra += [float(a.has_spike) for a in allies]  # Match the base encoder's geometry order.
        for i in range(3):
            p, mass = targets[i] if i < len(targets) else ((0, 0), 0.)
            extra += [p[0] / 26, p[1] / 44, mass / 5]
        beliefs = [analysis["placement"][i][n] for i in range(5) for n in (*self.scenario.names, "dead")]
        impact = pending_flash_impact(self.scenario, snapshot, ally)
        support = entry_features(self.scenario, snapshot, ally, destination, impact, coordination)
        observation = np.concatenate((base.observation, np.asarray(
            beliefs + extra + advice.features(ally, self.scenario.grid.shape) + support
            + (fire_support_features(self.scenario, snapshot, ally, tracks).tolist()
               if self.execution_version >= 10 else [])
            + support_values
            + (corner_utility_features(self.scenario,snapshot,ally,tracks,actions,legal).tolist()
               if self.execution_version >= 12 else [])
            + (self.flash_tracker.features(snapshot,tracks) if self.execution_version >= 12 else [])
            + (recon_cover_features(self.scenario,snapshot,ally,tracks,recon_plan).tolist()
               if self.execution_version >= 13 else [])
            + site_entry_features(snapshot, ally, goal, traffic, coordination).tolist()
            + [float(legal[PLANT_ACTION]), 0.], np.float32)))
        if len(observation) != observation_dim(self.execution_version) or not np.isfinite(observation).all():
            raise ValueError("Invalid plant observation")
        join_wait = (coordination and coordination['wait'] and ally.slot in coordination['main']
                     and ally.slot in coordination['ready'] and advice.reason == 'advance')
        flash_wait = impact is not None and advice.reason == 'advance'
        if legal[PLANT_ACTION] and plant_ready(ally, advice) and not join_wait and not flash_wait:
            teacher = PLANT_ACTION
        else:
            movement = np.flatnonzero(legal[:40])
            def score(index):
                dr, dc = MOVE_STEPS.get(MOVEMENTS[index // 8], (0, 0))
                p = ally.position[0] + dr, ally.position[1] + dc
                distance = routes[p]
                return (distance if distance >= 0 else 9999,
                        int(FACING[index % 8] != direction), index)
            teacher = int(min(movement, key=score))
            for i in movement:
                dr, dc = MOVE_STEPS.get(actions[i].kind, (0, 0))
                if (ally.position[0]+dr, ally.position[1]+dc) == advice.position and actions[i].facing == advice.facing:
                    teacher = int(i)
                    break
            active = {e.kind for e in snapshot.effects if e.phase in ("active", "flight")}
            threat = bool(advice.contacts) or bool(targets and targets[0][1] >= .8 and
                       max(abs(targets[0][0][0] - ally.position[0]), abs(targets[0][0][1] - ally.position[1])) <= 8)
            if self.execution_version>=16:
                threat=threat or bool(advice.future_contacts)
            choices = []
            if entry_plan or screen is not None:
                choices = [43]
            elif ally.ability_name in ("FLASH", "SMOKE", "ASH") and threat and ally.ability_name not in active:
                choices = [40 + i for i, p in enumerate(coordinates[:5]) if p is not None]
            elif ally.ability_name == "RECON" and not seen and threat and "RECON" not in active:
                choices = [43, 44, 45]
            elif ally.ability_name == "DANCE":
                choices = [40 + i for i, a in enumerate(allies) if a.alive and a.hp < .7 * a.max_hp]
            safe_cover_cast=(self.execution_version>=16 and recon_plan is not None
                             and recon_plan['steps']==0 and recon_plan['incoming']==0)
            if self.execution_version >= 10 and advice.reason == 'cover_retreat' and not safe_cover_cast:
                choices = []  # Teach the legal escape instead of a stationary cast in the open.
            if self.execution_version >= 11 and support_plan and support_plan['steps']:
                choices = []  # Show the reachable support path before an unrelated stationary cast.
            teacher = next((i for i in choices if legal[i]), teacher)
            if self.execution_version >= 12 and self.experimental_flash_teacher and ally.ability_name=='FLASH' and actions[teacher].kind=='ABILITY':
                threats=public_threats(snapshot,tracks)
                if threats and all(self.flash_tracker.suppressed(snapshot,s.position,tracks) for s in threats):
                    teacher=int(min(movement,key=score))  # Preserve the window rather than repeat known covered support.
            # Only pause before leaving cover; an exposed player should not be
            # taught to freeze. Recon flights never require an entry pause.
            if impact is not None and advice.reason == "advance":
                movement_index = int(min(movement, key=score))
                dr, dc = MOVE_STEPS.get(actions[movement_index].kind, (0, 0))
                destination = (ally.position[0] + dr, ally.position[1] + dc)
                blocked = {a.position for a in snapshot.allies if a.alive and a.slot != ally.slot}
                step = flash_entry_step(self.scenario, ally.position, destination, impact, snapshot.smoke_cells, blocked)
                for i in movement:
                    dr, dc = MOVE_STEPS.get(actions[i].kind, (0, 0))
                    if (ally.position[0] + dr, ally.position[1] + dc) == step and actions[i].facing == direction:
                        teacher = int(i)
                        break
            if (coordination and coordination['wait'] and ally.slot in coordination['main']
                    and ally.slot in coordination['ready'] and advice.reason == 'advance'
                    and actions[teacher].kind != 'ABILITY'):
                stays = [i for i in movement if actions[i].kind == 'STAY' and actions[i].facing == direction]
                if stays:
                    teacher = int(stays[0])
        support_field = traffic_distances(snapshot, ally, support_plan['goal']) if support_plan else None
        return PlantInputs(observation, legal, tuple(actions), teacher, base.distances, goal, advice, traffic,
                           support_plan, support_field)


def demonstration_kind_loss(values, teachers, masks, *, teacher_variant=False):
    """Teach move/stay/utility/plant independently of eight facing variants."""
    legal = values.masked_fill(~masks, -1e9)
    groups = torch.stack([legal[:,i:i+8].max(1).values for i in range(0,56,8)]
                         + [legal[:,56],legal[:,57]],dim=1)
    kinds = torch.where(teachers < 56, teachers//8, teachers-49)
    if teacher_variant:
        if not bool(masks.gather(1,teachers[:,None]).all()):
            raise ValueError('Teacher variant must be legal under the final mask')
        # A correct movement kind must not reinforce a different facing;
        # ability variants are targets, so preserve the demonstrated target too.
        groups=groups.scatter(1,kinds[:,None],legal.gather(1,teachers[:,None]))
    return nn.functional.cross_entropy(groups,kinds)


def sample_plant_batch(replay, rng, batch_size, carrier_fraction=0., carriers=None):
    if carriers is None:
        carriers = [row for row in replay if len(row[0]) > CARRIER_FEATURE_INDEX
                    and row[0][CARRIER_FEATURE_INDEX] > .5] if carrier_fraction else []
    count = int(batch_size*carrier_fraction) if carriers else 0
    return ([carriers[int(rng.integers(len(carriers)))] for _ in range(count)]
            + [replay[int(rng.integers(len(replay)))] for _ in range(batch_size-count)])


def learn_plant(model, target, optimizer, replay, rng, updates, batch_size, demonstration_weight=.05,
                demonstration_kind_weight=0., carrier_sample_fraction=0., demonstration_teacher_variant=False):
    if len(replay) < batch_size:
        return None
    replay = list(replay)  # Avoid repeated linear-time random indexing of a deque.
    losses = []
    carriers = [row for row in replay if row[0][CARRIER_FEATURE_INDEX] > .5] if carrier_sample_fraction else []
    model.train()
    for _ in range(updates):
        batch = sample_plant_batch(replay,rng,batch_size,carrier_sample_fraction,carriers)
        obs = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_obs = torch.tensor(np.stack([t[3] for t in batch]), dtype=torch.float32)
        next_masks = torch.tensor(np.stack([t[4] for t in batch]), dtype=torch.bool)
        done = torch.tensor([t[5] for t in batch], dtype=torch.float32)
        teachers = torch.tensor([t[6] for t in batch], dtype=torch.long)
        masks = torch.tensor(np.stack([t[7] for t in batch]), dtype=torch.bool)
        with torch.no_grad():
            chosen = model(next_obs).masked_fill(~next_masks, -1e9).argmax(1)
            expected = rewards + GAMMA * (1 - done) * target(next_obs).gather(1, chosen[:, None]).squeeze(1)
        values = model(obs)
        loss = nn.functional.smooth_l1_loss(values.gather(1, actions[:, None]).squeeze(1), expected)
        loss += demonstration_weight * nn.functional.cross_entropy(values.masked_fill(~masks, -1e9), teachers)
        if demonstration_kind_weight:
            loss += demonstration_kind_weight * demonstration_kind_loss(values,teachers,masks,
                teacher_variant=demonstration_teacher_variant)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        with torch.no_grad():
            for dst, src in zip(target.parameters(), model.parameters()):
                dst.lerp_(src, .02)
        losses.append(float(loss.detach()))
    model.eval()
    return float(np.mean(losses))
