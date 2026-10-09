"""Public combat demonstrations and deadline-aware defusing for retake only."""
from dataclasses import replace
import numpy as np
import torch
from game_core import DEFUSE_REQUIRED_TICKS
from frc_v1.actions import MOVE_STEPS
from toruAI_v4.tv4_defender_policy import PolicyEncoder, OBS_DIM, MOVEMENTS, DEFUSE_ACTION
from toruAI_v4.tv4_attacker_combat import AttackerCombatCoach, combat_schema, effective_utility, COMBAT_FEATURES, RECENT_CONTACT_TICKS
from toruAI_v4.tv4_attacker_entry_utility import projectile_impact

LEGACY_RETAKE_OBS_DIM = OBS_DIM + COMBAT_FEATURES + 4
RETAKE_OBS_DIM = LEGACY_RETAKE_OBS_DIM + 9
DEADLINE_MARGIN_TICKS = 3
DEADLINE_WAIT_RESERVE_TICKS = 6
VICTORY_REWARD = 8.
DEFEAT_REWARD = -8.
STOPPED_HIT_REWARD = .04
UTILITY_EFFECT_REWARD = .10
UTILITY_NO_EFFECT_PENALTY = .05
UNCONTESTED_DEFUSER_IDLE_PENALTY = .08
DEFUSER_PROGRESS_REWARD = .12
DEFUSER_DELAY_PENALTY = .20
DEFUSE_PROGRESS_REWARD = .12
DEFUSE_CANCEL_PENALTY = .15


def retake_combat_schema():
    from toruAI_v4.tv4_retake_coordination import MAX_RALLY_WAIT_TICKS
    return dict(combat=combat_schema(), obs_dim=RETAKE_OBS_DIM,
                rally_wait=MAX_RALLY_WAIT_TICKS,
                deadline_margin=DEADLINE_MARGIN_TICKS, wait_reserve=DEADLINE_WAIT_RESERVE_TICKS,
                reward=dict(win=VICTORY_REWARD, loss=DEFEAT_REWARD, stopped_hit=STOPPED_HIT_REWARD,
                            utility_effect=UTILITY_EFFECT_REWARD, utility_empty=UTILITY_NO_EFFECT_PENALTY),
                defuser_idle_penalty=UNCONTESTED_DEFUSER_IDLE_PENALTY,
                deadline_reward=dict(progress=DEFUSER_PROGRESS_REWARD, delay=DEFUSER_DELAY_PENALTY,
                                     net_defuse=DEFUSE_PROGRESS_REWARD, cancel=DEFUSE_CANCEL_PENALTY),
                role="persistent_public_nearest_legal_defuse_zone_v1",
                defuse_mask="living_active_defuser_only_v2",
                objective="defender_round_victory", selection="learned_legal_actions")


def initialize_retake(model, search):
    """Preserve search features; new combat columns start with zero weights."""
    weights = {k: v.clone() for k, v in search.state_dict().items()}
    old = weights['net.0.weight']
    expanded = torch.zeros_like(model.state_dict()['net.0.weight'])
    expanded[:, :old.shape[1]] = old
    weights['net.0.weight'] = expanded
    model.load_state_dict(weights)


def apply_retake_outcome(transitions, indices, alive, won):
    total = 0.
    for name, index in indices.items():
        # A dead participant does not inherit a teammate's later victory.
        bonus = VICTORY_REWARD if won and alive[name] else (0. if won else DEFEAT_REWARD)
        transitions[index][2] += bonus
        transitions[index][5] = 1.
        total += bonus
    return total


def effect_credit(record):
    effects = min(3, len(record['affected_enemies']) + int(record['blocked_lines'] > 0))
    return UTILITY_EFFECT_REWARD*effects if effects else -UTILITY_NO_EFFECT_PENALTY


class LegacyRetakeEncoder:
    def __init__(self, scenario):
        self.scenario = scenario
        self.base = PolicyEncoder(scenario)
        self.coach = AttackerCombatCoach(scenario)

    def encode(self, snapshot, ally, goal, probabilities, tracks, masks):
        inputs = self.base.encode(snapshot, ally, goal, probabilities, tracks, masks)
        legal_moves = np.flatnonzero(inputs.mask[:40])
        def rank(i):
            dr, dc = MOVE_STEPS.get(MOVEMENTS[i//8], (0, 0))
            distance = inputs.distances[ally.position[0]+dr, ally.position[1]+dc]
            return distance if distance >= 0 else 9999, i
        best = min(legal_moves, key=rank)
        dr, dc = MOVE_STEPS.get(MOVEMENTS[best//8], (0, 0))
        destination = ally.position[0]+dr, ally.position[1]+dc
        advice = self.coach.advise(snapshot, ally, destination, goal, tracks)
        remaining = max(0, DEFUSE_REQUIRED_TICKS-ally.defuse_progress)
        from grid_paths import distance_map
        routes = distance_map(self.scenario.grid, snapshot.spike_planted)
        distance = max(0, routes[ally.position]-1)
        slack = snapshot.detonate_timer - remaining - distance - DEADLINE_MARGIN_TICKS
        urgent = slack <= DEADLINE_WAIT_RESERVE_TICKS
        # Running out the clock cannot win. Do not teach waiting for assembly
        # when it consumes the time needed to reach and finish the defuse.
        if urgent and (advice.reason == 'assemble' or not inputs.mask[DEFUSE_ACTION]):
            advice = replace(advice, position=destination, reason='deadline_entry')
        allies = [a for a in snapshot.allies if a.alive]
        defuser = min(allies, key=lambda a: (not bool(a.defuse_progress),
                      routes[a.position] if routes[a.position] >= 0 else 9999, -a.hp, a.slot))
        may_defuse = (ally.slot == defuser.slot and inputs.mask[DEFUSE_ACTION]
                      and snapshot.detonate_timer >= remaining
                      and (urgent or (not ally.blind and (not advice.contacts or advice.supporters >= advice.contacts))))
        direction = ally.facing if ally.forced_facing else advice.facing
        inputs.actions = tuple(replace(a, facing=direction) if i >= 40 else a
                               for i, a in enumerate(inputs.actions))
        def movement_rank(i):
            dr, dc = MOVE_STEPS.get(MOVEMENTS[i//8], (0, 0))
            p = ally.position[0]+dr, ally.position[1]+dc
            return (p != advice.position, inputs.actions[i].facing != direction, *rank(i))
        inputs.teacher = int(min(legal_moves, key=movement_rank))
        # A teammate already defusing needs cover, not a second defuser
        # standing on the same tile. This changes examples, not legal masks.
        if ally.slot != defuser.slot and defuser.defuse_progress and not advice.contacts:
            stays = [i for i in legal_moves if inputs.actions[i].kind == 'STAY']
            inputs.teacher = int(min(stays, key=lambda i: inputs.actions[i].facing != direction))
        if may_defuse:
            inputs.teacher = DEFUSE_ACTION
        else:
            # Keep public remembered threats for effect geometry, without
            # treating a confirmed empty cell as an enemy.
            from frc_v1.perception import Sighting
            threats = list(snapshot.sightings)
            alive = {e.enemy_id for e in snapshot.enemies if e.alive}
            seen = {s.enemy_id for s in threats}
            for eid, (pos, tick, _) in tracks.items():
                if eid in alive and eid not in seen and snapshot.tick-tick <= RECENT_CONTACT_TICKS and pos not in snapshot.visible_cells:
                    threats.append(Sighting(eid, pos, 'history'))
            effect_snapshot = replace(snapshot, sightings=tuple(threats))
            active = {e.kind for e in snapshot.effects if e.phase in ('flight', 'active')}
            useful = []
            for i in np.flatnonzero(inputs.mask[40:48])+40:
                action = inputs.actions[i]
                if ally.ability_name in active:
                    continue
                if action.target is not None and ally.ability_name in ('FLASH', 'RECON', 'SMOKE', 'ASH'):
                    cast = dict(ability=ally.ability_name, target=action.target)
                    effective = effective_utility(self.scenario, effect_snapshot, ally, cast)
                    if ally.ability_name in ('FLASH', 'RECON'):
                        _, delay = projectile_impact(self.scenario, ally.position, action.target, ally.ability_name)
                        effective &= snapshot.detonate_timer > delay+remaining+DEADLINE_MARGIN_TICKS
                    if effective:
                        useful.append(int(i))
                elif ally.ability_name == 'DANCE' and action.ally_slot is not None:
                    target = next(a for a in snapshot.allies if a.slot == action.ally_slot)
                    if target.alive and target.hp < .6*target.max_hp:
                        useful.append(int(i))
            if useful:
                inputs.teacher = useful[0]
        inputs.observation = np.concatenate((inputs.observation,
            np.asarray(advice.features(ally, self.scenario.grid.shape)+[
                slack/55, remaining/DEFUSE_REQUIRED_TICKS, float(urgent), float(ally.slot == defuser.slot)], np.float32)))
        inputs.combat_contact = bool(advice.contacts)
        return inputs


class RetakeRoles:
    def __init__(self, scenario):
        self.scenario, self.slot = scenario, None

    def select(self, snapshot):
        from toruAI_v4.tv4_retake_progress_audit import defuse_distances
        field = defuse_distances(self.scenario, snapshot.spike_planted)
        alive = [a for a in snapshot.allies if a.alive and np.isfinite(field[a.position])]
        active = next((a for a in alive if a.defuse_progress), None)
        current = next((a for a in alive if a.slot == self.slot), None)
        best = min(alive, key=lambda a: (field[a.position], bool(a.blind), -a.hp, a.slot)) if alive else None
        # Keep the role stable; an active defuse takes priority. Reassign on
        # death or when a teammate can finish but the current member cannot.
        if active:
            current = active
        elif current is None or (field[current.position]+DEFUSE_REQUIRED_TICKS > snapshot.detonate_timer
                                 and best is not None and field[best.position]+DEFUSE_REQUIRED_TICKS <= snapshot.detonate_timer):
            current = best
        self.slot = current.slot if current else None
        return current, field

    def goals(self, snapshot, goals, launched):
        current, field = self.select(snapshot)
        if current is None or not launched:
            return goals
        plant = snapshot.spike_planted
        occupied = {a.position for a in snapshot.allies if a.alive and a.slot != current.slot}
        candidates = [(plant[0]+dr, plant[1]+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                      if 0 <= plant[0]+dr < self.scenario.grid.shape[0] and 0 <= plant[1]+dc < self.scenario.grid.shape[1]
                      and self.scenario.grid[plant[0]+dr, plant[1]+dc] != 1 and (plant[0]+dr, plant[1]+dc) not in occupied]
        from grid_paths import distance_map
        maps = {p: distance_map(self.scenario.grid, p) for p in candidates}
        reachable = [p for p in candidates if maps[p][current.position] >= 0]
        if reachable:
            goal = min(reachable, key=lambda p: (maps[p][current.position], p))
            return {**goals, current.slot: goal}
        return goals


class RetakeEncoder(LegacyRetakeEncoder):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.roles = RetakeRoles(scenario)

    def encode(self, snapshot, ally, goal, probabilities, tracks, masks):
        inputs = super().encode(snapshot, ally, goal, probabilities, tracks, masks)
        designated, field = self.roles.select(snapshot)
        is_defuser = designated is not None and ally.slot == designated.slot
        distance = float(field[ally.position])
        remaining = max(0, DEFUSE_REQUIRED_TICKS-ally.defuse_progress)
        slack = snapshot.detonate_timer-distance-remaining-DEADLINE_MARGIN_TICKS
        active = next((a for a in snapshot.allies if a.alive and a.defuse_progress), None)
        legal_defuse = bool(inputs.mask[DEFUSE_ACTION])
        # Once started, do not teach cancellation just because an enemy
        # appears. Cancelling throws away the engine's entire accumulated time.
        if is_defuser and legal_defuse and ally.defuse_progress:
            inputs.teacher = DEFUSE_ACTION
        elif is_defuser and legal_defuse and slack <= DEADLINE_WAIT_RESERVE_TICKS:
            inputs.teacher = DEFUSE_ACTION
        elif not is_defuser and inputs.teacher == DEFUSE_ACTION:
            stays = [i for i in np.flatnonzero(inputs.mask[:40]) if inputs.actions[i].kind == 'STAY']
            inputs.teacher = int(min(stays, key=lambda i: inputs.actions[i].facing != inputs.actions[DEFUSE_ACTION].facing))
        inputs.retake_context = dict(defuser=is_defuser, distance=distance, slack=slack,
                                     progress=ally.defuse_progress, active_slot=active.slot if active else None)
        # Correct old role flags too; retain all original columns for diagnostic
        # weight expansion while adding explicit legal-zone movement signals.
        inputs.observation[-4:] = [slack/55, remaining/DEFUSE_REQUIRED_TICKS,
                                   float(slack <= DEADLINE_WAIT_RESERVE_TICKS), float(is_defuser)]
        neighbours = []
        for kind in MOVEMENTS:
            dr, dc = MOVE_STEPS.get(kind, (0, 0)); p = ally.position[0]+dr, ally.position[1]+dc
            valid = 0 <= p[0] < field.shape[0] and 0 <= p[1] < field.shape[1] and np.isfinite(field[p])
            neighbours.append(float(np.clip(distance-field[p], -1, 1)) if valid and np.isfinite(distance) else 0.)
        inputs.observation = np.concatenate((inputs.observation, np.asarray([
            min(100, distance)/100, float(legal_defuse), float(active is not None),
            (active.defuse_progress if active else 0)/DEFUSE_REQUIRED_TICKS, *neighbours], np.float32)))
        return inputs


def deadline_action_reward(inputs, before, after, action, field, *, defused=False):
    context = getattr(inputs, 'retake_context', None)
    if context is None:
        return DEFUSE_PROGRESS_REWARD*max(0, after.defuse_timer-before.defuse_progress)
    progress_after = after.defuse_timer if after.is_alive or defused else 0
    reward = DEFUSE_PROGRESS_REWARD*(progress_after-before.defuse_progress)
    if before.defuse_progress and action.kind != 'DEFUSE' and not defused:
        reward -= DEFUSE_CANCEL_PENALTY
    if context['defuser'] and not before.defuse_progress:
        before_dist, after_dist = context['distance'], float(field[tuple(after.pos)])
        progress = float(np.clip(before_dist-after_dist, -1, 1)) if np.isfinite(before_dist+after_dist) else 0.
        reward += DEFUSER_PROGRESS_REWARD*progress
        if progress <= 0 and action.kind != 'DEFUSE':
            # Time is spent even while fighting or casting. Only the designated
            # role incurs delay costs; teammates remain free to provide cover.
            urgency = 2. if context['slack'] <= DEADLINE_WAIT_RESERVE_TICKS else 1.
            reward -= DEFUSER_DELAY_PENALTY*urgency
    return reward
