"""FRC/right utility teacher examples; action selection remains learned.

Rank existing legal casts by public coverage and usable allied firing lines.
No extra reward for merely casting, no candidate/mask/observation changes.
"""
import numpy as np
from abilities_los import AbilityLosMixin
from game_core import FACING_VECTORS, RECON_RADIUS
from grid_lines import line_cells
from touyama_v3.tv3_attacker_combat import LOW_HP_FRACTION
from touyama_v3.tv3_attacker_entry_utility import projectile_impact
from touyama_v3.tv3_learn_attacker_corner_utility import covered_threats
from touyama_v3.tv3_learn_attacker_fire_support import shot_clear

EXTENSION_SPEC=dict(id='attacker_plant_frc_v1_R',revision=2,
                    role='attacker_plant',opponent='frc_v1',site='R')

# 通常設定：公開射線を持つ味方がいる投擲だけを新しい教師例にする。
MIN_SUPPORTING_ALLIES = 1


class _PublicGeometry(AbilityLosMixin):
    """Engine geometry operating only on the public board."""
    def __init__(self, snapshot):
        self.grid = np.asarray(snapshot.grid)
        self.height, self.width = self.grid.shape

    def clear(self, origin, target):
        return self.check_cell_line_of_sight(origin, target, block_smoke=False)


def _supporters(geometry, snapshot, caster, target, *, through_smoke=False):
    smoke = set(snapshot.smoke_cells)
    count = 0
    for ally in snapshot.allies:
        if not ally.alive or ally.slot == caster.slot or ally.blind:
            continue
        dx, dy = FACING_VECTORS[ally.facing]
        if (target[1]-ally.position[1])*dx + (target[0]-ally.position[0])*dy < 0:
            continue
        if through_smoke:
            # Recon can remove smoke's visual obstruction, never walls/bodies.
            from dataclasses import replace
            without_smoke = replace(snapshot, smoke_cells=())
            if (not smoke.intersection(line_cells(ally.position, target))
                    or not shot_clear(geometry, without_smoke, ally.position, target, ally.slot)):
                continue
        elif not shot_clear(geometry, snapshot, ally.position, target, ally.slot):
            continue
        count += 1
    return count


def adjust_goal(context,goal):
    return goal


def adjust_inputs(context,inputs):
    ally, snapshot = context.ally, context.snapshot
    if (context.role != 'attacker_plant' or context.opponent != 'frc_v1' or context.site != 'R'
            or ally is None or ally.ability_name not in ('FLASH', 'RECON')
            or ally.blind or ally.hp <= ally.max_hp * LOW_HP_FRACTION
            or ally.has_spike or not snapshot.sightings):
        return inputs
    advice = inputs.combat
    if (advice is None or advice.reason == 'cover_retreat'
            or inputs.actions[inputs.teacher].kind in ('PLANT', 'ULTIMATE', 'COLLECT_ORB')
            or inputs.support and inputs.support['steps']):
        return inputs
    if any(e.kind == ally.ability_name and e.phase in ('flight', 'active') for e in snapshot.effects):
        return inputs
    geometry = _PublicGeometry(snapshot)
    candidates = []
    for index in range(40, 48):
        action = inputs.actions[index]
        if not inputs.mask[index] or action.target is None:
            continue
        if ally.ability_name == 'FLASH':
            _, delay, hits = covered_threats(geometry, snapshot, ally.position,
                                           action.target, snapshot.sightings)
            supported = [s for s in hits if _supporters(geometry, snapshot, ally, s.position)
                         >= MIN_SUPPORTING_ALLIES]
        else:
            impact, delay = projectile_impact(geometry, ally.position, action.target, 'RECON')
            hits = [s for s in snapshot.sightings if s.source != 'reveal'
                    and max(abs(impact[0]-s.position[0]), abs(impact[1]-s.position[1])) <= RECON_RADIUS]
            supported = [s for s in hits if _supporters(geometry, snapshot, ally, s.position,
                                                       through_smoke=True) >= MIN_SUPPORTING_ALLIES]
        # Arrival must precede the deadline; predicted coverage is not actual effect.
        if delay > 0 and delay + 1 < snapshot.round_timer and supported:
            candidates.append((-len(supported), -len(hits), delay, index))
    if candidates:
        inputs.teacher = min(candidates)[-1]
    return inputs


def adjust_reward(context,reward):
    # Actual effects retain the trainer's audited effect credit/ineffective-cast cost.
    return reward
