"""Preserve retreat intent when coordination removes a teacher action."""
from frc_v1.actions import MOVE_STEPS
from touyama_v3.tv3_learn_attacker_fire_support import public_threats, shot_clear
from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear


def masked_retreat_teacher(scenario, snapshot, ally, inputs, tracks):
    """Return a legal movement example using public risk, never select policy actions."""
    threats=public_threats(snapshot,tracks)
    advice=inputs.combat
    def rank(index):
        action=inputs.actions[index]
        dr,dc=MOVE_STEPS.get(action.kind,(0,0))
        position=(ally.position[0]+dr,ally.position[1]+dc)
        incoming=[t for t in threats if incoming_clear(scenario,snapshot,ally,position,t)]
        unreturnable=sum(not shot_clear(scenario,snapshot,position,t.position,ally.slot,
            potential_body_cover=getattr(snapshot,'temporary_body_cover',False)) for t in incoming)
        distance=abs(position[0]-advice.position[0])+abs(position[1]-advice.position[1])
        return len(incoming),unreturnable,distance,int(action.facing!=advice.facing),index
    candidates=[i for i in range(40) if inputs.mask[i]]
    if not candidates:
        raise ValueError('No legal movement for a masked retreat teacher')
    return min(candidates,key=rank)
