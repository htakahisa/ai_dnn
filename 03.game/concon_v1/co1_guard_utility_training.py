"""Training-only preference targets derived from guard IQ and recent sightings."""

import numpy as np
import torch
from game_core import FLASH_MAX_FLIGHT_TICKS, FLASH_SPEED_CELLS_PER_TICK, RECON_REVEAL_SIZE
from concon_v1.co1_guard_common import ACTION_DIM, ULTIMATE_ACTION, SELF_SMOKE_ACTION, MEMORY_TICKS, decode_action

UTILITY_LEARNING_VERSION = 1
DEFAULT_PREFERENCE_MARGIN = 0.25
DEFAULT_QUIET_LOSS_WEIGHT = 0.5
DEFAULT_COUNTER_LOSS_WEIGHT = 0.25


def learning_masks(controller,char,mask,context):
    """Never consult hidden enemy characters; sightings come from build_inputs."""
    moves = np.zeros(ACTION_DIM,dtype=bool)
    moves[:40] = mask[:40]
    abilities = np.asarray(mask,dtype=bool).copy()
    abilities[:40] = False
    abilities[ULTIMATE_ACTION:SELF_SMOKE_ACTION] = False
    quiet = abilities.copy() if not any(context[k] for k in ('tap','fireable','impaired')) else np.zeros_like(abilities)
    counter = np.zeros_like(abilities)
    game = getattr(controller,'game',None)
    if context['impaired'] and game is not None:
        threats = [pos for pos,tick in controller.sightings.values()
                   if 0 <= context['tick']-tick < MEMORY_TICKS]
        for action in np.flatnonzero(abilities):
            _,payload = decode_action(action,tuple(char.pos),context['targets'],context['ultimate_actions'])
            ability,target = payload['ability'],payload['target']
            if ability=='SMOKE':
                # Self smoke is cover for blindness, not for an active reveal.
                useful = (context['blind'] and not context['revealed']
                          and tuple(target)==tuple(char.pos)
                          and context['exposures'][tuple(char.pos)]>0)
            else:
                path = game._projectile_path(tuple(char.pos),tuple(target))
                limit = FLASH_MAX_FLIGHT_TICKS*FLASH_SPEED_CELLS_PER_TICK if ability=='FLASH' else len(path)-1
                impact = path[min(len(path)-1,limit)] if path else target
                useful = any(game.check_cell_line_of_sight(pos,impact,block_smoke=True)
                             if ability=='FLASH' else
                             max(abs(pos[0]-impact[0]),abs(pos[1]-impact[1]))<=RECON_REVEAL_SIZE//2
                             for pos in threats)
            counter[action] = useful
    return np.stack((moves,quiet,counter))


def preference_loss(values,masks,margin=DEFAULT_PREFERENCE_MARGIN):
    """Quiet casts below movement; useful counters above idle waiting, not cover."""
    masks = torch.as_tensor(masks,dtype=torch.bool,device=values.device)
    move,quiet,counter = masks.unbind(1)
    loss = values.sum()*0
    quiet_rows = move.any(1)&quiet.any(1)
    if quiet_rows.any():
        move_best = values[quiet_rows].masked_fill(~move[quiet_rows],-torch.inf).amax(1).detach()
        cast_best = values[quiet_rows].masked_fill(~quiet[quiet_rows],-torch.inf).amax(1)
        loss = loss+DEFAULT_QUIET_LOSS_WEIGHT*torch.relu(cast_best+margin-move_best).mean()
    wait = move.clone()
    wait[:,:32] = False
    counter_rows = wait.any(1)&counter.any(1)
    if counter_rows.any():
        wait_best = values[counter_rows].masked_fill(~wait[counter_rows],-torch.inf).amax(1).detach()
        cast_best = values[counter_rows].masked_fill(~counter[counter_rows],-torch.inf).amax(1)
        loss = loss+DEFAULT_COUNTER_LOSS_WEIGHT*torch.relu(wait_best+margin-cast_best).mean()
    return loss
