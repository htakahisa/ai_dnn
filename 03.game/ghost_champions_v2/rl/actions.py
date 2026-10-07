"""Legal residual actions around a rule proposal; no private enemy state."""
import math
import numpy as np
from gc_v1.gc_facing import FACING_DIRS
from ..geometry import CARDINAL,valid,route,projectile_path,recon_aim,watch_cells
from .observation import proposal_kind,public_defuse_features

ACTIONS=("KEEP_MOVE","KEEP_HOLD","KEEP_ABILITY","KEEP_OBJECTIVE",
         "UP","DOWN","LEFT","RIGHT",*("FACE_"+d for d in FACING_DIRS),
         "RECON_A","RECON_Mid","RECON_B","FLASH","DANCE")
ACTION_DIM=len(ACTIONS)
ABILITY_ACTIONS=(2,16,17,18,19,20)


def candidates(char,state,tactics,proposal):
    actions=[None]*ACTION_DIM
    kind=proposal_kind(proposal,char.pos)
    actions[kind]=proposal
    profile=tactics.config["profiles"][tactics.opponent]
    forced=bool(state.get("defender_setup_active") or profile.get("preserve_v1")
                or getattr(char,"plant_timer",0)>0)
    if forced:
        return actions,np.array([a is not None for a in actions],dtype=np.bool_)
    planted=bool(state.get("is_planted"))
    spike=tactics.planted_position(state) if planted else None
    occupied=tactics.occupied(char,state)
    allied=tactics.allies(char,state)
    support=any(c.name!=char.name and math.dist(tactics.ally_position(c),char.pos)<=tactics.config["entry"]["trade_radius"] for c in allied)
    contact=any(math.dist(c.pos,char.pos)<=tactics.config["entry"]["contact_radius"] for c in tactics.enemies(char,state))
    for i,(dr,dc) in enumerate(CARDINAL,4):
        point=char.pos[0]+dr,char.pos[1]+dc
        if not valid(state["grid"],point) or point in occupied:
            continue
        if spike and math.dist(char.pos,spike)<=tactics.config["hold"]["leash_radius"] and math.dist(point,spike)>tactics.config["hold"]["leash_radius"]:
            continue
        if not planted and contact and not support and len(allied)>1:
            continue
        if not planted and profile.get("confirm_defenders") and tactics.selected_site is None:
            # Keep scouting outside site-entry staging until a side is observed.
            if any(len(route(state["grid"],point,[tuple(target)]))-1<tactics.config["entry"]["staging_distance"]
                   for target in tactics.config["plant_targets"].values() if valid(state["grid"],target)):
                continue
        actions[i]=tactics.move_action(char,point)
    for i,facing in enumerate(FACING_DIRS,8):
        actions[i]=(list(char.pos),{"facing":facing})
    cfg=tactics.recon_config()
    if (not planted and getattr(char,"ability_name",None)=="RECON" and getattr(char,"recon_charges",0)>0
            and tactics.tick<=cfg["deadline"] and tactics.tick-tactics.last_cast.get(str(char.name),-cfg["gap_ticks"])>=cfg["gap_ticks"]):
        for i,axis in enumerate(("A","Mid","B"),16):
            aim=recon_aim(state["grid"],tuple(char.pos),axis,cfg["aims"][axis])
            if aim:
                actions[i]=(list(char.pos),{"ability":"RECON","target":aim})
    enemies=tactics.enemies(char,state)
    actions[19]=tactics._flash(char,state,[],watch_cells(state["grid"],spike)) if spike and public_defuse_features(state)[0]>0 else tactics._flash(char,state,enemies)
    actions[20]=tactics._heal(char,state)
    return actions,np.array([a is not None for a in actions],dtype=np.bool_)


def commit_recon(char,state,tactics,axis,action):
    from game_core import RECON_SPEED_CELLS_PER_TICK
    name=str(char.name)
    aim=action[1]["target"]
    path=projectile_path(state["grid"],tuple(char.pos),aim)
    impact=path[-1]
    reference=tactics.recon_config()["aims"][axis]
    tactics.last_cast[name]=tactics.tick
    tactics.recon_sequences[name]+=1
    tactics.last_recon_request[name]=dict(player=name,axis=axis,origin=tuple(char.pos),target=aim,
        impact=impact,covers_anchor=max(abs(impact[i]-reference[i]) for i in (0,1))<=4,
        ready_tick=tactics.tick+math.ceil((len(path)-1)/RECON_SPEED_CELLS_PER_TICK))
