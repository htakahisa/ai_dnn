"""Public spike geometry and real cross-team tap notifications, never enemy secrets."""
from functools import lru_cache
import math
import numpy as np
from grid_lines import line_cells
from game_core import FACING_VECTORS,DEFUSE_REQUIRED_TICKS
from ..gc3_observation import ARENA_SHAPE
from ..gc3_engine_actions import FACING_DIRS, facing_towards, capabilities
from ..preplant.gc3_features_attacker_navigation import walking_distances,proximity

OFFSETS=((0,0),(-1,0),(1,0),(0,-1),(0,1))
FEATURE_WIDTH=32


def tap_state(state):
    taps=[(float(p),float(required)) for p,required in (state.get('defender_defuse_info') or {}).values()
          if float(p)>0 and float(required)>0]
    progress=max((min(1.,p/required) for p,required in taps),default=0.)
    remaining=min((max(0.,required-p) for p,required in taps),default=float(DEFUSE_REQUIRED_TICKS))
    return bool(taps),progress,remaining,len(taps)


def alignment(origin,facing,point):
    dr,dc=point[0]-origin[0],point[1]-origin[1];length=math.hypot(dr,dc)
    fx,fy=FACING_VECTORS.get(facing,(0.,0.));norm=math.hypot(fx,fy)
    return max(0.,(fx*dc+fy*dr)/(length*norm)) if length and norm else 0.


@lru_cache(maxsize=32)
def spike_geometry(wall_bytes,spike):
    walls=np.frombuffer(wall_bytes,dtype=np.bool_).reshape(ARENA_SHAPE)
    zone=[(r,c) for r in range(max(0,spike[0]-1),min(ARENA_SHAPE[0],spike[0]+2))
          for c in range(max(0,spike[1]-1),min(ARENA_SHAPE[1],spike[1]+2)) if not walls[r,c]]
    ring=np.zeros(ARENA_SHAPE,dtype=bool);vantage=np.zeros_like(ring)
    for point in zone:ring[point]=True
    for r,c in np.argwhere(~walls):
        origin=int(r),int(c)
        vantage[origin]=any(not any(walls[p] for p in line_cells(origin,target)) for target in zone)
    return tuple(zone),walking_distances(wall_bytes,ring.tobytes()),walking_distances(wall_bytes,vantage.tobytes())


def public_coverage(origin,facing,zone,walls,smoke,occupied,ignore_smoke=False):
    scores=[]
    for target in zone:
        if target==origin:continue
        cells=line_cells(origin,target)
        clear=not any(walls[p] for p in cells) and not any(p in occupied for p in cells[1:-1])
        clear &= ignore_smoke or not any(p in smoke for p in cells)
        scores.append(alignment(origin,facing,target)/(1.+math.dist(origin,target)/12.) if clear else 0.)
    # An occupied endpoint may be the visible defuser. Only intervening bodies
    # block a shot. Covering one arbitrary empty cell must not count as covering
    # the entire possible defuse area.
    return float(np.mean(scores)) if scores else 0.


def public_response_scene(char,state):
    """Return only notified/known targets, otherwise the public possible area."""
    walls=np.asarray(state['grid']==1,dtype=bool)
    zone,_,_=spike_geometry(walls.tobytes(),tuple(map(int,state['planted_pos'])))
    known=[c for c in state.get('chars',()) if c.is_alive and c.name!=char.name
           and (c.team==char.team or getattr(c,'position_known',False))]
    taps=state.get('defender_defuse_info') or {}
    targets=[tuple(map(int,c.pos)) for c in known if c.team!=char.team
             and float(taps.get(c.name,(0,1))[0])>0]
    friendly={tuple(map(int,c.pos)) for c in known if c.team==char.team}
    return (targets or [p for p in zone if p not in friendly], walls,
            set(map(tuple,state.get('smoke_cells',()))),{tuple(map(int,c.pos)) for c in known},
            bool(getattr(char,'sees_through_smoke',False)))


def response_targets(char,state,masks):
    """Training labels for public firing opportunities, never a command override.

    Masks cover both WATCH and each legal MOVE_WATCH destination. Forced facing,
    blinds and completely blocked opportunities supply no turn supervision.
    """
    if not tap_state(state)[0] or not capabilities(char,state)['can_turn'] or getattr(char,'blind_remaining',0):return None
    zone,walls,smoke,occupied,ignore=public_response_scene(char,state)
    positions=np.flatnonzero(masks['position']);coords=np.indices(walls.shape).reshape(2,-1).T
    rows=[];valid_positions=[];move_good=[]
    for index in positions:
        origin=tuple(map(int,coords[index]))
        scores=np.asarray([public_coverage(origin,f,zone,walls,smoke,occupied,ignore) for f in FACING_DIRS])
        if float(scores.max())<=0:continue
        best={FACING_DIRS[i] for i in np.flatnonzero(scores>=.9*scores.max())}
        delta=np.asarray(origin)-np.asarray(char.pos)
        moved_facing=('N' if delta[0]<0 else 'S') if delta[0] else ('W' if delta[1]<0 else 'E') if delta[1] else char.facing
        mask=np.asarray([facing_towards(origin,p,char.facing) in best for p in coords],dtype=bool)&masks['watch']
        if mask.any():valid_positions.append(int(index));rows.append(mask);move_good.append(moved_facing in best)
    if not rows:return None
    return dict(origin=int(np.ravel_multi_index(tuple(map(int,char.pos)),walls.shape)),
                positions=valid_positions,watch=np.stack(rows),move=move_good)


def legacy_coverage(origin,facing,zone,walls,smoke,occupied,ignore):
    """Exact v1 input semantics for already-saved checkpoints."""
    best=0.
    for target in zone:
        if target==origin or target in occupied:continue
        cells=line_cells(origin,target)
        if any(walls[p] for p in cells) or any(p in occupied for p in cells[1:-1]):continue
        if not ignore and any(p in smoke for p in cells):continue
        best=max(best,alignment(origin,facing,target)/(1.+math.dist(origin,target)/12.))
    return best


def guard_features(char,state,previous=None,*,feature_version=2):
    values=np.zeros(FEATURE_WIDTH,dtype=np.float32)
    spike=state.get('planted_pos')
    if not state.get('is_planted') or spike is None:return values
    spike=tuple(map(int,spike));origin=tuple(map(int,char.pos));walls=np.asarray(state['grid']==1,dtype=bool)
    zone,distance,vantage=spike_geometry(walls.tobytes(),spike)
    active,progress,remaining,count=tap_state(state);prior=previous or (False,0.)
    smoke=set(map(tuple,state.get('smoke_cells',())))
    # Only known unit positions are used; an omitted defuser is never queried.
    known=[unit for unit in state.get('chars',()) if unit.name!=char.name and unit.is_alive
           and (unit.team==char.team or getattr(unit,'position_known',False))]
    occupied={tuple(map(int,u.pos)) for u in known}
    ignore=bool(getattr(char,'sees_through_smoke',False))
    if feature_version==1:coverage=legacy_coverage(origin,char.facing,zone,walls,smoke,occupied,ignore)
    else:
        targets,*_=public_response_scene(char,state)
        coverage=public_coverage(origin,char.facing,targets,walls,smoke,occupied,ignore)
    dr,dc=spike[0]-origin[0],spike[1]-origin[1]
    values[:12]=(1.,dr/ARENA_SHAPE[0],dc/ARENA_SHAPE[1],active,progress,remaining/DEFUSE_REQUIRED_TICKS,
        count/5.,active and not prior[0],progress-prior[1],state.get('detonate_timer',0)/55.,
        alignment(origin,char.facing,spike),coverage)
    for i,(dy,dx) in enumerate(OFFSETS):
        point=origin[0]+dy,origin[1]+dx
        valid=0<=point[0]<ARENA_SHAPE[0] and 0<=point[1]<ARENA_SHAPE[1] and not walls[point]
        values[12+4*i:16+4*i]=(valid,point in occupied,
            proximity(int(distance[point])) if valid else 0.,proximity(int(vantage[point])) if valid else 0.)
    if not np.isfinite(values).all():raise ValueError('Invalid public guard features')
    return values
