"""Public spike geometry and real cross-team tap notifications, never enemy secrets."""
from functools import lru_cache
import math
import numpy as np
from grid_lines import line_cells
from game_core import FACING_VECTORS,DEFUSE_REQUIRED_TICKS
from ..gc3_observation import ARENA_SHAPE
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
    best=0.
    for target in zone:
        if target==origin or target in occupied:continue
        cells=line_cells(origin,target)
        if any(walls[p] for p in cells) or any(p in occupied for p in cells[1:-1]):continue
        if not ignore_smoke and any(p in smoke for p in cells):continue
        best=max(best,alignment(origin,facing,target)/(1.+math.dist(origin,target)/12.))
    return best


def guard_features(char,state,previous=None):
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
    coverage=public_coverage(origin,char.facing,zone,walls,smoke,occupied,ignore)
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
