"""Danger estimates derived only from copied, visible display effects.

Projectile extrapolation is a guess from its visible direction and public
speed, never its engine path/target. Observed age is not a cast countdown.
Unknown affiliation is treated conservatively; hidden owners are never read.
"""
from collections import deque
import math
import numpy as np
from game_core import FLASH_SPEED_CELLS_PER_TICK
from .geometry import CARDINAL, valid, los

KINDS=("FLASH","RECON","ASH","TUNNEL","NEON","BALEMOON","DESTRUCTION",
       "SMOKE","ESCAPE","RAID","MONITOR")
PHASES=("flight","warning","active","other")
EFFECT_SLOTS=8
EFFECT_WIDTH=26
HAZARD_BLOCKS=(("public_effects",EFFECT_SLOTS*EFFECT_WIDTH),
               ("local_hazards",7*7*3),("hazard_summary",8))


class PublicHazards:
    def __init__(self):
        self.reset()

    def reset(self):
        self.first_seen={}
        self.effects=()
        self.tick=0
        self._risks={}
        self.casts={}
        self.known_own=set()

    def note_action(self,char,result,tick):
        name=str(char.name)
        previous=self.casts.get(name)
        if previous and previous[0]==tick:
            self.casts.pop(name)
        command=result[1] if isinstance(result,tuple) and len(result)>1 else None
        if not isinstance(command,dict):
            return
        kind=command.get("ability") or command.get("ultimate")
        if not isinstance(kind,str):
            return
        resource="ultimate_points" if command.get("ultimate") else kind.lower()+"_charges"
        target=command.get("target")
        self.casts[name]=(tick,kind,tuple(char.pos),tuple(target) if target is not None else None,
                          resource,int(getattr(char,resource,0)))

    def update(self,state):
        self.effects=tuple(state.get("public_effects",()))
        self.tick=int(state.get("battle_tick",0))
        self.grid=state["grid"]
        self.smoke=frozenset(state.get("smoke_cells",()))
        present={(e.handle,e.phase) for e in self.effects}
        self.first_seen={key:age for key,age in self.first_seen.items() if key in present}
        for key in present:
            self.first_seen.setdefault(key,self.tick)
        self.casts={name:cast for name,cast in self.casts.items() if self.tick-cast[0]<=6}
        allies={str(c.name):c for c in state.get("chars",())}
        self.known_own.intersection_update(e.handle for e in self.effects)
        for e in self.effects:
            for name,(tick,kind,origin,target,resource,before) in self.casts.items():
                ally=allies.get(name)
                if ally is None or int(getattr(ally,resource,before))>=before:
                    continue  # Requested but not actually consumed.
                direct=(self.tick-tick<=1 and e.kind==kind and (
                    (e.trail and e.trail[0]==origin) or
                    (e.kind=="BALEMOON" and e.position==origin) or
                    (target is not None and e.position==target)))
                ash_impact=(kind=="ASH" and e.kind=="DESTRUCTION" and e.position==target)
                if direct or ash_impact:
                    self.known_own.add(e.handle)
        self._risks={}
        self._areas={e.handle:frozenset(e.cells) for e in self.effects}
        self._projected={}
        for e in self.effects:
            if e.phase!="flight" or e.position is None:
                continue
            points=[e.position]
            for step in range(1,FLASH_SPEED_CELLS_PER_TICK+1):
                point=tuple(round(e.position[i]+step*e.direction[i]) for i in (0,1))
                if not valid(self.grid,point):
                    break
                points.append(point)
            self._projected[e.handle]=tuple(points)

    def channels(self,point):
        point=tuple(point)
        if point in self._risks:
            return self._risks[point]
        result=[0.,0.,0.]
        for effect in self.effects:
            if effect.handle in self.known_own:
                continue
            value=0.
            channel=0 if effect.phase=="warning" else 1
            if effect.kind in ("TUNNEL","NEON","BALEMOON","DESTRUCTION"):
                if point in self._areas[effect.handle]:
                    value=.8 if effect.phase=="warning" else 1.
            elif effect.phase=="flight":
                channel=2
                for predicted in self._projected.get(effect.handle,()):
                    if effect.kind=="FLASH" and los(self.grid,point,predicted,self.smoke):
                        # The engine has no flash distance attenuation.
                        value=max(value,.65)
                    elif effect.kind=="ASH" and max(abs(point[i]-predicted[i]) for i in (0,1))<=1:
                        value=max(value,.7)
                    elif effect.kind=="RECON" and max(abs(point[i]-predicted[i]) for i in (0,1))<=4:
                        value=max(value,.2)
            result[channel]=max(result[channel],value)
        self._risks[point]=tuple(result)
        return tuple(result)

    def risk(self,point):
        return max(self.channels(point))

    def features(self,char):
        origin=tuple(char.pos)
        height,width=self.grid.shape
        def distance(effect):
            return min((math.dist(origin,p) for p in effect.cells),default=(
                math.dist(origin,effect.position) if effect.position is not None else 999.))
        effects=sorted(self.effects,key=lambda e:(distance(e),e.kind,e.handle))
        slots=[]
        for e in effects[:EFFECT_SLOTS]:
            position=e.position
            slots.extend([1.,*[e.kind==kind for kind in KINDS],
                          *[(e.phase if e.phase in PHASES else "other")==phase for phase in PHASES],
                          position is not None,
                          (position[0]-origin[0])/height if position is not None else 0.,
                          (position[1]-origin[1])/width if position is not None else 0.,
                          *e.direction,min(1.,(self.tick-self.first_seen[(e.handle,e.phase)])/20),
                          origin in self._areas[e.handle],min(1.,distance(e)/max(height,width)),
                          min(1.,e.level/10),e.handle in self.known_own])
        slots.extend([0.]*(EFFECT_SLOTS*EFFECT_WIDTH-len(slots)))
        local=[value for dr in range(-3,4) for dc in range(-3,4)
               for value in self.channels((origin[0]+dr,origin[1]+dc))]
        choices=[origin,*[(origin[0]+dr,origin[1]+dc) for dr,dc in CARDINAL]]
        summary=[min(1.,len(effects)/16),min(1.,max(0,len(effects)-EFFECT_SLOTS)/16),
                 min(1.,sum(e.phase=="warning" for e in effects)/16),
                 *[self.risk(p) if valid(self.grid,p) else 1. for p in choices]]
        return np.asarray([*slots,*local,*summary],dtype=np.float32)

    def escape(self,char,occupied):
        """Shortest observed safe route; also handles multi-cell warning areas."""
        start=tuple(char.pos)
        if self.risk(start)<.35:
            return None
        queue=deque([(start,())])
        visited={start}
        routes=[]
        while queue:
            point,path=queue.popleft()
            if path:
                routes.append((self.risk(point),len(path),path))
            if len(path)>=6:
                continue
            for dr,dc in CARDINAL:
                nxt=point[0]+dr,point[1]+dc
                if nxt not in visited and nxt not in occupied and valid(self.grid,nxt):
                    visited.add(nxt)
                    queue.append((nxt,(*path,nxt)))
        if not routes:
            return None
        risk,_,path=min(routes,key=lambda item:(item[0],item[1]))
        return path[0] if risk+1e-6<self.risk(start) else None


def avoidance_reward(before,after,before_risk,after_risk,*,gamma,terminal,weight,hit_weight,kinds):
    """Potential shaping telescopes; status penalties require a visible hazard.

    No separate repeated 'successful dodge' bonus is awarded. The potential
    sums to zero from a safe start to terminal. Win/loss is the main objective.
    """
    reward=weight*(before_risk-gamma*(0. if terminal else after_risk))
    if kinds.intersection(("FLASH","TUNNEL")) and after[0]>max(0,before[0]-1):
        reward-=hit_weight
    if kinds.intersection(("ASH","BALEMOON","DESTRUCTION")) and after[1]>max(0,before[1]-1):
        reward-=hit_weight
    return reward
