"""Actual defuser damage/kills and public spike coverage; no command overrides."""
from collections import Counter
import numpy as np
from ..gc3_reward_phase import PhaseRewards,effective_hp,potential_difference
from .gc3_features_attacker_guard import spike_geometry,public_coverage

REWARD_VERSION='gc3-postplant-actual-antidefuse-v1'
WIN_REWARD=30.
LOSS_REWARD=-30.
DEFUSED_PENALTY=-10.
DEFUSER_DAMAGE_PER_HP=.02
DEFUSER_KILL_REWARD=2.
SPIKE_COVERAGE_WEIGHT=1.
ACTIVE_DEFUSE_COVERAGE_WEIGHT=2.


def reward_settings():
    from ..gc3_reward_phase import reward_settings as base
    return dict(base(),REWARD_VERSION=REWARD_VERSION,WIN_REWARD=WIN_REWARD,LOSS_REWARD=LOSS_REWARD,
        DEFUSED_PENALTY=DEFUSED_PENALTY,DEFUSER_DAMAGE_PER_HP=DEFUSER_DAMAGE_PER_HP,DEFUSER_KILL_REWARD=DEFUSER_KILL_REWARD,
        SPIKE_COVERAGE_WEIGHT=SPIKE_COVERAGE_WEIGHT,ACTIVE_DEFUSE_COVERAGE_WEIGHT=ACTIVE_DEFUSE_COVERAGE_WEIGHT,
        defuser_damage_cap='remaining enemy HP+shield at actual plant; no healing/repeated-tap farming',
        coverage='potential difference, zero terminal potential; public spike zone, walls, smoke and own bodies')


class GuardRewards(PhaseRewards):
    def __init__(self,game,gamma):
        super().__init__(game,'postplant',gamma)
        self.damage_budget={name:effective_hp(unit) for name,unit in self.enemies.items()}
        self.defuser_kills=set();self.tap_ticks=self.starts=0;self.previous_tap=False

    def __enter__(self):
        super().__enter__();self.original_move=self.game.move_character;self.game.move_character=self.observe_move
        return self

    def __exit__(self,*args):
        self.game.move_character=self.original_move
        return super().__exit__(*args)

    def observe_move(self,char):
        result=self.original_move(char)
        if char.team=='D' and char.is_alive and char.defuse_timer>0:self.active_targets.add(str(char.name))
        return result

    def begin_tick(self):
        self.active_targets=set();self.defuser_sources={}
        super().begin_tick()

    def contribution(self,owner,victim,tick,method):
        if method=='damage' and owner is not None and victim is not None and str(owner.name) in self.own and str(victim.name) in self.enemies:
            if getattr(victim,'defuse_timer',0)>0:
                target=str(victim.name);self.active_targets.add(target)
                self.defuser_sources.setdefault(target,Counter())[str(owner.name)]+=1
        return super().contribution(owner,victim,tick,method)

    def personal_potential(self,char):
        base=super().personal_potential(char)
        coverage=0.
        if char.is_alive:
            walls=np.asarray(self.game.grid==1,dtype=bool)
            zone,_,_=spike_geometry(walls.tobytes(),tuple(self.game.planted_pos))
            occupied={tuple(c.pos) for c in self.own.values() if c.is_alive and c is not char}
            smoke=set().union(*(s['cells'] for s in self.game.smokes if s['remaining_ticks']>0)) if self.game.smokes else set()
            coverage=public_coverage(tuple(char.pos),char.facing,zone,walls,smoke,occupied,
                bool(getattr(char,'sees_through_smoke',False)))
            active=any(c.is_alive and c.defuse_timer>0 for c in self.enemies.values())
            coverage*=SPIKE_COVERAGE_WEIGHT+(ACTIVE_DEFUSE_COVERAGE_WEIGHT if active else 0.)
        base['spike_coverage']=coverage
        return base

    def finish_tick(self,terminal,success):
        shared,personal=super().finish_tick(terminal,success)
        if terminal:
            shared['outcome']=WIN_REWARD if success else LOSS_REWARD
            if getattr(self.game,'is_defused',False):shared['defused_failure']=DEFUSED_PENALTY
        # Engine contribution hooks capture a tap begun and killed in one tick.
        # A cancelled tap without damage does not earn an interruption reward.
        for target,sources in self.defuser_sources.items():
            unit=self.enemies[target]
            lost=max(0.,self.before['enemies'][target]['hp']-effective_hp(unit))
            all_events=sum(self.damage_sources[target].values())
            eligible=lost*sum(sources.values())/max(1,all_events)
            credited=min(eligible,self.damage_budget[target]);self.damage_budget[target]-=credited
            if not sources:continue
            total=sum(sources.values())
            for name,count in sources.items():
                personal[name]['defuser_damage']=personal[name].get('defuser_damage',0.)+DEFUSER_DAMAGE_PER_HP*credited*count/total
            if self.before['enemies'][target]['alive'] and not unit.is_alive and target not in self.defuser_kills:
                self.defuser_kills.add(target)
                for name,count in sources.items():
                    personal[name]['defuser_kill']=personal[name].get('defuser_kill',0.)+DEFUSER_KILL_REWARD*count/total
        active=bool(self.active_targets) or any(c.is_alive and c.defuse_timer>0 for c in self.enemies.values())
        self.tap_ticks+=active;self.starts+=active and not self.previous_tap;self.previous_tap=active
        return shared,personal
