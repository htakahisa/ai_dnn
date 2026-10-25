"""Team planning once per tick, with learned directives for the preplant micro."""
from copy import deepcopy
import numpy as np
from ..macro_v1.gc3_learn_attacker_macro import MacroAttackerController as SiteMacroController
from ..preplant.gc3_learn_attacker_preplant import PreplantController
from ..preplant.gc3_observation_attacker_history import encode_public,command_features,ACTION_WIDTH
from ..gc3_observation import ARENA_SHAPE
from .gc3_candidates_attacker_macro import MacroCandidates
from .gc3_policy_attacker_macro import MacroPolicy
from .gc3_directives_attacker_macro import DirectedMicroPolicy,INSTRUCTION_LABELS,INSTRUCTIONS

MACRO_SEED=20261028


class DirectedPreplantController(PreplantController):
    def __init__(self,*,directive_provider,**kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.policy,DirectedMicroPolicy):
            raise ValueError('Macro v2 requires a directive-conditioned micro policy')
        self.directive_provider=directive_provider

    def remember(self,char,state,command,observation=None):
        if self.history.live:self.history.sync_round(getattr(self,'game',None))
        if self.history.round_id<0:self.history.begin_round()
        directive=None if state.get('is_planted') else self.directive_provider(str(char.name))
        self.history.append(encode_public(char,state,self.history) if observation is None else observation,
            command_features(char,command,state),name=char.name,tick=state.get('battle_tick',0),
            command=dict(micro_command=command,macro_directive=deepcopy(directive)))

    def decide_move(self,char,state):
        if not char.is_alive or state.get('is_planted'):return list(char.pos),'HOLD'
        if self.history.live:self.history.sync_round(getattr(self,'game',None))
        if self.history.round_id<0:self.history.begin_round()
        directive=self.directive_provider(str(char.name))
        if directive is None:raise RuntimeError('Prepare macro team orders before micro decisions')
        command,row=self.policy.select_directed(char,state,self.history,directive)
        self.remember(char,state,command,row['obs'])
        self.last_selections[str(char.name)]=dict(row['selections'])
        if self.recorder is not None:self.recorder.record(str(char.name),row)
        return command


class MacroAttackerController(SiteMacroController):
    def __init__(self,*,candidates=None,macro_policy=None,macro_checkpoint=None,preplant_policy=None,
                 preplant_checkpoint=None,guard_policy=None,guard_checkpoint=None,history=None,recorder=None):
        if candidates is None:
            from map_data import NEW_MAZE_STR
            arena=np.array([[int(v) for v in row] for row in NEW_MAZE_STR.strip().splitlines()])
            candidates=MacroCandidates.configured(arena)
        self.candidates=candidates
        metadata=None
        if macro_checkpoint:
            macro_policy,metadata=MacroPolicy.load(macro_checkpoint)
            if not metadata.get('trained'):raise ValueError('Select a trained macro v2 checkpoint')
            if metadata.get('candidate_signature')!=candidates.signature:raise ValueError('Macro v2 candidate map changed')
            from ..preplant.gc3_data_attacker_preplant import file_hash
            paths=dict(PREPLANT_CHECKPOINT=preplant_checkpoint,POSTPLANT_CHECKPOINT=guard_checkpoint)
            if any(path is None or metadata.get('lower_checkpoint_hashes',{}).get(key)!=file_hash(path) for key,path in paths.items()):
                raise ValueError('Macro v2 checkpoint requires its original lower checkpoints')
        if preplant_policy is None:
            if preplant_checkpoint is None:raise ValueError('Supply the directive-conditioned micro model explicitly')
            preplant_policy,_=DirectedMicroPolicy.load(preplant_checkpoint)
        if not isinstance(preplant_policy,DirectedMicroPolicy):raise ValueError('Old micro models do not understand macro v2 directives')
        if metadata is not None and not preplant_policy.directives_trained:
            raise ValueError('Train the micro directive inputs before using a trained macro v2')
        super().__init__(macro_policy=macro_policy if macro_policy is not None else MacroPolicy.fresh('attacker',MACRO_SEED),
            preplant_policy=preplant_policy,preplant_checkpoint=preplant_checkpoint,guard_policy=guard_policy,
            guard_checkpoint=guard_checkpoint,history=history,recorder=recorder)
        self.lower.preplant=DirectedPreplantController(policy=preplant_policy,checkpoint=preplant_checkpoint,
            history=self.history,directive_provider=self.directive_for)
        if metadata is not None:self.macro_step=metadata.get('step',0)

    def reset_macro(self):
        super().reset_macro()
        self.current_plan=None
        self.player_orders={}
        self.spawn_positions={}

    def directive_for(self,name):
        return deepcopy(self.player_orders.get(str(name)))

    def decide_macro(self,char,state):
        if not self.spawn_positions:
            # Own round-start positions are public; no enemy field is read.
            own=(c for c in (self._real_game.chars if self._real_game is not None else state['chars']) if c.team=='A')
            self.spawn_positions={str(c.name):tuple(map(int,c.pos)) for c in own}
        self.macro_history.sync_public_events()
        plan,row=self.policy.select_macro(char,state,self.macro_history,self.current_plan,self.candidates)
        goal=int(plan['plant'][1]>=ARENA_SHAPE[1]//2)
        if self.current_goal!=goal:
            self.switches+=self.current_goal is not None;self.goal_age=0
        else:self.goal_age+=1
        self.current_goal=goal
        self.current_plan=deepcopy(plan)
        self.player_orders={name:dict(destination=plan['destinations'][i],instruction=plan['instructions'][i],plant=plan['plant'],
                                     retreat_target=self.spawn_positions[name])
            for i,name in enumerate(row['roster']) if row['active'][i]}
        if not state.get('is_planted'):
            self.history.goal_sites[self.history.round_id]=goal
        command=dict(players=deepcopy(self.player_orders),planned_plant=plan['plant'])
        self.macro_history.events.append(dict(obs=row['obs'].copy(),action=np.zeros(ACTION_WIDTH,dtype=np.float32),
            round_id=self.history.round_id,name='macro',tick=state.get('battle_tick',0),command=command,
            goal_site=goal,macro_plan=deepcopy(plan)))
        if self.recorder is not None:self.recorder.record('macro',row)
        self.macro_decisions+=1;self.last_action=deepcopy(command)
        return command

    def decide_move(self,char,state):
        # Setup bypasses the engine's ordinary team-tick hook. The token in
        # prepare_team_tick still samples only once for all five players.
        self.prepare_team_tick()
        if state.get('defender_setup_active'):return self.lower.decide_move(char,state)
        return self.adapter.decide_move(char,state)

    def attacker_snapshot(self):
        orders={name:dict(order,instruction_label=INSTRUCTION_LABELS[INSTRUCTIONS.index(order['instruction'])])
            for name,order in self.player_orders.items()}
        return dict(controller=type(self).__name__,macro=dict(players=orders,
            planned_plant=None if self.current_plan is None else self.current_plan['plant'],decisions=self.macro_decisions,
            selection='sampled',candidate_signature=self.candidates.signature),lower=self.lower.attacker_snapshot())
