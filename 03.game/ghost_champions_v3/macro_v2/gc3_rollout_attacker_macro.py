"""Three-output macro episodes; only preplant macro decisions receive PPO credit."""
from collections import Counter
from contextlib import redirect_stdout
import io
import numpy as np
import torch
from ..gc3_policy import FreshPolicy
from ..gc3_rollout import Recorder,engine_directory,freeze_modules,module_digest
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_state_attacker_resources import MatchResources
from ..preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS,distinguish_same_roster
from .gc3_candidates_attacker_macro import MacroCandidates
from .gc3_directives_attacker_macro import DirectedMicroPolicy,INSTRUCTIONS
from .gc3_learn_attacker_macro import MacroAttackerController
from .gc3_reward_attacker_macro import MacroRewards,REWARD_VERSION
from ..macro_v1.gc3_opponents_attacker_macro import build_opponent


class PreplantMacroRecorder(Recorder):
    def record(self,name,row):
        # Postplant commands do not operate the existing guard. Their future
        # rewards belong to the preceding preplant plan, not ignored actions.
        if not row['obs'][2]:super().record(name,row)


def load_lower_policies(preplant_path,guard_path):
    preplant,before=DirectedMicroPolicy.load(preplant_path)
    guard,after=FreshPolicy.load(guard_path,role='attacker')
    if not before.get('trained') or before.get('stage')!='directive_rl' or not preplant.directives_trained or not after.get('trained'):
        raise ValueError('Use the trained directive micro and the trained postplant guard')
    return preplant.requires_grad_(False).eval(),guard.requires_grad_(False).eval()


def play_macro(code,seed,*,policy,preplant,guard,candidates,history=None,macro_history=None,gamma=.99,gae_lambda=.95):
    if code not in MAJOR_TEAMS:raise ValueError('Unknown major roster')
    history=history if history is not None else MatchHistory()
    if not hasattr(history,'match_resources'):history.match_resources=MatchResources()
    recorder=PreplantMacroRecorder()
    raw=MacroAttackerController(candidates=candidates,macro_policy=policy,preplant_policy=preplant,
        guard_policy=guard,history=history,recorder=recorder)
    if macro_history is not None:raw.macro_history=macro_history
    with engine_directory(),redirect_stdout(io.StringIO()):
        from run_game import VisualFPSBattle
        from run_competition_manager import seed_all
        from simulation_runtime import cpu_inference
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        seed_all(seed)
        own,enemy=get_preset('Ghost Champions'),get_preset(MAJOR_TEAMS[code])
        with cpu_inference(),torch.no_grad():
            game=VisualFPSBattle(NEW_MAZE_STR,DualRoleTeamAI('GC three-output macro',lambda:raw,DefaultDefenderController),
                build_opponent('default'),headless=True,disable_side_swap=True,
                attacker_roster=list(own.players),defender_roster=list(enemy.players),
                spike_holder_name=own.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                attacker_igl_name=own.igl,defender_igl_name=enemy.igl)
            game.stop_after_round=True;distinguish_same_roster(game)
            history.match_resources.restore(game,code)
            frozen=[preplant,guard]+freeze_modules(game.defender_controller,excluded=(game,policy))
            before=module_digest(frozen)
            neural=getattr(policy,'student',policy)
            macro_before=module_digest([neural])
            with MacroRewards(game) as rewards:
                for tick in range(400):
                    game._simulate_tick()
                    terminal=bool(game.round_over or game.match_over)
                    parts=rewards.finish_tick(terminal,bool(game.attacker_wins))
                    recorder.reward(sum(parts.values()),gamma,terminal,components=parts)
                    if terminal:break
                else:raise RuntimeError('Macro round exceeded engine limit')
            if module_digest(frozen)!=before or module_digest([neural])!=macro_before:
                raise RuntimeError('A macro rollout changed model weights')
            game.defender_controller.verify()
            history.match_resources.capture(game,code)
            rows=recorder.advantages(gae_lambda)
            for row in rows:row['reward_version']=REWARD_VERSION
            instructions=Counter();destination_changes=plant_changes=0;previous=None
            for row in rows:
                for i,active in enumerate(row['active']):
                    if active:instructions[INSTRUCTIONS[row['selections']['instruction'][i]]]+=1
                if previous is not None:
                    destination_changes+=sum(bool(a) and row['selections']['destination'][i]!=previous['destination'][i]
                        for i,a in enumerate(row['active']))
                    plant_changes+=row['selections']['plant']!=previous['plant']
                previous=row['selections']
            stats=dict(phase='macro_v2',seed=seed,opponent=code,opponent_ai='default',decisions=len(rows),ticks=tick+1,
                success=bool(game.attacker_wins),actual_round_success=bool(game.attacker_wins),planted=bool(game.is_planted),
                plant=list(game.planted_pos) if game.planted_pos is not None else None,
                reason=game.analytics_tracker.round_records[-1]['reason'],
                reward=REWARD_VERSION,reward_component_sums=dict(rewards.totals),environment_reward=sum(rewards.totals.values()),
                defuse_ever_started=rewards.defuse_ever_started,instructions=dict(instructions),
                destination_changes=destination_changes,plant_target_changes=int(plant_changes),site_switches=raw.switches,
                decisions_postplant=raw.macro_decisions-len(rows),lower_weights_unchanged=True,opponent_weights_unchanged=True,
                macro_only_learning=True)
            history.finish_round(stats);raw.macro_history.sync_public_events()
            return rows,stats,raw.macro_history
