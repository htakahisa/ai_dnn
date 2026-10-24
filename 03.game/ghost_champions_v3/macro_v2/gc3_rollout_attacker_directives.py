"""Micro-only instructional PPO episodes; guard/opponent finish real frozen rounds."""
from contextlib import redirect_stdout
import io
from collections import Counter
import torch
from ..gc3_rollout import Recorder,engine_directory,freeze_modules,module_digest
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS,distinguish_same_roster,credit_round_outcome
from ..preplant.gc3_state_attacker_resources import MatchResources
from .gc3_learn_attacker_macro import MacroAttackerController
from .gc3_curriculum_attacker_directives import directive_policy
from .gc3_reward_attacker_directives import DirectiveRewards,REWARD_VERSION


def play_directives(code,seed,*,policy,guard,candidates,config,history=None):
    if code not in MAJOR_TEAMS or config['FIXED_OPPONENT_AI']!='default':raise ValueError('Use fixed logic with a major roster')
    history=history if history is not None else MatchHistory()
    if not hasattr(history,'match_resources'):history.match_resources=MatchResources()
    recorder=Recorder()
    sampler=directive_policy(config,seed)
    raw=MacroAttackerController(candidates=candidates,macro_policy=sampler,preplant_policy=policy,guard_policy=guard,
        history=history)
    raw.lower.preplant.recorder=recorder
    # The controller freezes lower models by default. Only the new input layer
    # is trainable here; inference/rollouts themselves still never backpropagate.
    for parameter in policy.directive.parameters():parameter.requires_grad_(True)
    with engine_directory(),redirect_stdout(io.StringIO()):
        from run_game import VisualFPSBattle,_build_team_ai
        from run_competition_manager import seed_all
        from simulation_runtime import cpu_inference
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        seed_all(seed)
        own,enemy=get_preset('Ghost Champions'),get_preset(MAJOR_TEAMS[code])
        with cpu_inference(),torch.no_grad():
            game=VisualFPSBattle(NEW_MAZE_STR,DualRoleTeamAI('GC instruction training',lambda:raw,DefaultDefenderController),
                _build_team_ai('default',device='cpu'),headless=True,disable_side_swap=True,
                attacker_roster=list(own.players),defender_roster=list(enemy.players),
                spike_holder_name=own.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                attacker_igl_name=own.igl,defender_igl_name=enemy.igl)
            game.stop_after_round=True
            distinguish_same_roster(game)
            history.match_resources.restore(game,code)
            frozen=[guard]+freeze_modules(game.defender_controller,excluded=(game,policy))
            frozen_before=module_digest(frozen);micro_before=module_digest([policy])
            with DirectiveRewards(game,config['GAMMA'],raw,recorder,config) as reward:
                for tick in range(400):
                    reward.begin_tick();game._simulate_tick()
                    terminal=bool(game.is_planted or game.round_over)
                    shared,personal=reward.finish_tick(terminal,bool(game.attacker_wins))
                    deferred=terminal and game.is_planted and not game.round_over
                    recorder.reward(sum(shared.values()),config['GAMMA'],terminal and not deferred,
                        components=shared,individual_components=personal)
                    if terminal:break
                else:raise RuntimeError('Instruction preplant episode exceeded the engine limit')
            post_ticks=0
            while not game.round_over:
                game._simulate_tick();post_ticks+=1
                if post_ticks>=400:raise RuntimeError('Frozen guard did not finish the round')
            credit_round_outcome(recorder,post_ticks,config['GAMMA'],bool(game.attacker_wins))
            history.match_resources.capture(game,code)
            if module_digest(frozen)!=frozen_before or module_digest([policy])!=micro_before:
                raise RuntimeError('A rollout updated model weights')
            rows=recorder.advantages(config['GAE_LAMBDA'])
            for row in rows:row['reward_version']=REWARD_VERSION
            counts=Counter(row['macro_directive']['instruction'] for row in rows)
            stats=dict(phase='directed_preplant',opponent=code,opponent_ai='default',seed=seed,
                decisions=len(rows),ticks=tick+1,success=bool(game.attacker_wins),actual_round_success=bool(game.attacker_wins),
                planted=bool(game.is_planted),target_site=('A','B')[raw.current_goal],
                planned_plant=list(raw.current_plan['plant']),
                plant=list(game.planted_pos) if game.planted_pos is not None else None,
                planned_plant_distance=reward.distance(raw.current_plan['plant'],game.planted_pos) if game.planted_pos is not None else None,
                actual_round_reason=game.analytics_tracker.round_records[-1]['reason'],
                orb_collections=reward.completed_orbs,spike_pickups=reward.objectives.pickups,
                plant_completions=reward.objectives.plants,plant_cancellations=reward.objectives.cancellations,
                instructions=dict(counts),reward=REWARD_VERSION,
                instruction_measures=dict(reward.measures),
                reward_component_sums={key:sum(r['reward_components'].get(key,0.) for r in rows)
                    for key in sorted({key for row in rows for key in row['reward_components']})},
                guard_weights_unchanged=True,opponent_weights_unchanged=True,macro_learning=False,
                training_scope='new directive input layer only',postplant_ticks=post_ticks)
            stats['macro_directive_mode']=config.get('MACRO_DIRECTIVE_MODE','random')
            stats['retreat_practice_events']=getattr(sampler,'retreat_events',[])
            history.finish_round(stats)
            return rows,stats
