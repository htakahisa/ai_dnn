"""Actual plants by the frozen chosen macro/micro; only the new guard learns."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import numpy as np
import torch
from ..gc3_rollout import Recorder,engine_directory,module_digest,phase_start_snapshot,NoPostplantEntry
from ..gc3_policy import FreshPolicy
from ..preplant.gc3_data_attacker_preplant import file_hash
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_state_attacker_resources import MatchResources
from ..preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS,distinguish_same_roster
from ..macro_v1.gc3_opponents_attacker_macro import build_opponent
from ..macro_v2.gc3_policy_attacker_macro import MacroPolicy
from ..macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
from ..macro_v2.gc3_learn_attacker_macro import MacroAttackerController
from ..macro_v2.gc3_candidates_attacker_macro import MacroCandidates
from .gc3_learn_attacker_guard import GuardController
from .gc3_reward_attacker_guard import GuardRewards,REWARD_VERSION


def load_entry(config):
    macro,metadata=MacroPolicy.load(config['ENTRY_MACRO_CHECKPOINT'])
    micro,micro_saved=DirectedMicroPolicy.load(config['ENTRY_MICRO_CHECKPOINT'])
    guard,guard_saved=FreshPolicy.load(config['INITIAL_CHECKPOINT'],role='attacker')
    if not all(p.get('trained') for p in (metadata,micro_saved,guard_saved)) or not micro.directives_trained:
        raise ValueError('Entry requires trained macro/micro and the current guard')
    hashes=dict(PREPLANT_CHECKPOINT=file_hash(config['ENTRY_MICRO_CHECKPOINT']),POSTPLANT_CHECKPOINT=file_hash(config['INITIAL_CHECKPOINT']))
    if metadata['lower_checkpoint_hashes']!=hashes:raise ValueError('Chosen macro needs its original entry micro/guard')
    from map_data import NEW_MAZE_STR
    arena=np.array([[int(v) for v in row] for row in NEW_MAZE_STR.strip().splitlines()])
    candidates=MacroCandidates.configured(arena)
    if candidates.signature!=metadata['candidate_signature']:raise ValueError('Entry macro map changed')
    for model in (macro,micro,guard):model.requires_grad_(False).eval()
    return dict(macro=macro,micro=micro,guard=guard,candidates=candidates,macro_step=metadata['step'],micro_step=micro_saved['step'])


def prepare_entry(code,seed,entry,config,history=None,view=None,battle_class=None):
    history=history if history is not None else MatchHistory()
    if not hasattr(history,'match_resources'):history.match_resources=MatchResources()
    with engine_directory(),redirect_stdout(io.StringIO()):
        from run_game import VisualFPSBattle
        from run_competition_manager import seed_all
        from simulation_runtime import cpu_inference
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from map_data import NEW_MAZE_STR
        own,enemy=get_preset('Ghost Champions'),get_preset(MAJOR_TEAMS[code])
        for attempt in range(config['ENTRY_MAX_ATTEMPTS']):
            actual_seed=seed+1000000*attempt;seed_all(actual_seed)
            raw=MacroAttackerController(macro_policy=entry['macro'],preplant_policy=entry['micro'],guard_policy=entry['guard'],
                candidates=entry['candidates'],history=history)
            if view is not None:raw.macro_history=view
            with cpu_inference(),torch.no_grad():
                game=(battle_class or VisualFPSBattle)(NEW_MAZE_STR,
                    DualRoleTeamAI('GC frozen latest macro entry',lambda:raw,DefaultDefenderController),build_opponent('default'),
                    headless=True,disable_side_swap=True,attacker_roster=list(own.players),defender_roster=list(enemy.players),
                    spike_holder_name=own.spike_holder,defender_spike_holder_name=enemy.spike_holder,
                    attacker_igl_name=own.igl,defender_igl_name=enemy.igl)
                game.stop_after_round=True;distinguish_same_roster(game);history.match_resources.restore(game,code)
                for _ in range(400):
                    if game.is_planted or game.round_over:break
                    game._simulate_tick()
                if game.is_planted and not game.round_over and any(c.team=='A' and c.is_alive for c in game.chars):
                    return game,raw,dict(seed=actual_seed,attempts=attempt+1,state=phase_start_snapshot(game))
                if not game.round_over:
                    for _ in range(400):
                        game._simulate_tick()
                        if game.round_over:break
                if not game.round_over:raise RuntimeError('Frozen entry exceeded round limit')
                game.defender_controller.verify();history.match_resources.capture(game,code)
                history.finish_round(dict(success=bool(game.attacker_wins),planted=bool(game.is_planted),reason='skipped_guard_entry'))
                raw.macro_history.sync_public_events();view=raw.macro_history
        raise NoPostplantEntry(f'{code}, seed {seed}: no live plant from frozen latest macro in {config["ENTRY_MAX_ATTEMPTS"]} attempts')


def handoff(game,raw,policy,recorder=None,checkpoint=None):
    before=phase_start_snapshot(game)
    policy.reset_memory();raw.lower.postplant=GuardController(policy=policy,recorder=recorder,checkpoint=checkpoint)
    raw.lower.postplant.set_game(raw.lower.preplant.game)
    if phase_start_snapshot(game)!=before:raise RuntimeError('Guard handoff changed actual state')


def play_guard(code,seed,*,policy,entry,config,history=None,view=None):
    frozen_before=module_digest([entry['macro'],entry['micro'],entry['guard']])
    game,raw,start=prepare_entry(code,seed,entry,config,history=history,view=view)
    recorder=Recorder();handoff(game,raw,policy,recorder)
    with engine_directory(),redirect_stdout(io.StringIO()):
        from simulation_runtime import cpu_inference
        with cpu_inference(),torch.no_grad(),GuardRewards(game,config['GAMMA']) as rewards:
            for tick in range(400):
                rewards.begin_tick();game._simulate_tick();terminal=bool(game.round_over)
                shared,personal=rewards.finish_tick(terminal,bool(game.attacker_wins))
                recorder.reward(sum(shared.values()),config['GAMMA'],terminal,components=shared,individual_components=personal)
                if terminal:break
            else:raise RuntimeError('Guard round exceeded limit')
            outcome=game.analytics_tracker.round_records[-1]
            stats=dict(phase='postplant',opponent=code,opponent_ai='default',seed=seed,success=bool(game.attacker_wins),planted=True,
                plant=list(game.planted_pos),reason=outcome.get('reason'),ticks=tick+1,decisions=len(recorder.rows),
                phase_start=start,entry_macro_step=entry['macro_step'],entry_micro_step=entry['micro_step'],
                defuse_started=rewards.starts>0,defuse_starts=rewards.starts,defuse_active_ticks=rewards.tap_ticks,
                defused=bool(game.is_defused),reward=REWARD_VERSION,entry_weights_unchanged=True,opponent_weights_unchanged=True)
    if module_digest([entry['macro'],entry['micro'],entry['guard']])!=frozen_before:raise RuntimeError('Guard learning changed frozen entry')
    game.defender_controller.verify();raw.history.match_resources.capture(game,code)
    raw.history.finish_round(stats);raw.macro_history.sync_public_events()
    rows=recorder.advantages(config['GAE_LAMBDA'])
    for row in rows:row['reward_version']=REWARD_VERSION
    stats['reward_component_sums']={key:sum(row['reward_components'].get(key,0.) for row in rows)
        for key in sorted({k for row in rows for k in row['reward_components']})}
    return rows,stats,raw.macro_history
