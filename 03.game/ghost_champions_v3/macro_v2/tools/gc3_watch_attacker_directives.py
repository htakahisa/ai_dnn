"""Watch saved directive micro weights with their simple test macro; no learning."""
from functools import partial
from pathlib import Path
import argparse
import json
import sys

MACRO_ROOT=Path(__file__).resolve().parents[1]
AI_ROOT=MACRO_ROOT.parent
sys.path.insert(0,str(AI_ROOT.parent))

# Normal settings: edit here, then run without options from macro_v2.
RUN_NAME='attacker_micro_directives_v5'
CHECKPOINT_NAME='best.pt'
POSTPLANT_CHECKPOINT=AI_ROOT/'data/attacker_postplant_fixed_v1/best.pt'
OPPONENT='FNC'
FIXED_OPPONENT_AI='default'
SEED=20361030 # First evaluation seed: A. Next rounds alternate A/B.
TICK_MS=150
START_PAUSED=True
TORCH_THREADS=1
MODEL_LABEL='v5 best'
MACRO_CHECKPOINT=None # Set only by the learned-macro viewer; None keeps the simple test logic.


def prepare_round(model,opponent,seed,*,headless=True,tick_ms=TICK_MS):
    import numpy as np
    from ghost_champions_v3.gc3_rollout import engine_directory,freeze_modules
    from ghost_champions_v3.gc3_policy import FreshPolicy
    from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
    from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS,distinguish_same_roster
    from ghost_champions_v3.macro_v2.gc3_candidates_attacker_macro import MacroCandidates
    from ghost_champions_v3.macro_v2.gc3_curriculum_attacker_directives import directive_policy
    from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
    from ghost_champions_v3.macro_v2.gc3_learn_attacker_macro import MacroAttackerController
    if model!=MODEL_LABEL or opponent not in MAJOR_TEAMS or FIXED_OPPONENT_AI!='default':
        raise ValueError('Select the saved directive model and a major roster with fixed opponent logic')
    path=MACRO_ROOT/'data'/RUN_NAME
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    config=manifest['config']
    if config.get('MACRO_DIRECTIVE_MODE')!='simple':raise ValueError('This viewer requires a simple-macro run')
    checkpoint=path/CHECKPOINT_NAME
    policy,metadata=DirectedMicroPolicy.load(checkpoint)
    if not metadata.get('trained') or metadata.get('stage')!='directive_rl' or not policy.directives_trained:
        raise ValueError('Select trained directive micro weights')
    guard,guard_metadata=FreshPolicy.load(POSTPLANT_CHECKPOINT,role='attacker')
    if not guard_metadata.get('trained'):raise ValueError('Select the trained postplant guard')
    if manifest['input_hashes']['POSTPLANT_CHECKPOINT']!=file_hash(POSTPLANT_CHECKPOINT):
        raise ValueError('The postplant guard differs from the one used in training')
    with engine_directory():
        from run_game import _build_team_ai
        from run_competition_manager import seed_all
        from party_presets import get_preset
        from map_data import NEW_MAZE_STR
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController
        from ghost_champions_v3.tools import gc3_watch_attacker_preplant as viewer
        from ghost_champions_v3.tools.gc3_watch_attacker_postplant import PostplantBattle
        arena=np.array([[int(v) for v in row] for row in NEW_MAZE_STR.strip().splitlines()])
        candidates=MacroCandidates.configured(arena)
        if manifest['candidate_signature']!=candidates.signature:raise ValueError('The candidate map differs from this run')

        class DirectiveBattle(viewer.AttackerBattle):
            def draw(self):
                PostplantBattle.draw(self)
                if not hasattr(self,'viewer_status'):return
                ctrl=getattr(self.attacker_controller,'inner_controller',self.attacker_controller)
                phase='初期配置' if self.in_defender_setup_phase else '設置後（既存ガード）' if self.is_planted else '設置前'
                goal=getattr(ctrl,'current_goal',None)
                summary=f"{self.viewer_description} | {phase} tick={self.battle_tick}"
                if goal is not None:summary+=f" | 目的サイト={('A','B')[goal]}"
                if not self.is_planted:
                    from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import INSTRUCTIONS,INSTRUCTION_LABELS
                    orders=[]
                    for name,order in getattr(ctrl,'player_orders',{}).items():
                        target=order['retreat_target'] if order['instruction']=='RETREAT' else order['destination']
                        orders.append(f"{name}: {INSTRUCTION_LABELS[INSTRUCTIONS.index(order['instruction'])]} {tuple(target)}")
                    if orders:summary+='\n'+' / '.join(orders)
                if self.round_over:
                    reason=self.analytics_tracker.round_records[-1].get('reason','unknown')
                    summary+=f" | {'攻撃勝利' if self.attacker_wins else '防衛勝利'} ({reason})"
                self.viewer_status.set(summary)

        seed_all(seed)
        own,enemy=get_preset('Ghost Champions'),get_preset(MAJOR_TEAMS[opponent])
        logic=directive_policy(config,seed) if MACRO_CHECKPOINT is None else None
        factory=partial(MacroAttackerController,candidates=candidates,macro_policy=logic,macro_checkpoint=MACRO_CHECKPOINT,
            preplant_policy=policy,preplant_checkpoint=checkpoint,guard_policy=guard,guard_checkpoint=POSTPLANT_CHECKPOINT)
        game=DirectiveBattle(NEW_MAZE_STR,DualRoleTeamAI('GC directive viewer',factory,DefaultDefenderController),
            _build_team_ai(FIXED_OPPONENT_AI,device='cpu'),headless=True,disable_side_swap=True,tick_time_ms=tick_ms,
            attacker_roster=list(own.players),defender_roster=list(enemy.players),
            spike_holder_name=own.spike_holder,defender_spike_holder_name=enemy.spike_holder,
            attacker_igl_name=own.igl,defender_igl_name=enemy.igl)
        game.stop_after_round=True
        distinguish_same_roster(game)
        freeze_modules(game.attacker_controller,excluded=(game,))
        freeze_modules(game.defender_controller,excluded=(game,))
        game.replay_frames.clear();game._record_replay_frame()
        if not headless:game.attach_window()
        return game,dict(model=MODEL_LABEL,preplant_step=metadata['step'],preplant_stage=metadata['stage'],
            preplant_checkpoint=str(checkpoint),guard_checkpoint=str(POSTPLANT_CHECKPOINT),guard_step=guard_metadata.get('step'),
            macro_step=getattr(game.attacker_controller,'macro_step',None),
            macro_directive_mode='simple' if MACRO_CHECKPOINT is None else 'learned',opponent=opponent,opponent_ai=FIXED_OPPONENT_AI,seed=seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true',help='One frozen headless round for validation')
    args=parser.parse_args()
    import torch
    from ghost_champions_v3.gc3_rollout import engine_directory,module_digest
    from simulation_runtime import cpu_inference
    torch.set_num_threads(TORCH_THREADS)
    with engine_directory(),cpu_inference(),torch.no_grad():
        if args.verify_only:
            game,info=prepare_round(MODEL_LABEL,OPPONENT,SEED)
            ctrl=game.attacker_controller
            before=module_digest([ctrl.lower.preplant.policy,ctrl.lower.postplant.policy])
            game.run_headless_loop()
            if before!=module_digest([ctrl.lower.preplant.policy,ctrl.lower.postplant.policy]):
                raise RuntimeError('Viewing changed model weights')
            print(dict(info,planted=game.is_planted,success=bool(game.attacker_wins),
                result=game.analytics_tracker.round_records[-1]['reason'],weights_unchanged=True),flush=True)
            return
        from ghost_champions_v3.tools import gc3_watch_attacker_preplant as viewer
        viewer.DEFAULT_MODEL=MODEL_LABEL;viewer.MODEL_PATHS={MODEL_LABEL:MACRO_ROOT/'data'/RUN_NAME/CHECKPOINT_NAME}
        viewer.OPPONENT=OPPONENT;viewer.TICK_MS=TICK_MS;viewer.START_PAUSED=START_PAUSED
        viewer.prepare_round=prepare_round

        class DirectiveViewer(viewer.AttackerViewer):
            def __init__(self):
                super().__init__(seed=SEED)
                self.status_label.configure(wraplength=1400,justify='left')

            def next_round(self):
                if self.game is None:seed=self.seed
                else:
                    seed=self.seeds.getrandbits(32)
                    if MACRO_CHECKPOINT is None:seed=(seed&~1)|(1-(self.seed%2))
                self._mount(seed)

        DirectiveViewer().run()


if __name__=='__main__':main()
