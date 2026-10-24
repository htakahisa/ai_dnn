"""Browse actual postplants with the fixed latest macro/micro and the new guard."""
from pathlib import Path
import argparse
import sys

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_postplant_antidefuse_v1'
CHECKPOINT_NAME='latest.pt'
OPPONENT='FNC'
SEED=20361030
TICK_MS=150
START_PAUSED=True
SETTINGS_MODULE=None
sys.path.insert(0,str(GUARD_ROOT.parents[1]))


def main():
    from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard as settings
    settings=SETTINGS_MODULE or settings
    from ghost_champions_v3.postplant_v2.gc3_policy_attacker_guard import GuardPolicy
    from ghost_champions_v3.postplant_v2.gc3_rollout_attacker_guard import load_entry,prepare_entry,handoff
    from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
    from ghost_champions_v3.gc3_rollout import engine_directory,freeze_modules
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    config={name:value for name,value in vars(settings).items() if name.isupper()}
    checkpoint=GUARD_ROOT/'data'/RUN_NAME/CHECKPOINT_NAME
    if not checkpoint.exists():raise SystemExit(f'描画するチェックポイントはまだありません: {checkpoint}')
    policy,saved=GuardPolicy.load(checkpoint);entry=load_entry(config)
    required_inputs={key:file_hash(config[key]) for key in ('INITIAL_CHECKPOINT','ENTRY_MACRO_CHECKPOINT','ENTRY_MICRO_CHECKPOINT')}
    if config.get('START_CHECKPOINT') is not None:required_inputs['START_CHECKPOINT']=file_hash(config['START_CHECKPOINT'])
    if not saved.get('trained') or saved.get('inputs')!=required_inputs:
        raise ValueError('Guard viewer requires the original fixed entry checkpoints')
    policy.requires_grad_(False).eval()
    from ghost_champions_v3.tools import gc3_watch_attacker_preplant as viewer
    from ghost_champions_v3.tools.gc3_watch_attacker_postplant import PostplantBattle
    label=f'解除阻止ガード {Path(CHECKPOINT_NAME).stem}'
    class GuardBattle(PostplantBattle):
        def draw(self):
            super().draw()
            if hasattr(self,'viewer_status'):
                progress=max((c.defuse_timer for c in self.chars if c.team=='D' and c.is_alive),default=0)
                text=f'{label} ST={saved["step"]:,} | マクロlatest ST={entry["macro_step"]:,} | 爆発まで{self.detonate_timer} / 解除{progress}/6'
                if self.round_over:text+=f' | {self.analytics_tracker.round_records[-1]["reason"]}'
                self.viewer_status.set(text)
    def prepare(model,opponent,seed,*,tick_ms=TICK_MS):
        with engine_directory(),torch.no_grad():
            game,raw,start=prepare_entry(opponent,seed,entry,config,battle_class=GuardBattle)
            handoff(game,raw,policy,checkpoint=checkpoint);freeze_modules(game.attacker_controller,excluded=(game,))
            game.tick_time_ms=tick_ms;game.replay_frames.clear();game._record_replay_frame()
        return game,dict(model=label,preplant_step=saved['step'],opponent=opponent,opponent_ai='default',seed=start['seed'])
    if args.verify_only:
        game,info=prepare(label,OPPONENT,SEED)
        with engine_directory(),torch.no_grad():game.run_headless_loop()
        print(dict(info,success=bool(game.attacker_wins),reason=game.analytics_tracker.round_records[-1]['reason']));return
    viewer.DEFAULT_MODEL=label;viewer.MODEL_PATHS={label:checkpoint};viewer.OPPONENT=OPPONENT
    viewer.TICK_MS=TICK_MS;viewer.START_PAUSED=START_PAUSED;viewer.prepare_round=prepare
    viewer.AttackerViewer(seed=SEED).run()


if __name__=='__main__':main()
