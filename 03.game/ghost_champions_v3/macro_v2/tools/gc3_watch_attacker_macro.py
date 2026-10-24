"""Watch the learned three-output macro with the frozen v5 best micro."""
from pathlib import Path
import sys

MACRO_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_macro_three_outputs_v2'
CHECKPOINT_NAME='latest.pt' # Use best.pt here to compare the saved reward-best.
MICRO_RUN_NAME='attacker_micro_directives_v5'
MICRO_CHECKPOINT_NAME='best.pt'
OPPONENT='FNC'
SEED=20361030
TICK_MS=150
START_PAUSED=True
sys.path.insert(0,str(MACRO_ROOT.parents[1]))


def main():
    checkpoint=MACRO_ROOT/'data'/RUN_NAME/CHECKPOINT_NAME
    if not checkpoint.is_file():
        timing='最初の固定評価が終わってから' if CHECKPOINT_NAME=='best.pt' else '選択したチェックポイントが保存されてから'
        raise SystemExit(f'{RUN_NAME} の {CHECKPOINT_NAME} はまだ保存されていません。\n'
            f'学習開始後、{timing}描画してください。\n'
            f'読み込み先: {checkpoint}')
    from ghost_champions_v3.macro_v2.tools import gc3_watch_attacker_directives as viewer
    viewer.RUN_NAME=MICRO_RUN_NAME;viewer.CHECKPOINT_NAME=MICRO_CHECKPOINT_NAME
    viewer.MACRO_CHECKPOINT=checkpoint
    viewer.MODEL_LABEL=f'三出力マクロ {Path(CHECKPOINT_NAME).stem}';viewer.OPPONENT=OPPONENT;viewer.SEED=SEED
    viewer.TICK_MS=TICK_MS;viewer.START_PAUSED=START_PAUSED
    viewer.main()


if __name__=='__main__':main()
