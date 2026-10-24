"""Explicit postplant-only continuation, from actual plants by the chosen latest macro."""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parent
AI_ROOT=GUARD_ROOT.parent
sys.path.insert(0,str(AI_ROOT.parent))
RUN_NAME='attacker_postplant_antidefuse_v1'
INITIAL_CHECKPOINT=AI_ROOT/'data/attacker_postplant_fixed_v1/best.pt'
ENTRY_MACRO_CHECKPOINT=AI_ROOT/'macro_v2/data/attacker_macro_three_outputs_v2/latest.pt'
ENTRY_MICRO_CHECKPOINT=AI_ROOT/'macro_v2/data/attacker_micro_directives_v5/best.pt'
TRAINING_STEPS=20000 # Additional guard decisions, one actor action; excludes entry/macro decisions.
RESUME=True # Continue from this run's latest.pt, including optimizer and RNG state.
SEED=20261014
FIXED_OPPONENT_AI='default'
OPPONENT_CODES=('FNC','FRC','TYG','OMG','SPS','GRG','GC')
ENTRY_MAX_ATTEMPTS=24
ROUNDS_PER_MATCH=7
EVALUATE_EVERY_STEPS=2000
EVALUATION_ROUNDS_PER_OPPONENT=4
LEARNING_RATE=.00005
VALUE_LEARNING_RATE=.001
VALUE_WARMUP_EPOCHS=10
VALUE_EPOCHS=5
BATCH_SIZE=64
ROLLOUT_DECISIONS=512
PPO_EPOCHS=3
PPO_CLIP=.2
ENTROPY_WEIGHT=.01
TARGET_KL=.005
MAX_KL=.01
KL_MONITOR_ROWS=128
GAMMA=.99
GAE_LAMBDA=.95
TORCH_THREADS=1
RUN_PATH=GUARD_ROOT/'data'/RUN_NAME
LOG_PATH=GUARD_ROOT/'logs'/RUN_NAME

if __name__=='__main__':
    from ghost_champions_v3.postplant_v2.gc3_training_attacker_guard import train
    train({name:value for name,value in globals().copy().items() if name.isupper()})
