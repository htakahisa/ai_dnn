"""Recover the preserved BC macro with detached value fitting and guarded PPO."""
from pathlib import Path
import sys

MACRO_ROOT=Path(__file__).resolve().parent
AI_ROOT=MACRO_ROOT.parent
sys.path.insert(0,str(AI_ROOT.parent))
RUN_NAME='attacker_macro_three_outputs_v2'
INITIAL_CHECKPOINT=MACRO_ROOT/'data/attacker_macro_three_outputs_v1/best.pt' # Preserved BC actor, ST0.
BC_REPLAY_PATH=MACRO_ROOT/'data/attacker_macro_three_outputs_v1/bc_teacher' # Reuse; no collection.
PREPLANT_CHECKPOINT=MACRO_ROOT/'data/attacker_micro_directives_v5/best.pt'
POSTPLANT_CHECKPOINT=AI_ROOT/'data/attacker_postplant_fixed_v1/best.pt'
TRAINING_STEPS=20000 # Team macro PPO decisions, starting at 0; not micro ST.
RESUME=False
SEED=20261014
FIXED_OPPONENT_AI='default'
OPPONENT_CODES=('FNC','FRC','TYG','OMG','SPS','GRG','GC')
BC_WARMUP_STEPS=0 # The input already completed BC. No new collection or BC epochs.
BC_WARMUP_EPOCHS=8
BC_LEARNING_RATE=.0003
SIMPLE_RETREAT_AFTER_TICKS=4
SIMPLE_RETREAT_MIN_DISTANCE=6
SIMPLE_RETREAT_MAX_DISTANCE=10
SIMPLE_RETREAT_TICKS=20
SIMPLE_RETREAT_WAIT_TICKS=2
LEARNING_RATE=.00005 # Actor only.
VALUE_LEARNING_RATE=.001 # Separate critic only; no gradient into the actor.
VALUE_WARMUP_EPOCHS=10 # First rollout, fit value without changing the BC actor.
VALUE_EPOCHS=5
TARGET_KL=.005 # Stop actor updates at this joint KL.
MAX_KL=.01 # Reject updates exceeding the cap; also verify the entire batch.
KL_MONITOR_ROWS=128
BC_REPLAY_WEIGHT=.1 # Soft auxiliary imitation; PPO still samples learned commands.
BC_REPLAY_BATCH_SIZE=16
BATCH_SIZE=64
ROLLOUT_DECISIONS=512
PPO_EPOCHS=3
PPO_CLIP=.2
ENTROPY_WEIGHT=.01
GAMMA=.99
GAE_LAMBDA=.95
ROUNDS_PER_MATCH=7
EVALUATE_EVERY_STEPS=2000
EVALUATION_ROUNDS_PER_OPPONENT=2
TORCH_THREADS=1
RUN_PATH=MACRO_ROOT/'data'/RUN_NAME
LOG_PATH=MACRO_ROOT/'logs'/RUN_NAME

if __name__=='__main__':
    from ghost_champions_v3.macro_v2.gc3_training_attacker_macro import train
    train({k:v for k,v in globals().copy().items() if k.isupper()})
