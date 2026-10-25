"""Continue the completed micro weights with learned macro inputs; run without options."""
from pathlib import Path
import sys

MACRO_ROOT=Path(__file__).resolve().parent
AI_ROOT=MACRO_ROOT.parent
sys.path.insert(0,str(AI_ROOT.parent))
RUN_NAME='attacker_micro_directives_v5'
INITIAL_CHECKPOINT=AI_ROOT/'data/attacker_preplant_retrieval_v2/latest.pt'
DIRECTIVE_CHECKPOINT=MACRO_ROOT/'data/attacker_micro_directives_v4/latest.pt' # ST174991; retain learned advance, wait and retreat inputs.
POSTPLANT_CHECKPOINT=AI_ROOT/'data/attacker_postplant_fixed_v1/best.pt'
ADDITIONAL_TRAINING_STEPS=20000
RESUME=False
SEED=20261030
MACRO_DIRECTIVE_MODE='simple' # Fixed A/B plant plan and one short retreat per player.
SIMPLE_RETREAT_AFTER_TICKS=4 # Catch an actor near spawn before the team advances too far.
SIMPLE_RETREAT_MIN_DISTANCE=6 # Begin retreat only after leaving own spawn by this walking distance.
SIMPLE_RETREAT_MAX_DISTANCE=10 # Practice a return that can actually finish within its time budget.
SIMPLE_RETREAT_TICKS=20 # At most one player retreats at a time, once per round.
SIMPLE_RETREAT_WAIT_TICKS=2 # Briefly hold, then resume the same planting plan.
FIXED_OPPONENT_AI='default'
OPPONENT_CODES=('FNC','FRC','TYG','OMG','SPS','GRG','GC')
ORDER_DURATION_TICKS=40 # Used only if MACRO_DIRECTIVE_MODE='random'.
ORDER_WARMUP_TICKS=20 # First advance, then expose retreat away from spawn.
ORDER_PROBABILITIES=(.20,.50,.30) # WAIT, ADVANCE, RETREAT
ORDER_APPROACH_WEIGHT=4.0
ORDER_AWAY_PENALTY=.10 # Per walking-distance cell away from the assigned target.
ORDER_ARRIVAL_REWARD=.5 # Once per actor/order/target per round.
WAIT_HOLD_REWARD=.03
WAIT_MOVE_PENALTY=.10
INSTRUCTION_LOSS_WEIGHT=.5 # Training-only probability of all compliant legal choices.
INSTRUCTION_COMBAT_WEIGHT=.25 # Allow combat decisions to compete with orders.
MIN_INSTRUCTION_RATE=.75 # Readiness requires all three opportunity-adjusted rates.
LEARNING_RATE=.0003
BATCH_SIZE=128
ROLLOUT_DECISIONS=1024
PPO_EPOCHS=3
PPO_CLIP=.2
ENTROPY_WEIGHT=.01
BC_KL_WEIGHT=.02 # Retain the frozen base anchor without suppressing command adaptation.
GAMMA=.99
GAE_LAMBDA=.95
TORCH_THREADS=1
ROUNDS_PER_MATCH=7
EVALUATE_EVERY_STEPS=2000
EVALUATION_ROUNDS_PER_OPPONENT=2
RUN_PATH=MACRO_ROOT/'data'/RUN_NAME
LOG_PATH=MACRO_ROOT/'logs'/RUN_NAME

if __name__=='__main__':
    from ghost_champions_v3.macro_v2.gc3_training_attacker_directives import train
    train({key:value for key,value in globals().copy().items() if key.isupper()})
