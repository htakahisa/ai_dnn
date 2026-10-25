"""Run explicitly from ghost_champions_v3/macro_v1, normally without options."""
from pathlib import Path
import sys

MACRO_ROOT = Path(__file__).resolve().parent
AI_ROOT = MACRO_ROOT.parent
sys.path.insert(0, str(AI_ROOT.parent))
RUN_NAME = "attacker_macro_v1"
PREPLANT_CHECKPOINT = AI_ROOT/"data/attacker_preplant_omoko_rl_handoff_v3/best.pt"
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
TRAINING_STEPS = 20000  # Macro decisions, starting at ST 0, not lower-model ST.
RESUME = False
SEED = 20261022
FIXED_OPPONENT_AI = "default"
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
EVALUATE_EVERY_STEPS = 1000
EVALUATION_ROUNDS_PER_OPPONENT = 4
LEARNING_RATE = .0001
BATCH_SIZE = 128
ROLLOUT_DECISIONS = 1024
PPO_EPOCHS = 3
GAMMA = .99
GAE_LAMBDA = .95
PPO_CLIP = .2
ENTROPY_WEIGHT = .01
ROUNDS_PER_MATCH = 7
TORCH_THREADS = 1
RUN_PATH = MACRO_ROOT/"data"/RUN_NAME
LOG_PATH = MACRO_ROOT/"logs"/RUN_NAME

if __name__ == "__main__":
    from ghost_champions_v3.macro_v1.gc3_training_attacker_macro import train
    train({k:v for k,v in globals().copy().items() if k.isupper()})
