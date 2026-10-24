"""Train GC v3 from random weights. Normal execution: python gc3_train_attacker_guard.py."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3 import ATTACKER_SEED

RUN_NAME = "attacker_postplant_fixed_v1"
PHASE = "postplant"
SEED = ATTACKER_SEED
TRAINING_STEPS = 80066
EVALUATE_EVERY_STEPS = 5000
EVALUATION_ROUNDS_PER_OPPONENT = 4
OPPONENT_CODES = ("TYG", "OMG", "FRC", "FNC", "GG", "SPS")
# Frozen attackers only generate actual plant-entry states; no imitation/weight transfer.
POSTPLANT_ENTRY_ATTACKER_CODES = ("TYG", "OMG", "FRC", "FNC", "SPS")
POSTPLANT_ENTRY_MAX_ATTEMPTS = 24
RESUME = True
DATA_ROOT = AI_ROOT/"data"
LOG_ROOT = AI_ROOT/"logs"
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
ROLLOUT_DECISIONS = 1024
PPO_EPOCHS = 3
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ENTROPY_WEIGHT = 0.01
TORCH_THREADS = 1

if __name__ == "__main__":
    from ghost_champions_v3.gc3_training import main
    names = ("RUN_NAME", "PHASE", "SEED", "TRAINING_STEPS", "EVALUATE_EVERY_STEPS",
             "EVALUATION_ROUNDS_PER_OPPONENT", "OPPONENT_CODES", "RESUME", "DATA_ROOT",
             "LOG_ROOT", "LEARNING_RATE", "BATCH_SIZE", "ROLLOUT_DECISIONS", "PPO_EPOCHS",
             "GAMMA", "GAE_LAMBDA", "PPO_CLIP", "ENTROPY_WEIGHT", "TORCH_THREADS",
             "POSTPLANT_ENTRY_ATTACKER_CODES", "POSTPLANT_ENTRY_MAX_ATTEMPTS")
    main({name: globals()[name] for name in names})
