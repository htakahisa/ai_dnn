"""Restart attacker micro PPO from the existing BC, with repaired objective rewards."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT.parent))

# Edit these named settings; normally run from ghost_champions_v3 without options.
RUN_NAME = "attacker_preplant_objectives_v1"
BC_CHECKPOINT = AI_ROOT/"data/attacker_preplant_omoko_bc_site_v1/best.pt"
INITIAL_CHECKPOINT = None  # First run used BC; RESUME=True loads this run's latest.pt.
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
SITE_GOAL_POLICY = True
NAVIGATION_FEATURES = True
TRAINING_STEPS = 100000  # Cumulative target, including the completed ST 22,547.
RESUME = True
SEED = 20261024
FIXED_OPPONENT_AI = "default"
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
EVALUATE_EVERY_STEPS = 1000
EVALUATION_ROUNDS_PER_OPPONENT = 4
MIN_BC_PLANT_RATE = .01
MIN_BC_ORB_COLLECTIONS = 1
LEARNING_RATE = .0001
BATCH_SIZE = 128
ROLLOUT_DECISIONS = 1024
PPO_EPOCHS = 3
GAMMA = .99
GAE_LAMBDA = .95
PPO_CLIP = .2
ENTROPY_WEIGHT = .01
BC_KL_WEIGHT = .10
ROUNDS_PER_MATCH = 7
TORCH_THREADS = 1
RUN_PATH = AI_ROOT/"data"/RUN_NAME
LOG_PATH = AI_ROOT/"logs"/RUN_NAME

if __name__ == "__main__":
    from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import train_rl
    train_rl({key:value for key,value in globals().copy().items() if key.isupper()})
