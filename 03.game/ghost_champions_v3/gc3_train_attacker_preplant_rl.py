"""Continue the pre-plant BC actor with PPO against a fixed rule AI."""

from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_ROOT.parent))

RUN_NAME = "attacker_preplant_rl_history_v2"
BC_CHECKPOINT = AI_ROOT / "data/attacker_preplant_bc_history_v2/best.pt"
TRAINING_STEPS = 30000  # Cumulative policy decisions, including the saved 11,998 steps.
RESUME = True
SEED = 20261014
FIXED_OPPONENT_AI = "default"
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
EVALUATE_EVERY_STEPS = 1000
EVALUATION_ROUNDS_PER_OPPONENT = 4
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
ROLLOUT_DECISIONS = 1024
PPO_EPOCHS = 3
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ENTROPY_WEIGHT = 0.01
BC_KL_WEIGHT = 0.10  # Penalize erasing BC distributions, without forcing actions.
# With the default 28-round check, require at least one actual plant.
# The former 50% threshold was a discretionary gate, not a PPO prerequisite.
MIN_BC_PLANT_RATE = 0.01
ROUNDS_PER_MATCH = 7
TORCH_THREADS = 1
RUN_PATH = AI_ROOT / "data" / RUN_NAME
LOG_PATH = AI_ROOT / "logs" / RUN_NAME

if __name__ == "__main__":
    from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import train_rl

    train_rl({key: value for key, value in globals().copy().items() if key.isupper()})
