"""Fit one fresh pre-plant attacker policy to ConCon demonstrations."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_ROOT.parent))

RUN_NAME = "attacker_preplant_bc_history_v2"
DATASET_NAME = "attacker_preplant_concon_history_v1"
# Reuse the six committed sets of additional labels; do not collect again.
REUSE_DAGGER_RUN_PATH = AI_ROOT/"data/attacker_preplant_bc_history_v1"
EPOCHS = 10
RESUME = False
SEED = 20261013
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
TORCH_THREADS = 1
FIXED_OPPONENT_AI = "default"
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
EVALUATION_ROUNDS_PER_OPPONENT = 4
MIN_BC_PLANT_RATE = 0.50  # Readiness check; does not constrain in-game actions.
ENABLE_DAGGER = False
DAGGER_ROUNDS_PER_EPOCH = 7
DAGGER_TEACHER_MAPS = ("A1", "A2", "A3", "A4")
DATASET_PATH = AI_ROOT/"data"/DATASET_NAME
RUN_PATH = AI_ROOT/"data"/RUN_NAME
LOG_PATH = AI_ROOT/"logs"/RUN_NAME

if __name__ == "__main__":
    from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import train_bc
    train_bc({key:value for key, value in globals().copy().items() if key.isupper()})
