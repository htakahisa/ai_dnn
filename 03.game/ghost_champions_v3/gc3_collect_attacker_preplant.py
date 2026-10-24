"""Collect ConCon v1 actions only. Run from ghost_champions_v3 without options."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_ROOT.parent))

DATASET_NAME = "attacker_preplant_concon_history_v1"
COLLECTION_ROUNDS = 280
TEACHER_MAPS = ("A1", "A2", "A3", "A4")
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
FIXED_OPPONENT_AI = "default"
SEED = 2026101300
ROUNDS_PER_MATCH = 7
VALIDATION_EVERY_MATCHES = 5
MAX_SKIPPED_FRACTION = 0.05
TORCH_THREADS = 1
DATASET_PATH = AI_ROOT/"data"/DATASET_NAME
LOG_PATH = AI_ROOT/"logs"/DATASET_NAME

if __name__ == "__main__":
    from ghost_champions_v3.preplant.gc3_data_attacker_preplant import collect
    collect({key:value for key, value in globals().copy().items() if key.isupper()})
