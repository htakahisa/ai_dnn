"""Collect frozen Omoko Gaming demonstrations on the real GC roster."""
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_ROOT.parent))

DATASET_NAME = "attacker_preplant_omoko_site_v1"
TEACHER_KIND = "omoko"
TEACHER_MAPS = ()  # Omoko chooses its own attack site/route.
COLLECTION_ROUNDS = 280
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
FIXED_OPPONENT_AI = "default"
SEED = 2026101500
ROUNDS_PER_MATCH = 7
VALIDATION_EVERY_MATCHES = 5
MAX_SKIPPED_FRACTION = 0.05
REQUIRE_ORB_LABELS = True
TORCH_THREADS = 1
DATASET_PATH = AI_ROOT/"data"/DATASET_NAME
LOG_PATH = AI_ROOT/"logs"/DATASET_NAME

if __name__ == "__main__":
    from ghost_champions_v3.preplant.gc3_data_attacker_preplant import collect
    collect({k:v for k,v in globals().copy().items() if k.isupper()})
