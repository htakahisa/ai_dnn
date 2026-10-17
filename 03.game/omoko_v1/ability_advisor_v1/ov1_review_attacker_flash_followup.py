"""Separate smoke and recon changes while keeping the candidate flash adviser."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omoko_v1.ability_advisor_v1.ov1_review_attacker_flash_outcomes import (
    main, DEFAULT_FOLLOWUP_ARMS, DEFAULT_FOLLOWUP_OUTPUT,
)
if __name__ == '__main__':
    main(DEFAULT_FOLLOWUP_ARMS, DEFAULT_FOLLOWUP_OUTPUT)
