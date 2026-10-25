"""Bounded real-game validation, separate from the normal training run.

Run from postplant_v2: python tools/gc3_validate_attacker_guard_response.py
Settings for this validation are the constants below.
"""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='validation_response_v3_verified'
TRAINING_STEPS=64
EVALUATION_ROUNDS_PER_OPPONENT=1
EVALUATE_EVERY_STEPS=64
EVALUATION_PATIENCE=0
OPPONENT_CODES=('FNC',)
ROUNDS_PER_MATCH=1
RESUME=False
sys.path.insert(0,str(GUARD_ROOT.parents[1]))


def main():
    from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard_response as settings
    from ghost_champions_v3.postplant_v2.gc3_training_attacker_guard import train
    config={k:v for k,v in vars(settings).items() if k.isupper()}
    config.update({k:v for k,v in globals().items() if k in ('RUN_NAME','TRAINING_STEPS','EVALUATION_ROUNDS_PER_OPPONENT',
        'EVALUATE_EVERY_STEPS','EVALUATION_PATIENCE','RESUME','OPPONENT_CODES','ROUNDS_PER_MATCH')})
    config.update(RUN_PATH=GUARD_ROOT/'data'/RUN_NAME,LOG_PATH=GUARD_ROOT/'logs'/RUN_NAME)
    train(config)


if __name__=='__main__':main()
