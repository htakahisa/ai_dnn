"""Validate the configured planning map; never collect data or update weights."""
from pathlib import Path
import sys
import numpy as np

MACRO_ROOT=Path(__file__).resolve().parents[1]
GAME_ROOT=MACRO_ROOT.parents[1]
SHOW_COORDINATES=True
sys.path.insert(0,str(GAME_ROOT))


def main():
    from map_data import NEW_MAZE_STR
    from ghost_champions_v3.macro_v2.gc3_candidates_attacker_macro import MacroCandidates
    arena=np.array([[int(value) for value in row] for row in NEW_MAZE_STR.strip().splitlines()])
    try:candidates=MacroCandidates.configured(arena)
    except ValueError as error:
        print(f'[MAP] {error}')
        return 1
    print(candidates.summary())
    if SHOW_COORDINATES:
        print('Destination candidates:',[candidates.coordinate(i) for i in np.flatnonzero(candidates.destination_mask)])
        print('Plant candidates:',[candidates.coordinate(i) for i in np.flatnonzero(candidates.plant_mask)])
    return 0


if __name__=='__main__':raise SystemExit(main())
