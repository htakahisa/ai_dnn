"""Omoko scenario and frozen-policy provenance for fixed ability advice."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
ROLES = ('carry', 'escort', 'retrieve', 'guard')
DEFAULT_POLICY_PATHS = {
    role: AI_DIR / 'data' / ('attacker_' + role + '_data') /
          ('dqn_attacker_' + role + '_best_by_eval.pt') for role in ROLES
}


def policy_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts[0] == 'omoko_v1':
        return AI_DIR.parent / path
    return AI_DIR / path


def policy_digest(paths):
    from .ov1_learn_attacker_ability_advisor import sha256_file
    return hashlib.sha256(json.dumps({role: sha256_file(policy_path(path))
        for role, path in paths.items()}, sort_keys=True).encode()).hexdigest()


def scenario(name):
    from omoko_v1.ov1_map_data_carry import NEW_MAZE_STR as carry
    from omoko_v1.ov1_map_data_escort import NEW_MAZE_STR as escort
    from map_data import NEW_MAZE_STR as base
    from .ov1_learn_attacker_ability_advisor import MAPS
    if name not in MAPS:
        raise ValueError('unknown Omoko role/site: ' + name)
    rows = base.strip().splitlines()
    grid = np.array([[int(c) if c.isdigit() else 0 for c in row.strip()] for row in rows])
    # Include role policy and placement sources as well as the visible map.
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for pattern in ('ov1_map_data_*.py', 'ov1_learning_attacker_*.py', 'ov1_roster.py')
               for path in sorted(AI_DIR.glob(pattern))}
    signature = hashlib.sha256(json.dumps(dict(version='omoko_lineup_v1',
        role_site=name, carry=carry, escort=escort, base=base, sources=sources),
        sort_keys=True).encode()).hexdigest()
    return SimpleNamespace(map_name=name, grid=grid, signature=signature)


def defending_opponent(game):
    from .ov1_learn_attacker_ability_advisor import OPPONENT_NAMES
    ai = getattr(game, 'current_defender_team_ai', None)
    return OPPONENT_NAMES.get(getattr(ai, 'name', None))
