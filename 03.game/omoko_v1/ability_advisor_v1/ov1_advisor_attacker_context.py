"""Omoko scenario and frozen-policy provenance for fixed ability advice."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
ROLES = ('carry', 'escort', 'retrieve', 'guard')
ADVISOR_POLICY_ROLES = ('carry', 'escort')
# Verified from checkpoint bytes in Git c11afd437062944b7719516b44edce65a50bd5f9.
# Only this proven legacy bundle can be translated; unknown hashes still fail.
LEGACY_POLICY_HASHES = {
    'carry': '31dcb91ca1a3bd54771efcf4e63191d07baa021f0592e3a2dbdf0508f63a89d9',
    'escort': '0d38aeeebd812f45fb615a1f5c9077a5257ea247f8d51d9b94006196d7fceb81',
    'retrieve': 'c0fcbd23a5910f7d15aa6e0ff3b1438c7f927c4397d74e49fe63b10dd5114ffc',
    'guard': '08cc0852803338ae2614a877dc8cb6cdf4ea747364785bf358fcff00c5c93a2a',
}
# Change this only when the interpretation of the scenario inputs changes.
# Inference source text (including comments and guard-only changes) is not an
# input schema. Frozen checkpoint weights have their own provenance below.
SCENARIO_CONTRACT_VERSION = 'omoko_lineup_geometry_v2'
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
    return hashlib.sha256(json.dumps({role: sha256_file(policy_path(paths[role]))
        for role in ADVISOR_POLICY_ROLES}, sort_keys=True).encode()).hexdigest()


def normalize_policy_digest(digest):
    legacy = hashlib.sha256(json.dumps(LEGACY_POLICY_HASHES, sort_keys=True).encode()).hexdigest()
    if digest == legacy:
        return hashlib.sha256(json.dumps({role: LEGACY_POLICY_HASHES[role]
            for role in ADVISOR_POLICY_ROLES}, sort_keys=True).encode()).hexdigest()
    return digest


def compatible_route_metadata(recorded, current):
    if not isinstance(recorded, dict):
        return False
    normalized = dict(recorded)
    normalized['carry_sha256'] = normalize_policy_digest(normalized.get('carry_sha256'))
    return normalized == current


def scenario(name):
    from omoko_v1.ov1_map_data_carry import NEW_MAZE_STR as carry
    from omoko_v1.ov1_map_data_escort import NEW_MAZE_STR as escort
    from map_data import NEW_MAZE_STR as base
    from .ov1_learn_attacker_ability_advisor import MAPS
    if name not in MAPS:
        raise ValueError('unknown Omoko role/site: ' + name)
    rows = base.strip().splitlines()
    grid = np.array([[int(c) if c.isdigit() else 0 for c in row.strip()] for row in rows])
    # Hash the actual pre-plant layout inputs, never every policy/map source
    # file. Previously any edit to guard placement or inference Python text
    # invalidated all four routes even with exactly the same trained weights.
    manifest = dict(version=SCENARIO_CONTRACT_VERSION, role_site=name,
                    carry=carry.split(), escort=escort.split(), grid=grid.tolist())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return SimpleNamespace(map_name=name, grid=grid, signature=signature,
                           manifest=manifest)


def defending_opponent(game):
    from .ov1_learn_attacker_ability_advisor import OPPONENT_NAMES
    ai = getattr(game, 'current_defender_team_ai', None)
    return OPPONENT_NAMES.get(getattr(ai, 'name', None))


def preflight_default_advisor():
    """Validate every attacker route before a defender-start match can run."""
    from .ov1_learn_attacker_ability_advisor import DeployedAbilityAdvisor, MAPS
    advisor = DeployedAbilityAdvisor()
    digest = policy_digest(DEFAULT_POLICY_PATHS)
    for name in MAPS:
        advisor.bind_route(scenario(name), digest)
    return advisor
