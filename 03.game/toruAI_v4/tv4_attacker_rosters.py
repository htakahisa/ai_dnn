"""Roster selection for generic v4 attacker policies, including legacy metadata."""
from toruAI_v4.tv4_scenario import OPPONENTS


def eligible_attacker_presets(names, opponent):
    from party_presets import get_preset
    names = tuple(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Attacker preset lists must be nonempty and distinct")
    enemy = get_preset(OPPONENTS[opponent][1])
    valid = []
    for name in names:
        own = get_preset(name)
        if own is None or len(own.players) != 5 or len(set(own.players)) != 5:
            raise ValueError(f"Invalid five-player attacker preset: {name}")
        if not set(own.players) & set(enemy.players):
            valid.append(name)
    if not valid:
        raise ValueError(f"No disjoint attacker presets for {opponent}")
    return valid


def source_presets(config, opponent):
    """Old fixed-roster checkpoints remain readable; they are not called generic training."""
    if "train_presets" in config and "eval_presets" in config:
        training = eligible_attacker_presets(config["train_presets"], opponent)
        evaluation = eligible_attacker_presets(config["eval_presets"], opponent)
    elif "attacker_preset" in config:
        training = evaluation = eligible_attacker_presets((config["attacker_preset"],), opponent)
    else:
        raise ValueError("Plant checkpoint has no roster conditions")
    return training, evaluation
