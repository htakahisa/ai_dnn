"""Carry actual remaining ULT points across rounds in one rollout match."""
from functools import lru_cache
from pathlib import Path
from .gc3_observation_attacker_history import encode_public, command_features

POSTPLANT_CHECKPOINT = Path(__file__).resolve().parents[1]/"data/attacker_postplant_fixed_v1/best.pt"
MAX_POSTPLANT_TICKS = 400


class MatchResources:
    def __init__(self):
        self.points = {}

    @staticmethod
    def key(char, opponent):
        # Actual IDs, including mirror-team aliases, keep both sides separate.
        return ("GC" if char.team == "A" else opponent, char.team, str(char.name))

    def restore(self, game, opponent):
        for char in game.chars:
            value = self.points.get(self.key(char, opponent), 0)
            char.ultimate_points = min(char.ultimate_cost, max(0, int(value)))
            game._save_ultimate_points(char)

    def capture(self, game, opponent):
        if not game.round_over:
            raise RuntimeError("Carry ULT points only after the actual round has ended")
        for char in game.chars:
            self.points[self.key(char, opponent)] = int(char.ultimate_points)


class ObservedPostplant:
    requires_public_effects = True

    def __init__(self, controller, history):
        self.inner, self.history = controller, history

    def set_game(self, game):
        self.inner.set_game(game)

    def decide_move(self, char, state):
        observation = encode_public(char, state, self.history)
        command = self.inner.decide_move(char, state)
        self.history.append(observation, command_features(char, command, state),
                            name=char.name, tick=state.get("battle_tick", 0), command=command)
        return command

    def on_round_finished(self, result):
        self.history.finish_round(result)


@lru_cache(maxsize=2)
def frozen_guard(path, checkpoint_hash):
    from ..gc3_policy import FreshPolicy
    policy, metadata = FreshPolicy.load(path, role="attacker")
    if not metadata.get("trained"):
        raise ValueError("Resource rollout requires a trained, frozen postplant checkpoint")
    return policy.requires_grad_(False).eval()


def complete_round(game, history, checkpoint=POSTPLANT_CHECKPOINT):
    """Only the preplant actor learns; postplant actions finish the resource state."""
    if game.round_over:
        return 0
    if not game.is_planted:
        raise RuntimeError("A nonterminal preplant phase needs an actual plant")
    from .gc3_data_attacker_preplant import file_hash
    from ..gc3_learn_attacker_guard import GhostChampionsV3AttackerController
    from ..gc3_rollout import module_digest
    policy = frozen_guard(str(checkpoint), file_hash(checkpoint))
    digest = module_digest([policy])
    inner = ObservedPostplant(GhostChampionsV3AttackerController(policy=policy), history)
    # Preserve the engine's team perception; no new preplant recorder is made.
    game.attacker_controller.inner = inner
    for tick in range(MAX_POSTPLANT_TICKS):
        game._simulate_tick()
        if game.round_over:
            if module_digest([policy]) != digest:
                raise RuntimeError("Frozen postplant weights changed")
            return tick+1
    raise RuntimeError("Frozen postplant resource simulation did not finish the round")


def point_snapshot(game):
    return {str(c.name):int(c.ultimate_points) for c in game.chars}
