"""Record public defense-round information for the same team's future attacks."""
from .gc3_observation_attacker_history import encode_public, command_features


class ObservedDefender:
    requires_public_effects = True

    def __init__(self, controller, history):
        self.inner, self.history = controller, history

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def set_game(self, game):
        self.game = game
        self.inner.set_game(game)

    def reset_round(self):
        self.history.sync_round(getattr(self, "game", None))
        self.inner.reset_round()

    def decide_move(self, char, state):
        self.history.sync_round(self.game)
        observation = encode_public(char, state, self.history)
        command = self.inner.decide_move(char, state)
        self.history.append(observation, command_features(char, command, state), name=char.name,
                            tick=state.get("battle_tick", 0), command=command)
        return command

    def defender_snapshot(self):
        self.history.sync_round(self.game)
        return self.inner.defender_snapshot()

    def on_round_finished(self, result):
        self.history.finish_round(result)
