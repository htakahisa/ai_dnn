"""Only the five user-specified macro rewards, counted once per team event."""
from collections import Counter

REWARD_VERSION = "gc3-attacker-macro-user-events-v1"
KILL_REWARD = 1.0
DEATH_REWARD = -1.0
PLANT_REWARD = 10.0
WIN_REWARD = 30.0
LOSS_REWARD = -30.0
NO_DEFUSE_START_REWARD = 10.0


def reward_settings():
    return dict(REWARD_VERSION=REWARD_VERSION, KILL_REWARD=KILL_REWARD, DEATH_REWARD=DEATH_REWARD,
                PLANT_REWARD=PLANT_REWARD, WIN_REWARD=WIN_REWARD, LOSS_REWARD=LOSS_REWARD,
                NO_DEFUSE_START_REWARD=NO_DEFUSE_START_REWARD,
                scope="attacking team totals, including frozen postplant",
                no_defuse_start="actual plant, attacker round win and no actual defuse start at any time")


class MacroRewards:
    def __init__(self, game):
        self.game = game
        self.plant_credited = False
        self.defuse_ever_started = False
        self.outcome_credited = False
        self.totals = Counter()
        self.before = self.snapshot()

    def snapshot(self):
        return {str(c.name):(bool(c.is_alive), int(getattr(c, "kills", 0))) for c in self.game.chars if c.team == "A"}

    def __enter__(self):
        self.original_move = self.game.move_character
        self.game.move_character = self.observe_move
        return self

    def __exit__(self, *_):
        self.game.move_character = self.original_move

    def observe_move(self, char):
        result = self.original_move(char)
        # Check before battle resolution can kill the defuser/reset the timer
        # in this very same tick. An interrupted tap still disqualifies bonus.
        if char.team == "D" and getattr(char, "defuse_timer", 0) > 0:
            self.defuse_ever_started = True
        return result

    def finish_tick(self, terminal, success):
        current = self.snapshot()
        kills = sum(max(0, current[name][1]-kills) for name, (_, kills) in self.before.items())
        deaths = sum(alive and not current[name][0] for name, (alive, _) in self.before.items())
        parts = dict(kill=KILL_REWARD*kills, death=DEATH_REWARD*deaths)
        if self.game.is_planted and not self.plant_credited:
            parts["plant"] = PLANT_REWARD
            self.plant_credited = True
        if any(c.team == "D" and getattr(c, "defuse_timer", 0)>0 for c in self.game.chars):
            self.defuse_ever_started = True
        if terminal and not self.outcome_credited:
            parts["outcome"] = WIN_REWARD if success else LOSS_REWARD
            if success and self.plant_credited and not self.defuse_ever_started:
                parts["no_defuse_start"] = NO_DEFUSE_START_REWARD
            self.outcome_credited = True
        self.before = current
        self.totals.update(parts)
        return parts
