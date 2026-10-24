"""Synthetic public-world fixtures without games or training runs."""
from types import SimpleNamespace
import numpy as np
from game_core import Character
from touyama_v3.tv3_character_stats_touyama import TOUYAMA_ROSTER_ORDER
from touyama_v3.tv3_scenario import Scenario

def defender_world(scenario):
    class Game(Scenario):
        def _smoke_cells(self):
            return set()

        def _ramp_blocks_movement(self, char):
            return False
    game = Game()
    own = [Character(TOUYAMA_ROSTER_ORDER[i], "D", post.watch, "white", "green")
           for i, post in enumerate(scenario.posts)]
    enemies = [Character(f"enemy_{i}", "A", (22, 18 + i), "white", "red", has_spike=i == 0)
               for i in range(5)]
    game.chars = own + enemies
    game.defender_setup_phase = SimpleNamespace(active=False)
    game.current_round, game.battle_tick = 1, 0
    game.spike_pos = game.planted_pos = None
    game.is_planted = False
    game.round_timer, game.detonate_timer = 100, 55
    return game


def attacker_world():
    class Game(Scenario):
        def _smoke_cells(self):
            return set()

        def _ramp_blocks_movement(self, char):
            return False
    game = Game()
    cells = [tuple(map(int, p)) for p in np.argwhere(game.grid == 3)]
    game.chars = [Character(TOUYAMA_ROSTER_ORDER[i], "A", pos, "white", "red", has_spike=i == 2)
                  for i, pos in enumerate(cells)]
    game.chars += [Character(f"enemy_{i}", "D", post.watch, "white", "green")
                   for i, post in enumerate(game.posts)]
    game.current_round, game.battle_tick = 1, 0
    game.defender_setup_phase = SimpleNamespace(active=False)
    game.is_planted = False
    game.spike_pos = game.planted_pos = None
    game.round_timer, game.detonate_timer = 100, 55
    return game

