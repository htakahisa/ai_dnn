"""ConCon spike retrieval, shared by live rounds and A1 battle training."""

from controllers import BaseController

from concon_v1.co1_attacker_abilities import choose_ability
from concon_v1.co1_attacker_common import (
    choose_team_fire_target,
    facing_for_fire_target,
)


class ConconAttackerRetrieveController(BaseController):
    """Send the closest survivor to the spike and the others to support them."""

    def __init__(self):
        self.retriever_name = None

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.retriever_name = None

    def _retriever(self, chars, spike_pos, grid):
        alive = [other for other in chars if other.team == "A" and other.is_alive]
        current = next((other for other in alive if other.name == self.retriever_name), None)
        if current is None and alive:
            current = min(alive, key=lambda other: (
                self.shortest_path_distance(other.pos, spike_pos, grid), other.name,
            ))
            self.retriever_name = current.name
        return current

    def decide_move(self, char, game_state):
        spike_pos = game_state.get("spike_pos")
        if spike_pos is None:
            return list(char.pos)
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        game = getattr(self, "game", None)
        smoke = game_state.get("smoke_cells", ())

        # A visible, shootable enemy always stops movement for this tick.
        target = choose_team_fire_target(char, chars, grid, smoke, game)
        if target is not None:
            facing = facing_for_fire_target(char, target, chars, grid, smoke, game)
            return list(char.pos), {"facing": facing}

        # Reuse the carry phase's team-sighting and projectile targeting rules.
        ability = choose_ability(char, game, route_goal=tuple(spike_pos),
                                 allow_smoke=getattr(self, "allow_smoke", True),
                                 allow_flash=getattr(self, "allow_flash", True),
                                 allow_recon=getattr(self, "allow_recon", True))
        if ability is not None:
            return list(char.pos), ability

        retriever = self._retriever(chars, spike_pos, grid)
        if retriever is None:
            return list(char.pos)
        if char is retriever:
            return self.move_towards_target(
                char.pos, spike_pos, grid, chars=chars, moving_char=char,
            )
        return self.move_towards_target(
            char.pos, retriever.pos, grid, chars=chars, moving_char=char,
            allow_adjacent_goal=True,
        )
