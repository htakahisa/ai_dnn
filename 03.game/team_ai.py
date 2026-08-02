from __future__ import annotations

from typing import Callable, Any


class DualRoleTeamAI:
    def __init__(
        self,
        name: str,
        attacker_factory: Callable[[], Any],
        defender_factory: Callable[[], Any],
    ):
        self.name = name
        self.attacker_factory = attacker_factory
        self.defender_factory = defender_factory

        self._attacker_controller = None
        self._defender_controller = None
        self.game = None

    def get_attacker_controller(self):
        if self._attacker_controller is None:
            self._attacker_controller = self.attacker_factory()
            self._bind_controller(self._attacker_controller)

        return self._attacker_controller

    def get_defender_controller(self):
        if self._defender_controller is None:
            self._defender_controller = self.defender_factory()
            self._bind_controller(self._defender_controller)

        return self._defender_controller

    def bind_game(self, game):
        self.game = game

        if self._attacker_controller is not None:
            self._bind_controller(self._attacker_controller)

        if self._defender_controller is not None:
            self._bind_controller(self._defender_controller)

    def _bind_controller(self, controller):
        if self.game is not None and hasattr(controller, "set_game"):
            controller.set_game(self.game)

    def reset_round(self):
        for controller in (
            self._attacker_controller,
            self._defender_controller,
        ):
            if controller is not None and hasattr(controller, "reset_round"):
                controller.reset_round()
