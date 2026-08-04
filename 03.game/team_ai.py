from __future__ import annotations
from typing import Any, Callable

from iq_controller_adapter import IQAwareController
from iq_perception import IQPerceptionEngine


class DualRoleTeamAI:
    def __init__(
        self,
        name: str,
        attacker_factory: Callable[[], Any],
        defender_factory: Callable[[], Any],
        use_iq_perception: bool = True,
        perception_engine: IQPerceptionEngine | None = None,
    ):
        self.name = str(name)
        self.attacker_factory = attacker_factory
        self.defender_factory = defender_factory
        self.use_iq_perception = bool(use_iq_perception)
        self.perception_engine = perception_engine or IQPerceptionEngine()
        self._attacker_controller = None
        self._defender_controller = None
        self.game = None

    def _wrap(self, controller):
        if not self.use_iq_perception or isinstance(controller, IQAwareController):
            return controller
        return IQAwareController(controller, self.perception_engine)

    def get_attacker_controller(self):
        if self._attacker_controller is None:
            raw = self.attacker_factory()
            if raw is None:
                raise RuntimeError(f"{self.name}: attacker_factoryがNoneを返しました")
            self._attacker_controller = self._wrap(raw)
            self._bind_controller(self._attacker_controller)
        return self._attacker_controller

    def get_defender_controller(self):
        if self._defender_controller is None:
            raw = self.defender_factory()
            if raw is None:
                raise RuntimeError(f"{self.name}: defender_factoryがNoneを返しました")
            self._defender_controller = self._wrap(raw)
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
        self.perception_engine.clear_cache()
        for controller in (self._attacker_controller, self._defender_controller):
            if controller is not None and hasattr(controller, "reset_round"):
                controller.reset_round()
