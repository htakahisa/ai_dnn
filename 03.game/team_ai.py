from __future__ import annotations
from typing import Any, Callable

from iq_controller_adapter import IQAwareController
from iq_perception import (
    IQPerceptionEngine,
    PerceivedCharacter,
    PerceivedGameView,
)


class PrivateInfoController:
    """Hide enemy spike ownership without adding IQ position noise.

    Some training/evaluation team definitions intentionally disable IQ
    perception. They must still follow the same game-information rule as the
    normal runtime: own-team spike ownership is visible, enemy ownership is
    not. Dropped and planted spike positions remain available.
    """

    def __init__(self, inner):
        self.inner = inner
        self.real_game = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def set_game(self, game):
        self.real_game = game
        if hasattr(self.inner, "set_game"):
            self.inner.set_game(game)

    def reset_round(self):
        if hasattr(self.inner, "reset_round"):
            self.inner.reset_round()

    def decide_move(self, char, game_state):
        if self.real_game is None:
            raise RuntimeError("PrivateInfoController.set_game() was not called")

        proxies = []
        mapping = {}
        for real in self.real_game.chars:
            if getattr(real, "team", None) == getattr(char, "team", None):
                proxy = real
            else:
                proxy = PerceivedCharacter(real, has_spike=False)
            proxies.append(proxy)
            mapping[id(real)] = proxy

        private_game = PerceivedGameView(
            self.real_game,
            {"chars": proxies},
            mapping,
        )
        state = dict(game_state)
        state["chars"] = proxies
        if not bool(getattr(self.real_game, "is_planted", False)):
            state["spotted_info"] = {
                "spotted": 0.0,
                "site_r": 0.0,
                "site_c": 0.0,
            }

        if hasattr(self.inner, "set_game"):
            self.inner.set_game(private_game)
        return self.inner.decide_move(char, state)


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
        if isinstance(controller, IQAwareController):
            return controller
        if not self.use_iq_perception:
            return PrivateInfoController(controller)
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
