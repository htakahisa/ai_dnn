from __future__ import annotations
import contextlib
import io
from typing import Any, Callable

from iq_controller_adapter import IQAwareController
from iq_perception import (
    IQPerceptionEngine,
    PerceivedCharacter,
    PerceivedGameView,
)
from controllers import UserInputController
from environment import (
    GC_STDOUT_LOGS_ENABLED,
    OMOKO_V1_STDOUT_LOGS_ENABLED,
    TOUYAMA_V2_STDOUT_LOGS_ENABLED,
)


def _team_stdout_logs_enabled(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in {"ghost champions", "ghost champions v1", "gc", "gc_v1"}:
        return GC_STDOUT_LOGS_ENABLED
    if normalized in {"omoko gaming v1", "omoko_v1"}:
        return OMOKO_V1_STDOUT_LOGS_ENABLED
    if normalized in {"touyama gaming v2", "touyama_gaming_v2"}:
        return TOUYAMA_V2_STDOUT_LOGS_ENABLED
    return True


class _QuietController:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @staticmethod
    def _call(method, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return method(*args, **kwargs)

    def set_game(self, game):
        method = getattr(self.inner, "set_game", None)
        if callable(method):
            return self._call(method, game)

    def reset_round(self):
        method = getattr(self.inner, "reset_round", None)
        if callable(method):
            return self._call(method)

    def decide_move(self, char, game_state):
        return self._call(self.inner.decide_move, char, game_state)


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
        self.stdout_logs_enabled = _team_stdout_logs_enabled(self.name)
        self.attacker_factory = attacker_factory
        self.defender_factory = defender_factory
        self.use_iq_perception = bool(use_iq_perception)
        self.perception_engine = perception_engine or IQPerceptionEngine()
        self._attacker_controller = None
        self._defender_controller = None
        self.game = None

    def _wrap(self, controller):
        if getattr(controller, "handles_team_perception", False):
            wrapped = controller
        elif isinstance(controller, IQAwareController):
            wrapped = controller
        elif isinstance(controller, UserInputController):
            return controller
        elif not self.use_iq_perception:
            wrapped = PrivateInfoController(controller)
        else:
            wrapped = IQAwareController(controller, self.perception_engine)

        return wrapped if self.stdout_logs_enabled else _QuietController(wrapped)

    def _create_controller(self, factory):
        if self.stdout_logs_enabled:
            return factory()
        with contextlib.redirect_stdout(io.StringIO()):
            return factory()

    def get_attacker_controller(self):
        if self._attacker_controller is None:
            raw = self._create_controller(self.attacker_factory)
            if raw is None:
                raise RuntimeError(f"{self.name}: attacker_factoryがNoneを返しました")
            self._attacker_controller = self._wrap(raw)
            self._bind_controller(self._attacker_controller)
        return self._attacker_controller

    def get_defender_controller(self):
        if self._defender_controller is None:
            raw = self._create_controller(self.defender_factory)
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
