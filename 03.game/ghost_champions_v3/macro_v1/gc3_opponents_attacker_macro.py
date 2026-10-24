"""Explicit major-roster/main-AI pairs, used as frozen production opponents."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import torch
from ..gc3_rollout import freeze_modules, module_digest
from ..preplant.gc3_data_attacker_preplant import file_hash

MAIN_OPPONENT_AIS = dict(FNC="fnatic_v3", FRC="frc_v1", TYG="touyama_gaming_v2",
                         OMG="omoko_gaming_v1", SPS="toru_ai_v3.1", GRG="concon_v1", GC="ghost_champions_v2")


def opponent_settings(config, code):
    mode = config.get("OPPONENT_MODE", "fixed_logic")
    if mode == "fixed_logic":
        return dict(opponent_ai="default")
    if mode != "major_main":
        raise ValueError("Unknown macro opponent mode")
    keys = config.get("OPPONENT_AIS", MAIN_OPPONENT_AIS)
    if code not in keys or keys[code] != MAIN_OPPONENT_AIS[code]:
        raise ValueError("Major team must use its corresponding main AI")
    return dict(opponent_ai=keys[code], expected_opponent_files=config.get("_expected_opponent_files"))


@contextmanager
def capture_checkpoint_loads():
    """Identify actual model files, including models loaded lazily on a plant."""
    original = torch.load
    paths = {}
    def load(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if isinstance(path, (str, Path)) and Path(path).is_file():
            resolved = Path(path).resolve()
            paths[str(resolved)] = file_hash(resolved)
        return value
    with patch.object(torch, "load", load):
        yield paths


class FrozenOpponent:
    """Keep lazy models in eval/no-grad, and verify their weights at round end."""
    def __init__(self, inner):
        self.inner = inner
        self.models = {}
        self.game = None
        self.freeze_current()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def track(self, model):
        model.eval().requires_grad_(False)
        if id(model) not in self.models:
            self.models[id(model)] = (model, module_digest([model]))

    def freeze_current(self):
        for model in freeze_modules(self.inner, excluded=(self.game,)):
            self.track(model)

    def set_game(self, game):
        self.game = game
        if hasattr(self.inner, "set_game"):
            self.inner.set_game(game)

    def call(self, method, *args):
        self.freeze_current()
        # A retake model can be constructed inside decide_move. Freeze it
        # before its first forward, rather than only after the first action.
        hook = torch.nn.modules.module.register_module_forward_pre_hook(lambda model, inputs:self.track(model))
        try:
            with torch.no_grad():
                return method(*args)
        finally:
            hook.remove()
            self.freeze_current()

    def decide_move(self, char, state):
        return self.call(self.inner.decide_move, char, state)

    def reset_round(self):
        method = getattr(self.inner, "reset_round", None)
        if method:
            return self.call(method)

    def on_round_finished(self, result):
        method = getattr(self.inner, "on_round_finished", None)
        if method:
            return self.call(method, result)

    def prepare_team_tick(self):
        method = getattr(self.inner, "prepare_team_tick", None)
        if method:
            return self.call(method)

    def verify(self):
        if any(module_digest([model]) != digest for model,digest in self.models.values()):
            raise RuntimeError("Frozen major opponent updated model weights")


def build_opponent(key):
    from run_game import _build_team_ai
    team = _build_team_ai(key)
    factory = team.defender_factory
    team.defender_factory = lambda:FrozenOpponent(factory())
    return team
