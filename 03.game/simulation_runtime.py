"""Match-local randomness and consistent CPU inference for competition maps."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import random
import threading

import numpy as np

_cpu_match = ContextVar("cpu_match_inference", default=False)
_seed_scope = ContextVar("match_seed_scope", default=None)
_match_lock = threading.RLock()

# Competition inference uses the same device and arithmetic with/without UI.
COMPETITION_INFERENCE_DEVICE = "cpu"


def resolve_random_seed(seed=None, *, namespace):
    """Seed private RNGs from the match, preserving explicit/outside-match seeds."""
    scope = _seed_scope.get()
    if seed is not None or scope is None:
        return seed
    match_seed, counters = scope
    index = counters.get(namespace, 0)
    counters[namespace] = index + 1
    digest = hashlib.sha256(f"{match_seed}:{namespace}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


@contextmanager
def deterministic_match(seed):
    """Isolate competition randomness and inference settings from other work."""
    import torch
    with _match_lock, cpu_inference():
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        scope_token = _seed_scope.set((int(seed), {}))
        try:
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            # Competition runs on CPU; do not initialize or alter CUDA RNGs.
            torch.default_generator.manual_seed(int(seed))
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            _seed_scope.reset(scope_token)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class MatchRandomState:
    """Keep UI callbacks and other simulations out of a match's random stream."""
    def __init__(self):
        import torch
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.torch = torch.get_rng_state()
        scope = _seed_scope.get()
        self.scope = None if scope is None else (scope[0], dict(scope[1]))
        self.active = False

    @contextmanager
    def activate(self):
        if self.active:
            yield
            return
        import torch
        ambient = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        token = _seed_scope.set(self.scope)
        self.active = True
        try:
            random.setstate(self.python)
            np.random.set_state(self.numpy)
            torch.set_rng_state(self.torch)
            yield
        finally:
            self.python = random.getstate()
            self.numpy = np.random.get_state()
            self.torch = torch.get_rng_state()
            random.setstate(ambient[0])
            np.random.set_state(ambient[1])
            torch.set_rng_state(ambient[2])
            self.active = False
            _seed_scope.reset(token)


def isolated_match_randomness(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        state = getattr(self, "_match_random_state", None)
        if state is None:
            return method(self, *args, **kwargs)
        with state.activate():
            return method(self, *args, **kwargs)
    return wrapped


def competition_match(function):
    """Enter the seeded CPU scope before constructing even lazy controllers."""
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        with deterministic_match(arguments.arguments["seed"]):
            return function(*args, **kwargs)
    return wrapped


def resolve_inference_device(device=None, *, default="auto"):
    """Explicit device wins; otherwise use the match override and model default."""
    import torch
    selected = device
    if selected is None or str(selected) == "auto":
        selected = match_inference_device() or default
    if str(selected) == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(selected)


def match_inference_device():
    """Return the device shared by rendered and headless competition matches."""
    return COMPETITION_INFERENCE_DEVICE if _cpu_match.get() else None


@contextmanager
def cpu_inference(*, enabled=True):
    if not enabled:
        yield
        return
    import torch
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    token = _cpu_match.set(True)
    try:
        yield
    finally:
        _cpu_match.reset(token)
        if previous != 1:
            torch.set_num_threads(previous)
