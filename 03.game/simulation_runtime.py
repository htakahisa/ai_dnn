"""CPU settings for small, single-match inference batches, scoped to a map."""

from contextlib import contextmanager
from contextvars import ContextVar

_cpu_match = ContextVar("cpu_match_inference", default=False)


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
    """Override tiny per-character models only inside a headless match."""
    return "cpu" if _cpu_match.get() else None


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
