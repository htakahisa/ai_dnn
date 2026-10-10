"""Weights-only checkpoint compatibility and atomic publication."""
from collections import OrderedDict
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

SAVE_REPLACE_ATTEMPTS = 61  # Allow up to 30 seconds for Windows readers/scanners to release the file.
SAVE_REPLACE_DELAY_SECONDS = 0.5


def checkpoint_values(value):
    """Remove NumPy pickle globals, including scalars nested in metadata/replay."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, OrderedDict):
        result = OrderedDict((k, checkpoint_values(v)) for k, v in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = checkpoint_values(value._metadata)
        return result
    if isinstance(value, dict):
        return {checkpoint_values(k): checkpoint_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [checkpoint_values(v) for v in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_values(v) for v in value)
    return value


def load_checkpoint(path):
    """Read local legacy checkpoints without enabling unrestricted pickle."""
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import scalar

    # NumPy 1.x and 2.x use different pickle paths; dtype classes are built
    # dynamically and must also be explicitly allowed (PyTorch serialization docs).
    scalar_types = (np.bool_, np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64)
    allowed = [(scalar, "numpy.core.multiarray.scalar"),
               (scalar, "numpy._core.multiarray.scalar"), np.dtype]
    allowed.extend({type(np.dtype(t)) for t in scalar_types})
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint_values(state)


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique names prevent simultaneous writers from sharing latest.tmp.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    preserve_complete_save = False
    try:
        torch.save(checkpoint_values(state), temporary)
        for attempt in range(SAVE_REPLACE_ATTEMPTS):
            try:
                temporary.replace(path)
                return
            except PermissionError as error:
                if attempt + 1 == SAVE_REPLACE_ATTEMPTS:
                    preserve_complete_save = True
                    raise PermissionError(
                        f"Checkpoint replacement failed after {SAVE_REPLACE_ATTEMPTS} attempts: {path}. "
                        f"Previous checkpoint was retained; the complete new checkpoint is available at: {temporary}"
                    ) from error
                time.sleep(SAVE_REPLACE_DELAY_SECONDS)
    finally:
        if not preserve_complete_save:
            temporary.unlink(missing_ok=True)
