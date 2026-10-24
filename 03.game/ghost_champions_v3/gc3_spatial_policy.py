"""Soft map hints and sampled distributions; no movement plans or tactical overrides."""
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from grid_paths import distance_map

POSTPLANT_TEMPERATURE = 1.0
POSTPLANT_UNIFORM_MIX = 0.0
POSTPLANT_HINT_WEIGHT = 0.5
POSTPLANT_HINT_RADIUS = 1.5
DEFENDER_TEMPERATURE = 1.6
DEFENDER_UNIFORM_MIX = 0.15
DEFENDER_HINT_WEIGHT = 0.25
DEFENDER_HINT_RADIUS = 2.5
SPATIAL_SCHEMA = "gc3-scratch-map-5-6-v2"


def phase_settings(phase):
    if phase == "defender":
        return DEFENDER_TEMPERATURE, DEFENDER_UNIFORM_MIX, DEFENDER_HINT_WEIGHT, DEFENDER_HINT_RADIUS
    if phase == "postplant":
        return POSTPLANT_TEMPERATURE, POSTPLANT_UNIFORM_MIX, POSTPLANT_HINT_WEIGHT, POSTPLANT_HINT_RADIUS
    if phase == "standard":
        return POSTPLANT_TEMPERATURE, POSTPLANT_UNIFORM_MIX, 0., POSTPLANT_HINT_RADIUS
    raise ValueError(f"Unknown phase: {phase}")


def sampled_distribution(logits, mask, *, temperature=1., uniform_mix=0.):
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= uniform_mix <= 1:
        raise ValueError("Invalid sampling settings")
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("No legal action")
    probabilities = torch.softmax((logits / temperature).masked_fill(~mask, -torch.inf), dim=-1)
    uniform = mask.to(logits.dtype) / mask.sum(dim=-1, keepdim=True)
    return torch.distributions.Categorical(probs=(1-uniform_mix)*probabilities+uniform_mix*uniform)


def parse_map(text):
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Candidate map must be a nonempty rectangle")
    if any(ch not in "0123456" for row in rows for ch in row):
        raise ValueError("Candidate map accepts only 0/1/2/3/4/5/6")
    grid = np.array([[int(ch) for ch in row] for row in rows], dtype=np.int32)
    grid[np.isin(grid, (2, 3, 4))] = 0
    return grid


@lru_cache(maxsize=16)
def hint_fields(text, radius):
    grid = parse_map(text)
    fields = []
    for marker in (5, 6):
        field = np.zeros(grid.shape, dtype=np.float32)
        for point in np.argwhere(grid == marker):
            distances = distance_map(grid, tuple(map(int, point)))
            hint = np.exp(-np.maximum(0, distances).astype(float)**2/(2*radius**2))
            hint[distances < 0] = 0
            field = np.maximum(field, hint)
        fields.append(field)
    return tuple(fields)


class SpatialCandidatePolicy(nn.Module):
    """Neural scores for immediate movement, watch cells and ability targets."""
    def __init__(self, hidden, cells):
        super().__init__()
        self.position = nn.Linear(hidden, cells)
        self.watch = nn.Linear(hidden, cells)
        self.target = nn.Linear(hidden, cells)

    def distribution(self, context, mask, *, kind, phase, hints):
        head = getattr(self, kind)
        logits = head(context)
        temperature, mix, weight, _ = phase_settings(phase)
        if kind in ("position", "watch"):
            field = np.asarray(hints)[..., 0 if kind == "position" else 1, :]
            logits = logits + weight*torch.as_tensor(field, dtype=logits.dtype, device=logits.device)
        return sampled_distribution(logits, mask, temperature=temperature, uniform_mix=mix)
