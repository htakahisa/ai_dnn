"""Trainable spatial distributions with soft map hints and legal-cell masks."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from grid_paths import distance_map
from ghost_champions_v2.geometry import los

# Normal deployment settings. Initial defense retains more exploration.
POSTPLANT_TEMPERATURE = 1.0
POSTPLANT_UNIFORM_MIX = 0.0
POSTPLANT_HINT_WEIGHT = 0.5
POSTPLANT_HINT_RADIUS = 1.5
POSTPLANT_ARRIVAL_RADIUS = 2.0
DEFENDER_TEMPERATURE = 1.6
DEFENDER_UNIFORM_MIX = 0.15
DEFENDER_HINT_WEIGHT = 0.25
DEFENDER_HINT_RADIUS = 2.5
DEFENDER_ARRIVAL_RADIUS = 3.0
CONTEXT_DIM = 10
CELL_DIM = 8
SPATIAL_SCHEMA = "gc3-spatial-soft-5-6-v1"


def phase_settings(phase):
    if phase == "postplant":
        return POSTPLANT_TEMPERATURE, POSTPLANT_UNIFORM_MIX, POSTPLANT_HINT_WEIGHT, POSTPLANT_HINT_RADIUS
    if phase == "defender":
        return DEFENDER_TEMPERATURE, DEFENDER_UNIFORM_MIX, DEFENDER_HINT_WEIGHT, DEFENDER_HINT_RADIUS
    raise ValueError(f"Unknown deployment phase: {phase}")


def sampled_distribution(logits, mask, *, temperature=1., uniform_mix=0.):
    """Never argmax; mixing stays inside legal cells, with correct log-probs."""
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= uniform_mix <= 1:
        raise ValueError("Invalid spatial sampling settings")
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if not bool(mask.any()):
        raise ValueError("No legal spatial action")
    scaled = (logits / temperature).masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(scaled, dim=-1)
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
def _hint_fields(text, radius):
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


def public_context(char, state):
    height, width = state["grid"].shape
    spike = state.get("planted_pos") if state.get("is_planted") else None
    own = [c for c in state.get("chars", ()) if c.team == char.team]
    # Setup exposes raw engine objects: do not inspect enemy positions there.
    known = 0 if state.get("defender_setup_active") else sum(
        c.is_alive and getattr(c, "position_known", False) for c in state.get("chars", ()) if c.team != char.team)
    return np.array([char.pos[0]/height, char.pos[1]/width,
                     getattr(char, "hp", 100)/max(1, getattr(char, "max_hp", 100)),
                     sum(c.is_alive for c in own)/5, known/5,
                     spike[0]/height if spike else 0, spike[1]/width if spike else 0,
                     bool(state.get("is_planted")), bool(state.get("defender_setup_active")),
                     state.get("detonate_timer", 0)/55], dtype=np.float32)


class SpatialCandidatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(CONTEXT_DIM+CELL_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        self.watch = nn.Sequential(nn.Linear(CONTEXT_DIM+CELL_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        # Until trained, hints are a weak prior rather than random neural scores.
        for head in (self.position, self.watch):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def distribution(self, context, features, mask, *, kind, phase):
        head = self.position if kind == "position" else self.watch if kind == "watch" else None
        if head is None:
            raise ValueError("Spatial kind must be position or watch")
        device = head[0].weight.device
        context = torch.as_tensor(context, dtype=torch.float32, device=device)
        features = torch.as_tensor(features, dtype=torch.float32, device=device)
        inputs = torch.cat((context.expand(len(features), -1), features), dim=-1)
        temperature, mix, weight, _ = phase_settings(phase)
        logits = head(inputs).squeeze(-1) + weight*features[:, 5 if kind == "position" else 6]
        return sampled_distribution(logits, mask, temperature=temperature, uniform_mix=mix)

    def sample(self, char, state, map_text, *, kind, phase, legal_grid=None, blocked=(), center=None, radius=None,
               max_steps=None, origin=None):
        grid = np.asarray(state["grid"])
        markers = parse_map(map_text)
        if markers.shape != grid.shape or not np.array_equal(markers == 1, grid == 1):
            raise ValueError("GC v3 candidate map does not match arena walls/dimensions")
        legal_grid = grid if legal_grid is None else legal_grid
        origin = tuple(char.pos) if origin is None else tuple(origin)
        distances = distance_map(legal_grid, origin)
        coords = np.indices(grid.shape).reshape(2, -1).T
        temperature, mix, _, hint_radius = phase_settings(phase)
        position_hint, watch_hint = _hint_fields(map_text, hint_radius)
        height, width = grid.shape
        features = np.array([[r/height, c/width, (r-origin[0])/height, (c-origin[1])/width,
                              min(44, max(0, distances[r, c]))/44, position_hint[r, c], watch_hint[r, c],
                              los(grid, origin, (int(r), int(c)), state.get("smoke_cells", ()))]
                             for r, c in coords], dtype=np.float32)
        mask = (np.asarray(legal_grid) != 1).reshape(-1) & (distances >= 0).reshape(-1)
        for point in blocked:
            mask &= (coords != point).any(axis=1)
        if max_steps is not None:
            mask &= distances.reshape(-1) <= max_steps
        if center is not None and radius is not None:
            mask &= np.linalg.norm(coords-np.array(center), axis=1) <= radius
        if kind == "watch":
            mask &= features[:, 7] > 0
            mask &= (coords != origin).any(axis=1)
        if not mask.any():
            raise ValueError(f"No legal {phase}/{kind} spatial action")
        context = public_context(char, state)
        with torch.no_grad():
            distribution = self.distribution(context, features, mask, kind=kind, phase=phase)
            selected = int(distribution.sample().item())
            logp = float(distribution.log_prob(torch.tensor(selected, device=distribution.probs.device)))
        record = dict(context=context, features=features, mask=mask, kind=kind, phase=phase,
                      action=selected, logp=logp, temperature=temperature, uniform_mix=mix)
        return tuple(map(int, coords[selected])), record

    def log_prob(self, record):
        distribution = self.distribution(record["context"], record["features"], record["mask"],
                                         kind=record["kind"], phase=record["phase"])
        return distribution.log_prob(torch.tensor(record["action"], device=distribution.probs.device)), distribution.entropy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(schema=SPATIAL_SCHEMA, model_state_dict=self.state_dict()), path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != SPATIAL_SCHEMA:
            raise ValueError("GC v3 spatial checkpoint schema mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls()
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model
