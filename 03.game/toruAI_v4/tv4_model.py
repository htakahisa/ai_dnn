"""Opponent-specific supervised classifier and stable rotation decisions."""
import numpy as np
import torch
from torch import nn

VERSION = 3  # Public allied capabilities and multiple-roster training.


class SiteModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size, 128), nn.ReLU(), nn.Linear(128, 64),
                                 nn.ReLU(), nn.Linear(64, 2))
        # Before learning, unknown observations must not trigger a rotation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def probabilities(self, features):
        return self(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)).softmax(-1)[0].numpy()


class DecisionGate:
    def __init__(self, threshold=.8, confirm=3):
        self.threshold, self.confirm = threshold, confirm
        self.side = self.tick = self.candidate = None
        self.streak = self.changes = 0

    def update(self, probabilities, tick):
        index = int(np.argmax(probabilities))
        candidate = ("L", "R")[index] if probabilities[index] >= self.threshold else None
        self.streak = self.streak + 1 if candidate is not None and candidate == self.candidate else int(candidate is not None)
        self.candidate = candidate
        if candidate is not None and self.streak >= self.confirm and candidate != self.side:
            self.changes += int(self.side is not None)
            self.side, self.tick = candidate, tick
            return True
        return False


def optimize(model, optimizer, replay, rng, updates, batch_size):
    if not replay:
        return None
    losses = []
    model.train()
    for _ in range(updates):
        # Sample rounds uniformly, then a tick uniformly. A long round must
        # not outweigh short rounds simply because it has more observations.
        sampled = [replay[int(rng.integers(len(replay)))] for _ in range(batch_size)]
        x = np.stack([r["features"][int(rng.integers(len(r["features"])))] for r in sampled])
        y = torch.tensor([r["label"] for r in sampled], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(torch.tensor(x, dtype=torch.float32)), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return float(np.mean(losses))
