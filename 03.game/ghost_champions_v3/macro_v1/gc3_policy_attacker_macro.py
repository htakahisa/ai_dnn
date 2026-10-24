"""Stochastic KEEP/A/B actor with causal, complete public match memory."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..preplant.gc3_policy_attacker_history import HistoryPolicy, DATA_SCHEMA_HASH
from ..preplant.gc3_observation_attacker_history import encode_public, EXTRA_WIDTH
from ..gc3_spatial_policy import sampled_distribution
from ..gc3_io import replace_file

ACTIONS = ("KEEP", "A", "B")
POLICY_TYPE = "gc3_attacker_macro_v1"
SCHEMA_HASH = hashlib.sha256((DATA_SCHEMA_HASH+"macro-keep-a-b-causal-v1").encode()).hexdigest()


class MacroHistoryView:
    """All public lower events plus macro events, without changing lower memory."""
    def __init__(self, history):
        self.history = history
        self.events = []
        self._copied = 0
        self._encoded = self._cache_policy = None

    def __getattr__(self, name):
        return getattr(self.history, name)

    def sync_public_events(self):
        self.events.extend(dict(e) for e in self.history.events[self._copied:])
        self._copied = len(self.history.events)

    def save(self, path):
        from ..preplant.gc3_observation_attacker_history import MatchHistory
        self.sync_public_events()
        ledger = MatchHistory()
        ledger.events = self.events
        ledger.round_id = self.history.round_id
        ledger.results = self.history.results
        ledger.goal_sites = self.history.goal_sites
        ledger.save(path)


class MacroPolicy(HistoryPolicy):
    def __init__(self, role="attacker", hidden=128):
        super().__init__(role, hidden)
        # Reuse the public-history encoder, not the lower actor or its weights.
        del self.mode, self.ally, self.spatial, self.move
        self.actor = nn.Linear(hidden, len(ACTIONS))
        self.goal_merge = nn.Sequential(nn.Linear(hidden+4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.past_goal = nn.Embedding(3, hidden)
        self.past_macro_action = nn.Embedding(4, hidden, padding_idx=3)
        self.schema_hash = SCHEMA_HASH

    def event_features(self, events):
        base = super().event_features(events)
        goal = torch.tensor([2 if e.get("goal_site") is None else e["goal_site"] for e in events])
        action = torch.tensor([e.get("macro_action", 3) for e in events])
        return base+self.past_goal(goal)+self.past_macro_action(action)

    def contexts(self, rows, *, cache=False):
        context = super().contexts(rows, cache=cache)
        extra = np.zeros((len(rows), 4), dtype=np.float32)
        for i, row in enumerate(rows):
            extra[i, 2 if row["current_goal"] is None else row["current_goal"]] = 1.
            extra[i, 3] = row["goal_age"]/100.
        return self.goal_merge(torch.cat((context, torch.from_numpy(extra)), -1))

    def select_macro(self, char, state, history, current_goal, goal_age):
        observation = encode_public(char, state, history)
        observation[-EXTRA_WIDTH+6] = 2.  # Macro event, not a lower movement decision.
        mask = np.array([current_goal is not None, True, True], dtype=np.bool_)
        row = dict(obs=observation, phase="macro", history=history, history_index=len(history.events),
                   current_goal=current_goal, goal_age=goal_age, masks={"macro":mask}, selections={})
        with torch.no_grad():
            context = self.contexts([row], cache=True)[0]
            distribution = sampled_distribution(self.actor(context), mask)
            action = distribution.sample()
            row.update(selections={"macro":int(action)}, logp=float(distribution.log_prob(action)),
                       value=float(self.critic(context)))
        return int(action), row

    def evaluate_records(self, rows):
        context = self.contexts(rows)
        dist = sampled_distribution(self.actor(context), np.stack([r["masks"]["macro"] for r in rows]))
        choices = torch.tensor([r["selections"]["macro"] for r in rows])
        return dist.log_prob(choices), dist.entropy(), self.critic(context).squeeze(-1)

    def save(self, path, **metadata):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(policy_type=POLICY_TYPE, schema_hash=SCHEMA_HASH, hidden=self.hidden,
                        role="attacker", model_state_dict=self.state_dict(), **metadata), temporary)
        replace_file(temporary, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("policy_type") != POLICY_TYPE or data.get("schema_hash") != SCHEMA_HASH or data.get("role") != "attacker":
            raise ValueError("Attacker macro v1 checkpoint mismatch")
        with torch.random.fork_rng(devices=[]):
            policy = cls(hidden=data["hidden"])
            policy.load_state_dict(data["model_state_dict"])
        return policy.eval(), data
