"""One pre-plant actor with causal attention over every preceding match event."""
import hashlib
from bisect import bisect_left
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..gc3_policy import FreshPolicy, SCHEMA_HASH as BASE_SCHEMA, ObservationFeatures
from ..gc3_observation import OBS_DIM, CELLS, CONTEXT_WIDTH, BOARD_CHANNELS, UNIT_WIDTH, ARENA_SHAPE, map_hints, sampling_phase
from ..gc3_engine_actions import MODES, CARDINAL, legal_actions, decode_command
from ..gc3_spatial_policy import sampled_distribution, phase_settings
from ..gc3_io import replace_file
from .gc3_observation_attacker_history import EXTRA_WIDTH, HISTORY_OBS_DIM, ACTION_WIDTH, encode_public, command_features

# The input format is unchanged. Model revisions must not invalidate collected data.
DATA_SCHEMA_HASH = hashlib.sha256((BASE_SCHEMA+"full-public-match-attention-v1-"+str(HISTORY_OBS_DIM)).encode()).hexdigest()
SCHEMA_HASH = hashlib.sha256((DATA_SCHEMA_HASH+"relative-movement-location-self-history-v2").encode()).hexdigest()
MOVE_OFFSETS = ((0, 0), *CARDINAL)


def preplant_legal_actions(char, state):
    masks, caps = legal_actions(char, state)
    # A facing command can point into a wall. Watch targets encode direction,
    # rather than movement or projectile destinations.
    masks["watch"] = np.ones(np.asarray(state["grid"]).size, dtype=np.bool_)
    return masks, caps


def actor_key(observation):
    """Identity and round from the existing public input, never a teacher label."""
    start = CONTEXT_WIDTH+BOARD_CHANNELS*CELLS
    flags = observation[start+1:start+5*UNIT_WIDTH:UNIT_WIDTH]
    return (int(round(float(observation[19]))), int(np.argmax(flags)),
            int(round(float(observation[-EXTRA_WIDTH+5])*30)))


class PublicFeatures(nn.Module):
    def __init__(self, hidden, *, legacy=False):
        super().__init__()
        self.base = ObservationFeatures(hidden)
        self.extra = nn.Linear(EXTRA_WIDTH, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.location = None if legacy else nn.Embedding(CELLS, hidden)

    def forward(self, observation):
        encoded = self.base(observation[..., :OBS_DIM])+self.extra(observation[..., OBS_DIM:])
        if self.location is not None:
            point = (observation[..., :2]*observation.new_tensor(ARENA_SHAPE)).round().long()
            encoded = encoded+self.location(point[..., 0]*ARENA_SHAPE[1]+point[..., 1])
        return self.norm(encoded)


class HistoryPolicy(FreshPolicy):
    def __init__(self, role="attacker", hidden=128, *, legacy=False):
        if role != "attacker":
            raise ValueError("History policy is the unified pre-plant attacker")
        super().__init__(role, hidden)
        self.legacy = legacy
        self.features = PublicFeatures(hidden, legacy=legacy)
        self.action_features = nn.Linear(ACTION_WIDTH, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.memory_merge = nn.Sequential(nn.Linear(hidden*(2 if legacy else 3), hidden), nn.LayerNorm(hidden), nn.SiLU())
        if not legacy:
            # The same five learned outputs apply at every cell. Staying is still legal.
            self.move = nn.Linear(hidden, len(MOVE_OFFSETS))
            self.spatial.position = None
            self.register_buffer("cells", torch.tensor(np.indices(ARENA_SHAPE).reshape(2, -1).T), persistent=False)
        self.schema_hash = DATA_SCHEMA_HASH if legacy else SCHEMA_HASH

    def distribution(self, context, record, kind):
        if self.legacy or kind != "position":
            return super().distribution(context, record, kind)
        observation = torch.as_tensor(record["obs"], dtype=context.dtype, device=context.device)
        origin = (observation[..., :2]*context.new_tensor(ARENA_SHAPE)).round().long()
        delta = self.cells-origin[..., None, :]
        choices = (delta[..., None, :] == delta.new_tensor(MOVE_OFFSETS)).all(-1)
        mask = torch.as_tensor(record["masks"][kind], dtype=torch.bool, device=context.device)
        if bool((mask & ~choices.any(-1)).any()):
            raise ValueError("Movement mask contains a nonlocal action")
        directions = choices.long().argmax(-1)
        logits = self.move(context).gather(-1, directions)
        temperature, mix, _, _ = phase_settings(record["phase"])
        return sampled_distribution(logits, mask, temperature=temperature, uniform_mix=mix)

    def event_features(self, events):
        obs = torch.from_numpy(np.stack([e["obs"] for e in events]))
        actions = torch.from_numpy(np.stack([e["action"] for e in events]))
        if not self.legacy:
            actions = actions.clone()
            displacement = (actions[:, len(MODES):len(MODES)+2]-obs[:, :2])*obs.new_tensor(ARENA_SHAPE)
            actions[:, len(MODES):len(MODES)+2] = displacement*(actions[:, :len(MODES)].sum(-1)>0)[:, None]
        return self.features(obs)+self.action_features(actions)

    def contexts(self, rows, *, cache=False):
        current = self.features(torch.from_numpy(np.stack([r["obs"] for r in rows])))
        outputs = [None]*len(rows)
        groups = {}
        for i, row in enumerate(rows):
            history = row.get("history")
            if history is None:
                raise ValueError("History checkpoint needs chronological public history")
            groups.setdefault(id(history), (history, []))[1].append(i)
        for history, indices in groups.values():
            ends = [int(rows[i]["history_index"]) for i in indices]
            maximum = max(ends)
            if any(end < 0 or end > len(history.events) for end in ends):
                raise ValueError("Invalid causal history index")
            memory = None
            if maximum:
                if cache and not torch.is_grad_enabled():
                    if history._cache_policy is not self:
                        history._encoded, history._cache_policy = None, self
                    length = 0 if history._encoded is None else len(history._encoded)
                    if length < maximum:
                        added = self.event_features(history.events[length:maximum])
                        history._encoded = added if length == 0 else torch.cat((history._encoded, added))
                    memory = history._encoded[:maximum]
                else:
                    memory = self.event_features(history.events[:maximum])
            query = current[indices]
            if memory is None:
                attended = torch.zeros_like(query)
            else:
                scores = self.query(query)@self.key(memory).T/(self.hidden**.5)
                allowed = torch.arange(maximum)[None, :] < torch.tensor(ends)[:, None]
                # Zero-history queries must avoid an all-minus-infinity softmax.
                safe = scores.masked_fill(~allowed, -1e9).softmax(-1)*allowed
                attended = safe@self.value(memory)
            parts = [query, attended]
            if not self.legacy:
                # Full-history attention remains. Add direct access to the last
                # decision of this same actor in this round, excluding this action.
                events_by_actor = {}
                for j, event in enumerate(history.events[:maximum]):
                    if event["obs"][-EXTRA_WIDTH+6] == 0:
                        events_by_actor.setdefault(actor_key(event["obs"]), []).append(j)
                recent = []
                for i, end in zip(indices, ends):
                    past = events_by_actor.get(actor_key(rows[i]["obs"]), ())
                    slot = bisect_left(past, end)-1
                    recent.append(memory[past[slot]] if slot >= 0 else torch.zeros_like(current[i]))
                parts.append(torch.stack(recent))
            merged = self.memory_merge(torch.cat(parts, -1))
            for i, output in zip(indices, merged):
                outputs[i] = output
        return torch.stack(outputs)

    def select(self, char, state, history):
        hints = map_hints(char, state)
        masks, _ = preplant_legal_actions(char, state)
        record = dict(obs=encode_public(char, state, history), hints=hints, masks=masks,
                      phase=sampling_phase(char, state), selections={}, history=history,
                      history_index=len(history.events))
        with torch.no_grad():
            context = self.contexts([record], cache=True)[0]
            logp = 0.
            def draw(kind):
                nonlocal logp
                distribution = self.distribution(context, record, kind)
                action = distribution.sample()
                record["selections"][kind] = int(action)
                logp += float(distribution.log_prob(action))
            draw("mode")
            mode = MODES[record["selections"]["mode"]]
            if mode in ("MOVE", "MOVE_WATCH"):
                draw("position")
            if mode in ("WATCH", "MOVE_WATCH"):
                draw("watch")
            if mode == "ABILITY":
                if char.ability_name == "DANCE": draw("ally")
                elif char.ability_name != "RAMP": draw("target")
            if mode == "ULTIMATE":
                if char.ultimate_name in ("NEON", "ESCAPE"):
                    record["masks"]["target"] = masks["ultimate_target"]
                    draw("target")
                elif char.ultimate_name in ("RAID", "TUNNEL"): draw("watch")
            record.update(logp=logp, value=float(self.critic(context)))
        return decode_command(char, state, record["selections"]), record

    def evaluate_contexts(self, records, contexts):
        logp, entropy = contexts.new_zeros(len(records)), contexts.new_zeros(len(records))
        for kind in ("mode", "position", "watch", "target", "ally"):
            indices = [i for i,r in enumerate(records) if kind in r["selections"]]
            if not indices: continue
            index = torch.tensor(indices)
            batch = dict(phase="standard", obs=np.stack([records[i]["obs"] for i in indices]), hints=np.stack([records[i]["hints"] for i in indices]),
                         masks={kind:np.stack([records[i]["masks"][kind] for i in indices])})
            dist = self.distribution(contexts[index], batch, kind)
            actions = torch.tensor([records[i]["selections"][kind] for i in indices])
            branch = dist.log_prob(actions)
            if kind == "watch":
                options = np.stack([records[i].get("watch_options") if records[i].get("watch_options") is not None
                                    else np.eye(1, dist.probs.shape[-1], int(actions[j]), dtype=bool)[0]
                                    for j,i in enumerate(indices)])
                branch = (dist.probs*torch.from_numpy(options)).sum(-1).clamp_min(1e-12).log()
            logp = logp.index_add(0, index, branch)
            entropy = entropy.index_add(0, index, dist.entropy())
        return logp, entropy, self.critic(contexts).squeeze(-1)

    def evaluate_records(self, records):
        return self.evaluate_contexts(records, self.contexts(records))

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(policy_type="gc3_public_match_history", schema_hash=self.schema_hash, role=self.role,
                        hidden=self.hidden, model_state_dict=self.state_dict(), **metadata), temporary)
        replace_file(temporary, path)

    @classmethod
    def load(cls, path, *, role="attacker", **kwargs):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("policy_type") != "gc3_public_match_history" or payload.get("schema_hash") not in (SCHEMA_HASH, DATA_SCHEMA_HASH):
            raise ValueError("Need a newly trained GC v3 public-history checkpoint; old flat BC/RL cannot resume")
        if payload["role"] != role: raise ValueError("Checkpoint role mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls(role, payload["hidden"], legacy=payload["schema_hash"] == DATA_SCHEMA_HASH)
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload


def load_preplant_policy(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("policy_type") == "gc3_public_match_site_history":
        from .gc3_policy_attacker_site import SiteHistoryPolicy
        return SiteHistoryPolicy.load(path)
    if payload.get("policy_type") == "gc3_public_match_history":
        return HistoryPolicy.load(path)
    # Old checkpoints remain available for explicit inference/comparison only.
    return FreshPolicy.load(path, role="attacker")
