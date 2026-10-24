"""Fresh actor/critic with learned movement, watch and ability-target distributions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .gc3_engine_actions import MODES, legal_actions, decode_command
from .gc3_observation import OBS_DIM, CELLS, BOARD_CHANNELS, CONTEXT_WIDTH, encode, map_hints, sampling_phase
from .gc3_spatial_policy import SpatialCandidatePolicy, sampled_distribution, phase_settings, SPATIAL_SCHEMA
from .gc3_io import replace_file

HIDDEN_WIDTH = 128
LEGACY_SCHEMA_HASH = hashlib.sha256(json.dumps(["gc3-independent-scratch-v1", OBS_DIM, CELLS, MODES, SPATIAL_SCHEMA]).encode()).hexdigest()
SCHEMA_HASH = hashlib.sha256(json.dumps(["gc3-split-layernorm-silu-v2", OBS_DIM, CELLS, MODES, SPATIAL_SCHEMA]).encode()).hexdigest()


class ObservationFeatures(nn.Module):
    """Give the small changing state its own normalized branch beside the map."""
    def __init__(self, hidden):
        super().__init__()
        self.board_end = CONTEXT_WIDTH+BOARD_CHANNELS*CELLS
        self.board = nn.Sequential(nn.Linear(BOARD_CHANNELS*CELLS, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.state = nn.Sequential(nn.Linear(OBS_DIM-BOARD_CHANNELS*CELLS, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.merge = nn.Sequential(nn.Linear(2*hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())

    def forward(self, observation):
        board = observation[..., CONTEXT_WIDTH:self.board_end]
        state = torch.cat((observation[..., :CONTEXT_WIDTH], observation[..., self.board_end:]), dim=-1)
        return self.merge(torch.cat((self.board(board), self.state(state)), dim=-1))


class FreshPolicy(nn.Module):
    def __init__(self, role, hidden=HIDDEN_WIDTH, *, legacy=False):
        super().__init__()
        if role not in ("attacker", "defender"):
            raise ValueError("Unknown GC v3 policy role")
        self.role, self.hidden = role, hidden
        self.schema_hash = LEGACY_SCHEMA_HASH if legacy else SCHEMA_HASH
        self.features = (nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
                         if legacy else ObservationFeatures(hidden))
        self.mode = nn.Linear(hidden, len(MODES))
        self.ally = nn.Linear(hidden, 5)
        self.critic = nn.Linear(hidden, 1)
        self.spatial = SpatialCandidatePolicy(hidden, CELLS)

    @classmethod
    def fresh(cls, role, seed):
        # Initialization neither loads another model nor consumes match RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(role)
        return model

    def distribution(self, context, record, kind):
        temperature, mix, _, _ = phase_settings(record["phase"])
        if kind in ("mode", "ally"):
            return sampled_distribution(getattr(self, kind)(context), record["masks"][kind],
                                        temperature=temperature, uniform_mix=mix)
        return self.spatial.distribution(context, record["masks"][kind], kind=kind,
                                         phase=record["phase"], hints=record["hints"])

    def select(self, char, state):
        hints = map_hints(char, state)
        masks, caps = legal_actions(char, state)
        record = dict(obs=encode(char, state, hints), hints=hints, masks=masks,
                      phase=sampling_phase(char, state), selections={})
        with torch.no_grad():
            context = self.features(torch.from_numpy(record["obs"]))
            logp = 0.
            def draw(kind):
                nonlocal logp
                dist = self.distribution(context, record, kind)
                action = dist.sample()
                record["selections"][kind] = int(action.item())
                logp += float(dist.log_prob(action).item())
            draw("mode")
            mode = MODES[record["selections"]["mode"]]
            if mode in ("MOVE", "MOVE_WATCH"):
                draw("position")
            if mode in ("WATCH", "MOVE_WATCH"):
                draw("watch")
            if mode == "ABILITY":
                if char.ability_name == "DANCE":
                    draw("ally")
                elif char.ability_name != "RAMP":
                    draw("target")
            if mode == "ULTIMATE":
                if char.ultimate_name in ("NEON", "ESCAPE"):
                    record["masks"]["target"] = record["masks"]["ultimate_target"]
                    draw("target")
                elif char.ultimate_name in ("RAID", "TUNNEL"):
                    draw("watch")
            record["logp"] = logp
            record["value"] = float(self.critic(context).item())
        return decode_command(char, state, record["selections"]), record

    def evaluate_record(self, record):
        context = self.features(torch.as_tensor(record["obs"], dtype=torch.float32))
        logp, entropy = context.new_zeros(()), context.new_zeros(())
        for kind, action in record["selections"].items():
            dist = self.distribution(context, record, kind)
            logp = logp+dist.log_prob(torch.tensor(action))
            entropy = entropy+dist.entropy()
        return logp, entropy, self.critic(context).squeeze(-1)

    def evaluate_records(self, records):
        context = self.features(torch.from_numpy(np.stack([r["obs"] for r in records])))
        logp, entropy = context.new_zeros(len(records)), context.new_zeros(len(records))
        for phase in ("standard", "defender", "postplant"):
            for kind in ("mode", "position", "watch", "target", "ally"):
                indices = [i for i, r in enumerate(records) if r["phase"] == phase and kind in r["selections"]]
                if not indices:
                    continue
                index = torch.tensor(indices)
                batch = dict(phase=phase, hints=np.stack([records[i]["hints"] for i in indices]),
                             masks={kind: np.stack([records[i]["masks"][kind] for i in indices])})
                dist = self.distribution(context[index], batch, kind)
                actions = torch.tensor([records[i]["selections"][kind] for i in indices])
                logp = logp.index_add(0, index, dist.log_prob(actions))
                entropy = entropy.index_add(0, index, dist.entropy())
        return logp, entropy, self.critic(context).squeeze(-1)

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(policy_type="gc3_independent_scratch", schema_hash=self.schema_hash,
                        role=self.role, hidden=self.hidden, model_state_dict=self.state_dict(), **metadata), temporary)
        replace_file(temporary, path)

    @classmethod
    def load(cls, path, *, role=None, allow_legacy=False):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        legacy = payload.get("schema_hash") == LEGACY_SCHEMA_HASH
        if payload.get("policy_type") != "gc3_independent_scratch" or not (
                payload.get("schema_hash") == SCHEMA_HASH or (allow_legacy and legacy)):
            raise ValueError("Checkpoint is not a fresh-architecture GC v3 model")
        if role is not None and role != payload["role"]:
            raise ValueError("GC v3 checkpoint role mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls(payload["role"], payload["hidden"], legacy=legacy)
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload


class ScratchController:
    requires_public_effects = True

    def __init__(self, *, role, seed, policy=None, checkpoint=None, recorder=None, record_phase=None):
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.policy = policy if policy is not None else FreshPolicy.load(checkpoint, role=role, allow_legacy=True)[0] if checkpoint else FreshPolicy.fresh(role, seed)
        if self.policy.role != role:
            raise ValueError("GC v3 controller policy role mismatch")
        self.role, self.recorder, self.record_phase = role, recorder, record_phase
        self.last_selections = {}

    def set_game(self, game):
        # Keep only the perception view supplied by the game interface.
        self.game = game

    def reset_round(self):
        self.last_selections = {}

    def decide_move(self, char, state):
        if not char.is_alive:
            return list(char.pos), "HOLD"
        result, record = self.policy.select(char, state)
        self.last_selections[str(char.name)] = dict(record["selections"])
        should_record = self.record_phase is None or (
            self.record_phase == "search" and not state.get("is_planted")) or (
            self.record_phase == "postplant" and state.get("is_planted") and not state.get("defender_setup_active"))
        if self.recorder is not None and should_record:
            self.recorder.record(str(char.name), record)
        return result

    def snapshot(self):
        return dict(controller=type(self).__name__, initialization="scratch", checkpoint=self.checkpoint,
                    selection="sampled", decisions=dict(self.last_selections))

    def attacker_snapshot(self):
        return self.snapshot()

    def defender_snapshot(self):
        return self.snapshot()
