"""Restartable, local plant-boundary cases with shared tensor storage.

These Python snapshots are for this project's code/environment, not an
interchange format. Only load case files produced by a trusted collector.
"""

import gzip
import hashlib
import os
from pathlib import Path
import pickle
import random
import uuid

import cloudpickle
import numpy as np
import torch

from iq_controller_adapter import IQAwareController
from iq_perception import IQPerceptionEngine, PerceivedCharacter, PerceivedGameView
from team_ai import PrivateInfoController
from concon_v1.co1_retake_scenarios import plant_site
from concon_v1.co1_battle_training import _run_from_project_root

CASE_VERSION = 1


def _new_object(cls):
    return object.__new__(cls)


def _restore_state(obj, state):
    # Delegating __getattr__ methods cannot run before their owner is restored.
    if isinstance(obj, PerceivedGameView):
        state = dict(state)
        state["_map"] = {id(real): proxy for real, proxy in state["_map"]}
    for name, value in state.items():
        object.__setattr__(obj, name, value)


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CasePickler(cloudpickle.CloudPickler):
    def __init__(self, handle, tensor_dir):
        super().__init__(handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.tensor_dir = Path(tensor_dir)
        self.tensors = {}

    def persistent_id(self, obj):
        if not isinstance(obj, torch.Tensor):
            return None
        identity = id(obj)
        if identity in self.tensors:
            return self.tensors[identity][1]
        cpu = obj.detach().cpu().contiguous()
        parameter = isinstance(obj, torch.nn.Parameter)
        header = (str(cpu.dtype), tuple(cpu.shape), parameter, obj.requires_grad)
        digest = hashlib.sha256(repr(header).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        key = digest.hexdigest()
        path = self.tensor_dir / (key + ".pt")
        if not path.exists():
            value = (torch.nn.Parameter(cpu, requires_grad=obj.requires_grad) if parameter
                     else cpu.requires_grad_(obj.requires_grad))
            _atomic_write(path, lambda temporary: torch.save(value, temporary))
        token = ("tensor", key, len(self.tensors), str(obj.device))
        # Keep the object alive: a temporary tensor's id must not be reused.
        self.tensors[identity] = obj, token
        return token

    def reducer_override(self, obj):
        if isinstance(obj, (IQAwareController, PrivateInfoController, IQPerceptionEngine)):
            state = dict(vars(obj))
            if isinstance(obj, IQPerceptionEngine):
                # Tick-local caches contain old object ids; observations are
                # rebuilt on the next tick. Defuse memory is retained below.
                state.update(_cache={}, _last_tick=None)
            return _new_object, (type(obj),), state, None, None, _restore_state
        if isinstance(obj, PerceivedCharacter):
            state = {name: object.__getattribute__(obj, name) for name in obj.__slots__}
            return _new_object, (type(obj),), state, None, None, _restore_state
        if isinstance(obj, PerceivedGameView):
            state = {name: object.__getattribute__(obj, name) for name in obj.__slots__}
            mapping = state["_map"]
            state["_map"] = [(char, mapping[id(char)]) for char in state["_real"].chars
                             if id(char) in mapping]
            return _new_object, (type(obj),), state, None, None, _restore_state
        return super().reducer_override(obj)


class CaseUnpickler(pickle.Unpickler):
    def __init__(self, handle, tensor_dir, tensor_cache=None):
        super().__init__(handle)
        self.tensor_dir, self.tensors = Path(tensor_dir), {}
        self.tensor_cache = tensor_cache

    def persistent_load(self, token):
        kind, key, index, device = token
        if kind != "tensor" or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid case tensor reference")
        if index not in self.tensors:
            path = self.tensor_dir / (key + ".pt")
            cache_key = str(path.resolve()), device
            if self.tensor_cache is None:
                value = torch.load(path, map_location=device, weights_only=False)
            else:
                if cache_key not in self.tensor_cache:
                    self.tensor_cache[cache_key] = torch.load(path, map_location=device, weights_only=False)
                template = self.tensor_cache[cache_key]
                value = template.detach().clone()
                if isinstance(template, torch.nn.Parameter):
                    value = torch.nn.Parameter(value, requires_grad=template.requires_grad)
                else:
                    value.requires_grad_(template.requires_grad)
            self.tensors[index] = value
        return self.tensors[index]


def validate_case_game(game):
    if not game.headless or not game.is_planted or game.planted_pos is None:
        raise ValueError("a case requires a headless game with a real planted spike")
    if game.round_over or game.match_over or game.is_defused or game.detonate_timer <= 0:
        raise ValueError("finished rounds cannot be retake start cases")
    if not any(char.team == "D" and char.is_alive for char in game.chars):
        raise ValueError("a case requires a surviving defender")
    if game.defender_setup_phase.active:
        raise ValueError("setup must finish before capturing a retake case")


def case_metadata(game, opponent):
    validate_case_game(game)
    def actor(char):
        return dict(name=char.name, team=char.team, alive=bool(char.is_alive),
                    pos=list(map(int, char.pos)), hp=float(char.hp), facing=char.facing,
                    ability=getattr(char, "ability_name", ""),
                    charges={kind: int(getattr(char, kind + "_charges", 0))
                             for kind in ("smoke", "flash", "recon")})
    return dict(version=CASE_VERSION, opponent=opponent,
                site=plant_site(game.planted_pos, game.grid),
                planted_pos=list(map(int, game.planted_pos)), battle_tick=int(game.battle_tick),
                detonate_timer=float(game.detonate_timer),
                attacker_alive=sum(char.team == "A" and char.is_alive for char in game.chars),
                defender_alive=sum(char.team == "D" and char.is_alive for char in game.chars),
                actors=[actor(char) for char in game.chars])


def _identity_objects(game):
    objects = [game, *game.chars, *getattr(game, "smokes", ())]
    for wrapper in (game.attacker_controller, game.defender_controller):
        controller = vars(wrapper).get("inner", wrapper)
        owner = vars(controller).get("game")
        if owner is not None:
            objects.append(owner)
        smoke_recon = vars(controller).get("smoke_recon")
        if smoke_recon is not None:
            objects.extend(smoke_recon.smokes.values())
    return [(id(obj), obj) for obj in objects]


def _restore_opponent_ids(game, identities):
    # Fnatic's confirmed smoke-use/history records use Python object ids.
    # Keeping their old integers would cause a fresh recon or a history reset.
    for wrapper in (game.attacker_controller, game.defender_controller):
        controller = vars(wrapper).get("inner", wrapper)
        smoke_recon = vars(controller).get("smoke_recon")
        if smoke_recon is not None:
            smoke_recon.smokes = {id(smoke): smoke for smoke in smoke_recon.smokes.values()}
            smoke_recon.completed = {name: {identities.get(key, key) for key in done}
                                     for name, done in smoke_recon.completed.items()}
            smoke_recon.pending = {name: (identities.get(row[0], row[0]), *row[1:])
                                   for name, row in smoke_recon.pending.items()}
        gate = vars(controller).get("recon_gate")
        if gate is not None:
            def keys(conditions):
                return {("smoke", identities.get(key[1], key[1])) if key[0] == "smoke" else key
                        for key in conditions}
            gate.used = {name: keys(conditions) for name, conditions in gate.used.items()}
            gate.pending = {name: (*row[:2], keys(row[2])) for name, row in gate.pending.items()}
        history = vars(controller).get("opponent_history")
        if history is not None:
            for name in ("scope", "round_key"):
                value = getattr(history, name)
                if value is not None:
                    setattr(history, name, (identities.get(value[0], value[0]), *value[1:]))


def save_case(path, game, metadata):
    """Save the full object graph; immutable tensor contents are deduplicated."""
    validate_case_game(game)
    payload = dict(version=CASE_VERSION, game=game, metadata=metadata,
                   identity_objects=_identity_objects(game),
                   random_state=random.getstate(), numpy_state=np.random.get_state(),
                   torch_state=torch.get_rng_state(),
                   cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    path = Path(path)
    def write(temporary):
        with gzip.open(temporary, "wb") as handle:
            CasePickler(handle, path.parent / "tensors").dump(payload)
    _atomic_write(path, write)


def load_case(path, *, restore_rng=True, tensor_cache=None):
    """Return (game, metadata) without init_round/reset_round or model reloads.

    A future learner can replace the defender controller after loading while
    leaving the attacker's controller and both teams' perception state intact.
    Set restore_rng=False when the learner owns randomness across samples.
    """
    return _load_case(Path(path).resolve(), restore_rng=restore_rng, tensor_cache=tensor_cache)


@_run_from_project_root
def _load_case(path, *, restore_rng, tensor_cache):
    # Opponent imports contain legacy paths relative to the game's root. Resolve
    # the user's case path first, then perform imports/deserialization there.
    with gzip.open(path, "rb") as handle:
        payload = CaseUnpickler(handle, path.parent / "tensors", tensor_cache).load()
    if payload.get("version") != CASE_VERSION:
        raise ValueError("unsupported retake case version")
    game = payload["game"]
    identities = {old: id(obj) for old, obj in payload["identity_objects"]}
    for team in (game.current_attacker_team_ai, game.current_defender_team_ai):
        engine = team.perception_engine
        engine._defuse_touched_viewers = {identities[old] for old in engine._defuse_touched_viewers
                                        if old in identities}
    _restore_opponent_ids(game, identities)
    validate_case_game(game)
    if restore_rng:
        random.setstate(payload["random_state"])
        np.random.set_state(payload["numpy_state"])
        torch.set_rng_state(payload["torch_state"])
        if payload["cuda_states"]:
            torch.cuda.set_rng_state_all(payload["cuda_states"])
    return game, payload["metadata"]
