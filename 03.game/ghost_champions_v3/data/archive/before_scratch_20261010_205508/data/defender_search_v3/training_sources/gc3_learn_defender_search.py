"""Stochastic preplant defense with jointly learned deployment and watch choices."""
import hashlib
from pathlib import Path
import math
import numpy as np
import torch
from gc_v1.gc_facing import facing_towards
from ghost_champions_v2.config import load_config
from ghost_champions_v2.tactics import AttackerTactics
from ghost_champions_v2.geometry import route, los, recon_aim
from ghost_champions_v2.rl.observation import ObservationEncoder, proposal_kind
from ghost_champions_v2.rl.actions import candidates, constrain_proposal
from ghost_champions_v2.rl.unit_observation import action_capabilities
from .gc3_learn_attacker_guard import GuardPolicy, ACTIONS, SCHEMA_HASH as GUARD_SCHEMA
from .gc3_learn_defender_initial import GhostChampionsV3DefenderController
from .gc3_spatial_policy import DEFENDER_ARRIVAL_RADIUS, sampled_distribution
from . import gc3_map_defender_initial as candidate_map

SCHEMA_HASH = hashlib.sha256((GUARD_SCHEMA + "defender-search-v1").encode()).hexdigest()
SEARCH_TEMPERATURE = 1.2
SEARCH_UNIFORM_MIX = .05


class DefenderSearchPolicy(GuardPolicy):
    def forward(self, obs, mask):
        features = self.features(obs)
        return sampled_distribution(self.actor(features), mask,
                    temperature=SEARCH_TEMPERATURE, uniform_mix=SEARCH_UNIFORM_MIX), self.critic(features).squeeze(-1)

    @classmethod
    def from_v2(cls, path):
        # The v2 defender supplies demonstrations, rather than attacker weights.
        return cls()

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(policy_type="gc_v3_defender_search", schema_hash=SCHEMA_HASH,
                        actions=ACTIONS, hidden=self.hidden, model_state_dict=self.state_dict(), **metadata), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_hash") != SCHEMA_HASH or tuple(payload.get("actions", ())) != ACTIONS:
            raise ValueError("GC v3 defender search checkpoint schema mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls(payload["hidden"])
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload


class GhostChampionsV3SearchController(GhostChampionsV3DefenderController):
    requires_public_effects = True

    def __init__(self, *, search_policy=None, search_checkpoint=None, search_mode="stochastic", search_recorder=None, **kwargs):
        self.search_policy = search_policy
        if search_policy is None and search_checkpoint:
            self.search_policy, _ = DefenderSearchPolicy.load(search_checkpoint)
        self.search_checkpoint = str(search_checkpoint) if search_checkpoint else None
        self.search_mode = search_mode
        self.search_recorder = search_recorder
        self.search_encoder = ObservationEncoder()
        self.search_tactics = AttackerTactics(load_config())
        self.search_recorded_initial = set()
        self.search_cast_ticks = {}
        super().__init__(initial_policy=self.search_policy.spatial if self.search_policy is not None else None, **kwargs)

    def reset_round(self):
        super().reset_round()
        self.search_encoder.reset()
        self.search_tactics.reset_round()
        self.search_recorded_initial.clear()
        self.search_cast_ticks.clear()

    def decide_move(self, char, state):
        if state.get("defender_setup_active") or state.get("is_planted") or not char.is_alive:
            return super().decide_move(char, state)
        tactics = self.search_tactics
        tactics.observe(char, state)
        name = str(char.name)
        grid = state["grid"]
        if name not in self.initial_goals:
            point, position = self.initial_policy.sample(char, state, candidate_map.DEFENDER_SETUP_POSITION_MAP_GC_STR,
                kind="position", phase="defender", blocked=tactics.occupied(char, state) | set(self.initial_goals.values()))
            aim, watch = self.initial_policy.sample(char, state, candidate_map.DEFENDER_SETUP_POSITION_MAP_GC_STR,
                kind="watch", phase="defender", origin=point)
            self.initial_goals[name], self.initial_aims[name] = point, aim
            self.initial_records[name] = [position, watch]
        goal, aim = self.initial_goals[name], self.initial_aims[name]
        enemies = tactics.enemies(char, state)
        visible = [e for e in enemies if los(grid, tuple(char.pos), tuple(e.pos), state.get("smoke_cells", ()))]
        target = tuple(min(visible, key=lambda e: math.dist(e.pos, char.pos)).pos) if visible else aim
        caps = action_capabilities(char, state)
        hold = (list(char.pos), {"facing": facing_towards(char.pos, target) or char.facing}) if caps["can_turn"] else (list(char.pos), "HOLD")
        region = [tuple(map(int, p)) for p in np.argwhere(grid != 1) if math.dist(p, goal) <= DEFENDER_ARRIVAL_RADIUS]
        path = route(grid, tuple(char.pos), region, tactics.occupied(char, state))
        position = tactics.move_action(char, path[1]) if len(path) > 1 else hold
        # Preserve the existing learned combat/search proposal when contact is public.
        proposal = super().decide_move(char, state) if visible or state.get("spike_pos") else position
        choices, _ = candidates(char, state, tactics, proposal)
        choices += [None] * (len(ACTIONS)-len(choices))
        choices[22], choices[23] = constrain_proposal(char, state, position), hold
        ready = caps["can_use_ability"] and tactics.tick-self.search_cast_ticks.get(name, -20) >= 5
        if ready:
            lanes = candidate_map.get_gc_setup_watch_candidates() or [aim]
            if char.ability_name == "RECON" and char.recon_charges > 0:
                lane = min(lanes, key=lambda p: math.dist(p, char.pos))
                cast = recon_aim(grid, tuple(char.pos), "A" if lane[1] < 20 else "B", lane)
                if cast:
                    choices[24] = (list(char.pos), {"ability": "RECON", "target": cast})
            if char.ability_name == "SMOKE" and char.smoke_charges > 0 and visible:
                choices[21] = (list(char.pos), {"ability": "SMOKE", "target": target})
            if char.ability_name == "ASH" and char.ash_charges > 0 and visible:
                from game_core import ASH_RANGE_CELLS
                if math.dist(char.pos, target) <= ASH_RANGE_CELLS:
                    choices[25] = (list(char.pos), {"ability": "ASH", "target": target})
        else:
            choices[16:21] = [None]*5
        if caps["can_use_ultimate"]:
            from gc_v1.ultimate_tactics_gc import ultimate_ready, build_ultimate_action
            if ultimate_ready(char):
                command = build_ultimate_action(grid, char, state.get("chars", ()), target)
                if command:
                    choices[26] = (list(char.pos), command)
        if not visible and not state.get("spike_pos"):
            proposal = choices[20] or choices[24] or position
        dodge = tactics.dodge(char, state)
        proposal = constrain_proposal(char, state, tactics.avoid_entry(char, state, dodge or proposal))
        teacher = proposal_kind(proposal, char.pos)
        choices[teacher] = proposal
        if not caps["can_act"] or dodge is not None:
            choices = [proposal if i == teacher else None for i in range(len(ACTIONS))]
        mask = np.array([c is not None for c in choices], dtype=np.bool_)
        base = self.search_encoder.encode(char, state, tactics, proposal)
        extra = [*(np.array(goal)-char.pos)/44, *(np.array(target)-char.pos)/44,
                 math.dist(char.pos, goal)/20, float(math.dist(char.pos, goal) <= DEFENDER_ARRIVAL_RADIUS),
                 float(char.facing == facing_towards(char.pos, target)), float(bool(visible)),
                 min(1., (tactics.tick-self.search_cast_ticks.get(name, -20))/20), float(goal[1] >= 20)]
        obs = np.concatenate((base, np.array(extra, dtype=np.float32))).astype(np.float32)
        selected, logp, value = teacher, 0., 0.
        if self.search_policy is not None and self.search_mode != "teacher":
            with torch.no_grad():
                dist, val = self.search_policy(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                selected = int(dist.sample().item())
                logp, value = float(dist.log_prob(torch.tensor([selected])).item()), float(val.item())
        result = choices[selected]
        command = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if isinstance(command, dict) and command.get("ability"):
            self.search_cast_ticks[name] = tactics.tick
        tactics.remember_result(char, result)
        self.macro.phase, self.macro.goals = "v3_search", dict(self.initial_goals)
        if self.search_recorder is not None:
            self.search_recorder.record(name, obs, mask, selected, logp, value, teacher)
            if self.search_mode != "teacher" and name not in self.search_recorded_initial:
                records = self.initial_records.get(name, [])
                row = self.search_recorder.rows[-1]
                row["spatial_choices"] = records
                row["logp"] += sum(record["logp"] for record in records)
                self.search_recorded_initial.add(name)
        return result

    def defender_snapshot(self):
        snapshot = super().defender_snapshot()
        snapshot["search_v3"] = dict(checkpoint=self.search_checkpoint, selection="sampled", mode=self.search_mode)
        return snapshot
