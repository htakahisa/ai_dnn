"""Separate guard policy; the preserved v2 policy handles all preplant actions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from grid_lines import line_cells
from gc_v1.gc_facing import facing_towards
from ghost_champions_v2.rl.controller import LearnedAttackerController
from ghost_champions_v2.rl.actions import candidates as v2_candidates, constrain_proposal
from ghost_champions_v2.rl.observation import OBS_DIM as V2_DIM, SCHEMA_HASH as V2_SCHEMA, proposal_kind, public_defuse_features
from ghost_champions_v2.rl.unit_observation import action_capabilities
from ghost_champions_v2.geometry import los, route, recon_aim, watch_cells, projectile_path
from .gc3_map_attacker_guard import Post
from . import gc3_map_attacker_postplant as candidate_map
from .gc3_spatial_policy import SpatialCandidatePolicy, POSTPLANT_ARRIVAL_RADIUS, SPATIAL_SCHEMA

ACTIONS = ("KEEP_MOVE", "KEEP_HOLD", "KEEP_ABILITY", "KEEP_OBJECTIVE",
           "UP", "DOWN", "LEFT", "RIGHT", "FACE_N", "FACE_NE", "FACE_E", "FACE_SE",
           "FACE_S", "FACE_SW", "FACE_W", "FACE_NW", "RECON_A", "RECON_Mid", "RECON_B",
           "FLASH", "DANCE", "SMOKE_LANE", "GUARD_POSITION", "PREAIM", "RECON_LANE", "ASH", "ULTIMATE")
OBS_DIM = V2_DIM + 10
SCHEMA_HASH = hashlib.sha256(json.dumps([V2_SCHEMA, ACTIONS, SPATIAL_SCHEMA, "soft-arrival-v2"]).encode()).hexdigest()


class GuardPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.features = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(ACTIONS))
        self.critic = nn.Linear(hidden, 1)
        self.spatial = SpatialCandidatePolicy()

    def forward(self, obs, mask):
        features = self.features(obs)
        return torch.distributions.Categorical(logits=self.actor(features).masked_fill(~mask, -1e9)), self.critic(features).squeeze(-1)

    @classmethod
    def from_v2(cls, path):
        from ghost_champions_v2.rl.policy import ResidualPolicy
        baseline, _ = ResidualPolicy.load(path)
        model = cls(baseline.hidden)
        with torch.no_grad():
            model.features[0].weight.zero_()
            model.features[0].weight[:, :V2_DIM].copy_(baseline.features[0].weight)
            model.features[0].bias.copy_(baseline.features[0].bias)
            model.features[2].load_state_dict(baseline.features[2].state_dict())
            model.actor.weight[:21].copy_(baseline.actor.weight)
            model.actor.bias[:21].copy_(baseline.actor.bias)
        return model

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(policy_type="gc_v3_guard", schema_hash=SCHEMA_HASH, actions=ACTIONS,
                        hidden=self.hidden, model_state_dict=self.state_dict(), **metadata), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_hash") != SCHEMA_HASH or tuple(payload.get("actions", ())) != ACTIONS:
            raise ValueError("GC v3 guard checkpoint schema mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls(payload["hidden"])
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload


class GuardPlan:
    def __init__(self, spatial_policy=None):
        self.spatial_policy = spatial_policy if spatial_policy is not None else SpatialCandidatePolicy()
        self.reset()

    def reset(self):
        self.key = None
        self.posts = {}
        self.cast_ticks = {}
        self.lane_casts = {}
        self.pending_choices = {}

    def assign(self, char, state, tactics, spike):
        allies = tactics.allies(char, state)
        key = spike, tuple(str(c.name) for c in allies), candidate_map.NEW_MAZE_STR
        if self.key != key:
            self.key = key
            self.posts = {}
            self.pending_choices = {}
        name = str(char.name)
        if name not in self.posts:
            occupied = tactics.occupied(char, state) | {p.position for p in self.posts.values()}
            point, position_record = self.spatial_policy.sample(char, state, candidate_map.NEW_MAZE_STR,
                kind="position", phase="postplant", blocked=occupied, center=spike, radius=10)
            aim, watch_record = self.spatial_policy.sample(char, state, candidate_map.NEW_MAZE_STR,
                kind="watch", phase="postplant", origin=point)
            self.posts[name] = Post("sampled_guard", point, aim)
            self.pending_choices[name] = [position_record, watch_record]
        return self.posts[str(char.name)]

    def choices(self, char, state, tactics):
        spike = tactics.planted_position(state)
        if spike is None:
            raise ValueError("Guard phase requires a publicly known plant position")
        post = self.assign(char, state, tactics, spike)
        occupied = tactics.occupied(char, state)
        grid, smoke = state["grid"], state.get("smoke_cells", ())
        enemies = tactics.enemies(char, state)
        visible = [e for e in enemies if los(grid, tuple(char.pos), tuple(e.pos),
                   () if getattr(e, "reveal_remaining", 0) or getattr(char, "sees_through_smoke", False) else smoke, occupied)]
        taps = public_defuse_features(state)[0] > 0
        target = spike if taps else tuple(min(visible, key=lambda e: math.dist(e.pos, spike)).pos) if visible else post.aim
        caps = action_capabilities(char, state)
        def hold(aim):
            if not caps["can_turn"]:
                return list(char.pos), "HOLD"
            direction = facing_towards(char.pos, aim) or getattr(char, "facing", "N")
            return list(char.pos), {"facing": direction}
        # Arrival anywhere in the sampled region is acceptable. Do not correct
        # a player back to the exact hint cell every tick.
        goals = [tuple(map(int, p)) for p in np.argwhere(grid != 1)
                 if math.dist(p, post.position) <= POSTPLANT_ARRIVAL_RADIUS]
        path = route(grid, tuple(char.pos), goals, occupied, spike, 10)
        position = tactics.move_action(char, path[min(1, len(path)-1)]) if len(path) > 1 else hold(target)
        response = position
        if taps:
            goals = [p for p in watch_cells(grid, spike) if p != spike]
            response_path = route(grid, tuple(char.pos), goals, occupied, spike, 10)
            response = tactics.move_action(char, response_path[min(1, len(response_path)-1)]) if len(response_path) > 1 else hold(spike)
        proposal = hold(target) if visible or taps else position
        choices, _ = v2_candidates(char, state, tactics, proposal)
        choices += [None] * (len(ACTIONS)-len(choices))
        choices[22] = constrain_proposal(char, state, position)
        choices[23] = constrain_proposal(char, state, hold(target))
        lanes = tuple(candidate_map.get_gc_postplant_watch_candidates()) or (spike,)
        name = str(char.name)
        ready = caps["can_use_ability"] and tactics.tick-self.cast_ticks.get(name, -20) >= 5
        if ready:
            if getattr(char, "ability_name", None) == "RECON" and char.recon_charges > 0:
                lane = min(lanes, key=lambda p: (self.lane_casts.get(p, -100), math.dist(p, char.pos)))
                aim = recon_aim(grid, tuple(char.pos), "A" if spike[1] < 20 else "B", lane)
                if aim:
                    choices[24] = list(char.pos), {"ability": "RECON", "target": aim}
            choices[19] = tactics._flash(char, state, visible, watch_cells(grid, spike) if taps else lanes)
            choices[20] = tactics._heal(char, state)
            if getattr(char, "ability_name", None) == "SMOKE" and char.smoke_charges > 0:
                # AbilityLosMixin casts a 3x3 smoke in the current engine.
                smoke_half_width = 1
                protected = {p for hold_post in self.posts.values() for q in watch_cells(grid, spike)
                             if los(grid, hold_post.position, q) for p in line_cells(hold_post.position, q)}
                for lane in lanes:
                    cells = {(lane[0]+dr, lane[1]+dc) for dr in range(-smoke_half_width, smoke_half_width+1)
                             for dc in range(-smoke_half_width, smoke_half_width+1)}
                    if not cells.intersection(protected) and not cells.intersection(occupied | {tuple(char.pos)}):
                        choices[21] = list(char.pos), {"ability": "SMOKE", "target": lane}
                        break
            if getattr(char, "ability_name", None) == "ASH" and char.ash_charges > 0 and visible:
                from game_core import ASH_RANGE_CELLS
                target_enemy = min(visible, key=lambda e: math.dist(e.pos, char.pos))
                if math.dist(target_enemy.pos, char.pos) <= ASH_RANGE_CELLS:
                    choices[25] = list(char.pos), {"ability": "ASH", "target": tuple(target_enemy.pos)}
        else:
            choices[19] = choices[20] = None
        if caps["can_use_ultimate"]:
            from gc_v1.ultimate_tactics_gc import ultimate_ready, build_ultimate_action
            if ultimate_ready(char):
                command = build_ultimate_action(grid, char, [*tactics.allies(char, state), *enemies], target)
                if command:
                    choices[26] = list(char.pos), command
        # Teacher commits only the returned action, never rejected proposals.
        if taps:
            proposal = choices[19] or response
        elif visible:
            proposal = choices[25] or choices[19] or hold(target)
        else:
            proposal = choices[20] or choices[24] or choices[21] or position
        dodge = tactics.dodge(char, state)
        proposal = dodge or proposal
        proposal = constrain_proposal(char, state, tactics.avoid_entry(char, state, proposal))
        kind = proposal_kind(proposal, char.pos)
        choices[kind] = proposal
        if not caps["can_act"] or dodge is not None:
            choices = [proposal if i == kind else None for i in range(len(ACTIONS))]
        mask = np.array([a is not None for a in choices], dtype=np.bool_)
        return choices, mask, proposal, kind, post, target

    def commit(self, char, action, tactics, spike):
        command = action[1] if isinstance(action, tuple) and len(action) > 1 else None
        if isinstance(command, dict) and command.get("ability"):
            self.cast_ticks[str(char.name)] = tactics.tick
            if command["ability"] == "RECON":
                end = projectile_path(self.grid, tuple(char.pos), command["target"])[-1]
                lane = min(candidate_map.get_gc_postplant_watch_candidates() or [spike], key=lambda p: math.dist(p, end))
                self.lane_casts[lane] = tactics.tick


class GhostChampionsV3AttackerController(LearnedAttackerController):
    def __init__(self, *, guard_policy=None, guard_checkpoint=None, guard_mode="stochastic", guard_recorder=None, **kwargs):
        self.guard_policy = guard_policy
        self.guard_checkpoint = Path(guard_checkpoint).resolve() if guard_checkpoint else None
        if self.guard_policy is None and self.guard_checkpoint is not None:
            self.guard_policy, _ = GuardPolicy.load(self.guard_checkpoint)
        if guard_mode not in {"teacher", "stochastic", "greedy"}:
            raise ValueError("Unknown guard mode")
        # Historical 'greedy' callers also use the sampled deployment policy.
        self.guard_mode = "teacher" if guard_mode == "teacher" else "stochastic"
        self.guard_recorder = guard_recorder
        self.guard_plan = GuardPlan(self.guard_policy.spatial if self.guard_policy is not None else None)
        super().__init__(**kwargs)

    def reset_round(self):
        super().reset_round()
        self.guard_plan.reset()

    def decide_move(self, char, state):
        if not state.get("is_planted") or state.get("defender_setup_active") or not char.is_alive:
            return super().decide_move(char, state)
        self.tactics.observe(char, state)
        choices, mask, proposal, teacher, post, target = self.guard_plan.choices(char, state, self.tactics)
        self.guard_plan.grid = state["grid"]
        base = self.encoder.encode(char, state, self.tactics, proposal)
        spike = self.tactics.planted_position(state)
        extra = [*(np.array(post.position)-char.pos)/44, *(np.array(target)-char.pos)/44,
                 math.dist(char.pos, post.position)/20, float(math.dist(char.pos, post.position) <= POSTPLANT_ARRIVAL_RADIUS),
                 float(getattr(char, "facing", None) == facing_towards(char.pos, target)),
                 float(any(los(state["grid"], tuple(char.pos), p, state.get("smoke_cells", ())) for p in watch_cells(state["grid"], spike))),
                 min(1., (self.tactics.tick-self.guard_plan.cast_ticks.get(str(char.name), -20))/20),
                 float(spike[1] >= 20)]
        obs = np.concatenate((base, np.array(extra, dtype=np.float32))).astype(np.float32)
        selected, logp, value = teacher, 0., 0.
        if self.guard_policy is not None and self.guard_mode != "teacher":
            with torch.no_grad():
                dist, val = self.guard_policy(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                selected = int(dist.sample().item())
                logp, value = float(dist.log_prob(torch.tensor([selected])).item()), float(val.item())
        result = choices[selected]
        self.guard_plan.commit(char, result, self.tactics, spike)
        self.tactics.remember_result(char, result)
        self.last_policy_action = ACTIONS[selected]
        if self.guard_recorder is not None:
            self.guard_recorder.record(str(char.name), obs, mask, selected, logp, value, teacher)
            if self.guard_mode != "teacher":
                spatial = self.guard_plan.pending_choices.pop(str(char.name), [])
                self.guard_recorder.rows[-1]["spatial_choices"] = spatial
                self.guard_recorder.rows[-1]["logp"] += sum(item["logp"] for item in spatial)
        return result

    def attacker_snapshot(self):
        snapshot = super().attacker_snapshot()
        snapshot["guard_v3"] = dict(checkpoint=str(self.guard_checkpoint) if self.guard_checkpoint else None,
                                    mode=self.guard_mode, action=self.last_policy_action,
                                    posts={name: dict(position=p.position, aim=p.aim) for name, p in self.guard_plan.posts.items()})
        return snapshot
