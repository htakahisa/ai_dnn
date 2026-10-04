"""Joint FRC actor and a training-only centralized critic."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from frc_v1 import FACING
from frc_v1.actions import KINDS, PHASES, INTENTS, FrcAction, TeamDecision, target_required, ability_for
from frc_v1.baseline import plant_sites
from frc_v1.observation import (GRID_FIELDS, VECTOR_FIELDS, TOKEN_FIELDS, ALLY_FIELDS, GLOBAL_FIELDS,
                                metadata, validate_metadata)

CRITIC_SIZE = 100
TEAM_SIZES = (2, len(PHASES), 5, 5, 5) + (len(INTENTS),) * 5


class FrcActorCritic(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.cells = shape[0] * shape[1]
        self.grid_encoder = nn.Sequential(nn.Conv2d(len(GRID_FIELDS), 24, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.token_encoder = nn.Sequential(nn.Linear(len(TOKEN_FIELDS), 48), nn.ReLU())
        self.encoder = nn.Sequential(nn.Linear(32 * 16 + 48 + len(VECTOR_FIELDS), 128), nn.ReLU())
        self.team_head = nn.Linear(128, sum(TEAM_SIZES))
        self.role_heads = nn.ModuleList(nn.Sequential(nn.Linear(128 + sum(TEAM_SIZES), 128), nn.ReLU(),
            nn.Linear(128, len(KINDS) + 8 + 2 * self.cells)) for _ in range(5))
        self.critic = nn.Sequential(nn.Linear(128 + CRITIC_SIZE, 128), nn.ReLU(), nn.Linear(128, 1))

    def features(self, observations, *, runtime=False):
        device = next(self.parameters()).device
        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.stack(values), dtype=dtype, device=device)
        grid = tensor([o.grid for o in observations])
        vector = tensor([o.vector for o in observations])
        if runtime and not any(o.token_mask.any() for o in observations):
            tokens = torch.zeros((len(observations), 48), dtype=grid.dtype, device=device)
        else:
            tokens = self.token_encoder(tensor([o.tokens for o in observations]))
            presence = tensor([o.token_mask for o in observations]).unsqueeze(-1)
            tokens = (tokens * presence).sum(1) / presence.sum(1).clamp(min=1)
        return self.encoder(torch.cat((self.grid_encoder(grid), vector, tokens), dim=1))

    def distribution(self, observations, *, records=None, deterministic=False, critic=None, collect_statistics=True):
        feature = self.features(observations)
        device, batch = feature.device, len(observations)
        log_prob = torch.zeros(batch, device=device)
        entropy = torch.zeros(batch, device=device)

        def draw(logits, mask, fixed=None):
            nonlocal log_prob, entropy
            if deterministic and not collect_statistics and device.type == "cpu":
                legal = np.asarray(mask, dtype=bool)
                if not legal.any(axis=1).all():
                    raise ValueError("empty FRC action mask")
                chosen = (np.asarray(fixed, dtype=np.int64) if fixed is not None else
                          np.where(legal, logits.detach().numpy(), -1e9).argmax(axis=1))
                return torch.from_numpy(chosen)
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if not mask.any(dim=1).all():
                raise ValueError("empty FRC action mask")
            masked = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=masked) if collect_statistics or not deterministic else None
            chosen = (torch.as_tensor(fixed, dtype=torch.long, device=device) if fixed is not None else
                      masked.argmax(1) if deterministic else dist.sample())
            if collect_statistics:
                log_prob = log_prob + dist.log_prob(chosen)
                entropy = entropy + dist.entropy()
            return chosen

        team_logits = self.team_head(feature).split(TEAM_SIZES, dim=1)
        alive = np.array([[o.vector[len(GLOBAL_FIELDS) + i * len(ALLY_FIELDS)] > 0 for i in range(5)] for o in observations])
        alive[~alive.any(1), 0] = True
        team = []
        for i, logits in enumerate(team_logits):
            mask = np.ones((batch, TEAM_SIZES[i]), bool)
            if i in (2, 3, 4):
                mask = alive
            fixed = None if records is None else [r["team"][i] for r in records]
            team.append(draw(logits, mask, fixed))
        communication = torch.cat([torch.nn.functional.one_hot(chosen, size).float()
                                   for chosen, size in zip(team, TEAM_SIZES)], dim=1)
        role_context = torch.cat((feature, communication), dim=1)
        kinds, facings, targets = [], [], []
        for slot, head in enumerate(self.role_heads):
            logits = head(role_context)
            kind_logits, facing_logits, target_logits = torch.split(logits, (len(KINDS), 8, 2 * self.cells), dim=1)
            kind = draw(kind_logits, np.stack([o.masks.kind[slot] for o in observations]),
                        None if records is None else [r["kind"][slot] for r in records])
            kind_numpy = kind.detach().cpu().numpy()
            facing_mask = np.stack([o.masks.ultimate_facing[slot] if k == 9 else o.masks.facing[slot]
                                    for o, k in zip(observations, kind_numpy)])
            facing = draw(facing_logits, facing_mask,
                          None if records is None else [r["facing"][slot] for r in records])
            needs_target = np.array([target_required(slot, KINDS[k], o.masks)
                                     for o, k in zip(observations, kind_numpy)])
            selected_logits = target_logits.reshape(batch, 2, self.cells)[torch.arange(batch, device=device),
                torch.as_tensor(kind_numpy == 9, device=device, dtype=torch.long)]
            target_mask = np.stack([o.masks.target[slot, int(k == 9)] for o, k in zip(observations, kind_numpy)]).copy()
            target_mask[~needs_target] = False
            target_mask[~needs_target, 0] = True
            fixed = None if records is None else [max(0, r["target"][slot]) for r in records]
            # A no-target row is a single deterministic sentinel, with zero log-probability.
            target = draw(selected_logits, target_mask, fixed)
            kinds.append(kind)
            facings.append(facing)
            targets.append(torch.where(torch.as_tensor(needs_target, device=device), target, -1))
        full_state = torch.zeros((batch, CRITIC_SIZE), device=device) if critic is None else torch.as_tensor(
            np.stack(critic), device=device, dtype=torch.float32)
        value = self.critic(torch.cat((feature, full_state), dim=1)).squeeze(-1) if collect_statistics else torch.zeros(batch, device=device)
        packed = {"team": torch.stack(team, dim=1), "kind": torch.stack(kinds, dim=1),
                  "facing": torch.stack(facings, dim=1), "target": torch.stack(targets, dim=1)}
        return packed, log_prob, entropy, value

    def greedy_record(self, observation):
        """Decode one CPU inference without training tensors or sentinel draws."""
        feature = self.features([observation], runtime=True)
        def choose(logits, legal):
            if not legal.any():
                raise ValueError("empty FRC action mask")
            return int(np.where(legal, logits, -1e9).argmax())

        alive = observation.vector[len(GLOBAL_FIELDS)::len(ALLY_FIELDS)] > 0
        if not alive.any():
            alive = alive.copy()
            alive[0] = True
        team = np.empty(len(TEAM_SIZES), np.int64)
        communication = np.zeros(sum(TEAM_SIZES), np.float32)
        offset = 0
        team_logits = np.split(self.team_head(feature)[0].detach().numpy(), np.cumsum(TEAM_SIZES)[:-1])
        for i, (size, logits) in enumerate(zip(TEAM_SIZES, team_logits)):
            team[i] = choose(logits, alive) if i in (2, 3, 4) else int(logits.argmax())
            communication[offset + team[i]] = 1
            offset += size
        context = torch.cat((feature, torch.from_numpy(communication).unsqueeze(0)), dim=1)
        kinds = np.empty(5, np.int64)
        facings = np.empty(5, np.int64)
        targets = np.full(5, -1, np.int64)
        for slot, head in enumerate(self.role_heads):
            if (observation.masks.kind[slot, 0] and np.count_nonzero(observation.masks.kind[slot]) == 1
                    and np.count_nonzero(observation.masks.facing[slot]) == 1):
                kinds[slot] = 0
                facings[slot] = int(observation.masks.facing[slot].argmax())
                continue
            hidden = head[1](head[0](context))
            output = head[2]
            action_size = len(KINDS) + 8
            logits = torch.nn.functional.linear(hidden, output.weight[:action_size],
                                                  output.bias[:action_size])[0].detach().numpy()
            kind = kinds[slot] = choose(logits[:len(KINDS)], observation.masks.kind[slot])
            facings[slot] = choose(logits[len(KINDS):len(KINDS) + 8],
                observation.masks.ultimate_facing[slot] if kind == 9 else observation.masks.facing[slot])
            if target_required(slot, KINDS[kind], observation.masks):
                offset = len(KINDS) + 8 + int(kind == 9) * self.cells
                target_logits = torch.nn.functional.linear(hidden, output.weight[offset:offset + self.cells],
                    output.bias[offset:offset + self.cells])[0].detach().numpy()
                targets[slot] = choose(target_logits, observation.masks.target[slot, int(kind == 9)])
        return {"team": team, "kind": kinds, "facing": facings, "target": targets}


class FrcPolicy:
    def __init__(self, grid, side, *, device="cpu", deterministic=True, effects_mode="all"):
        if side not in ("A", "D"):
            raise ValueError("FRC policy side must be A or D")
        self.grid, self.side = tuple(tuple(map(int, row)) for row in grid), side
        if len(plant_sites(self.grid)) != 2:
            raise ValueError("FRC policy requires the fixed two-site map")
        self.model = FrcActorCritic((len(grid), len(grid[0]))).to(device)
        self.deterministic = deterministic
        self.collect_statistics = True
        self.effects_mode = effects_mode
        self.last_sample = None
        self.reset()

    def reset(self):
        self._navigation_round = None
        self._navigation_phase = None
        self._navigation_tick = -1
        self._navigation_site = None
        self._navigation_waypoint = None
        self._navigation_attack_episode = 0
        self._navigation_defense_episode = 0
        self._defense_assignments = None
        self._defense_anchors = None
        self._postplant_plan = None
        self._postplant_recon_tick = -100
        self._last_tactical_utility = {}
        self._tactical_cast_history = {4: []}
        self._attack_setup_anchors = None

    def sample(self, observation, *, critic=None):
        with torch.no_grad():
            if not self.collect_statistics and self.deterministic and next(self.model.parameters()).device.type == "cpu":
                record = self.model.greedy_record(observation)
                self.last_sample = None
            else:
                packed, log_prob, _, value = self.model.distribution([observation], deterministic=self.deterministic,
                    critic=None if critic is None else [critic], collect_statistics=self.collect_statistics)
                record = {key: array[0].cpu().numpy().copy() for key, array in packed.items()}
                self.last_sample = (record, float(log_prob[0]), float(value[0])) if self.collect_statistics else None
        actions = []
        for slot in range(5):
            kind = KINDS[record["kind"][slot]]
            target = int(record["target"][slot])
            ally_slot = target if observation.masks.abilities[slot] == "DANCE" and kind == "ABILITY" else None
            cell = divmod(target, len(self.grid[0])) if target >= 0 and ally_slot is None else None
            actions.append(FrcAction(kind, FACING[record["facing"][slot]], cell, ally_slot))
        t = record["team"]
        return TeamDecision(tuple(actions), int(t[0]), PHASES[t[1]], int(t[2]), int(t[3]), int(t[4]), tuple(map(int, t[5:])))

    def act(self, observation, snapshot, belief, *, critic=None, navigate=True):
        if snapshot.side != self.side or snapshot.grid != self.grid:
            raise ValueError("FRC policy side/map changed")
        decision = self.sample(observation, critic=critic)
        if navigate:
            from frc_v1.navigation import (EAST_LONG_WAYPOINT, attack_setup_positions,
                                           defense_anchor_positions, defense_site_assignments,
                                           guard_attack_navigation, guard_defense_navigation,
                                           guard_tactical_utility)
            same_round = snapshot.round_number == self._navigation_round
            new_episode = (not same_round or
                self._navigation_phase == "live" and snapshot.phase == "setup" or
                self._navigation_phase == snapshot.phase == "live" and snapshot.tick < self._navigation_tick)
            if new_episode:
                self._postplant_plan = None
                self._postplant_recon_tick = -100
                self._last_tactical_utility = {}
                self._tactical_cast_history = {4: []}
                self._attack_setup_anchors = None
                if self.side == "A":
                    if same_round:
                        self._navigation_attack_episode += 1
                    else:
                        self._navigation_attack_episode = snapshot.round_number - 1
                    attack_plan = self._navigation_attack_episode % 5
                    self._navigation_site = (1, 0, 1, 0, 1)[attack_plan]
                    self._navigation_waypoint = EAST_LONG_WAYPOINT if attack_plan == 3 else None
                else:
                    if same_round:
                        self._navigation_defense_episode += 1
                    else:
                        self._navigation_defense_episode = snapshot.round_number
                    self._defense_assignments = defense_site_assignments(
                        snapshot, variation=self._navigation_defense_episode)
                    self._defense_anchors = defense_anchor_positions(
                        snapshot, self._defense_assignments,
                        variation=self._navigation_defense_episode)
            elif (self.side == "D" and self._navigation_phase == "setup" and
                  snapshot.phase == "live"):
                self._defense_anchors = defense_anchor_positions(
                    snapshot, self._defense_assignments,
                    variation=self._navigation_defense_episode)
            if self.side == "A":
                if snapshot.is_planted and snapshot.spike_planted is not None:
                    from frc_v1.postplant import guard_attack_postplant, postplant_positions
                    if self._postplant_plan is None:
                        self._postplant_plan = postplant_positions(snapshot)
                    decision = guard_attack_postplant(decision, snapshot, observation.masks,
                                                       plan=self._postplant_plan,
                                                       last_recon_tick=self._postplant_recon_tick)
                    for slot, recon in enumerate(decision.actions):
                        if (ability_for(snapshot, slot) == "RECON" and recon.kind == "ABILITY" and recon.target is not None and
                                max(abs(recon.target[0] - snapshot.spike_planted[0]),
                                    abs(recon.target[1] - snapshot.spike_planted[1])) <= 2):
                            self._postplant_recon_tick = snapshot.tick
                else:
                    if snapshot.phase == "setup" and self._attack_setup_anchors is None:
                        self._attack_setup_anchors = attack_setup_positions(
                            snapshot, self._navigation_site, route_waypoint=self._navigation_waypoint)
                    decision = guard_attack_navigation(decision, snapshot, observation.masks,
                                                       site_index=self._navigation_site,
                                                       setup_positions=self._attack_setup_anchors,
                                                       route_waypoint=self._navigation_waypoint,
                                                       belief=belief)
            else:
                decision = guard_defense_navigation(decision, snapshot, observation.masks,
                                                    site_assignments=self._defense_assignments,
                                                    anchor_positions=self._defense_anchors)
            decision = guard_tactical_utility(decision, snapshot, observation.masks,
                                              last_cast=self._last_tactical_utility,
                                              cast_history=self._tactical_cast_history)
            for slot, action in enumerate(decision.actions):
                if ability_for(snapshot, slot) in ("SMOKE", "ASH") and action.kind == "ABILITY":
                    self._last_tactical_utility[slot] = snapshot.tick
                    if ability_for(snapshot, slot) == "ASH" and action.target is not None:
                        self._tactical_cast_history.setdefault(slot, []).append(action.target)
            self._navigation_round = snapshot.round_number
            self._navigation_phase = snapshot.phase
            self._navigation_tick = snapshot.tick
        return decision

    def save(self, path, *, training=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata(self.grid, self.side), "effects_mode": self.effects_mode,
                    "state_dict": self.model.state_dict(), "training": training or {}}, path)

    @classmethod
    def load(cls, path, *, side, device="cpu", deterministic=True):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"FRC learned checkpoint missing: {path}. Train it first or select FRC v1（基礎ルール）.")
        from map_data import NEW_MAZE_STR
        grid = tuple(tuple(map(int, row.strip())) for row in NEW_MAZE_STR.strip().splitlines())
        saved = torch.load(path, map_location=device, weights_only=True)
        validate_metadata(saved["metadata"], grid, side)
        policy = cls(grid, side, device=device, deterministic=deterministic, effects_mode=saved.get("effects_mode", "all"))
        policy.model.load_state_dict(saved["state_dict"], strict=True)
        policy.model.eval()
        return policy
