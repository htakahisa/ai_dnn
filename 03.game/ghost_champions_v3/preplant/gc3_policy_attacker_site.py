"""One attacker actor: sample a shared site intention, then sample its actions."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .gc3_policy_attacker_history import HistoryPolicy, SCHEMA_HASH as HISTORY_SCHEMA, preplant_legal_actions
from .gc3_observation_attacker_history import encode_public
from ..gc3_observation import map_hints, sampling_phase
from ..gc3_engine_actions import MODES, legal_actions, decode_command
from ..gc3_spatial_policy import sampled_distribution
from ..gc3_io import replace_file

SCHEMA_HASH = hashlib.sha256((HISTORY_SCHEMA+"shared-sampled-site-intention-v1").encode()).hexdigest()
NAVIGATION_SCHEMA_HASH = hashlib.sha256((SCHEMA_HASH+"public-navigation-residual-v1").encode()).hexdigest()
RETRIEVAL_SCHEMA_HASH = hashlib.sha256((NAVIGATION_SCHEMA_HASH+"public-dropped-spike-residual-v1").encode()).hexdigest()
POLICY_TYPE = "gc3_public_match_site_history"
SITES = ("A", "B")


class SiteHistoryPolicy(HistoryPolicy):
    def __init__(self, role="attacker", hidden=128, *, navigation=False, retrieval=False):
        super().__init__(role, hidden)
        self.site = nn.Linear(hidden, 2)
        self.goal = nn.Embedding(3, hidden, padding_idx=2)
        nn.init.normal_(self.goal.weight[:2], std=.02)
        self.schema_hash = SCHEMA_HASH
        self.navigation = None
        self.retrieval_navigation = None
        if navigation:
            from .gc3_features_attacker_navigation import FEATURE_WIDTH
            self.navigation = nn.Sequential(nn.Linear(FEATURE_WIDTH, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            # A transferred checkpoint samples exactly its former distribution
            # until explicit learning updates this new residual branch.
            nn.init.zeros_(self.navigation[-1].weight)
            nn.init.zeros_(self.navigation[-1].bias)
            self.schema_hash = NAVIGATION_SCHEMA_HASH
        if retrieval:
            if not navigation:
                raise ValueError("Dropped-spike inputs require public navigation")
            from .gc3_features_attacker_navigation import DROP_FEATURE_WIDTH
            self.retrieval_navigation=nn.Sequential(nn.Linear(DROP_FEATURE_WIDTH,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
            nn.init.zeros_(self.retrieval_navigation[-1].weight)
            nn.init.zeros_(self.retrieval_navigation[-1].bias)
            self.schema_hash=RETRIEVAL_SCHEMA_HASH

    @classmethod
    def fresh(cls, role, seed, *, navigation=False, retrieval=False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            result=cls(role,navigation=navigation,retrieval=retrieval)
        return result.eval()

    @classmethod
    def with_navigation(cls, policy):
        if not isinstance(policy, cls):
            raise ValueError("Navigation transfer needs a site-history policy")
        if policy.navigation is not None:
            return policy
        result = cls(policy.role, policy.hidden, navigation=True)
        missing, unexpected = result.load_state_dict(policy.state_dict(), strict=False)
        if unexpected or set(missing) != {"navigation.0.weight", "navigation.0.bias", "navigation.2.weight", "navigation.2.bias"}:
            raise ValueError("Unexpected navigation transfer architecture")
        return result

    @classmethod
    def with_retrieval(cls, policy):
        policy=cls.with_navigation(policy)
        if policy.retrieval_navigation is not None:
            return policy
        result=cls(policy.role,policy.hidden,navigation=True,retrieval=True)
        missing,unexpected=result.load_state_dict(policy.state_dict(),strict=False)
        expected={"retrieval_navigation.0.weight","retrieval_navigation.0.bias",
                  "retrieval_navigation.2.weight","retrieval_navigation.2.bias"}
        if unexpected or set(missing)!=expected:
            raise ValueError("Unexpected dropped-spike input architecture")
        return result

    @classmethod
    def from_history(cls, policy):
        if policy.legacy:
            raise ValueError("Use the corrected GC v3 history model as the BC starting point")
        result = cls(policy.role, policy.hidden)
        missing, unexpected = result.load_state_dict(policy.state_dict(), strict=False)
        if unexpected or set(missing) != {"site.weight", "site.bias", "goal.weight"}:
            raise ValueError("Unexpected initial policy architecture")
        return result

    def goal_features(self, sites):
        indices = torch.as_tensor([2 if s is None else int(s) for s in sites], dtype=torch.long)
        return self.goal(indices)

    def event_features(self, events):
        return super().event_features(events)+self.goal_features([e.get("goal_site") for e in events])

    def contexts(self, rows, *, cache=False):
        return (super().contexts(rows, cache=cache)+self.goal_features([r.get("goal_site") for r in rows])
                +self.navigation_context(rows))

    def navigation_context(self, rows, *, neutral=False):
        if self.navigation is None:
            return 0.
        from .gc3_features_attacker_navigation import navigation_features
        features = np.stack([navigation_features(r, None if neutral else r.get("goal_site")) for r in rows])
        context=self.navigation(torch.from_numpy(features))
        if self.retrieval_navigation is not None:
            from .gc3_features_attacker_navigation import dropped_spike_features
            drop=np.stack([dropped_spike_features(row) for row in rows])
            context=context+self.retrieval_navigation(torch.from_numpy(drop))
        return context

    def neutral_context(self, contexts, rows):
        return (contexts-self.goal_features([r.get("goal_site") for r in rows])
                -self.navigation_context(rows)+self.navigation_context(rows, neutral=True))

    def distribution(self, context, record, kind):
        if kind != "site":
            return super().distribution(context, record, kind)
        goals = record.get("goal_site")
        single = context.ndim == 1
        ids = [goals] if single else list(goals)
        # A site decision must not see its own selected label as an input.
        neutral = context-(self.goal_features(ids)[0] if single else self.goal_features(ids))
        if self.navigation is not None:
            # Batched PPO distributions carry the same derived feature vectors
            # as the causally reconstructed contexts, including the neutral ones.
            if single:
                from .gc3_features_attacker_navigation import navigation_features
                chosen = torch.from_numpy(navigation_features(record, goals))
                initial = torch.from_numpy(navigation_features(record, None))
            else:
                chosen = torch.as_tensor(record["navigation_features"], dtype=context.dtype)
                initial = torch.as_tensor(record["neutral_navigation_features"], dtype=context.dtype)
            neutral = neutral-self.navigation(chosen)+self.navigation(initial)
        return sampled_distribution(self.site(neutral), torch.as_tensor(record["masks"]["site"], dtype=torch.bool))

    def select(self, char, state, history):
        hints = map_hints(char, state)
        masks, _ = preplant_legal_actions(char, state)
        goal = history.goal_sites.get(history.round_id)
        record = dict(obs=encode_public(char, state, history), hints=hints, masks=masks,
                      phase=sampling_phase(char, state), selections={}, history=history,
                      history_index=len(history.events), goal_site=goal)
        with torch.no_grad():
            context = self.contexts([record], cache=True)[0]
            logp = 0.
            def draw(kind):
                nonlocal logp
                dist = self.distribution(context, record, kind)
                action = dist.sample()
                record["selections"][kind] = int(action)
                logp += float(dist.log_prob(action))
            value_context = context
            if goal is None:
                masks["site"] = np.ones(2, dtype=np.bool_)
                draw("site")
                goal = record["selections"]["site"]
                history.goal_sites[history.round_id] = goal
                record["goal_site"] = goal
                context = context+self.goal_features([goal])[0]
                if self.navigation is not None:
                    context = context+self.navigation_context([record])[0]-self.navigation_context([record], neutral=True)[0]
            draw("mode")
            mode = MODES[record["selections"]["mode"]]
            if mode in ("MOVE", "MOVE_WATCH"): draw("position")
            if mode in ("WATCH", "MOVE_WATCH"): draw("watch")
            if mode == "ABILITY":
                if char.ability_name == "DANCE": draw("ally")
                elif char.ability_name != "RAMP": draw("target")
            if mode == "ULTIMATE":
                if char.ultimate_name in ("NEON", "ESCAPE"):
                    masks["target"] = masks["ultimate_target"]
                    draw("target")
                elif char.ultimate_name in ("RAID", "TUNNEL"): draw("watch")
            # The initial site's critic baseline is evaluated before that choice.
            record.update(logp=logp, value=float(self.critic(value_context)))
        return decode_command(char, state, record["selections"]), record

    def evaluate_contexts(self, records, contexts):
        logp, entropy, values = super().evaluate_contexts(records, contexts)
        indices = [i for i,r in enumerate(records) if "site" in r["selections"]]
        if indices:
            idx = torch.tensor(indices)
            goals = [records[i]["goal_site"] for i in indices]
            batch = dict(goal_site=goals, masks={"site":np.stack([records[i]["masks"]["site"] for i in indices])})
            if self.navigation is not None:
                from .gc3_features_attacker_navigation import navigation_features
                batch.update(navigation_features=np.stack([navigation_features(records[i], records[i].get("goal_site")) for i in indices]),
                             neutral_navigation_features=np.stack([navigation_features(records[i], None) for i in indices]))
            dist = self.distribution(contexts[idx], batch, "site")
            chosen = torch.tensor([records[i]["selections"]["site"] for i in indices])
            logp = logp.index_add(0, idx, dist.log_prob(chosen))
            entropy = entropy.index_add(0, idx, dist.entropy())
            neutral_values = self.critic(self.neutral_context(contexts[idx], [records[i] for i in indices])).squeeze(-1)
            values = values.index_copy(0, idx, neutral_values)
        return logp, entropy, values

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(policy_type=POLICY_TYPE, schema_hash=self.schema_hash, role=self.role,
                        hidden=self.hidden, model_state_dict=self.state_dict(), **metadata), temporary)
        replace_file(temporary, path)

    @classmethod
    def load(cls, path, *, role="attacker", **kwargs):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("policy_type") != POLICY_TYPE or payload.get("schema_hash") not in (SCHEMA_HASH, NAVIGATION_SCHEMA_HASH, RETRIEVAL_SCHEMA_HASH) or payload.get("role") != role:
            raise ValueError("Site-intention checkpoint mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls(role, payload["hidden"], navigation=payload["schema_hash"] in (NAVIGATION_SCHEMA_HASH,RETRIEVAL_SCHEMA_HASH),
                        retrieval=payload["schema_hash"]==RETRIEVAL_SCHEMA_HASH)
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload
