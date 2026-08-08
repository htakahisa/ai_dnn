from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from train_bc import NUM_ACTIONS

class PPOActorCritic(nn.Module):
    def __init__(self, obs_size:int, num_actions:int=NUM_ACTIONS):
        super().__init__()
        self.obs_size=int(obs_size); self.num_actions=int(num_actions)
        self.encoder=nn.Sequential(
            nn.Linear(self.obs_size,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(256,128), nn.ReLU(),
        )
        self.policy_head=nn.Linear(128,self.num_actions)
        self.value_head=nn.Sequential(nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,obs):
        f=self.encoder(obs)
        return self.policy_head(f), self.value_head(f).squeeze(-1)

def _unwrap(loaded:Any):
    if not isinstance(loaded,dict): raise TypeError("モデルファイルが辞書形式ではありません")
    for key in ("model_state_dict","state_dict","policy_state_dict"):
        if isinstance(loaded.get(key),dict): return loaded[key],loaded
    return loaded,loaded

def load_actor_critic_from_bc(model_path,device):
    loaded=torch.load(Path(model_path), map_location=device, weights_only=False)
    sd,meta=_unwrap(loaded)
    sd={str(k).removeprefix("module."):v for k,v in sd.items()}
    first,final=sd.get("net.0.weight"),sd.get("net.10.weight")
    if first is None or final is None: raise KeyError("net.0.weight / net.10.weight がありません")
    obs=int(meta.get("obs_size",first.shape[1])); acts=int(meta.get("num_actions",final.shape[0]))
    if acts!=NUM_ACTIONS: raise ValueError(f"アクション数不一致: {acts} != {NUM_ACTIONS}")
    model=PPOActorCritic(obs,acts).to(device)
    mapping={
      "net.0.weight":"encoder.0.weight","net.0.bias":"encoder.0.bias",
      "net.1.weight":"encoder.1.weight","net.1.bias":"encoder.1.bias",
      "net.4.weight":"encoder.4.weight","net.4.bias":"encoder.4.bias",
      "net.5.weight":"encoder.5.weight","net.5.bias":"encoder.5.bias",
      "net.8.weight":"encoder.8.weight","net.8.bias":"encoder.8.bias",
      "net.10.weight":"policy_head.weight","net.10.bias":"policy_head.bias",
    }
    target=model.state_dict()
    for s,t in mapping.items():
        if s not in sd: raise KeyError(s)
        target[t]=sd[s]
    model.load_state_dict(target)
    return model


def is_ppo_checkpoint(model_path, device):
    """指定ファイルがPPOActorCritic形式のチェックポイントか判定する。"""
    loaded = torch.load(Path(model_path), map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        return False

    state_dict = loaded.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return False

    normalized = {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }
    return (
        "encoder.0.weight" in normalized
        and "policy_head.weight" in normalized
        and "value_head.0.weight" in normalized
    )


def load_ppo_checkpoint(model_path, device):
    """PPOチェックポイントをモデル・学習状態ごと読み込む。"""
    model_path = Path(model_path)
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError("PPOチェックポイントが辞書形式ではありません")

    raw_state = loaded.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise KeyError("model_state_dictがありません")

    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in raw_state.items()
    }

    first_weight = state_dict.get("encoder.0.weight")
    policy_weight = state_dict.get("policy_head.weight")
    if first_weight is None or policy_weight is None:
        raise KeyError(
            "PPOモデルにencoder.0.weight / policy_head.weightがありません"
        )

    obs_size = int(loaded.get("obs_size", first_weight.shape[1]))
    num_actions = int(loaded.get("num_actions", policy_weight.shape[0]))
    if num_actions != NUM_ACTIONS:
        raise ValueError(
            f"アクション数不一致: {num_actions} != {NUM_ACTIONS}"
        )

    model = PPOActorCritic(obs_size, num_actions).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "optimizer_state_dict": loaded.get("optimizer_state_dict"),
        "update": int(loaded.get("update", 0)),
        "episodes": int(loaded.get("episodes", 0)),
        "best_win_rate": float(loaded.get("best_win_rate", 0.0)),
        "evaluation": loaded.get("evaluation"),
        "accepted_evaluation": loaded.get("accepted_evaluation"),
        "baseline_evaluation": loaded.get("baseline_evaluation"),
        "rollout": loaded.get("rollout", []),
        "best_rounds": loaded.get("best_rounds"),
        "accepted_result": loaded.get("accepted_result"),
        "interrupted": bool(loaded.get("interrupted", False)),
        "latest": bool(loaded.get("latest", False)),
        "model_type": loaded.get("model_type", "ppo_actor_critic"),
    }
    return model, metadata

def save_ppo_checkpoint(path,model,optimizer,update,episodes,best_win_rate,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict() if optimizer else None,
             "obs_size":model.obs_size,"num_actions":model.num_actions,"update":int(update),
             "episodes":int(episodes),"best_win_rate":float(best_win_rate),"model_type":"ppo_actor_critic"}
    if extra: payload.update(extra)
    tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); tmp.replace(path)
