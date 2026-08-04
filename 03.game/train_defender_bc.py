from __future__ import annotations
import json, math, random
from collections import Counter
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader,TensorDataset
from defender_policy_common import *

def main():
    src=Path("demos/defender_dagger_aggregated.json")
    demos=json.loads(src.read_text(encoding="utf-8")); xs=[]; ys=[]
    for d in demos:
        if d.get("action",{}).get("team")!="D" or d.get("teacher_action_valid") is False: continue
        xs.append(defender_observation_to_vector(d["observation"])); ys.append(encode_defender_action(d["action"],d["observation"]))
    x=np.stack(xs); y=np.asarray(ys,np.int64); print("samples",len(y),"obs",x.shape[1],Counter(y.tolist()))
    rng=np.random.default_rng(42); idx=np.arange(len(y)); rng.shuffle(idx); cut=int(len(idx)*.9); tr,va=idx[:cut],idx[cut:]
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DefenderPolicyNetwork(x.shape[1]).to(dev)

    initial_model_path=Path("policy_fnatic_defender_final.pt")
    if not initial_model_path.exists():
        raise FileNotFoundError(
            f"継続学習元モデルが見つかりません: {initial_model_path}"
        )

    checkpoint=torch.load(initial_model_path,map_location=dev)
    if not isinstance(checkpoint,dict):
        raise TypeError("継続学習元モデルが辞書形式ではありません")

    state_dict=checkpoint.get("model_state_dict",checkpoint)
    checkpoint_obs_size=int(
        checkpoint.get(
            "obs_size",
            state_dict["net.0.weight"].shape[1],
        )
    )
    checkpoint_num_actions=int(
        checkpoint.get("num_actions",NUM_ACTIONS)
    )

    if checkpoint_obs_size!=x.shape[1]:
        raise ValueError(
            "継続学習元モデルとAGGデータの観測次元が一致しません: "
            f"model={checkpoint_obs_size}, data={x.shape[1]}"
        )
    if checkpoint_num_actions!=NUM_ACTIONS:
        raise ValueError(
            "継続学習元モデルと現在コードのアクション数が一致しません: "
            f"model={checkpoint_num_actions}, code={NUM_ACTIONS}"
        )

    model.load_state_dict(state_dict)
    print("continue from",initial_model_path)
    counts=np.bincount(y[tr],minlength=NUM_ACTIONS); weights=np.zeros(NUM_ACTIONS,np.float32); total=counts.sum()
    for i,c in enumerate(counts): weights[i]=math.sqrt(total/c) if c else 0
    nz=weights[weights>0]; weights=weights/nz.mean() if len(nz) else weights; weights=np.clip(weights,0,5)
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,device=dev)); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    loader=DataLoader(TensorDataset(torch.tensor(x[tr]),torch.tensor(y[tr])),batch_size=64,shuffle=True)
    xv=torch.tensor(x[va],device=dev); yv=torch.tensor(y[va],device=dev); best=1e9; patience=0
    for ep in range(1,81):
        model.train(); good=tot=0; ls=0
        for xb,yb in loader:
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(); z=model(xb); l=loss_fn(z,yb); l.backward(); opt.step(); ls+=l.item()*len(yb); good+=(z.argmax(1)==yb).sum().item(); tot+=len(yb)
        model.eval()
        with torch.no_grad(): zv=model(xv); vl=loss_fn(zv,yv).item(); acc=(zv.argmax(1)==yv).float().mean().item()
        print(f"Epoch {ep:3d} train={ls/tot:.5f} acc={good/tot:.4f} val={vl:.5f} val_acc={acc:.4f}")
        if vl<best-1e-5:
            best=vl; patience=0; torch.save({"model_state_dict":model.state_dict(),"obs_size":x.shape[1],"num_actions":NUM_ACTIONS,"action_names":ACTION_NAMES},"policy_fnatic_defender_dagger_best.pt")
        else:
            patience+=1
            if patience>=15: break
    bestck=torch.load("policy_fnatic_defender_dagger_best.pt",map_location="cpu"); torch.save(bestck,"policy_fnatic_defender_final.pt")
if __name__=="__main__": main()
