# diagnose_stall.py
import torch
import numpy as np
from train_attacker_retrieve import (
    RetrieveEnv, DuelingQNet, N_ACTIONS, DEVICE, masked_argmax,
)

MODEL_PATH = "data/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt"

policy_net = DuelingQNet(RetrieveEnv.OBS_DIM, N_ACTIONS).to(DEVICE)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy_net.eval()

env = RetrieveEnv()
n_episodes = 300
results = {"reached": 0, "died": 0, "timeout": 0}
timeout_min_dist_ratio = []  # timeout時、どこまで近づけたか(0=ゴール直前, 1=全く進まず)
enemy_seen_in_timeout = 0

for _ in range(n_episodes):
    obs, mask = env.reset()
    start_dist = env.dist_map[tuple(env.pos)]
    min_dist_reached = start_dist
    saw_enemy = False
    done = False
    while not done:
        if env._visible_enemy():
            saw_enemy = True
        state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)
        with torch.no_grad():
            q = policy_net(state_t).squeeze(0)
            action = masked_argmax(q, mask_t)
        obs, reward, done, mask, info = env.step(action)
        cur_dist = env.dist_map[tuple(env.pos)]
        min_dist_reached = min(min_dist_reached, cur_dist)

    result = info.get("result", "unknown")
    results[result] = results.get(result, 0) + 1

    if result == "timeout":
        ratio = min_dist_reached / max(1, start_dist)
        timeout_min_dist_ratio.append(ratio)
        if saw_enemy:
            enemy_seen_in_timeout += 1

print(results)
if timeout_min_dist_ratio:
    arr = np.array(timeout_min_dist_ratio)
    print(f"timeout時の最接近割合(0=ゴール到達寸前, 1=ほぼ前進なし): "
          f"mean={arr.mean():.2f} median={np.median(arr):.2f}")
    print(f"timeoutのうち、敵を一度でも見た割合: "
          f"{enemy_seen_in_timeout}/{len(timeout_min_dist_ratio)} "
          f"({enemy_seen_in_timeout/len(timeout_min_dist_ratio):.1%})")