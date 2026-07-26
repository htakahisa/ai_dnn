# debug_loop_exhaustive.py
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from train_attacker_ability import AttackerAbilityEnv, DuelingQNetwork, OBS_DIM, N_ACTIONS, mask_invalid_actions
from train_attacker_combined import bfs_distances
import random

lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
plant_rows, plant_cols = np.where(grid == 2)
plant_candidates = list(zip(plant_rows, plant_cols))

env = AttackerAbilityEnv(grid, plant_candidates)
device = torch.device("cpu")
q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
q_net.load_state_dict(torch.load("attacker_ability_data/dqn_attacker_ability_ep1950.pt", map_location=device))
q_net.eval()

fail_pairs = []

# debug_loop_exhaustive_with_escort.py (前のスクリプトのループ本体だけ変更)
for start in env.attacker_spawn_candidates:
    for goal in plant_candidates:
        env.reset(phase="carry")
        env.player_pos = start
        env.goal_pos = goal
        # 💡変更: 護衛を1体、別のスポーン地点に配置して実際に動かす
        other_spawns = [s for s in env.attacker_spawn_candidates if s != start]
        env.escort_positions = {"escort0": random.choice(other_spawns)} if other_spawns else {}
        env.dist_map = bfs_distances(goal, grid)
        env.last_action = None
        obs = env._get_obs()

        pos_history = []
        for step in range(150):
            with torch.no_grad():
                q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            q_values = mask_invalid_actions(q_values, env)
            action = int(np.argmax(q_values))
            obs, reward, term, trunc, _ = env.step(action)  # 💡step内で_move_escortsも呼ばれ護衛が動く
            pos_history.append(tuple(env.player_pos))
            if len(pos_history) >= 8 and len(set(pos_history[-8:])) <= 2:
                fail_pairs.append((start, goal, "loop"))
                break
            if term or trunc:
                if env.carry_end_reason != "planted":
                    fail_pairs.append((start, goal, env.carry_end_reason))
                break

print(f"total combos={len(env.attacker_spawn_candidates) * len(plant_candidates)}")
print(f"failed={len(fail_pairs)}")
for s, g, reason in fail_pairs[:30]:
    print(f"start={s} goal={g} reason={reason}")