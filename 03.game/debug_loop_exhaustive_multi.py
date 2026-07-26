import random
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from train_attacker_multi import AttackerMultiEnv, DuelingQNetwork, OBS_DIM, N_ACTIONS
from train_attacker_combined import bfs_distances

lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
plant_rows, plant_cols = np.where(grid == 2)
plant_candidates = list(zip(plant_rows, plant_cols))

env = AttackerMultiEnv(grid, plant_candidates)
device = torch.device("cpu")
q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
q_net.load_state_dict(torch.load("dqn_attacker_multi_best_by_eval.pt", map_location=device))
q_net.eval()

def mask(q, env):
    q = q.copy()
    if env.grid[env.player_pos[0], env.player_pos[1]] != 2:
        q[4] = -np.inf
    return q

fail_pairs = []
for start in env.attacker_spawn_candidates:
    for goal in plant_candidates:
        env.reset(phase="carry")
        env.player_pos = start
        env.goal_pos = goal
        other = [s for s in env.attacker_spawn_candidates if s != start]
        env.escort_positions = {"front": random.choice(other), "back": random.choice(other)} if other else {}
        env.dist_map = bfs_distances(goal, grid)
        env.last_action = None
        obs = env._get_obs()

        pos_history = []
        for step in range(150):
            with torch.no_grad():
                q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            q_values = mask(q_values, env)
            action = int(np.argmax(q_values))
            obs, reward, term, trunc, _ = env.step(action)
            pos_history.append(tuple(env.player_pos))
            if len(pos_history) >= 8 and len(set(pos_history[-8:])) <= 2:
                fail_pairs.append((start, goal, "loop"))
                break
            if term or trunc:
                if env.carry_end_reason != "planted":
                    fail_pairs.append((start, goal, env.carry_end_reason))
                break

print(f"total={len(env.attacker_spawn_candidates)*len(plant_candidates)} failed={len(fail_pairs)}")
for s, g, r in fail_pairs[:30]:
    print(s, g, r)