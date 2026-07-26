# debug_step_from_2222.py
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from train_attacker_ability import AttackerAbilityEnv, DuelingQNetwork, OBS_DIM, N_ACTIONS, get_action_mask, mask_invalid_actions, bfs_distances

lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
plant_rows, plant_cols = np.where(grid == 2)
plant_candidates = list(zip(plant_rows, plant_cols))

env = AttackerAbilityEnv(grid, plant_candidates)
device = torch.device("cpu")
q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
q_net.load_state_dict(torch.load("dqn_attacker_ability_best_by_eval.pt", map_location=device))
q_net.eval()

env.reset(phase="carry")
# 護衛は"front"/"back"固定キーが必須なので、ダミーで遠くに配置しておく
env.escort_positions = {
    "front": env.attacker_spawn_candidates[0],
    "back": env.attacker_spawn_candidates[0],
}
env.player_pos = (22, 40)
env.goal_pos = plant_candidates[0]
env.dist_map = bfs_distances(env.goal_pos, env.grid)
env.last_action = 1
obs = env._get_obs()

for step in range(30):
    with torch.no_grad():
        q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    mask = get_action_mask(env)
    q_values = mask_invalid_actions(q_values, mask)
    action = int(np.argmax(q_values))
    obs, reward, term, trunc, _ = env.step(action)
    print(f"step={step} action={action} pos={env.player_pos} q={np.round(q_values,1)} reward={reward:.1f}")
    if term or trunc:
        print(f"END reason={env.carry_end_reason}")
        break