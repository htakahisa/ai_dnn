# debug_carry_no_escort.py
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from train_attacker_ability import AttackerAbilityEnv, DuelingQNetwork, OBS_DIM, N_ACTIONS, mask_invalid_actions

lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
plant_rows, plant_cols = np.where(grid == 2)
plant_candidates = list(zip(plant_rows, plant_cols))

env = AttackerAbilityEnv(grid, plant_candidates)
env.escort_positions = {}  # 💡強制的に護衛なし状態にする(実ゲームの1人プレイを模擬)

device = torch.device("cpu")
q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
q_net.load_state_dict(torch.load("attacker_ability_data/dqn_attacker_ability_ep1950.pt", map_location=device))
q_net.eval()

obs, _ = env.reset(phase="carry")
env.escort_positions = {}
env._move_escorts = lambda: None   # 💡追加: 空のescort_positionsで動く_move_escortsが無いので、一時的に無効化
obs = env._get_obs()
print(f"start={env.player_pos} goal={env.goal_pos}")

for step in range(60):
    with torch.no_grad():
        q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    q_values = mask_invalid_actions(q_values, env)
    action = int(np.argmax(q_values))

    pr, pc = env.player_pos
    d_before = env.dist_map[pr][pc]
    obs, reward, term, trunc, _ = env.step(action)
    env.escort_positions = {}  # 💡毎tick護衛なしを維持
    obs = env._get_obs()
    pr2, pc2 = env.player_pos
    d_after = env.dist_map[pr2][pc2]

    print(f"step={step:3d} action={action} pos={env.player_pos} dist={d_after:5.1f}(prev={d_before:5.1f}) reward={reward:6.2f}")
    if term or trunc:
        print(f"END reason={env.carry_end_reason}")
        break