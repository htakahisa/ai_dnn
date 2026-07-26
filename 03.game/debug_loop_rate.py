# debug_loop_rate.py
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from train_attacker_ability import AttackerAbilityEnv, DuelingQNetwork, OBS_DIM, N_ACTIONS, mask_invalid_actions

lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
plant_rows, plant_cols = np.where(grid == 2)
plant_candidates = list(zip(plant_rows, plant_cols))

env = AttackerAbilityEnv(grid, plant_candidates)
device = torch.device("cpu")
q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
q_net.load_state_dict(torch.load("attacker_ability_data/dqn_attacker_ability_ep1950.pt", map_location=device))
q_net.eval()

N_TRIALS = 300
success, timeout, loop_detected = 0, 0, 0

for trial in range(N_TRIALS):
    obs, _ = env.reset(phase="carry")
    pos_history = []
    looped = False

    for step in range(150):
        with torch.no_grad():
            q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        q_values = mask_invalid_actions(q_values, env)
        action = int(np.argmax(q_values))
        obs, reward, term, trunc, _ = env.step(action)

        pos_history.append(tuple(env.player_pos))
        if len(pos_history) >= 6:
            recent = pos_history[-6:]
            if len(set(recent)) <= 2:
                looped = True

        if term or trunc:
            if env.carry_end_reason == "planted":
                success += 1
            else:
                timeout += 1
            if looped:
                loop_detected += 1
            break

print(f"total={N_TRIALS}")
print(f"success={success} ({success/N_TRIALS*100:.1f}%)")
print(f"timeout={timeout} ({timeout/N_TRIALS*100:.1f}%)")
print(f"loop_detected={loop_detected} ({loop_detected/N_TRIALS*100:.1f}%)  <- 直近6手が2地点以内で往復したエピソードの割合")