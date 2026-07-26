# debug_obs_diff.py
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from train_attacker_ability import (
    AttackerAbilityEnv, DuelingQNetwork, OBS_DIM, N_ACTIONS,
    get_action_mask, mask_invalid_actions, bfs_distances,
)

lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
plant_rows, plant_cols = np.where(grid == 2)
plant_candidates = list(zip(plant_rows, plant_cols))
spawn_rows, spawn_cols = np.where(grid == 3)
spawn_candidates = list(zip(spawn_rows.tolist(), spawn_cols.tolist()))

print(f"spawn候補数={len(spawn_candidates)} plant候補数={len(plant_candidates)} 総組み合わせ={len(spawn_candidates)*len(plant_candidates)}")

env = AttackerAbilityEnv(grid, plant_candidates)
device = torch.device("cpu")
q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
q_net.load_state_dict(torch.load("dqn_attacker_ability_best_by_eval.pt", map_location=device))
q_net.eval()

MAX_STEPS = 150
LOOP_WINDOW = 8       # 直近何手を見て往復判定するか
LOOP_UNIQUE_MAX = 2   # 直近手の中でユニークな座標がこの数以下なら往復とみなす

loop_cases = []
timeout_cases = []
success_count = 0

for start in spawn_candidates:
    for goal in plant_candidates:
        env.reset(phase="carry")
        # 護衛はダミーで固定位置に置く("front"/"back"キーが必須なので空にはできない)
        env.escort_positions = {
            "front": spawn_candidates[0],
            "back": spawn_candidates[0],
        }
        env.player_pos = start
        env.goal_pos = goal
        env.dist_map = bfs_distances(env.goal_pos, env.grid)
        env.last_action = None
        obs = env._get_obs()

        pos_history = []
        looped = False
        loop_at_step = None

        for step in range(MAX_STEPS):
            with torch.no_grad():
                q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            mask = get_action_mask(env)
            q_values = mask_invalid_actions(q_values, mask)
            action = int(np.argmax(q_values))
            obs, reward, term, trunc, _ = env.step(action)

            pos_history.append(tuple(env.player_pos))
            if not looped and len(pos_history) >= LOOP_WINDOW:
                recent = pos_history[-LOOP_WINDOW:]
                if len(set(recent)) <= LOOP_UNIQUE_MAX:
                    looped = True
                    loop_at_step = step

            if term or trunc:
                if env.carry_end_reason == "planted":
                    success_count += 1
                else:
                    timeout_cases.append((start, goal))
                break

        if looped:
            loop_cases.append((start, goal, loop_at_step, env.carry_end_reason))

total = len(spawn_candidates) * len(plant_candidates)
print(f"\n=== 結果 ===")
print(f"total={total} success={success_count} timeout={len(timeout_cases)} loop_detected={len(loop_cases)}")

if loop_cases:
    print(f"\n--- 往復ループが検出された組み合わせ ({len(loop_cases)}件) ---")
    for start, goal, loop_step, reason in loop_cases:
        print(f"start={start} goal={goal} loop_at_step={loop_step} final_reason={reason}")

if timeout_cases:
    print(f"\n--- タイムアウトした組み合わせ ({len(timeout_cases)}件) ---")
    for start, goal in timeout_cases:
        print(f"start={start} goal={goal}")