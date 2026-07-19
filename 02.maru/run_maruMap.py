import argparse
import torch
import tkinter as tk
from collections import deque as pos_deque
from train_maruMap import GridWorldEnv, QNetwork, parse_fixed_maze, MAZE_STR


def run_continuous(env, policy_net, device, delay_ms=30, pause_ms=500, loop_window=7):
    root = tk.Tk()
    root.title("DQN Visualization - Continuous")
    cell_size = 18
    canvas = tk.Canvas(root, width=env.width * cell_size, height=env.height * cell_size)
    canvas.pack()
    label = tk.Label(root, text="Episode: 0 | Steps: 0")
    label.pack()

    state = {"obs": None, "steps": 0, "episode": 0, "done": False}
    state["obs"], _ = env.reset()
    recent_positions = pos_deque(maxlen=loop_window)

    def draw():
        canvas.delete("all")
        goal_candidate_set = {tuple(g) for g in env.goal_candidates} if env.goal_candidates is not None else set()
        for r in range(env.height):
            for c in range(env.width):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                color = "white"
                if env.grid[r, c] == 1:
                    color = "#34495e"
                elif (r, c) in goal_candidate_set:
                    color = "#F1DE94"  # 黄色：ゴール候補エリア全体
                if r == env.goal_pos[0] and c == env.goal_pos[1]:
                    color = "#2ecc71"  # 緑：今回選ばれた実際のゴール
                elif (r, c) == (env.player_pos[0], env.player_pos[1]):
                    color = "#e74c3c"  # 赤：現在地
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#eee")
        label.config(text=f"Episode: {state['episode']} | Steps: {state['steps']}")

    def choose_action():
        with torch.no_grad():
            state_t = torch.tensor(state["obs"], dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_t).squeeze(0)

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        r, c = env.player_pos

        ranked_actions = torch.argsort(q_values, descending=True).tolist()

        for action in ranked_actions:
            dr, dc = moves[action]
            next_pos = (r + dr, c + dc)
            if next_pos not in recent_positions:
                return action

        return ranked_actions[0]

    def step():
        if not root.winfo_exists():
            return
        if not state["done"]:
            action = choose_action()
            obs, reward, term, trunc, _ = env.step(action)
            recent_positions.append(tuple(env.player_pos))
            state["obs"] = obs
            state["steps"] += 1
            state["done"] = term or trunc
            draw()
            root.after(delay_ms, step)
        else:
            state["episode"] += 1
            state["obs"], _ = env.reset()
            state["steps"] = 0
            state["done"] = False
            recent_positions.clear()
            root.after(pause_ms, step)

    step()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みDQNモデルで迷路を連続実行")
    parser.add_argument("--model", type=str, default="dqn_gridworld_fixedmap.pt",
                         help="読み込むモデルファイルのパス (default: dqn_gridworld_fixedmap.pt)")
    parser.add_argument("--delay", type=int, default=30,
                         help="1ステップごとの表示間隔ms (default: 30)")
    parser.add_argument("--pause", type=int, default=500,
                         help="ゴール後、次の迷路までの待機ms (default: 500)")
    parser.add_argument("--loop-window", type=int, default=6,
                         help="直近何手分を移動禁止として記録するか (default: 7)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed_grid, goal_candidates = parse_fixed_maze(MAZE_STR)
    env = GridWorldEnv(fixed_grid=fixed_grid, goal_candidates=goal_candidates)

    policy_net = QNetwork(env.obs_dim, 4).to(device)
    policy_net.load_state_dict(torch.load(args.model, map_location=device))
    policy_net.eval()

    print(f"Loaded model: {args.model}")
    run_continuous(env, policy_net, device, delay_ms=args.delay, pause_ms=args.pause, loop_window=args.loop_window)