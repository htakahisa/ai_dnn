import argparse
import torch
import tkinter as tk
from train_4426 import GridWorldEnv, QNetwork

def run_continuous(env, policy_net, device, delay_ms=30, pause_ms=500):
    root = tk.Tk()
    root.title("DQN Visualization - Continuous")
    cell_size = 18
    canvas = tk.Canvas(root, width=env.width * cell_size, height=env.height * cell_size)
    canvas.pack()

    label = tk.Label(root, text="Episode: 0 | Steps: 0")
    label.pack()

    state = {"obs": None, "steps": 0, "episode": 0, "done": False}
    state["obs"], _ = env.reset()

    def draw():
        canvas.delete("all")
        for r in range(env.height):
            for c in range(env.width):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                color = "white"
                if env.grid[r, c] == 1:
                    color = "#34495e"
                elif r == env.goal_pos[0] and c == env.goal_pos[1]:
                    color = "#2ecc71"
                elif (r, c) == (env.player_pos[0], env.player_pos[1]):
                    color = "#e74c3c"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#eee")
        label.config(text=f"Episode: {state['episode']} | Steps: {state['steps']}")

    def step():
        if not root.winfo_exists():
            return
        if not state["done"]:
            with torch.no_grad():
                state_t = torch.tensor(state["obs"], dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_t).argmax(dim=1).item()
            obs, reward, term, trunc, _ = env.step(action)
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
            root.after(pause_ms, step)

    step()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みDQNモデルで迷路を連続実行")
    parser.add_argument("--model", type=str, default="dqn_gridworld.pt",
                         help="読み込むモデルファイルのパス (default: dqn_gridworld.pt)")
    parser.add_argument("--delay", type=int, default=30,
                         help="1ステップごとの表示間隔ms (default: 30)")
    parser.add_argument("--pause", type=int, default=500,
                         help="ゴール後、次の迷路までの待機ms (default: 500)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv()

    policy_net = QNetwork(env.obs_dim, 4).to(device)
    policy_net.load_state_dict(torch.load(args.model, map_location=device))
    policy_net.eval()

    print(f"Loaded model: {args.model}")
    run_continuous(env, policy_net, device, delay_ms=args.delay, pause_ms=args.pause)