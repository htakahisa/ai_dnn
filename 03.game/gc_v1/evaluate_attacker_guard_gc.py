
import os, random
from collections import Counter, defaultdict
import numpy as np
import torch
import train_attacker_guard_gc as T

MODEL = r"data\attacker_guard_gc_data\dqn_attacker_guard_gc_8000_backup.pt"
EPISODES = 1000
SEED = 20260811

def win(reason):
    return reason in ("attacker_win_wipe", "attacker_win_detonate")

def evaluate():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not os.path.exists(MODEL):
        raise FileNotFoundError(MODEL)

    net = T.AttackerGuardDuelingDQN().to(T.DEVICE)
    obj = torch.load(MODEL, map_location=T.DEVICE, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj: obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj: obj = obj["model_state_dict"]
    net.load_state_dict(obj, strict=True)
    net.eval()

    env = T.GuardEnv()
    reasons = Counter()
    pats = defaultdict(lambda: Counter())
    total_ticks = 0
    arrive_n = arrived_n = dist_n = 0
    dist_sum = 0.0

    def sample(pattern):
        nonlocal arrive_n, arrived_n, dist_n, dist_sum
        for a in env.attackers:
            if not a.is_alive or a.assigned_guard_dist_map is None: continue
            r,c = map(int,a.pos)
            d = int(a.assigned_guard_dist_map[r,c])
            if d < 0: continue
            arrive_n += 1; dist_n += 1; dist_sum += d
            pats[pattern]["arrive_n"] += 1
            pats[pattern]["dist_n"] += 1
            pats[pattern]["dist_sum"] += d
            if d <= T.GUARD_POS_REACH_RADIUS:
                arrived_n += 1; pats[pattern]["arrived_n"] += 1

    with torch.no_grad():
        for ep in range(1, EPISODES+1):
            obs, masks = env.reset()
            p = int(env.pattern_marker)
            pats[p]["episodes"] += 1
            sample(p)
            ticks = 0
            for _ in range(T.MAX_TICKS):
                actions = {n:T.select_action(net,o,masks[n],0.0) for n,o in obs.items()}
                obs, masks, rewards, done = env.step(actions)
                ticks += 1; sample(p)
                if done or not obs: break
            reason = env.match_over_reason or "unknown"
            reasons[reason] += 1
            pats[p][reason] += 1
            pats[p]["ticks"] += ticks
            if win(reason): pats[p]["wins"] += 1
            total_ticks += ticks
            if ep % 100 == 0:
                w = reasons["attacker_win_wipe"] + reasons["attacker_win_detonate"]
                print(f"[EVAL {ep}/{EPISODES}] win_rate={w/ep:.3f} avg_ticks={total_ticks/ep:.2f} reasons={dict(reasons)}")

    wins = reasons["attacker_win_wipe"] + reasons["attacker_win_detonate"]
    print("\n"+"="*78)
    print("GC ATTACKER GUARD - GREEDY EVALUATION")
    print("="*78)
    print("model:", MODEL)
    print(f"episodes={EPISODES} epsilon=0.0")
    print(f"win_rate={wins/EPISODES:.3f} ({wins}/{EPISODES})")
    print(f"detonated={reasons['attacker_win_detonate']} defenders_wiped={reasons['attacker_win_wipe']} defused={reasons['defused']} attackers_wiped={reasons['attacker_wipe']} unknown={reasons['unknown']}")
    print(f"avg_ticks={total_ticks/EPISODES:.2f}")
    print(f"guard_arrive_rate={arrived_n/arrive_n if arrive_n else 0:.3f}")
    print(f"guard_avg_bfs_dist={dist_sum/dist_n if dist_n else float('nan'):.3f}")
    print("\n[PER-PATTERN]")
    for p in sorted(pats):
        s=pats[p]; n=s["episodes"]
        print(f"pattern={p} episodes={n} win_rate={s['wins']/n if n else 0:.3f} detonated={s['attacker_win_detonate']} defenders_wiped={s['attacker_win_wipe']} defused={s['defused']} attackers_wiped={s['attacker_wipe']} avg_ticks={s['ticks']/n if n else 0:.2f} guard_arrive_rate={s['arrived_n']/s['arrive_n'] if s['arrive_n'] else 0:.3f} guard_avg_bfs_dist={s['dist_sum']/s['dist_n'] if s['dist_n'] else float('nan'):.3f}")
    print("="*78)

if __name__ == "__main__":
    evaluate()
