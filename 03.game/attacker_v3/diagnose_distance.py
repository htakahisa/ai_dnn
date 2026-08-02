# diagnose_distance.py (train_attacker_retrieve.py と同じ階層で実行)
import random
from train_attacker_retrieve import WALKABLE, bfs_distance_map, MAX_TICKS

samples = []
for _ in range(2000):
    goal = random.choice(WALKABLE)
    dist_map = bfs_distance_map(goal)
    reachable = [p for p in WALKABLE if dist_map[p] > 0]
    if not reachable:
        continue
    start = random.choice(reachable)
    samples.append(dist_map[start])

import numpy as np
samples = np.array(samples)
print(f"mean={samples.mean():.1f} median={np.median(samples):.1f} "
      f"p75={np.percentile(samples,75):.1f} p90={np.percentile(samples,90):.1f} max={samples.max()}")
print(f"MAX_TICKS={MAX_TICKS} を超える割合: {(samples > MAX_TICKS).mean():.1%}")