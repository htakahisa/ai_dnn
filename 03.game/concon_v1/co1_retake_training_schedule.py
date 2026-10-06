"""Collect equal team quotas while retaining the natural plant-site distribution."""

from collections import Counter


def iter_training_windows(env, opponents, episodes, checkpoint_interval, rng, epsilon_fn,
                          on_step=None, on_progress=None):
    if not opponents or min(episodes, checkpoint_interval) < 1:
        raise ValueError("a nonempty roster and positive episode counts are required")
    if episodes % len(opponents) or checkpoint_interval % len(opponents):
        raise ValueError("episodes and checkpoint interval must be divisible by the opponent count for equal team quotas")
    completed, attempts, site_completed = 0, 0, dict(L=0, R=0)
    for start in range(0, episodes, checkpoint_interval):
        end = min(start + checkpoint_interval, episodes)
        order = list(opponents)
        rng.shuffle(order)
        quotas = Counter(order[index % len(order)] for index in range(end - start))
        counts = Counter()
        queue, team_attempts = [], Counter()
        while any(counts[opponent] < required for opponent, required in quotas.items()):
            if not queue:
                queue = [opponent for opponent, required in quotas.items()
                         if counts[opponent] < required]
                rng.shuffle(queue)
            opponent = queue.pop()
            # A quota can become complete while this opponent is still queued.
            if counts[opponent] >= quotas[opponent]:
                continue
            epsilon = epsilon_fn(completed + 1)
            env.reset(opponent=opponent)
            ticks = 0
            while not env.done:
                transitions, _, _ = env.step(epsilon)
                ticks += 1
                if on_step is not None:
                    on_step(transitions, ticks)
            raw = env.result()
            site = raw.get("site")
            retake = bool(raw.get("planted") and not raw.get("excluded_from_retake", False))
            counted = retake and site in site_completed
            if counted:
                counts[opponent] += 1
                completed += 1
                site_completed[site] += 1
            attempts += 1
            team_attempts[opponent] += 1
            record = dict(raw, round=attempts, episode=completed,
                          training_episodes=dict(site_completed), epsilon=epsilon,
                          counted_episode=counted, excluded_training_quota=False)
            if not counted and team_attempts[opponent] % 10 == 0 and on_progress is not None:
                on_progress(opponent, counts[opponent], quotas[opponent], team_attempts[opponent])
            boundary = end if all(counts[team] >= required for team, required in quotas.items()) else None
            yield record, boundary
