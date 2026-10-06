"""Fixed dataset selection and balanced replay of real plant snapshots."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
from concon_v1.co1_retake_cases import CASE_VERSION


class RetakeCaseDataset:
    def __init__(self, directory, opponents, seed=0):
        self.directory = Path(directory).resolve()
        self.opponents = tuple(opponents)
        config = json.loads((self.directory / "collection.json").read_text(encoding="utf-8"))
        self.provenance = config["provenance"]
        if self.provenance["version"] != CASE_VERSION:
            raise ValueError("unsupported retake dataset version")
        if self.provenance["map_sha256"] != hashlib.sha256(GAME_MAZE_STR.encode()).hexdigest():
            raise ValueError("retake dataset map differs from this game's map")
        groups = defaultdict(list)
        self.tensor_cache = {}
        seen = set()
        # Freeze complete manifest lines at startup. A collector can append new
        # cases concurrently; they join training on the next invocation.
        with (self.directory / "cases.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                if row["opponent"] not in self.opponents:
                    continue
                filename = row["file"]
                path = (self.directory / filename).resolve()
                if path.parent != self.directory or not path.is_file() or filename in seen:
                    raise ValueError(f"missing, duplicate or invalid retake case: {filename}")
                if row["version"] != CASE_VERSION or row["site"] not in ("L", "R"):
                    raise ValueError(f"invalid retake case metadata: {filename}")
                seen.add(filename)
                groups[row["opponent"], row["site"]].append(row)
        keys = [(opponent, site) for opponent in self.opponents for site in ("L", "R")]
        missing = [key for key in keys if not groups[key]]
        if missing:
            raise ValueError(f"collect at least one case for every requested opponent/site; missing={missing}")
        self.available_counts = {f"{opponent}/{site}": len(groups[opponent, site]) for opponent, site in keys}
        self.per_group = min(len(groups[key]) for key in keys)
        self.available_size = sum(self.available_counts.values())
        rng = random.Random(seed)
        self.groups = {}
        for key in keys:
            rows = sorted(groups[key], key=lambda row: row["file"])
            rng.shuffle(rows)
            self.groups[key] = rows[:self.per_group]
        self.size = self.per_group * len(keys)
        self.signature = hashlib.sha256(json.dumps(
            [row["file"] for rows in self.groups.values() for row in rows], sort_keys=True).encode()).hexdigest()

    def validate_search(self, path):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != self.provenance["search_model_sha256"]:
            raise ValueError("dataset search weights differ; use --search-model with the collection's search checkpoint")

    def iter_cases(self, rng):
        """Shuffle each epoch and use each selected case exactly once per epoch."""
        while True:
            queues = {key: rng.sample(rows, len(rows)) for key, rows in self.groups.items()}
            for _ in range(self.per_group):
                order = list(queues)
                rng.shuffle(order)
                for key in order:
                    yield queues[key].pop()


def iter_case_training_windows(env, dataset, episodes, checkpoint_interval, rng, epsilon_fn, on_step=None):
    group_count = len(dataset.groups)
    if min(episodes, checkpoint_interval) < 1 or episodes % group_count or checkpoint_interval % group_count:
        raise ValueError("case episodes and checkpoint interval must be divisible by opponent count * 2 sites")
    site_completed = dict(L=0, R=0)
    cases = dataset.iter_cases(rng)
    for episode in range(1, episodes + 1):
        row = next(cases)
        path = dataset.directory / row["file"]
        env.reset_case(path, dataset.tensor_cache)
        if env.opponent != row["opponent"] or env.site != row["site"]:
            raise ValueError(f"snapshot differs from its manifest: {path}")
        epsilon, ticks = epsilon_fn(episode), 0
        while not env.done:
            transitions, _, _ = env.step(epsilon)
            ticks += 1
            if on_step is not None:
                on_step(transitions, ticks)
        raw = env.result()
        if raw.get("excluded_from_retake", False) or not raw.get("planted"):
            raise ValueError(f"saved case produced no retake decisions: {path}")
        site_completed[row["site"]] += 1
        record = dict(raw, round=episode, episode=episode, training_episodes=dict(site_completed),
                      epsilon=epsilon, counted_episode=True, excluded_training_quota=False,
                      case_epoch=(episode - 1) // dataset.size + 1, case_file=row["file"])
        boundary = episode if episode % checkpoint_interval == 0 or episode == episodes else None
        yield record, boundary
