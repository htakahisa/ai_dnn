"""GC v3 public observations and an untruncated, causal match history.

The original encoder is kept intact for existing guard/search checkpoints.
Only already-perceived observations and our own issued commands enter memory.
"""
import json
import gzip
import uuid
from pathlib import Path
import numpy as np
from ..gc3_observation import encode, map_hints, OBS_DIM, ARENA_SHAPE
from ..gc3_engine_actions import MODES, FACING_DIRS

EXTRA_WIDTH = 13  # Five allied spike flags + round/event/sequence/result metadata.
HISTORY_OBS_DIM = OBS_DIM+EXTRA_WIDTH
ACTION_WIDTH = len(MODES)+6+5


def encode_public(char, state, ledger=None):
    base = encode(char, state, map_hints(char, state))
    own = sorted((c for c in state.get("chars", ()) if c.team == char.team), key=lambda c: str(c.name))[:5]
    flags = [float(bool(getattr(c, "has_spike", False))) for c in own]
    flags += [0.]*(5-len(flags))
    round_id = ledger.round_id if ledger else 0
    count = len(ledger.events) if ledger else 0
    previous = ledger.results[-1] if ledger and ledger.results else {}
    # Never include the current action, teacher route, current-round outcome or reward.
    metadata = [round_id/30, 0., count/1000, np.sin(count/10), np.cos(count/10),
                float(previous.get("success", False)), float(previous.get("planted", False)),
                float(bool(previous))]
    return np.concatenate((base, np.array(flags+metadata, dtype=np.float32)))


def command_features(char, command, state=None):
    """Facing is encoded as a direction, never the arbitrary BC watch-cell label."""
    result = np.zeros(ACTION_WIDTH, dtype=np.float32)
    point, payload = command, "MOVE"
    if isinstance(command, tuple) and len(command) >= 2 and isinstance(command[1], (str, dict)):
        point, payload = command[:2]
    mode = payload if isinstance(payload, str) else "ABILITY" if "ability" in payload else "ULTIMATE" if "ultimate" in payload else "WATCH" if tuple(point) == tuple(char.pos) else "MOVE_WATCH"
    if mode == "HOLD":
        mode = "MOVE"
    if mode in MODES:
        result[MODES.index(mode)] = 1.
    offset = len(MODES)
    result[offset:offset+2] = np.asarray(point, dtype=np.float32)/ARENA_SHAPE
    if isinstance(payload, dict):
        direction = payload.get("facing")
        if direction in FACING_DIRS:
            from game_core import FACING_VECTORS
            result[offset+2:offset+4] = FACING_VECTORS[direction]
        if payload.get("target") is not None:
            result[offset+4:offset+6] = np.asarray(payload["target"])/ARENA_SHAPE
        own = sorted((c for c in (state or {}).get("chars", ()) if c.team == char.team and c.is_alive), key=lambda c:str(c.name))
        for index, ally in enumerate(own[:5]):
            if ally.name == payload.get("target_name"):
                result[offset+6+index] = 1.
    return result


class MatchHistory:
    def __init__(self, journal_path=None):
        self.events, self.results = [], []
        self.journal_path = Path(journal_path) if journal_path else None
        self.round_id = -1
        self.goal_sites = {}  # Own sampled intentions, indexed by round; no teacher route/future outcome.
        self._encoded = None
        self._cache_policy = None
        self.live = False
        self._game_round = None
        self._finished_count = 0

    @classmethod
    def live_match(cls, log_root):
        history = cls(Path(log_root)/(uuid.uuid4().hex+".jsonl.gz"))
        history.live = True
        return history

    def sync_round(self, view):
        """Share the same ledger across attack/defense controllers and side swaps."""
        records = getattr(getattr(view, "analytics_tracker", None), "round_records", ())
        if len(records) > self._finished_count:
            result = records[-1]
            self.finish_round(dict(success=result.get("winner") == "attacker",
                planted=bool(result.get("planted", False)), reason=result.get("reason", "unknown")))
            self._finished_count = len(records)
        token = int(getattr(view, "current_round", 1))
        if token != self._game_round:
            self.begin_round()
            self._game_round = token

    def begin_round(self):
        self.round_id += 1

    def append(self, observation, action, *, name="", tick=0, command=None):
        index = len(self.events)
        self.events.append(dict(obs=np.array(observation, dtype=np.float32, copy=True), action=np.array(action, dtype=np.float32, copy=True),
                               round_id=self.round_id, tick=int(tick), name=str(name), command=command,
                               goal_site=self.goal_sites.get(self.round_id)))
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.journal_path, "at", encoding="utf-8") as handle:
                handle.write(json.dumps(self.events[-1], ensure_ascii=False,
                    default=lambda v:v.item() if isinstance(v, np.generic) else list(v))+"\n")
        return index

    def finish_round(self, result):
        if self.results and self.results[-1]["round_id"] == self.round_id:
            return
        public = {key:result[key] for key in ("success", "planted", "reason") if key in result}
        public["round_id"] = self.round_id
        self.results.append(public)
        if self.events:
            observation = self.events[-1]["obs"].copy()
            observation[-EXTRA_WIDTH+6] = 1.  # Boundary, not a policy decision.
            observation[-3:] = [float(public.get("success", False)), float(public.get("planted", False)), 1.]
            self.append(observation, np.zeros(ACTION_WIDTH, dtype=np.float32), name="round_end", tick=self.events[-1]["tick"],
                        command={"round_result":public})

    def save(self, path):
        """All events survive; there is no last-N frame/window truncation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, obs=np.stack([e["obs"] for e in self.events]),
                action=np.stack([e["action"] for e in self.events]),
                metadata=np.array(json.dumps(dict(results=self.results, round_id=self.round_id,
                    events=[{k:v for k,v in e.items() if k not in ("obs", "action")} for e in self.events]),
                    ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else list(v))))

    @classmethod
    def load(cls, path):
        history = cls()
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            obs, actions = data["obs"], data["action"]
            if obs.shape != (len(actions), HISTORY_OBS_DIM) or actions.shape[1:] != (ACTION_WIDTH,):
                raise ValueError("Incompatible match history")
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError("Non-finite history")
            history.events = [dict(event, obs=o.copy(), action=a.copy())
                              for event,o,a in zip(metadata["events"], obs, actions)]
            history.results, history.round_id = metadata["results"], metadata["round_id"]
            history.goal_sites = {int(e["round_id"]):int(e["goal_site"]) for e in history.events if e.get("goal_site") is not None}
        return history
