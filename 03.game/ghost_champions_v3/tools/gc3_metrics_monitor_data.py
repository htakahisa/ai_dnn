"""Read-only snapshots of training logs, safe while JSONL files are appended."""
import json
from pathlib import Path

LOG_FILENAMES = ("episodes.jsonl", "training_metrics.jsonl", "status.json",
                 "initial_evaluation.json", "final_evaluation.json", "bc_metrics.jsonl")

def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def read_complete_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    # A writer may have appended a partial JSON object or UTF-8 character.
    # Newline is the commit boundary used by the training logger.
    end = raw.rfind(b"\n")
    if end < 0:
        return []
    lines = raw[:end+1].decode("utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} の {number} 行目を読み込めません") from exc
    return rows

def log_signature(root):
    root = Path(root)
    paths = [root/name for name in LOG_FILENAMES]
    paths.extend(root.glob("evaluation_*.json"))
    signature = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
            signature.append((path.name, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append((path.name, None, None))
    return tuple(signature)

def snapshot(root, window=20):
    if window < 1:
        raise ValueError("Moving-average window must be positive")
    root = Path(root)
    episodes = read_complete_lines(root/"episodes.jsonl")
    metrics = read_complete_lines(root/"training_metrics.jsonl")
    evaluations = []
    initial = read_json(root/"initial_evaluation.json")
    if initial is not None:
        evaluations.append(dict(initial, step=int(initial.get("step", 0))))
    for path in root.glob("evaluation_*.json"):
        evaluations.append(dict(read_json(path), step=int(path.stem.split("_")[-1])))
    evaluations.sort(key=lambda row: row["step"])
    status = read_json(root/"status.json") or {}
    steps, rolling, accumulated = [], [], int(initial.get("step", 0)) if initial else 0
    for index, episode in enumerate(episodes):
        accumulated += int(episode["decisions"])
        steps.append(accumulated)
        recent = episodes[max(0,index-window+1):index+1]
        rolling.append(100*sum(bool(row["success"]) for row in recent)/len(recent))
    current_step = max(accumulated, int(status.get("step",0)),
                       max((int(row["step"]) for row in metrics), default=0))
    complete = bool(status.get("complete")) and int(status.get("step",0)) >= current_step
    final = read_json(root/"final_evaluation.json") if complete else None
    measured_from = min((int(row["start_step"]) for row in metrics),default=None)
    return dict(episodes=episodes, metrics=metrics, evaluations=evaluations, status=status,
                steps=steps, rolling_success=rolling, step=current_step, complete=complete,
                measured_from=measured_from, final=final)


def bc_snapshot(root):
    root = Path(root)
    rows = read_complete_lines(root/"bc_metrics.jsonl")
    status = read_json(root/"status.json") or {}
    epoch = max(int(status.get("epoch", 0)), max((int(row["epoch"]) for row in rows), default=0))
    step = max(int(status.get("step", 0)), max((int(row["step"]) for row in rows), default=0))
    return dict(metrics=rows, epoch=epoch, step=step, status=status,
                complete=bool(status.get("complete")) and int(status.get("epoch", 0)) >= epoch)
