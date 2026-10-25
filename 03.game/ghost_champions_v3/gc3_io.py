"""Atomic output with retries for transient Windows reader locks."""
import json
from pathlib import Path
import time

WRITE_RETRY_ATTEMPTS = 40
WRITE_RETRY_SECONDS = .05


def replace_file(source, destination):
    for attempt in range(WRITE_RETRY_ATTEMPTS):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == WRITE_RETRY_ATTEMPTS-1:
                raise
            time.sleep(WRITE_RETRY_SECONDS)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    replace_file(temporary, path)
