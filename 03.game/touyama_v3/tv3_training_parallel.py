"""Bounded opponent workers with isolated RNG, cwd, stdout redirection and logs."""
from pathlib import Path
import subprocess
import sys
import threading


def run_opponent_workers(script, argv, opponents, max_workers, *, log_dir=None):
    """Return None for ordinary execution, otherwise the workers' exit status.

    Children use the same interpreter/source constants and explicit CLI overrides.
    Model/data paths stay unchanged; shared log files get opponent directories.
    No shell, game, model or collector runs inside the supervising process.
    """
    if max_workers < 1:
        raise ValueError("MAX_PARALLEL_WORKERS must be at least 1")
    opponents = tuple(opponents)
    if not opponents or len(opponents) != len(set(opponents)):
        raise ValueError("Opponent workers require nonempty distinct opponents")
    if max_workers == 1 or len(opponents) == 1:
        return None
    script = Path(script).resolve()
    arguments = list(sys.argv[1:] if argv is None else argv)
    pending = iter(opponents)
    active = {}
    limit = min(max_workers, len(opponents))
    print(f"[並列学習] 最大同時実行={limit} 相手AI={','.join(opponents)}", flush=True)
    stopped = threading.Event()
    try:
        while True:
            while len(active) < limit:
                opponent = next(pending, None)
                if opponent is None:
                    break
                command = [sys.executable, "-u", str(script), *arguments,
                           "--opponents", opponent, "--max-workers", "1"]
                if log_dir is not None:
                    command += ["--log-dir", str(Path(log_dir).resolve() / opponent)]
                process = subprocess.Popen(command, cwd=str(script.parent))
                active[process] = opponent
                print(f"[並列学習] 開始 {opponent} pid={process.pid}", flush=True)
            if not active:
                return 0
            for process, opponent in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                del active[process]
                print(f"[並列学習] {'完了' if status == 0 else '失敗'} {opponent} exit={status}", flush=True)
                if status:
                    print("[並列学習] 他のworkerを停止します。完了セットのlatestから再開できます。", flush=True)
                    return status
            stopped.wait(.1)
    except KeyboardInterrupt:
        print("[並列学習] 中断。完了セットのlatestは保持します。", flush=True)
        return 130
    finally:
        # Includes launch failures and interrupts; never leave orphan learners.
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
