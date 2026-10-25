"""Learn public defuse responses in a new run. Run from postplant_v2 with no options."""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parent
AI_ROOT=GUARD_ROOT.parent
sys.path.insert(0,str(AI_ROOT.parent))
from ghost_champions_v3.postplant_v2.gc3_train_attacker_guard import *

RUN_NAME='attacker_postplant_response_v3'
START_CHECKPOINT=GUARD_ROOT/'data/attacker_postplant_antidefuse_v1/best.pt'
RUN_PATH=GUARD_ROOT/'data'/RUN_NAME
LOG_PATH=GUARD_ROOT/'logs'/RUN_NAME
RESUME=True # The v3 run has started; retain optimizer and RNG state on the next launch.
TRAINING_STEPS=20000
RESPONSE_LOSS_WEIGHT=1.
RESPONSE_SAMPLE_FRACTION=.5
ANCHOR_KL_WEIGHT=2.
EVALUATION_PATIENCE=2
KL_BACKTRACK_STEPS=6


def process_is_running(pid):
    if pid<=0:return False
    import os
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return False
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def show_existing_run(lock):
    import json
    try:pid=int(lock.read_text(encoding='utf-8'))
    except (OSError,ValueError):pid=0
    if not process_is_running(pid):
        raise SystemExit(f'学習ロックが残っています（PID={pid or "不明"}）。停止済みか確認が必要です: {lock}')
    print(f'学習はすでに実行中です（PID={pid}、run={RUN_NAME}）。',flush=True)
    try:
        status=json.loads((LOG_PATH/'status.json').read_text(encoding='utf-8'))
        print(f'保存済み ST={status["step"]:,} / 目標 {TRAINING_STEPS:,}',flush=True)
    except (OSError,ValueError,KeyError):pass
    print('二重起動を避けるため、このコマンドは終了します。実行中の学習は継続します。',flush=True)
    print('進捗表示: python tools/gc3_watch_attacker_guard_response_metrics.py',flush=True)


if __name__=='__main__':
    lock=RUN_PATH/'training.lock'
    if lock.exists():show_existing_run(lock)
    else:
        from ghost_champions_v3.postplant_v2.gc3_training_attacker_guard import train
        try:train({name:value for name,value in globals().copy().items() if name.isupper()})
        except FileExistsError as error:
            if error.filename is None or Path(error.filename).resolve()!=lock.resolve():raise
            show_existing_run(lock)
