"""View recorded final-evaluation matches. Run from ghost_champions_v2/."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

AI_ROOT=Path(__file__).resolve().parents[1]
ROOT=AI_ROOT.parent
sys.path.insert(0,str(ROOT))
OPPONENTS=("TYG","OMG","FRC","FNC","GG","SPS")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="attacker_rl_defuse_20261007_eval2")
    parser.add_argument("--opponent",choices=OPPONENTS,default="OMG")
    parser.add_argument("--series",type=int,default=0,help="Zero-based series index")
    parser.add_argument("--map",type=int,default=1,help="One-based map number")
    parser.add_argument("--round",type=int)
    parser.add_argument("--check",action="store_true",help="Validate the replay without opening a window")
    args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in (".",".."):
        parser.error("run must be a directory name")
    status=json.loads((AI_ROOT/"logs"/args.run/"status.json").read_text(encoding="utf-8"))
    if not status.get("complete"):
        parser.error("This run has not completed its final evaluation")
    evaluation=Path(status["evaluation_output"])
    path=evaluation/f"series_{args.opponent}_{args.series:03d}.json"
    if not path.is_file():
        parser.error(f"Replay not found: {path}")
    series=json.loads(path.read_text(encoding="utf-8"))
    maps=[SimpleNamespace(**item) for item in series.get("maps",[])]
    if not 1<=args.map<=len(maps):
        parser.error(f"map must be between 1 and {len(maps)}")
    selected=maps[args.map-1]
    if not selected.replay_frames:
        parser.error("The selected map has no replay frames")
    if args.round is not None and not any(frame.get("round")==args.round for frame in selected.replay_frames):
        parser.error("The requested round is not present in this map")
    print(f"Replay: {path}\nMap {args.map}: {len(selected.replay_frames)} recorded frames; training step {status['step']}",flush=True)
    if args.check:
        return
    # The recorded frames supply positions and effects; no policy is rerun here.
    import tkinter as tk
    from tkinter import ttk
    from analytics.replay_viewer import ReplayViewer
    snapshot=AI_ROOT/"data/runtime_fixed_20261007"
    if (snapshot/"map_data.py").is_file():
        import importlib.util
        spec=importlib.util.spec_from_file_location("gc_replay_map",snapshot/"map_data.py")
        map_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(map_module)
        import analytics.replay_viewer as replay_module
        replay_module.NEW_MAZE_STR=map_module.NEW_MAZE_STR
    root=tk.Tk()
    root.withdraw()
    viewer=ReplayViewer(root,selected.replay_frames,map_options=maps,round_records=selected.round_records,
                        title=f"GC v2 学習後 vs {args.opponent} — 最終評価リプレイ")
    caption=ttk.Label(viewer,text="赤：攻撃側　緑：防衛側（攻守交代で色が変わります）　Playで再生／Round ±でラウンド移動",
                      padding=(8,8))
    caption.pack(before=viewer.canvas,fill=tk.X)
    if len(maps)>1:
        viewer.map_combo.current(args.map-1)
        viewer._map_changed()
    if args.round is not None:
        viewer.seek(next(i for i,frame in enumerate(viewer.frames) if frame.get("round")==args.round))
    def close():
        viewer.playing=False
        root.destroy()
    viewer.protocol("WM_DELETE_WINDOW",close)
    root.update_idletasks()
    viewer.lift()
    print(f"REPLAY_READY window={viewer.winfo_id()} paused=True",flush=True)
    root.mainloop()


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
