"""Live BC + pre-plant PPO charts. Run from ghost_champions_v3 without options."""
import argparse
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
BC_RUN = "attacker_preplant_retrieval_bc_v2"
RL_RUN = "attacker_preplant_retrieval_v2"
LOG_ROOT = AI_ROOT/"logs"
OUTPUT_ROOT = LOG_ROOT/"training_graphs"
REPORT_NAME = "gc3_attacker_micro_retrieval_training_progress"
REFRESH_SECONDS = 5
ROLLING_EPISODES = 20
WINDOW_GEOMETRY = "1280x900"
FIGURE_DPI = 160

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.tools.gc3_metrics_monitor_data import bc_snapshot, snapshot, log_signature
from ghost_champions_v3.tools.gc3_preplant_metrics_plot import render_metrics


def configure_fonts():
    import matplotlib
    from matplotlib import font_manager
    installed = {font.name for font in font_manager.fontManager.ttflist}
    japanese = [name for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP") if name in installed]
    matplotlib.rcParams.update({"font.family":japanese+["DejaVu Sans"], "axes.unicode_minus":False,
                               "font.size":9, "axes.spines.top":False, "axes.spines.right":False})


def save_charts(figure):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = OUTPUT_ROOT/f"{REPORT_NAME}.{suffix}"
        figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")


class PreplantMonitor:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.root, self.signature = root, None
        root.title("GC v3 設置前：BCと強化学習のグラフ")
        root.geometry(WINDOW_GEOMETRY)
        self.automatic = tk.BooleanVar(root, value=True)
        self.status = tk.StringVar(root, value="ログを読み込みます")
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        ttk.Checkbutton(controls, text=f"{REFRESH_SECONDS}秒ごとに自動更新", variable=self.automatic).pack(side="left")
        ttk.Button(controls, text="今すぐ更新", command=lambda:self.refresh(force=True)).pack(side="left", padx=10)
        ttk.Label(root, textvariable=self.status, padding=6).pack(fill="x")
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        ttk.Label(root, text=f"BC：{BC_RUN} ／ RL：{RL_RUN}　|　RL成功＝設置後まで含む実際のラウンド勝利", padding=6).pack(fill="x")
        self.refresh(force=True)
        self.after_id = root.after(int(REFRESH_SECONDS*1000), self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def refresh(self, force=False):
        try:
            signature = (log_signature(LOG_ROOT/BC_RUN), log_signature(LOG_ROOT/RL_RUN))
            if not force and signature == self.signature:
                return
            bc, rl = bc_snapshot(LOG_ROOT/BC_RUN), snapshot(LOG_ROOT/RL_RUN, ROLLING_EPISODES)
            render_metrics(self.figure, bc, rl, rolling_episodes=ROLLING_EPISODES)
            self.canvas.draw_idle()
            self.toolbar.update()
            last = rl["evaluations"][-1] if rl["evaluations"] else None
            evaluation = f"最新評価：成功 {100*last['success_rate']:.1f}%" if last else "評価記録待ち"
            if last is not None and "plant_rate" in last:
                evaluation += f"／設置 {100*last['plant_rate']:.1f}%"
            last_bc = bc["metrics"][-1] if bc["metrics"] else None
            orb_report = last if last is not None else last_bc
            if orb_report is not None and "orb_collections" in orb_report:
                evaluation += f"／オーブ回収完了 {orb_report['orb_collections']}回"
            self.status.set(f"BC {bc['epoch']} epoch ({'完了' if bc['complete'] else '監視中'})　|　"
                            f"RL {rl['step']:,} step ({'完了' if rl['complete'] else '監視中'})　|　{evaluation}")
            save_charts(self.figure)
            self.signature = signature
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.signature = None
            self.status.set(f"ログを読み直します：{error}")

    def poll(self):
        if self.automatic.get():
            self.refresh()
        self.after_id = self.root.after(int(REFRESH_SECONDS*1000), self.poll)

    def close(self):
        self.root.after_cancel(self.after_id)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true", help="Save PNG/SVG without a desktop window")
    args = parser.parse_args()
    if REFRESH_SECONDS <= 0 or ROLLING_EPISODES < 1:
        raise ValueError("Refresh interval and rolling window must be positive")
    import matplotlib
    matplotlib.use("Agg" if args.save_only else "TkAgg")
    configure_fonts()
    if args.save_only:
        from matplotlib.figure import Figure
        figure = Figure(figsize=(12, 8))
        render_metrics(figure, bc_snapshot(LOG_ROOT/BC_RUN), snapshot(LOG_ROOT/RL_RUN, ROLLING_EPISODES),
                       rolling_episodes=ROLLING_EPISODES)
        save_charts(figure)
        print(OUTPUT_ROOT/f"{REPORT_NAME}.png")
    else:
        import tkinter as tk
        root = tk.Tk()
        PreplantMonitor(root)
        root.mainloop()


if __name__ == "__main__":
    main()
