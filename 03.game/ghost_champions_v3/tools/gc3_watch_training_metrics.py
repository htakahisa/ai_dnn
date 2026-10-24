"""Live desktop charts; start from GC v3 with python tools/gc3_watch_training_metrics.py."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import font_manager
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = AI_ROOT/"logs"
RUNS = (("attacker_postplant_fixed_v1", "アタッカーポストプラント（修正後）", "#ea580c"),
        ("attacker_preplant_rl_history_v2", "アタッカー設置前（移動修正後・履歴BC後の強化学習）", "#16a34a"),
        ("attacker_preplant_rl_history_v1", "アタッカー設置前（移動修正前）", "#94a3b8"),
        ("attacker_preplant_rl_v1", "アタッカー設置前（旧・履歴なし）", "#94a3b8"),
        ("defender_search_fixed_v1", "ディフェンダーサーチ（修正後）", "#2563eb"),
        ("attacker_postplant_reward_v1", "アタッカー（修正前）", "#94a3b8"),
        ("defender_search_reward_v1", "ディフェンダー（修正前）", "#64748b"))
DEFAULT_RUN = "attacker_postplant_fixed_v1"
REFRESH_SECONDS = 5
ROLLING_EPISODES = 20
WINDOW_GEOMETRY = "1280x900"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gc3_metrics_monitor_data import log_signature, snapshot

def configure_fonts():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    japanese = [name for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP") if name in installed]
    matplotlib.rcParams.update({"font.family": japanese+["DejaVu Sans"], "axes.unicode_minus": False,
                               "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def waiting(ax, text):
    ax.text(.5,.5,text,ha="center",va="center",transform=ax.transAxes,color="#64748b",fontsize=11)
    ax.set_yticks([])

class TrainingMonitor:
    def __init__(self, root):
        self.root = root
        root.title("GC v3 学習グラフ（run別）")
        root.geometry(WINDOW_GEOMETRY)
        root.minsize(900,650)
        self.last_signature = None
        self.last_snapshot = None
        self.last_loaded = None
        self.automatic = tk.BooleanVar(value=True)
        self.selected = tk.StringVar(value=next(label for name,label,_ in RUNS if name==DEFAULT_RUN))
        controls = ttk.Frame(root,padding=(12,10))
        controls.pack(fill="x")
        ttk.Label(controls,text="対象").pack(side="left",padx=(0,6))
        choices = ttk.Combobox(controls,textvariable=self.selected,state="readonly",
                               values=[label for _,label,_ in RUNS],width=30)
        choices.pack(side="left")
        choices.bind("<<ComboboxSelected>>",self.change_run)
        ttk.Checkbutton(controls,text=f"{REFRESH_SECONDS}秒ごとに自動更新",variable=self.automatic).pack(side="left",padx=16)
        ttk.Button(controls,text="今すぐ更新",command=lambda:self.refresh(force=True)).pack(side="left")
        self.status = tk.StringVar(value="ログを読み込んでいます")
        ttk.Label(root,textvariable=self.status,padding=(12,0)).pack(fill="x")
        self.figure = Figure(figsize=(12,7.5),dpi=100)
        self.axes = self.figure.subplots(2,2).reshape(-1)
        self.canvas = FigureCanvasTkAgg(self.figure,master=root)
        self.canvas.get_tk_widget().pack(fill="both",expand=True)
        toolbar_frame = ttk.Frame(root)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas,toolbar_frame)
        ttk.Label(root,text="reward：収集行動の報酬平均 ／ return：価値学習の目標平均 ／ エントロピー：PPO更新前の行動分布",
                  padding=(12,6)).pack(fill="x")
        self.refresh(force=True)
        root.after(int(REFRESH_SECONDS*1000),self.poll)

    def run_info(self):
        return next(row for row in RUNS if row[1]==self.selected.get())

    def change_run(self,event=None):
        self.last_signature = None
        self.last_snapshot = None
        self.last_loaded = None
        for ax in self.axes:
            ax.clear()
            waiting(ax,"読み込み中")
        self.canvas.draw_idle()
        self.refresh(force=True)

    def poll(self):
        if self.automatic.get():
            self.refresh()
        self.root.after(int(REFRESH_SECONDS*1000),self.poll)

    def refresh(self,force=False):
        name,label,color = self.run_info()
        root = LOG_ROOT/name
        try:
            signature = log_signature(root)
            now = datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M:%S")
            if force or signature != self.last_signature:
                data = snapshot(root,ROLLING_EPISODES)
                self.render(data,color)
                self.last_snapshot = data
                self.last_signature = signature
                self.last_loaded = now
            data = self.last_snapshot
            measured = f"実測：{data['measured_from']:,} step以降" if data["measured_from"] is not None else "実測値の記録待ち"
            phase = "学習完了" if data["complete"] else "ログを監視中"
            self.status.set(f"{label} [{name}]  |  {data['step']:,} step  |  {len(data['episodes'])}試合  |  {phase}  |  {measured}"
                            f"  |  読込 {self.last_loaded}／確認 {now}")
        except (OSError,ValueError,KeyError,TypeError) as exc:
            # Keep the last complete chart while an atomic replacement/append
            # is in progress; retry at the next poll.
            self.last_signature = None
            self.status.set(f"ログを読み直します（{exc}）")

    def render(self,data,color):
        reward_ax,return_ax,entropy_ax,win_ax = self.axes
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=.2)
            ax.set_xlabel("学習step（学習側の行動選択回数）")
            ax.ticklabel_format(axis="x",style="plain",useOffset=False)
        metrics = data["metrics"]
        for ax,field,title in ((reward_ax,"reward_mean","AIに渡したrewardの平均"),
                               (return_ax,"return_mean","価値学習のreturnの平均"),
                               (entropy_ax,"entropy_mean","行動分布のエントロピー")):
            ax.set_title(title)
            if metrics:
                ax.plot([row["step"] for row in metrics],[row[field] for row in metrics],color=color,marker=".",lw=1.8)
                first = data["measured_from"]
                last = max(data["step"],metrics[-1]["step"])
                ax.set_xlim(first,max(first+1,last)+(last-first)*.03)
            else:
                waiting(ax,"実測値の記録を待っています")
            ax.set_ylabel("nats" if field=="entropy_mean" else "収集バッチの平均値")
        win_ax.set_title("フェーズ成功率")
        if data["steps"]:
            win_ax.plot(data["steps"],data["rolling_success"],color=color,lw=1.7,
                        label=f"学習中：直近{ROLLING_EPISODES}試合")
        if data["evaluations"]:
            win_ax.plot([row["step"] for row in data["evaluations"]],
                        [100*row["success_rate"] for row in data["evaluations"]],
                        color="#0f172a",marker="o",label="定期評価")
        candidates = [row for row in data["evaluations"] if row["step"]>0]
        if data["final"] is not None and candidates:
            best = max(candidates,key=lambda row:row["success_rate"])
            win_ax.scatter([best["step"]],[100*data["final"]["success_rate"]],color="#dc2626",marker="X",s=65,
                           label="保存bestの最終評価",zorder=6)
        if data["steps"] or data["evaluations"]:
            win_ax.legend(fontsize=8)
        else:
            waiting(win_ax,"試合結果の記録を待っています")
        win_ax.set_ylabel("成功率（%）")
        win_ax.set_ylim(-3,max(35,max(data["rolling_success"],default=0)+5,
                             max((100*row["success_rate"] for row in data["evaluations"]),default=0)+5))
        win_ax.set_xlim(0,max(1,data["step"])*1.03)
        self.figure.tight_layout(pad=2)
        self.toolbar.update()
        self.canvas.draw_idle()

def main():
    if REFRESH_SECONDS <= 0 or ROLLING_EPISODES < 1:
        raise ValueError("Refresh interval and rolling window must be positive")
    configure_fonts()
    root = tk.Tk()
    TrainingMonitor(root)
    root.mainloop()

if __name__=="__main__":
    main()
