"""Live read-only macro charts; run from macro_v1 without options."""
from pathlib import Path
import argparse
import sys

MACRO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "attacker_macro_v1"
LOG_ROOT = MACRO_ROOT/"logs"
OUTPUT_ROOT = LOG_ROOT/"training_graphs"
REPORT_NAME = "gc3_attacker_macro_training_progress"
REFRESH_SECONDS = 5
ROLLING_EPISODES = 20
WINDOW_GEOMETRY = "1280x900"
FIGURE_DPI = 160

sys.path.insert(0, str(MACRO_ROOT.parents[1]))
from ghost_champions_v3.tools.gc3_metrics_monitor_data import snapshot
from ghost_champions_v3.tools.gc3_watch_attacker_preplant_metrics import configure_fonts


def render(figure, data):
    figure.clear()
    axes = figure.subplots(2, 2).reshape(-1)
    for ax, field, title in zip(axes[:3], ("reward_mean", "return_mean", "entropy_mean"),
                              ("AIに渡したrewardの平均", "価値学習のreturnの平均", "マクロ行動分布のエントロピー")):
        ax.set_title(title)
        ax.set_ylabel("nats" if field == "entropy_mean" else "収集バッチの平均")
        if data["metrics"]:
            ax.plot([r["step"] for r in data["metrics"]], [r[field] for r in data["metrics"]],
                    color="#7c3aed", marker=".")
        else:
            ax.text(.5, .5, "実測値の記録待ち", ha="center", transform=ax.transAxes)
    rates = axes[3]
    rates.set_title("実勝率・設置率・解除未開始の勝利ボーナス率")
    rates.set_ylabel("%")
    if data["steps"]:
        rates.plot(data["steps"], data["rolling_success"], color="#16a34a", alpha=.65,
                   label=f"学習中勝率：直近{ROLLING_EPISODES}ラウンド")
    evaluated = data["evaluations"]
    for field, label, color, style in (("success_rate", "固定評価：実勝率", "#0f172a", "-"),
                                        ("plant_rate", "固定評価：設置率", "#ea580c", "--"),
                                        ("no_defuse_bonus_rate", "固定評価：追加+10の獲得率", "#0284c7", ":")):
        rows = [r for r in evaluated if field in r]
        if rows:
            rates.plot([r["step"] for r in rows], [100*r[field] for r in rows],
                       color=color, ls=style, marker=".", label=label)
    candidates = [r for r in evaluated if r["step"]>0]
    if data["final"] is not None and candidates:
        # Macro best is selected by the user's reward, not by peak win rate.
        best = max(candidates, key=lambda r:(r["reward_mean"], r["success_rate"], r["plant_rate"]))
        rates.scatter([best["step"]], [100*data["final"]["success_rate"]],
                      color="#dc2626", marker="X", s=65, label="保存bestの最終実勝率", zorder=6)
    if rates.has_data():
        rates.legend(fontsize=8)
    else:
        rates.text(.5, .5, "ラウンド・評価記録待ち", ha="center", transform=rates.transAxes)
    rates.set_ylim(-3, 103)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlabel("マクロST（チーム共通の上位判断回数）")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    figure.suptitle(f"アタッカーマクロ v1：{RUN_NAME}", fontsize=13)
    figure.tight_layout(pad=2)
    return axes


def save(figure):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(OUTPUT_ROOT/f"{REPORT_NAME}.{suffix}", dpi=FIGURE_DPI, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true", help="Temporary export without a window")
    args = parser.parse_args()
    if REFRESH_SECONDS <= 0 or ROLLING_EPISODES < 1:
        raise ValueError("Refresh interval and rolling window must be positive")
    import matplotlib
    if args.save_only:
        matplotlib.use("Agg")
        configure_fonts()
        from matplotlib.figure import Figure
        figure = Figure(figsize=(12, 8))
        render(figure, snapshot(LOG_ROOT/RUN_NAME, ROLLING_EPISODES))
        save(figure)
        print(OUTPUT_ROOT/f"{REPORT_NAME}.png")
        return
    from ghost_champions_v3.tools import gc3_watch_training_metrics as monitor
    monitor.LOG_ROOT = LOG_ROOT
    monitor.RUNS = ((RUN_NAME, "アタッカーマクロ v1", "#7c3aed"),)
    monitor.DEFAULT_RUN = RUN_NAME
    monitor.REFRESH_SECONDS = REFRESH_SECONDS
    monitor.ROLLING_EPISODES = ROLLING_EPISODES
    monitor.WINDOW_GEOMETRY = WINDOW_GEOMETRY
    configure_fonts()

    class MacroMonitor(monitor.TrainingMonitor):
        def render(self, data, color):
            self.axes = render(self.figure, data)
            self.toolbar.update()
            self.canvas.draw_idle()
            save(self.figure)

    root = monitor.tk.Tk()
    MacroMonitor(root)
    root.title("GC v3 アタッカーマクロ：学習グラフ")
    root.mainloop()


if __name__ == "__main__":
    main()
