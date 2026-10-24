"""Plot actual PPO transition metrics and recorded phase success rates."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

AI_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = AI_ROOT/"logs"
OUTPUT_ROOT = LOG_ROOT/"training_graphs"
RUNS = (("defender_search_fixed_v1", "ディフェンダーサーチ（修正後）", "#2563eb"),
        ("attacker_postplant_fixed_v1", "アタッカーポストプラント（修正後）", "#ea580c"))
ROLLING_EPISODES = 20
REPORT_NAME = "gc3_training_progress"
FIGURE_DPI = 160

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gc3_metrics_monitor_data import snapshot

def read_run(name):
    root = LOG_ROOT/name
    data = snapshot(root, ROLLING_EPISODES)
    evaluations = [(row["step"], row) for row in data["evaluations"]]
    for _, report in evaluations:
        actual = sum(bool(row["success"]) for row in report["rounds"])/len(report["rounds"])
        if not np.isclose(actual, report["success_rate"]):
            raise ValueError(f"Inconsistent evaluation: {name}")
    return dict(episodes=data["episodes"], steps=np.asarray(data["steps"]),
                moving_win=np.asarray(data["rolling_success"])/100., evaluations=evaluations,
                metrics=data["metrics"], final=data["final"], step=data["step"], complete=data["complete"])

def configure_fonts():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    japanese = [name for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP") if name in installed]
    plt.rcParams.update({"font.family": japanese+["DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def unavailable(ax, message):
    ax.text(.5,.5,message,ha="center",va="center",transform=ax.transAxes,fontsize=11,color="#64748b")
    ax.set_yticks([])

def main():
    if ROLLING_EPISODES < 1:
        raise ValueError("ROLLING_EPISODES must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    fig, axes = plt.subplots(3, len(RUNS), figsize=(13,11.5), squeeze=False)
    summaries = {}
    for column, (name, title, color) in enumerate(RUNS):
        data = read_run(name)
        steps = data["steps"]
        evaluations = data["evaluations"]
        reward_ax, entropy_ax, win_ax = axes[:, column]
        reward_ax.set_title(f"{title}\n{name}", fontsize=11, weight="bold")
        metrics = data["metrics"]
        if metrics:
            metric_steps = [row["step"] for row in metrics]
            reward_ax.plot(metric_steps, [row["reward_mean"] for row in metrics], color=color,marker=".",
                           label="収集行動のreward平均")
            reward_ax.plot(metric_steps, [row["return_mean"] for row in metrics], color="#0f172a",marker=".",
                           label="価値学習のreturn平均")
            reward_ax.legend(fontsize=8)
            entropy_ax.plot(metric_steps, [row["entropy_mean"] for row in metrics],color=color,marker=".")
        else:
            unavailable(reward_ax,"実測reward・returnの記録を待っています")
            unavailable(entropy_ax,"エントロピーの記録を待っています")
        reward_ax.set_ylabel("AIが使う学習値（収集バッチの平均）")
        entropy_ax.set_ylabel("行動分布のエントロピー平均（nats）")
        eval_steps = [step for step, _ in data["evaluations"]]
        eval_rates = [100*report["success_rate"] for _, report in data["evaluations"]]
        counts = sorted({len(report["rounds"]) for _,report in data["evaluations"]})
        count_label = "/".join(str(count) for count in counts)
        if len(steps):
            win_ax.plot(steps, data["moving_win"]*100, color=color, lw=1.8, alpha=.7,
                        label=f"学習中：直近{ROLLING_EPISODES}試合")
        if evaluations:
            win_ax.plot(eval_steps, eval_rates, color="#0f172a", marker="o", lw=1.6,
                        label=f"固定評価：各{count_label}試合")
        candidates = [(step, report) for step, report in data["evaluations"] if step>0]
        if candidates and data["final"] is not None:
            best_step = max(candidates, key=lambda item: item[1]["success_rate"])[0]
            win_ax.scatter([best_step],[100*data["final"]["success_rate"]],marker="X",s=80,color="#dc2626",
                           zorder=6,label="保存bestモデルの最終再評価")
        for step, rate in zip(eval_steps,eval_rates):
            win_ax.annotate(f"{rate:.1f}%",(step,rate),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
        if not len(steps) and not evaluations:
            unavailable(win_ax,"試合結果の記録を待っています")
        win_ax.set_ylim(-4,max(35,max(eval_rates, default=0)+10,
                              max(data["moving_win"], default=0)*100+5))
        win_ax.set_ylabel("フェーズ成功率（%）")
        if len(steps) or evaluations:
            win_ax.legend(loc="upper left",fontsize=8)
        for ax in (reward_ax,entropy_ax,win_ax):
            ax.set_xlabel("学習step（学習側の行動選択回数）")
            ax.grid(alpha=.2)
            ax.set_xlim(0,max(1,data["step"])*1.03)
            ax.ticklabel_format(axis="x",style="plain",useOffset=False)
        with (OUTPUT_ROOT/f"{name}_episodes.csv").open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.writer(handle)
            writer.writerow(("episode","step","success","rolling_success_rate"))
            for index,row in enumerate(data["episodes"]):
                writer.writerow((index+1,int(steps[index]),int(row["success"]),data["moving_win"][index]))
        with (OUTPUT_ROOT/f"{name}_evaluation.csv").open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.writer(handle)
            writer.writerow(("step","success_rate","successes","rounds"))
            for step,report in data["evaluations"]:
                writer.writerow((step,report["success_rate"],sum(row["success"] for row in report["rounds"]),len(report["rounds"])))
        with (OUTPUT_ROOT/f"{name}_actual_metrics.csv").open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.writer(handle)
            fields=("step","transitions","reward_mean","return_mean","advantage_mean","value_mean","entropy_mean")
            writer.writerow(fields)
            writer.writerows([row[field] for field in fields] for row in metrics)
        initial = next((report for step, report in evaluations if step==0), None)
        summaries[name]=dict(step=data["step"],episodes=len(data["episodes"]),complete=data["complete"],
                             training_success_rate=float(np.mean([row["success"] for row in data["episodes"]])) if data["episodes"] else None,
                             initial_evaluation_rate=initial["success_rate"] if initial else None,
                             final_best_evaluation_rate=data["final"]["success_rate"] if data["final"] else None,
                             actual_reward_recorded=bool(metrics),entropy_recorded=bool(metrics),
                             recorded_batches=len(metrics))
    fig.suptitle("GC v3 新報酬の学習値と成功率",fontsize=18,weight="bold",y=.985)
    fig.text(.5,.025,"報酬とreturnは行動データから直接記録した値だけを使用します。未記録区間は補完しません。\n"
             "returnは価値学習の目標。エントロピーはPPO更新前の選択分岐の各分布の和です。",
             ha="center",fontsize=9,color="#475569")
    fig.tight_layout(rect=(0,.065,1,.95))
    for suffix in ("png","svg"):
        destination=OUTPUT_ROOT/f"{REPORT_NAME}.{suffix}"
        fig.savefig(destination,dpi=FIGURE_DPI,facecolor="white")
        print(destination)
    plt.close(fig)
    (OUTPUT_ROOT/"summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
