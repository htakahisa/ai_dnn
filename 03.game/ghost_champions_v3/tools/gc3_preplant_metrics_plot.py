"""Plot only observed BC losses and actual pre-plant PPO logs."""


def render_metrics(figure, bc, rl, *, rolling_episodes=20):
    figure.clear()
    bc_ax, reward_ax, entropy_ax, success_ax = figure.subplots(2, 2).reshape(-1)
    for ax in figure.axes:
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    bc_ax.set_title("BC：教師行動の訓練・検証損失")
    bc_ax.set_xlabel("epoch（訓練データの反復回数）")
    bc_ax.set_ylabel("負の対数尤度")
    if bc["metrics"]:
        epochs = [row["epoch"] for row in bc["metrics"]]
        bc_ax.plot(epochs, [row["train_loss"] for row in bc["metrics"]], marker="o", label="訓練")
        bc_ax.plot(epochs, [row["validation_loss"] for row in bc["metrics"]], marker="o", label="検証（別試合）")
        bc_ax.legend(fontsize=8, loc="upper left")
        evaluated = [row for row in bc["metrics"] if "plant_rate" in row]
        if evaluated:
            plants_ax = bc_ax.twinx()
            plants_ax.plot([row["epoch"] for row in evaluated], [100*row["plant_rate"] for row in evaluated],
                           color="#ea580c", ls="--", marker="s", label="BC実動作の設置率")
            plants_ax.plot([row["epoch"] for row in evaluated], [100*row["success_rate"] for row in evaluated],
                           color="#16a34a", ls=":", marker=".", label="BC実動作の成功率")
            plants_ax.set_ylabel("BC実動作の設置率・成功率 (%)")
            orbs = [row for row in evaluated if "orb_round_rate" in row]
            if orbs:
                plants_ax.plot([row["epoch"] for row in orbs], [100*row["orb_round_rate"] for row in orbs],
                               color="#0284c7", ls="-.", label="BCオーブ回収ラウンド率")
            plants_ax.set_ylim(-3, 103)
            plants_ax.legend(fontsize=8, loc="lower right")
    else:
        bc_ax.text(.5, .5, "BCの損失記録を待っています", ha="center", transform=bc_ax.transAxes)
    reward_ax.set_title("RL：AIに渡したreward・returnの平均")
    reward_ax.set_ylabel("収集バッチの平均")
    entropy_ax.set_title("RL：行動分布のエントロピー（更新前）")
    entropy_ax.set_ylabel("nats")
    actual_outcome = any("actual_round_win_rate" in row for row in rl["evaluations"])
    success_label = "勝率" if actual_outcome else "成功率"
    success_ax.set_title(f"RL：{success_label}・設置率")
    success_ax.set_ylabel("%")
    for ax in (reward_ax, entropy_ax, success_ax):
        ax.set_xlabel("RL step（5人の方策判断の合計）")
    metrics = rl["metrics"]
    if metrics:
        steps = [row["step"] for row in metrics]
        reward_ax.plot(steps, [row["reward_mean"] for row in metrics], color="#16a34a", label="reward")
        reward_ax.plot(steps, [row["return_mean"] for row in metrics], color="#0f172a", label="return")
        reward_ax.legend(fontsize=8)
        entropy_ax.plot(steps, [row["entropy_mean"] for row in metrics], color="#7c3aed")
    else:
        for ax in (reward_ax, entropy_ax):
            ax.text(.5, .5, "実測値の記録を待っています", ha="center", transform=ax.transAxes)
    if rl["steps"]:
        success_ax.plot(rl["steps"], rl["rolling_success"], color="#16a34a", alpha=.7,
                        label=f"学習中{success_label}：直近{rolling_episodes}ラウンド")
    evaluations = rl["evaluations"]
    if evaluations:
        success_ax.plot([row["step"] for row in evaluations], [100*row["success_rate"] for row in evaluations],
                        color="#0f172a", marker="o", label=f"固定評価：{success_label}")
        plants = [row for row in evaluations if "plant_rate" in row]
        if plants:
            success_ax.plot([row["step"] for row in plants], [100*row["plant_rate"] for row in plants],
                            color="#ea580c", ls="--", marker=".", label="固定評価：設置率")
        orbs = [row for row in evaluations if "orb_round_rate" in row]
        if orbs:
            success_ax.plot([row["step"] for row in orbs], [100*row["orb_round_rate"] for row in orbs],
                           color="#0284c7", ls="-.", label="固定評価：オーブ回収ラウンド率")
    if rl["steps"] or evaluations:
        success_ax.legend(fontsize=8)
    else:
        success_ax.text(.5, .5, "ラウンド・評価結果を待っています", ha="center", transform=success_ax.transAxes)
    success_ax.set_ylim(-3, 103)
    figure.tight_layout(pad=2)
