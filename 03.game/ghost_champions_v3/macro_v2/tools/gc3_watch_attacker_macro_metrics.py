"""Read only three-output macro logs, including the initial BC warmup."""
from pathlib import Path
import sys

MACRO_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_macro_three_outputs_v2'
REFRESH_SECONDS=5
ROLLING_EPISODES=20
WINDOW_GEOMETRY='1280x900'
UPDATE_METRICS_FILENAME='ppo_updates.jsonl'
sys.path.insert(0,str(MACRO_ROOT.parents[1]))


def main():
    from ghost_champions_v3.macro_v1.tools import gc3_watch_attacker_macro_metrics as plots
    from ghost_champions_v3.tools import gc3_watch_training_metrics as charts
    from ghost_champions_v3.tools.gc3_metrics_monitor_data import read_complete_lines
    plots.RUN_NAME=RUN_NAME;plots.ROLLING_EPISODES=ROLLING_EPISODES
    charts.LOG_ROOT=MACRO_ROOT/'logs'
    charts.RUNS=((RUN_NAME,'三出力アタッカーマクロ','#7c3aed'),);charts.DEFAULT_RUN=RUN_NAME
    charts.REFRESH_SECONDS=REFRESH_SECONDS;charts.ROLLING_EPISODES=ROLLING_EPISODES;charts.WINDOW_GEOMETRY=WINDOW_GEOMETRY

    class MacroMonitor(charts.TrainingMonitor):
        def render(self,data,color):
            self.axes=plots.render(self.figure,dict(data,final=None))
            self.figure.suptitle('GC v3 三出力マクロ：'+RUN_NAME,fontsize=13)
            if not data['metrics']:
                bc=read_complete_lines(charts.LOG_ROOT/RUN_NAME/'bc_metrics.jsonl')
                if bc:
                    ax=self.axes[0];ax.clear();ax.grid(alpha=.2);ax.set_title('初期BCの模倣損失')
                    ax.plot([r['epoch'] for r in bc],[r['train_loss'] for r in bc],color=color,marker='.')
                    ax.set_xlabel('BC epoch');ax.set_ylabel('joint action loss')
            final=data['final']
            if final and final.get('step') is not None:
                label='BC後bestの最終勝率' if final['step']==0 else '保存bestの最終勝率'
                self.axes[3].scatter([final['step']],[100*final['success_rate']],color='#dc2626',marker='X',s=65,
                    label=label,zorder=6);self.axes[3].legend(fontsize=8)
            updates=read_complete_lines(charts.LOG_ROOT/RUN_NAME/UPDATE_METRICS_FILENAME)
            updates=[row for row in updates if row['step']<=data['status'].get('step',0)]
            if updates:
                layout=self.figure.add_gridspec(3,2)
                for index,ax in enumerate(self.axes):ax.set_subplotspec(layout[index])
                kl_ax=self.figure.add_subplot(layout[4]);value_ax=self.figure.add_subplot(layout[5])
                steps=[row['step'] for row in updates]
                kl_ax.plot(steps,[row['joint_kl'] for row in updates],color=color,label='11出力全体のKL')
                kl_ax.axhline(updates[-1]['target_kl'],color='#d97706',ls='--',label='更新停止の目安')
                kl_ax.axhline(updates[-1]['max_kl'],color='#dc2626',ls=':',label='更新を戻す上限')
                kl_ax.set_title('行動分布の変化量');kl_ax.legend(fontsize=8)
                value_ax.plot(steps,[row['value_mse_before'] for row in updates],color='#94a3b8',label='価値更新前')
                value_ax.plot(steps,[row['value_mse_after'] for row in updates],color='#0284c7',label='価値更新後')
                value_ax.set_title('価値予測の平均二乗誤差');value_ax.legend(fontsize=8)
                for ax in (kl_ax,value_ax):ax.grid(alpha=.2);ax.set_xlabel('マクロST')
                self.axes=list(self.axes)+[kl_ax,value_ax]
            self.figure.tight_layout(pad=2);self.toolbar.update();self.canvas.draw_idle()

        def refresh(self,force=False):
            super().refresh(force)
            if self.last_snapshot and self.last_snapshot['status'].get('stage')=='macro_bc':
                status=self.last_snapshot['status']
                self.status.set(f"マクロBC 教師={status.get('teacher_steps',0)}件 epoch={status.get('bc_epoch',0)} "
                    f"loss={status.get('bc_loss',0):.4f} | PPO ST=0 | {RUN_NAME}")
    charts.TrainingMonitor=MacroMonitor
    charts.main()


if __name__=='__main__':main()
