"""Auto-updating guard reward, actual defuse outcomes and PPO stability charts."""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_postplant_antidefuse_v1'
REFRESH_SECONDS=5
ROLLING_EPISODES=20
WINDOW_GEOMETRY='1280x900'
sys.path.insert(0,str(GUARD_ROOT.parents[1]))


def main():
    from ghost_champions_v3.tools import gc3_watch_training_metrics as charts
    from ghost_champions_v3.tools.gc3_metrics_monitor_data import read_complete_lines
    charts.LOG_ROOT=GUARD_ROOT/'logs';charts.RUNS=((RUN_NAME,'ポストプラント：解除阻止','#ea580c'),);charts.DEFAULT_RUN=RUN_NAME
    charts.REFRESH_SECONDS=REFRESH_SECONDS;charts.ROLLING_EPISODES=ROLLING_EPISODES;charts.WINDOW_GEOMETRY=WINDOW_GEOMETRY
    class GuardMonitor(charts.TrainingMonitor):
        def render(self,data,color):
            self.figure.clear();self.axes=self.figure.subplots(3,2).reshape(-1)
            for ax,key,title in zip(self.axes[:3],('reward_mean','return_mean','entropy_mean'),('実際に渡した報酬の平均','価値学習のreturn平均','行動分布のエントロピー')):
                ax.set_title(title)
                if data['metrics']:ax.plot([r['step'] for r in data['metrics']],[r[key] for r in data['metrics']],color=color)
            ax=self.axes[3];ax.set_title('ポストプラント実勝率と解除への対応');ax.set_ylim(-3,103)
            for key,label,style in [('success_rate','固定評価：実勝率','-'),('defused_rate','解除された率','--'),('defuse_started_win_rate','解除開始された試合の勝率',':')]:
                rows=[r for r in data['evaluations'] if r.get(key) is not None]
                if rows:ax.plot([r['step'] for r in rows],[100*r[key] for r in rows],ls=style,marker='.',label=label)
            if ax.has_data():ax.legend(fontsize=8)
            updates=read_complete_lines(charts.LOG_ROOT/RUN_NAME/'ppo_updates.jsonl')
            updates=[r for r in updates if r['step']<=data['status'].get('step',0)]
            self.axes[4].set_title('行動分布全体のKL');self.axes[5].set_title('価値予測の平均二乗誤差')
            if updates:
                steps=[r['step'] for r in updates]
                self.axes[4].plot(steps,[r['joint_kl'] for r in updates],color=color)
                self.axes[4].axhline(updates[-1]['max_kl'],color='#dc2626',ls='--',label='更新を戻す上限');self.axes[4].legend(fontsize=8)
                for key,label in [('value_mse_before','更新前'),('value_mse_after','更新後')]:
                    self.axes[5].plot(steps,[r[key] for r in updates],label=label)
                self.axes[5].legend(fontsize=8)
            for ax in self.axes:ax.grid(alpha=.2);ax.set_xlabel('追加ガードST（プレイヤーの行動判断数）')
            self.figure.suptitle('GC v3 ポストプラント解除阻止：'+RUN_NAME)
            self.figure.tight_layout(pad=2);self.toolbar.update();self.canvas.draw_idle()
    charts.TrainingMonitor=GuardMonitor;charts.main()


if __name__=='__main__':main()
