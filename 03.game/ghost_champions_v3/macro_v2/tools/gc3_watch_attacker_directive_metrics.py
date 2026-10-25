"""Monitor only the additional micro instruction run; run from macro_v2."""
from pathlib import Path
import sys

MACRO_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_micro_directives_v5'
REFRESH_SECONDS=5
ROLLING_EPISODES=20
WINDOW_GEOMETRY='1380x1040'
sys.path.insert(0,str(MACRO_ROOT.parents[1]))


def render_metrics(figure,axes,data,color):
    """Six independent panels; rates and retreat counts use separate axes."""
    def waiting(ax,label):
        ax.text(.5,.5,label,ha='center',va='center',transform=ax.transAxes,color='#64748b')
    evaluations=data['evaluations'];metrics=data['metrics']
    starts=[r['step'] for r in evaluations]+[r['start_step'] for r in metrics]
    first=min(starts) if starts else max(0,data['step'])
    last=max(data['step'],max(starts,default=first))
    right=max(first+1,last)+max(1,last-first)*.03
    for ax in axes:
        ax.clear();ax.grid(alpha=.2);ax.set_xlabel('累計学習step（判断回数）')
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.set_xlim(first,right)
    for ax,key,title in ((axes[0],'reward_mean','AIに渡した報酬の平均'),
                         (axes[1],'return_mean','returnの平均'),
                         (axes[2],'entropy_mean','行動分布のエントロピー')):
        ax.set_title(title)
        if metrics:ax.plot([r['step'] for r in metrics],[r[key] for r in metrics],color=color,marker='.',lw=1.6)
        else:waiting(ax,'このrunの学習ログを待っています')
        ax.set_ylabel('nats' if key=='entropy_mean' else '収集バッチの平均値')
    win=axes[3];win.set_title('設置率・勝率')
    if data['steps']:
        win.plot(data['steps'],data['rolling_success'],color='#94a3b8',lw=1.1,
            label=f'学習中の勝率（直近{ROLLING_EPISODES}試合）')
    for key,label,tint in (('plant_rate','設置率（定期評価）','#2563eb'),('success_rate','勝率（定期評価）','#16a34a')):
        rows=[r for r in evaluations if r.get(key) is not None]
        if rows:win.plot([r['step'] for r in rows],[100*r[key] for r in rows],color=tint,marker='o',lw=1.5,label=label)
    final=data['final']
    if final and final.get('step') is not None:
        for key,tint in (('plant_rate','#2563eb'),('success_rate','#16a34a')):
            if final.get(key) is not None:
                win.scatter([final['step']],[100*final[key]],color=tint,marker='X',s=60,zorder=6,
                    label='保存best '+('設置率' if key=='plant_rate' else '勝率'))
    if not win.get_legend_handles_labels()[0]:waiting(win,'試合結果を待っています')
    orders=axes[4];orders.set_title('指示への対応率（実行可能な判断のみ）')
    for order,label,tint in (('WAIT','待機','#9333ea'),('ADVANCE','進軍','#2563eb'),('RETREAT','撤退','#ea580c')):
        rows=[r for r in evaluations if order in r.get('instruction_compliance_rates',{})
            and r.get('instruction_measures',{}).get(order+'_applicable',0)>0]
        if rows:orders.plot([r['step'] for r in rows],[100*r['instruction_compliance_rates'][order] for r in rows],
            color=tint,marker='.',lw=1.6,label=label)
    if orders.get_legend_handles_labels()[0]:
        threshold=next((r['minimum_instruction_rate'] for r in reversed(evaluations) if 'minimum_instruction_rate' in r),None)
        if threshold is not None:orders.axhline(100*threshold,color='#64748b',ls=':',lw=1,label=f'目標 {100*threshold:g}%')
    else:waiting(orders,'指示対応率の測定を待っています')
    for ax in (win,orders):ax.set_ylim(-3,103);ax.set_ylabel('割合（%）')
    retreat=axes[5];retreat.set_title('撤退練習の開始・終了件数（定期評価）')
    rows=[]
    for report in evaluations:
        rounds=report.get('rounds',[])
        if not rounds or not any('retreat_practice_events' in r for r in rounds):continue
        events=[event for r in rounds for event in r.get('retreat_practice_events',[])]
        rows.append(dict(step=report['step'],started=len(events),
            arrived=sum(e.get('reason')=='reached_spawn' for e in events),
            timed_out=sum(e.get('reason')=='timeout' for e in events),
            interrupted=sum(e.get('reason') in ('died','objective') for e in events)))
    if rows:
        for key,label,tint in (('started','開始','#64748b'),('arrived','スポーン到達','#16a34a'),
                                ('timed_out','時間切れ','#ea580c'),('interrupted','設置等で中断・死亡','#9333ea')):
            retreat.plot([r['step'] for r in rows],[r[key] for r in rows],color=tint,marker='.',lw=1.4,label=label)
        retreat.set_ylim(-.5,max(1,max(r['started'] for r in rows))*1.12)
        retreat.text(.02,.02,'各評価の合計件数。時間切れは到達成功に含めません。',transform=retreat.transAxes,
            fontsize=8,color='#64748b')
    else:waiting(retreat,'このrunには撤退練習の件数記録がまだありません')
    retreat.set_ylabel('件数／評価')
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    figure.tight_layout(pad=2)


def main():
    from ghost_champions_v3.tools import gc3_watch_training_metrics as charts
    charts.LOG_ROOT=MACRO_ROOT/'logs'
    charts.RUNS=((RUN_NAME,'指示対応ミクロ（近距離の撤退練習）','#16a34a'),
                ('attacker_micro_directives_v4','指示対応ミクロ（撤退練習・調整前）','#94a3b8'),
                ('attacker_micro_directives_v3','指示対応ミクロ（撤退練習なし）','#94a3b8'),
                ('attacker_micro_directives_v2','指示対応ミクロ（ランダム指示）','#94a3b8'),
                ('attacker_micro_directives_v1','指示対応ミクロ（調整前）','#94a3b8'))
    charts.DEFAULT_RUN=RUN_NAME
    charts.REFRESH_SECONDS=REFRESH_SECONDS
    charts.ROLLING_EPISODES=ROLLING_EPISODES
    charts.WINDOW_GEOMETRY=WINDOW_GEOMETRY
    class DirectiveMonitor(charts.TrainingMonitor):
        def __init__(self,root):
            super().__init__(root)
            root.title('GC v3 指示対応学習グラフ')

        def render(self,data,color):
            if len(self.axes)!=6:
                self.figure.clear()
                self.axes=self.figure.subplots(3,2).reshape(-1)
            render_metrics(self.figure,self.axes,data,color)
            self.toolbar.update();self.canvas.draw_idle()
    charts.TrainingMonitor=DirectiveMonitor
    charts.main()


if __name__=='__main__':main()
