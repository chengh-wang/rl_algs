"""Plot the same physical evaluation reward/metrics across the three methods."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'mbdac':'MBDAC','fastsac':'Holosoma FastSAC','wm_fastsac':'WM + FastSAC'}


def run_summary(path, rows):
    timing=json.loads((path/'timing.json').read_text())
    cfg=json.loads((path/'config.json').read_text())
    first={}
    for gate in ('pass','all_command_pass'):
        passing=next((i for i,r in enumerate(rows) if r[gate]),None)
        first[gate]=None if passing is None else dict(
            first_observed=rows[passing],previous_sample=rows[passing-1] if passing else None)
    return dict(run=str(path.resolve()),method=rows[0]['method'],recipe=cfg.get('recipe','original'),
                timing=timing,first_passing=first,last_evaluation=rows[-1])


def training_rewards(runs, summaries, output):
    """Native episode reward divided by the configured maximum episode time."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');records=[]
    for path,summary in zip(runs,summaries):
        events=EventAccumulator(str(path),size_guidance={'scalars':0}).Reload()
        tags=[key for key in events.Tags()['scalars'] if key.startswith(
            ('Episode_Reward/','Episode/Episode_Reward/'))]
        by_step={}
        for tag in tags:
            for event in events.Scalars(tag):
                row=by_step.setdefault(event.step,dict(step=event.step,wall_time=event.wall_time,values={}))
                row['values'][tag.rsplit('/',1)[-1]]=event.value
        rows=sorted(by_step.values(),key=lambda r:r['step'])
        # All terms must be present: never turn missing measurements into zero.
        rows=[r for r in rows if len(r['values'])==len(tags)]
        label=LABELS[summary['method']]
        if summary['recipe']=='real_only':label='Same integration, real SAC only'
        for row in rows:
            row.update(run=str(path),total_process_s=row['wall_time']-summary['timing']['process_start_unix'],
                       episode_reward_rate=sum(row['values'].values()))
        ax.plot([r['total_process_s']/60 for r in rows],
                [r['episode_reward_rate'] for r in rows],label=label,lw=1)
        records.extend(rows)
    ax.set(xlabel='Total process elapsed (minutes)',ylabel='Episode reward / maximum episode seconds',
           title='Training episodes: policy-dependent command curriculum and episode lengths')
    ax.grid(alpha=.25);ax.legend();fig.savefig(output/'training_episode_rewards.png',dpi=180);plt.close(fig)
    (output/'training_episode_rewards.json').write_text(json.dumps(records,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=[];summaries=[]
    for path in args.runs:
        rows=[json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()]
        for r in rows:r['run']=str(path)
        records+=rows
        summaries.append(run_summary(path,rows))
    if len({r['task_sha256'] for r in records}) != 1:raise ValueError('Native task fingerprints differ')
    (args.out/'raw_evaluations.json').write_text(json.dumps(records,indent=2)+'\n')
    (args.out/'run_summaries.json').write_text(json.dumps(summaries,indent=2)+'\n')
    columns=('run','method','robot','iteration','real_transitions','active_process_s','process_elapsed_s',
             'mean_return','xy_rmse_m_s','yaw_rmse_rad_s','survival','pass','all_command_pass')
    with (args.out/'evaluations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    panels=(('mean_return','Fixed evaluation return',None),('xy_rmse_m_s','XY velocity RMSE (m/s)',.2),
            ('yaw_rmse_rad_s','Yaw rate RMSE (rad/s)',.3),('survival','20 s survival fraction',1.))
    for axis,axis_label,scale in [('active_process_s','Elapsed excluding evaluation (minutes)',60),
                                 ('process_elapsed_s','Total process elapsed (minutes)',60),
                                 ('real_transitions','Real environment transitions (millions)',1e6)]:
        fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
        for path,summary in zip(args.runs,summaries):
            rows=[r for r in records if r['run']==str(path)]
            x=[r[axis]/scale for r in rows];label=LABELS[rows[0]['method']]
            if summary['recipe']=='real_only':label='Same integration, real SAC only'
            if 'status' not in summary['timing']:label+=' (running)'
            for ax,(key,title,threshold) in zip(axes.flat,panels):
                ax.plot(x,[r[key] for r in rows],marker='o',markersize=4,label=label)
                ax.set(xlabel=axis_label,ylabel=title);ax.grid(alpha=.25)
        for ax,(_,_,threshold) in zip(axes.flat,panels):
            if threshold is not None:ax.axhline(threshold,color='.4',ls='--',lw=1)
        robot=records[0]['robot'].upper();seeds=sorted({s['timing']['seed'] for s in summaries})
        axes[0,0].legend();fig.suptitle(f'Native {robot} velocity: controlled cold-start pilot (seeds {seeds})')
        fig.savefig(args.out/f'comparison_{axis}.png',dpi=180);plt.close(fig)
    training_rewards(args.runs,summaries,args.out)


if __name__=='__main__':main()
