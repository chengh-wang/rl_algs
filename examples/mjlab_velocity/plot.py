"""Plot the same physical evaluation reward/metrics across the three methods."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'mbdac':'MBDAC','fastsac':'Holosoma FastSAC','wm_fastsac':'WM + FastSAC'}


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in args.runs:
        rows=[json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()]
        for r in rows:r['run']=str(path)
        records+=rows
    if len({r['task_sha256'] for r in records}) != 1:raise ValueError('Native task fingerprints differ')
    (args.out/'raw_evaluations.json').write_text(json.dumps(records,indent=2)+'\n')
    panels=(('mean_return','Fixed evaluation return',None),('xy_rmse_m_s','XY velocity RMSE (m/s)',.2),
            ('yaw_rmse_rad_s','Yaw rate RMSE (rad/s)',.3),('survival','20 s survival fraction',1.))
    for axis,axis_label,scale in [('active_process_s','Elapsed excluding evaluation (minutes)',60),
                                 ('process_elapsed_s','Total process elapsed (minutes)',60),
                                 ('real_transitions','Real environment transitions (millions)',1e6)]:
        fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
        for path in args.runs:
            rows=[r for r in records if r['run']==str(path)]
            x=[r[axis]/scale for r in rows];label=LABELS[rows[0]['method']]
            for ax,(key,title,threshold) in zip(axes.flat,panels):
                ax.plot(x,[r[key] for r in rows],marker='o',markersize=4,label=label)
                ax.set(xlabel=axis_label,ylabel=title);ax.grid(alpha=.25)
        for ax,(_,_,threshold) in zip(axes.flat,panels):
            if threshold is not None:ax.axhline(threshold,color='.4',ls='--',lw=1)
        axes[0,0].legend();fig.suptitle('Native Go1 velocity: controlled cold-start pilot (seed 42)')
        fig.savefig(args.out/f'comparison_{axis}.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
