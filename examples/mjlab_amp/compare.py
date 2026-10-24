"""Compare measured policies after checking their evaluation protocol."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('baseline',type=Path); p.add_argument('candidate',type=Path)
    p.add_argument('--baseline-training',type=Path); p.add_argument('--candidate-launch',type=Path)
    p.add_argument('--output',type=Path,default=Path('logs/velocity_comparison')); args=p.parse_args()
    baseline=json.loads(args.baseline.read_text()); candidate=json.loads(args.candidate.read_text())
    for key in ('task','seed','initialization','requested_horizon_s','physics_dt','control_dt',
                'native_physics','auto_reset','domain_randomization','observation_noise','steady_start_s'):
        if baseline.get(key)!=candidate.get(key):
            raise ValueError(f'Evaluation protocol differs for {key}')
    if not baseline.get('native_physics') or baseline.get('auto_reset'):
        raise ValueError('Comparison requires native physics without automatic reset')
    old={r['name']:r for r in baseline['cases']}; new={r['name']:r for r in candidate['cases']}
    if old.keys()!=new.keys(): raise ValueError('Evaluated command cases differ')
    rows=[]
    for name,a in old.items():
        b=new[name]; np.testing.assert_allclose(a['command'],b['command'],atol=1e-6,rtol=0)
        row=dict(name=name,command=a['command'])
        for label,item in [('baseline',a),('candidate',b)]:
            row[label+'_survived']=item['survived_horizon']
            row[label+'_horizontal_rmse_m_s']=item.get('horizontal_velocity_rmse')
            row[label+'_yaw_rmse_rad_s']=item.get('rmse_per_axis',[None,None,None])[2]
            row[label+'_first_failure_s']=item.get('first_failure_s')
            for metric in ('foot_slip_rms_m_s','joint_acceleration_rms_rad_s2','liftoffs_per_foot'):
                row[label+'_'+metric]=item.get(metric)
        rows.append(row)
    timing=dict(candidate_trainer_elapsed_s=candidate.get('training_elapsed_s'))
    if args.baseline_training:
        timing['baseline_launch_to_checkpoint_s']=json.loads(args.baseline_training.read_text())['launch_to_checkpoint_wall_s']
    if args.candidate_launch:
        launch=json.loads(args.candidate_launch.read_text())
        timing['candidate_launch_to_checkpoint_s']=Path(candidate['checkpoint']).stat().st_mtime-launch['started_unix']
        timing['candidate_training_source_commit']=launch['source_commit']
        if timing['candidate_launch_to_checkpoint_s']<0: raise ValueError('Checkpoint predates candidate launch')
    result=dict(baseline_evaluation=str(args.baseline.resolve()),candidate_evaluation=str(args.candidate.resolve()),
        baseline_checkpoint=baseline['checkpoint'],candidate_checkpoint=candidate['checkpoint'],
        baseline_iteration=baseline['iteration'],candidate_iteration=candidate['iteration'],
        task=baseline['task'],same_evaluation_protocol=True,timing=timing,cases=rows,
        caveats=['Report different training environment/batch counts and policy architectures.',
                 'Two checkpoints alone do not establish time to matched quality.',
                 'Missing RMSE means insufficient pre-failure steady data, not zero error.',
                 'Launch costs include setup/compilation; trainer elapsed excludes environment construction.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    def number(value): return '—' if value is None else f'{value:.4f}'
    lines=['# Matched native MJLab velocity evaluation','',
        '| Command | PPO survived | Candidate survived | PPO XY RMSE (m/s) | Candidate XY RMSE (m/s) | PPO yaw RMSE (rad/s) | Candidate yaw RMSE (rad/s) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["name"]} | {row["baseline_survived"]} | {row["candidate_survived"]} | {number(row["baseline_horizontal_rmse_m_s"])} | {number(row["candidate_horizontal_rmse_m_s"])} | {number(row["baseline_yaw_rmse_rad_s"])} | {number(row["candidate_yaw_rmse_rad_s"])} |')
    lines+=['','Training cost:',*['- '+k+': '+str(v) for k,v in timing.items()],
            '',*['- '+note for note in result['caveats']]]
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
