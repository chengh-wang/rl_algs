"""Make physical videos and plots after the sequential benchmark releases the GPU.

This reads existing trials; it never trains, resumes or overwrites a checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def wait_for_matrix(pid):
    while True:
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            status=Path(f'/proc/{pid}/stat').read_text().split()[2]
        except FileNotFoundError:
            return
        if b'native_velocity_matrix.py' not in command or status=='Z':return
        time.sleep(10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    args.runs=[path.resolve() for path in args.runs];args.out=args.out.resolve()
    if args.wait_pid:wait_for_matrix(args.wait_pid)
    args.out.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'MUJOCO_GL':'egl'};selected=[]
    def run(command):
        print(json.dumps({'command':command}),flush=True)
        subprocess.run(command,cwd=root,env=environment,check=True)
    for path in args.runs:
        timing=json.loads((path/'timing.json').read_text())
        if timing.get('status') not in ('converged','completed','budget_reached'):
            raise RuntimeError(f'Trial has not ended normally: {path}: {timing.get("status")}')
        rows=[json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()]
        # Predeclared selection: first strict pass, otherwise final measured policy.
        row=next((r for r in rows if r['all_command_pass']),rows[-1])
        checkpoint=path/f'eval_model_{row["iteration"]}.pt'
        target=args.out/path.name;target.mkdir(parents=True,exist_ok=True)
        output=target/'metrics.json'
        reusable=False
        if output.exists() and (target/'trajectory.npz').exists():
            reusable=json.loads(output.read_text()).get('checkpoint')==str(checkpoint)
        if not reusable:
            run([sys.executable,'-m','examples.mjlab_velocity.evaluate',str(checkpoint),
                 '--robot',row['robot'],'--out',str(target),'--trajectory'])
        label={'mbdac':'MBDAC','fastsac':'Holosoma FastSAC','wm_fastsac':'WM + FastSAC'}[row['method']]
        config=json.loads((path/'config.json').read_text())
        if config.get('recipe')=='real_only':label='Same integration, real SAC only'
        label+=f' | update {row["iteration"]}'
        if not row['all_command_pass']:label+=' | threshold not reached'
        run([sys.executable,'-m','examples.mjlab_velocity.render',str(target/'trajectory.npz'),
             '--robot',row['robot'],'--out',str(target/'forward.mp4'),'--label',label])
        selected.append(dict(run=str(path),checkpoint=str(checkpoint),label=label,
            selection='first_all_command_pass' if row['all_command_pass'] else 'last_evaluation',
            online_evaluation=row,physical_metrics=json.loads(output.read_text()),video=str(target/'forward.mp4')))
    (args.out/'selected_policies.json').write_text(json.dumps(selected,indent=2)+'\n')
    primary=[r for r in selected if r['online_evaluation']['robot']=='go1' and 'real_only' not in r['run']]
    if len(primary)==3:
        run([sys.executable,'-m','examples.mjlab_velocity.plot',*[r['run'] for r in primary],
             '--out',str(args.out/'primary')])
        command=['ffmpeg','-y']
        for row in primary:command+=['-i',row['video']]
        command+=['-filter_complex',
            '[0:v]scale=640:384[a];[1:v]scale=640:384[b];[2:v]scale=640:384[c];[a][b][c]hstack=inputs=3[v]',
            '-map','[v]','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-an',str(args.out/'three_methods_forward.mp4')]
        run(command)
    print(json.dumps({'complete':str(args.out/'selected_policies.json')}),flush=True)


if __name__=='__main__':main()
