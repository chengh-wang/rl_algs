"""Run GPU benchmarks sequentially, preserving failed and completed trials."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.wait_pid:
        while True:
            try:command=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'examples.mjlab_velocity.train' not in command:break
            time.sleep(10)
    jobs=[('go1','mbdac','native_go1_mbdac_v3',100000,3600,'hybrid'),
          ('go1','wm_fastsac','native_go1_wm_fastsac_v3',100000,3600,'hybrid'),
          ('go1','wm_fastsac','native_go1_real_only_v3',100000,1200,'real_only'),
          ('g1','wm_fastsac','native_g1_wm_fastsac_v3',3000,1800,'hybrid')]
    record=root/'logs/native_velocity_matrix.json';results=[]
    for robot,method,name,iterations,budget,recipe in jobs:
        target=root/'logs'/name
        if (target/'config.json').exists():
            raise RuntimeError(f'Refusing to overwrite {target}; choose new trial names deliberately')
        cmd=[sys.executable,'-u','-m','examples.mjlab_velocity.train','--robot',robot,'--method',method,
             '--num-envs','1024','--iterations',str(iterations),'--max-train-seconds',str(budget),
             '--compile','--recipe',recipe,'--logger','wandb','--log-dir',str(target)]
        started=time.time();print(json.dumps({'start':name,'command':cmd}),flush=True)
        with (root/f'logs_{name}.console.log').open('w') as log:
            process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            interrupted=False
            try:code=process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT);code=process.wait();interrupted=True
        results.append(dict(name=name,robot=robot,method=method,command=cmd,started_unix=started,
                            finished_unix=time.time(),process_s=time.time()-started,exit_code=code))
        record.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results[-1]),flush=True)
        if interrupted:raise KeyboardInterrupt()
    runs=[root/'logs'/n for n in ('native_go1_mbdac_v3','native_go1_fastsac_v2','native_go1_wm_fastsac_v3')]
    subprocess.run([sys.executable,'-m','examples.mjlab_velocity.plot',*map(str,runs),
                    '--out',str(root/'logs/native_go1_benchmark_v3_report')],cwd=root,check=True)


if __name__=='__main__':main()
