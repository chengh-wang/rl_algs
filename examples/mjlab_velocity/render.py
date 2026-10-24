"""Render recorded native GPU trajectories; no smoothing or extra physics."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from mjlab.scene import Scene
from .config import velocity_cfg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trajectory',type=Path)
    p.add_argument('--robot',choices=('g1','go1'),default='go1');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--world',type=int,default=8,help='World 8 is the first +0.8 m/s command replica')
    args=p.parse_args();trajectory=np.load(args.trajectory)
    scene=Scene(velocity_cfg(args.robot,1,evaluation=True).scene,device='cpu')
    model=scene.compile();data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=480,width=800)
    camera=mujoco.MjvCamera();camera.distance=2. if args.robot=='go1' else 3.;camera.elevation=-16;camera.azimuth=135
    dt=float(trajectory['dt']);world=args.world;origin=trajectory['env_origins'][world];command=trajectory['command'][world]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    try:
        with imageio.get_writer(args.out,fps=25,codec='libx264',quality=8) as writer:
            for i in range(0,len(trajectory['qpos']),round(1/25/dt)):
                q=trajectory['qpos'][i,world].copy();q[:3]-=origin
                data.qpos[:]=q;mujoco.mj_forward(model,data)
                camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera=camera)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,800,49),fill=(18,24,32))
                draw.text((12,9),f'{args.robot.upper()} | recorded native MJLab GPU physics | t={i*dt:.2f}s',fill='white')
                draw.text((12,28),f'command vx={command[0]:.2f} vy={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s',fill='white')
                writer.append_data(np.asarray(frame))
    finally:renderer.close()
    print(args.out.resolve())


if __name__=='__main__':main()
