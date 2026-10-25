"""One-step random-goal position control: additive bias versus potential shaping.

Run from this checkout with ``python -m examples.precision_control.benchmark``.
This is an analytic counterexample and numerical code-path test, not evidence
about real robots. All config and evaluation grids are frozen before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mbdac.runners import FlashSACRunner
from mbdac.rewards import PotentialShapingConfig, PotentialRewardProvider

MODES = ('task_only', 'additive', 'potential')
STYLE_WEIGHT = .4
GAMMA = .99


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def objective_optimum(goal, mode):
    target = goal.clone()
    if mode == 'additive':
        target[:, 0] += STYLE_WEIGHT / 4
    return target


class PrecisionBandit:
    """Real one-step transitions; target generator never touches learner RNG."""
    num_actions = 2

    def __init__(self, mode, seed, steps, num_envs, device, forbidden_grid_axis=None):
        self.mode, self.num_envs, self.device = mode, num_envs, torch.device(device)
        generator = torch.Generator(device='cpu').manual_seed(seed + 100_000)
        self.goals = .6 * torch.rand(steps + 1, num_envs, 2, generator=generator) - .3
        if forbidden_grid_axis is not None:
            collision = torch.isin(self.goals[..., 0], forbidden_grid_axis) & torch.isin(self.goals[..., 1], forbidden_grid_axis)
            while collision.any():
                self.goals[collision] = .6 * torch.rand(int(collision.sum()), 2, generator=generator) - .3
                collision = torch.isin(self.goals[..., 0], forbidden_grid_axis) & torch.isin(self.goals[..., 1], forbidden_grid_axis)
        self.goal_stream_sha256 = hashlib.sha256(self.goals.numpy().tobytes()).hexdigest()
        self.index = 0
        self.current = self.goals[0].to(self.device)
        self.diagnostics = {}
        self.potential_config = PotentialShapingConfig(
            potential_id='precision_initial_position_style', version='1', gamma=GAMMA,
            beta=1., max_abs_potential=STYLE_WEIGHT)
        if mode == 'potential':
            self.reward_provider = PotentialRewardProvider(self.potential_config, next_potential_is_pre_reset=True)

    def get_observations(self):
        return {'goal': self.current.clone()}

    def reset(self):
        # Initial goal already drawn independently; no learner-RNG reset.
        return self.get_observations()

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.num_envs, 2) or not torch.isfinite(action).all():
            raise ValueError('Expected finite normalized position actions [N,2]')
        if bool((action.abs() > 1.000001).any()):
            raise ValueError('Policy actions exceed normalized position box')
        task = -(action - self.current).square().sum(-1)
        style = (action[:, 0] + 1) / 2
        reward = task + STYLE_WEIGHT * style if self.mode == 'additive' else task
        terminal = self.get_observations()
        done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        extras = dict(terminated=done, truncated=torch.zeros_like(done),
                      terminal_observation=terminal)
        if self.mode == 'potential':
            # Initial state x=0 is action-independent: Phi=.4*S(0)=.2.
            # Terminal endpoint Phi=0, hence shaping is the constant -.2.
            extras['reward_potential'] = dict(version='1', current=torch.full_like(task, STYLE_WEIGHT / 2),
                                             next=torch.zeros_like(task))
        self.diagnostics = dict(train_raw_task_return=float(task.mean()),
            train_style=float(style.mean()), train_task_l2_rmse=float((-task.mean()).sqrt()),
            train_bias_x=float((action[:, 0] - self.current[:, 0]).mean()))
        self.index += 1
        self.current = self.goals[self.index].to(self.device)
        return self.get_observations(), reward, done, extras


class DiagnosticRunner(FlashSACRunner):
    def _log(self, logs):
        super()._log({**logs, **self.env.diagnostics})


def evaluate(policy, grid, mode):
    with torch.inference_mode():
        action = policy({'goal': grid}).detach()
    if not torch.isfinite(action).all():
        raise RuntimeError('Nonfinite evaluation actions')
    own = objective_optimum(grid, mode)
    task_error, own_error = action-grid, action-own
    task_distance, own_distance = task_error.norm(dim=-1), own_error.norm(dim=-1)
    style = (action[:, 0]+1)/2
    raw = -task_error.square().sum(-1)
    composed = raw + STYLE_WEIGHT*style if mode == 'additive' else raw - (STYLE_WEIGHT/2 if mode == 'potential' else 0)
    def errors(error, distance):
        return dict(mean_bias_xy=error.mean(0).cpu().tolist(), xy_rmse=error.square().mean(0).sqrt().cpu().tolist(),
                    l2_rmse=float(error.square().sum(-1).mean().sqrt()),
                    mean_l2_error=float(distance.mean()), success_fraction_l2_le_0p02=float((distance <= .02).float().mean()))
    return dict(samples=len(grid), task_optimum=errors(task_error, task_distance),
                own_objective_optimum=errors(own_error, own_distance),
                raw_task_return=float(raw.mean()), style_mean=float(style.mean()),
                composed_return=float(composed.mean())), action


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=4096)
    parser.add_argument('--capacity', type=int, default=65536)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--compile-mode', default='default')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--grid-size', type=int, default=41)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(argv)
    if min(args.steps,args.num_envs,args.updates,args.log_every) <= 0 or args.grid_size < 2:
        parser.error('Positive training dimensions and grid-size>=2 required')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error('Seeds and modes must be unique')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'config.json').exists():
        raise FileExistsError('Use a fresh benchmark output directory')
    algorithm = dict(class_name='FlashSAC', capacity=args.capacity,warmup=args.warmup,
        batch_size=args.batch_size,updates_per_step=args.updates,
        schedule_updates=max(1,(args.steps-(args.warmup+args.num_envs-1)//args.num_envs+1)*args.updates),
        gamma=GAMMA,critic_min_v=-5.,critic_max_v=5.,critic_num_bins=101,
        reward_normalization='fixed',reward_scale=1.,target_sigma=.05,initial_temperature=.001,
        use_compile=not args.no_compile,compile_mode=args.compile_mode,use_amp=not args.no_amp,
        critic_diagnostics=True)
    axis = torch.linspace(-.3,.3,args.grid_size)
    grid = torch.cartesian_prod(axis, axis)
    np.save(args.output/'evaluation_goals.npy', grid.numpy())
    frozen = dict(arguments={**vars(args),'output':str(args.output)}, algorithm=algorithm,
        task='one-step terminal normalized 2D position, goal uniform[-.3,.3]^2',
        raw_task='-sum((action-goal)^2)', style='(action_x+1)/2',style_weight=STYLE_WEIGHT,
        additive_optimum='(goal_x+0.1,goal_y)',task_and_potential_optimum='goal',
        optimum_semantics='Analytic unregularized deterministic reward optimum; SAC also optimizes its configured entropy term',
        potential='Phi(initial x=0)=0.2; Phi(terminal)=0; reward=task-0.2',
        success_threshold=dict(l2=.02,meaning='descriptive threshold, not acceptance criterion'),
        goal_stream='CPU generator seed=learner seed+100000; matched across modes, isolated from learner RNG',
        evaluation='fixed grid frozen before training, excluded from sampled training goals; no evaluation-based tuning',
        limitations='Analytic counterexample and public runner code-path test; no real-robot generalization claim')
    write_json(args.output/'config.json', frozen)
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            env = PrecisionBandit(mode,seed,args.steps,args.num_envs,args.device,axis)
            cfg = dict(algorithm={**algorithm,'seed':seed},obs_groups={'actor':['goal'],'critic':['goal']},
                logger='json',log_interval=args.log_every,save_interval=args.steps+1,
                task_contract=dict(kind='precision_bandit',mode=mode,style_weight=STYLE_WEIGHT,
                                   goal_stream_sha256=env.goal_stream_sha256))
            if mode == 'potential': cfg['reward']={'source':'provider'}
            directory = args.output/f'seed{seed}'/mode
            write_json(args.output/f'run_config_seed{seed}_{mode}.json',cfg)
            started = time.perf_counter()
            runner = DiagnosticRunner(env,cfg,log_dir=directory,device=args.device)
            runner.learn(args.steps)
            runner.save(directory/'checkpoint.pt')
            metrics,actions = evaluate(runner.get_inference_policy(),grid.to(args.device),mode)
            row = dict(seed=seed,mode=mode,vector_steps=runner.alg.vector_steps,
                transitions=runner.alg.transitions,updates=runner.alg.update_step,
                elapsed_s=time.perf_counter()-started,goal_stream_sha256=env.goal_stream_sha256,**metrics)
            write_json(directory/'evaluation.json',row)
            np.save(directory/'evaluation_actions.npy',actions.cpu().numpy())
            results.append(row)
            write_json(args.output/'results.json',dict(config_frozen_before_training=True,runs=results))
            print(json.dumps(row),flush=True)
            del runner,env
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    aggregates = {}
    for mode in args.modes:
        rows=[r for r in results if r['mode']==mode]
        aggregates[mode]=dict(seeds=len(rows),task_l2_rmse_mean=float(np.mean([r['task_optimum']['l2_rmse'] for r in rows])),
            own_l2_rmse_mean=float(np.mean([r['own_objective_optimum']['l2_rmse'] for r in rows])),
            task_bias_xy_mean=np.mean([r['task_optimum']['mean_bias_xy'] for r in rows],axis=0).tolist())
    write_json(args.output/'summary.json',dict(aggregates=aggregates,runs=results,
        interpretation='Task-optimum errors and own-objective errors answer different questions; report both.'))
    if args.plot:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3))
        for axis,key in zip(axes,('task_l2_rmse_mean','own_l2_rmse_mean')):
            axis.bar(list(aggregates),[aggregates[m][key] for m in aggregates]);axis.set_ylabel(key)
        fig.tight_layout();fig.savefig(args.output/'errors.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
