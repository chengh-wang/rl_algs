"""G1 task bookkeeping and experiment configuration for the WM/FastSAC learner."""
import torch
from .model_task import G1ModelTask
from .robot import G1_JOINT_NAMES, action_scale


class G1FastSACTask(G1ModelTask):
    def advance(self, condition, state=None, context=None):
        # Native physics clamps its final reference frame before reset. Q still
        # bootstraps time limits using this pre-reset physical state/reference.
        return torch.minimum(condition+1,self.motion.clip_end[self.motion.clip_id[condition]])


def training_config(task, horizon=3, compile=False, logger='wandb', run_name=None):
    pd_scale=torch.tensor([action_scale(n) for n in G1_JOINT_NAMES],device=task.default_q.device)
    # Exact Holosoma boundary formula, in this native task's raw action units.
    bounds=torch.maximum((task.kin.q_lo-task.default_q).abs(),(task.kin.q_hi-task.default_q).abs())/pd_scale
    return dict(algorithm=dict(class_name='MBDACFastSAC',history=10,horizon=horizon,
        compile=compile,compile_task=False,wm_batch=128,pi_batch=256,wm_only_iters=500,
        action_clip=max(20.,float(bounds.max())),fast_sac=dict(action_scale=bounds.cpu().tolist())),
        logger=logger,wandb_project='mbdac-fastsac-tracking',run_name=run_name,
        save_interval=500,log_interval=10,task='g1_lafan1_single_clip_tracking',
        source='Holosoma d18d6cc math + MBDAC WM; new model-based actor objective',
        physics_dt=.005,control_dt=.02,real_critic_only=True)
