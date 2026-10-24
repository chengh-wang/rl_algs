"""Keep the external AMP baseline's native physics, managers and reward terms."""
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.utils.rotations import quat_to_mat


def capture_transition(env):
    if env._pending is not None:
        robot=env.scene['robot']; feet=env.scene['feet_ground_contact'].data
        history=env.scene['self_collision'].data.force_history
        env._captured=dict(s1=env.physical_state().clone(),condition_reward=env.condition(reward_phase=True),
            **{'primitive/joint_acceleration':robot.data.joint_acc.clone(),
               'primitive/foot_contact':(feet.found>0).float().reshape(env.num_envs,-1).clone(),
               'primitive/self_contact_force_history':history.reshape(env.num_envs,-1).clone()})
        if env.audit_snapshots:
            names=['body_link_pos_w','body_link_quat_w','body_link_lin_vel_w','body_link_ang_vel_w',
                   'site_lin_vel_w','joint_pos','joint_vel','joint_acc','soft_joint_pos_limits','default_root_state']
            env.audit_snapshot={k:getattr(robot.data,k).clone() for k in names}
    return torch.zeros(env.num_envs,device=env.device)


class BaselineVelocityEnv(ManagerBasedRlEnv):
    def __init__(self,num_envs=1024,device='cuda:0',seed=42,play=False,cfg=None,render_mode=None):
        # Import the installed external repository; no vendoring or sys.path patch.
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
        if cfg is None: cfg=g1_amp_flat_env_cfg(play=play)
        cfg.scene.num_envs=num_envs; cfg.seed=seed
        cfg.rewards={'_model_capture':RewardTermCfg(func=capture_transition,weight=1.),**cfg.rewards}
        self._pending=None; self.model_task=None; self.audit_snapshots=False
        super().__init__(cfg,device=device,render_mode=render_mode)
        assert self.sim.mj_model.nq==36 and self.sim.mj_model.nv==35

    def physical_state(self):
        q,v=self.sim.data.qpos,self.sim.data.qvel; R=quat_to_mat(q[:,3:7])
        return torch.cat((q[:,:3]-self.scene.env_origins,q[:,3:7],v[:,:3],(R@v[:,3:6,None]).squeeze(-1),q[:,7:],v[:,6:]),-1)

    def condition(self,reward_phase=False):
        tm=self.termination_manager; command=self.command_manager.get_command('twist')
        mask=getattr(tm,'_delay_env_mask',torch.zeros(self.num_envs,device=self.device,dtype=torch.bool))
        counters=getattr(tm,'_delay_counters',torch.zeros_like(mask,dtype=torch.long))
        ready=tm.terminated.float() if reward_phase else torch.zeros_like(counters,dtype=torch.float)
        return torch.cat((command.clone(),mask[:,None].float(),counters[:,None].float(),ready[:,None]),-1)

    def physical_context(self):
        robot=self.scene['robot']; m=self.sim.mj_model
        foot=m.geom('robot/left_foot1_collision').id; torso=m.body('robot/torso_link').id
        friction=self.sim.model.geom_friction[:,foot,:1]
        com=self.sim.model.body_ipos[:,torso,:]-torch.as_tensor(m.body_ipos[torso],device=self.device,dtype=torch.float32)
        return torch.cat((friction,robot.data.encoder_bias,com),-1)

    @torch.no_grad()
    def step(self,action):
        self._pending=dict(s=self.physical_state().clone(),a=action.clone(),
            a_prev=self.action_manager.action.clone(),context=self.physical_context().clone(),
            condition=self.condition(),ep_step=self.episode_length_buf.clone())
        obs,r,terminated,truncated,extras=super().step(action)
        s1=self._captured['s1']; done=terminated|truncated
        # Interval events (e.g. velocity pushes) occur after the reward snapshot.
        # Keep their actual states, and stop WM windows at this exogenous boundary.
        boundary=((self.physical_state()-s1).abs().amax(-1)>1e-5)&~done
        extras['model_transition']={**self._pending,**self._captured,'reward':r.clone(),
            'terminated':terminated.clone(),'truncated':truncated.clone(),'model_boundary':boundary}
        self._pending=None
        return obs,r,terminated,truncated,extras


class VelocityModelAdapter(MjlabModelAdapter):
    def __init__(self,env,task):
        self._model_obs=None
        super().__init__(env,task)

    def get_observations(self):
        if self._model_obs is None:
            e=self.env; c=e.condition()
            # Reuse the original manager's sensor values, order and noise draw.
            # History for the MBDAC actor is maintained by the learning core.
            o=e.obs_buf['actor'].reshape(self.num_envs,-1,self.model_task.obs_dim)[:,-1].clone()
            self._model_obs=TensorDict(dict(actor=o,goal=self.model_task.goal(c)),batch_size=[self.num_envs])
        return self._model_obs

    def step(self,actions):
        _,reward,dones,extras=super().step(actions); self._model_obs=None
        return self.get_observations(),reward,dones,extras

    def randomize_episode_lengths(self):
        self.env.episode_length_buf.random_(self.max_episode_length)
