"""Native ManagerBasedRlEnv plus a zero-valued pre-reset measurement hook."""
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.utils.rotations import quat_to_mat
from .config import velocity_cfg


def capture_transition(env):
    if env._pending is not None:
        env._captured={'s1':env.physical_state().clone(),'a':env.action_manager.action.clone()}
        if env.model_task is not None:
            env._captured.update(env.model_task.signals.capture(env,env.model_task.physical_dependencies))
        env._captured['native_terms']=env.reward_manager._step_reward[:,:-1].clone()*env.step_dt
        if env.audit:
            r=env.scene['robot'].data
            env._captured['audit']={k:getattr(r,k).clone() for k in
                ('root_link_lin_vel_b','root_link_ang_vel_b','projected_gravity_b','joint_pos','joint_vel',
                 'body_link_quat_w','body_link_pos_w','body_link_lin_vel_w','body_link_ang_vel_w',
                 'site_pos_w','site_lin_vel_w')}
            if 'angular_momentum' in env.reward_manager.active_terms:
                env._captured['audit']['angular_momentum']=env.scene['robot/root_angmom'].data.clone()
    return torch.zeros(env.num_envs,device=env.device)


class NativeVelocityEnv(ManagerBasedRlEnv):
    def __init__(self,robot='go1',num_envs=1024,device='cuda:0',seed=42,evaluation=False):
        self.robot_name=robot; self._pending=None; self._captured=None; self.model_task=None; self.audit=False
        cfg=velocity_cfg(robot,num_envs,seed,evaluation)
        cfg.rewards['_model_capture']=RewardTermCfg(func=capture_transition,weight=1.)
        super().__init__(cfg,device)
        self.reset()

    def physical_state(self):
        q,v=self.sim.data.qpos,self.sim.data.qvel; R=quat_to_mat(q[:,3:7])
        memory=self.reward_manager.get_term_cfg('foot_swing_height').func.peak_heights
        air=self.scene['feet_ground_contact'].data.current_air_time
        return torch.cat((q[:,:3]-self.scene.env_origins,q[:,3:7],v[:,:3],(R@v[:,3:6,None]).squeeze(-1),q[:,7:],v[:,6:],memory,air),-1)

    def condition(self): return self.command_manager.get_command('twist').clone()

    def physical_context(self):
        task=self.model_task
        if task is None: return torch.empty(self.num_envs,0,device=self.device)
        m=self.sim.mj_model; robot=self.scene['robot']; ids=robot.indexing.body_ids
        geom=int(robot.indexing.geom_ids[task.friction_geom])
        body=int(ids[task.com_body]); base=torch.tensor(m.body_ipos[body],device=self.device,dtype=torch.float32)
        return torch.cat((self.sim.model.geom_friction[:,geom,:1],robot.data.encoder_bias,self.sim.model.body_ipos[:,body]-base),-1)

    @torch.no_grad()
    def step(self,action):
        self._pending=dict(s=self.physical_state().clone(),a=action.clone(),a_prev=self.action_manager.action.clone(),
                           context=self.physical_context().clone(),condition=self.condition(),ep_step=self.episode_length_buf.clone(),
                           critic_features=self.obs_buf['critic'].clone())
        obs,r,term,trunc,extras=super().step(action)
        captured=self._captured; s1=captured['s1']; done=term|trunc
        boundary=((self.physical_state()-s1).abs().amax(-1)>1e-5)&~done
        transition={**self._pending,**{k:v for k,v in captured.items() if k not in ('audit','native_terms')},
                    'condition1':torch.where(done[:,None],self._pending['condition'],self.condition()),
                    'terminated':term.clone(),'truncated':trunc.clone(),'model_boundary':boundary}
        # Timeouts bootstrap physical terminal observations, never reset states.
        if self.model_task is not None:
            task=self.model_task
            terminal=task.observation(s1,action,transition['context'],transition['condition'])
            transition['terminal_actor_observation']=task.add_noise(terminal)
            labels={k.split('/',1)[1]:v for k,v in transition.items() if k.startswith('primitive/')}
            final_critic=task.controlled_critic_features(s1,transition['condition'],transition['context'],transition['a'],primitives=labels)
            transition['next_critic_features']=torch.where(done[:,None],final_critic,obs['critic']).clone()
        extras['model_transition']=transition; self._pending=None
        return obs,r,term,trunc,extras


class VelocityAdapter(MjlabModelAdapter):
    def get_observations(self):
        return TensorDict({'actor':self.env.obs_buf['actor'],'goal':self.env.condition()},batch_size=[self.num_envs])
    def step(self,action):
        _,r,d,extras=super().step(action)
        return self.get_observations(),r,d,extras
