"""Reuse the external baseline's Torch rewards through a differentiable data view.

Physics/rewards during real collection remain the unchanged native environment.
Only hard contact and failure gates are softened for imagined policy gradients.
"""
import math
from types import SimpleNamespace
import numpy as np
import torch
from mbdac.modules.codecs import ArticulatedCodec
from mbdac.modules.primitives import PrimitiveSpec,PhysicalSignals
from mbdac.utils.rotations import mat_to_quat
from examples.mjlab_g1.kinematics import G1Kinematics
from examples.mjlab_g1.model_task import G1ModelTask
from examples.mjlab_g1.robot import G1_JOINT_NAMES


class BaselineVelocityTask:
    action_dim=29; context_dim=33; obs_dim=96; goal_dim=3; critic_dim=305; amp_feature_dim=195
    analytic_primitives=G1ModelTask.analytic_primitives
    amp_features=G1ModelTask.amp_features

    def __init__(self,env):
        self.device=env.device; self.dt=env.step_dt; self.env=env
        robot=env.scene['robot']; m=env.sim.mj_model
        assert list(robot.joint_names)==G1_JOINT_NAMES
        self.codec=ArticulatedCodec(29).to(self.device)
        self.kin=G1Kinematics(self.device,model=m,prefix='robot/')
        self.default_q=robot.data.default_joint_pos[0].clone()
        self.soft_limits=robot.data.soft_joint_pos_limits[0].clone()
        self.default_root=robot.data.default_root_state[0].clone()
        self.num_bodies=len(robot.body_names)
        self.moving_ids=torch.tensor([list(robot.body_names).index(n) for n in self.kin.body_names],device=self.device)
        sites=robot.indexing.site_ids.cpu().numpy()
        self.site_body=torch.tensor([self.kin.body_names.index(m.body(int(m.site_bodyid[i])).name.removeprefix('robot/')) for i in sites],device=self.device)
        self.site_pos=torch.tensor(m.site_pos[sites],device=self.device,dtype=torch.float32)
        self.self_force_shape=tuple(env.scene['self_collision'].data.force_history.shape[1:])
        self.physical_dependencies=('joint_acceleration','foot_contact','self_contact_force_history')
        self.signals=PhysicalSignals()
        self.signals.register(PrimitiveSpec('joint_acceleration',29,'rad/s^2','joint',transition=True),
                              measure=lambda e:e.scene['robot'].data.joint_acc)
        self.signals.register(PrimitiveSpec('foot_contact',2,'probability','left/right foot',binary=True,transition=True),
                              measure=lambda e:e.scene['feet_ground_contact'].data.found>0)
        self.signals.register(PrimitiveSpec('self_contact_force_history',int(np.prod(self.self_force_shape)),'N','world; physics substep history',transition=True),
                              measure=lambda e:e.scene['self_collision'].data.force_history)
        self.primitive_specs=self.signals.learned_specs(self.physical_dependencies)
        self.reward_cfg={k:env.reward_manager.get_term_cfg(k) for k in env.reward_manager.active_terms if k!='_model_capture'}
        self.noise=torch.cat((torch.full((3,),.2),torch.full((3,),.05),torch.zeros(3),torch.full((29,),.01),
                              torch.full((29,),.5),torch.zeros(29))).to(self.device)
        self.expert=None; self._expert_iterator=None; self._expert_batch_size=None

    def normalize_context(self,x):
        return torch.cat(((x[...,:1]-.75)/.45,x[...,1:30]/.015,x[...,30:33]/.03),-1)

    def observation(self,state,previous_action,context,condition,primitives=None):
        p,R,v,w,q,qd=self.codec.unpack(state)
        return torch.cat(((R.transpose(-1,-2)@w.unsqueeze(-1)).squeeze(-1),-R[...,2,:],condition[...,:3],
                          q-self.default_q,qd,previous_action),-1)

    def add_noise(self,o): return o+(2*torch.rand_like(o)-1)*self.noise
    def goal(self,c): return c[...,:3]
    def valid_start(self,c,h): return torch.ones(c.shape[0],device=c.device,dtype=torch.bool)

    def advance(self,c,state=None,context=None):
        result=c.clone(); R=self.codec.unpack(state)[1]
        bad=(state[...,2]<.5)|(R[...,2,2]<math.cos(math.radians(70)))
        count=torch.where((c[...,3]>0)&bad,c[...,4]+1,0.)
        ready=count>=250
        result[...,4]=torch.where(ready,0.,count); result[...,5]=ready.float()
        # External velocity commands are held over the short imagined horizon.
        # Real collection keeps the baseline's heading controller and resampling.
        return result

    def termination_probability(self,state,c,primitives=None):
        R=self.codec.unpack(state)[1]
        pheight=torch.sigmoid((.5-state[...,2])/.02)
        ptilt=torch.sigmoid((math.cos(math.radians(70))-R[...,2,2])/.04)
        failure=1-(1-pheight)*(1-ptilt)
        return failure*torch.where(c[...,3]>0,c[...,5],1.)

    def critic_features(self,state,c,context,primitives=None):
        # Delayed recovery changes reward/termination at the same physical state.
        # Give the privileged critic that native bookkeeping as well.
        bookkeeping=c[...,3:6]/c.new_tensor([1.,250.,1.])
        return torch.cat((self.codec.features(state),self.goal(c),self.normalize_context(context),self.amp_features(state),bookkeeping),-1)

    def data_view(self,state,c,action,previous,primitives,overrides=None,soft=True):
        B=len(state); p,R,v,w,q,qd=self.codec.unpack(state)
        bp,bR,bv,bw=self.kin.forward(p,R,v,w,q,qd,all_bodies=True)
        fill=lambda x: x.new_zeros((B,self.num_bodies,*x.shape[2:])).index_copy(1,self.moving_ids,x)
        site_offset=(bR[:,self.site_body]@self.site_pos[None,:,:,None]).squeeze(-1)
        site_velocity=bv[:,self.site_body]+torch.cross(bw[:,self.site_body],site_offset,dim=-1)
        fields=dict(body_link_pos_w=fill(bp),body_link_quat_w=fill(mat_to_quat(bR)),body_link_lin_vel_w=fill(bv),
                    body_link_ang_vel_w=fill(bw),site_lin_vel_w=site_velocity,joint_pos=q,joint_vel=qd,
                    joint_acc=primitives['joint_acceleration'],soft_joint_pos_limits=self.soft_limits[None].expand(B,-1,-1),
                    default_root_state=self.default_root[None].expand(B,-1))
        if overrides: fields.update(overrides)
        failure=self.termination_probability(state,c) if soft else c[:,5]
        return SimpleNamespace(num_envs=B,step_dt=self.dt,extras={'log':{}},
            scene={'robot':SimpleNamespace(data=SimpleNamespace(**fields))},
            command_manager=SimpleNamespace(get_command=lambda name:c[:,:3]),
            action_manager=SimpleNamespace(action=action,prev_action=previous),
            termination_manager=SimpleNamespace(terminated=failure,_delay_env_mask=c[:,3]>0,_delay_counters=c[:,4]))

    def reward(self,state,c,action,previous,primitives=None,previous_state=None,soft=True,overrides=None,terms=False):
        view=self.data_view(state,c,action,previous,primitives,overrides,soft)
        values={}
        for name,cfg in self.reward_cfg.items():
            if name=='foot_slip':
                ids=cfg.params['asset_cfg'].site_ids
                active=(c[:,:2].norm(dim=-1)+c[:,2].abs()>cfg.params['command_threshold']).float()
                value=(view.scene['robot'].data.site_lin_vel_w[:,ids,:2].square().sum(-1)*primitives['foot_contact']).sum(-1)*active
            elif name=='self_collisions':
                force=primitives['self_contact_force_history'].reshape(len(state),*self.self_force_shape).norm(dim=-1)
                threshold=cfg.params.get('force_threshold',10.)
                value=(1-torch.prod(1-torch.sigmoid((force-threshold)/2.),dim=1)).sum(-1) if soft else (force>threshold).any(1).sum(-1).float()
            else: value=cfg.func(view,**cfg.params)
            values[name]=value*cfg.weight*self.dt
        total=torch.stack(list(values.values())).sum(0)
        return (total,values) if terms else total

    def sample_expert_features(self,count):
        if self.expert is None:
            from rsl_rl.utils import AMPLoader
            from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
            cfg=g1_amp_ppo_runner_cfg()
            self.expert=AMPLoader(cfg.amp_motion_files,cfg.amp_body_names,cfg.amp_anchor_name,
                                  self.env.scene['robot'].body_names,device=self.device)
            self._expert_batches=cfg.algorithm.num_learning_epochs*cfg.algorithm.num_mini_batches
        # Upstream cycles clips by batch index *inside* its generator. Creating
        # a new generator for every update would silently sample only clip 0.
        if self._expert_iterator is None or self._expert_batch_size!=count:
            self._expert_iterator=self.expert.feed_forward_generator(self._expert_batches,count)
            self._expert_batch_size=count
        try: return next(self._expert_iterator)
        except StopIteration:
            self._expert_iterator=self.expert.feed_forward_generator(self._expert_batches,count)
            return next(self._expert_iterator)
