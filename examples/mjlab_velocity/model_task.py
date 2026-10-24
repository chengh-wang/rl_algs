"""Analytic native reward/observation views with learned physical contact heads.

No reward totals are learned. The augmented state retains the native swing-height
memory and air-time counters, which cannot be reconstructed from an instantaneous
joint pose. Hard contact predicates are softened only for imagined gradients.
"""
import copy
import math
from types import SimpleNamespace as NS
import numpy as np
import torch
from mbdac.modules.codecs import ArticulatedCodec
from mbdac.modules.primitives import PhysicalSignals, PrimitiveSpec
from mbdac.utils.rotations import mat_to_quat, quat_to_mat
from .kinematics import TreeKinematics


class VelocityCodec(ArticulatedCodec):
    def __init__(self, joints, feet):
        super().__init__(joints)
        self.physical_dim = self.state_dim; self.memory_dim = 2*feet
        self.state_dim += self.memory_dim; self.feature_dim += self.memory_dim
        # Native contact forces are world-frame vectors. Retain world yaw for
        # their physical heads instead of giving them only yaw-invariant inputs.
        self.feature_dim += 2
        self.delta_dim += self.memory_dim
        self.error_floor = torch.cat((self.error_floor, torch.full((feet,), .003), torch.full((feet,), .02)))

    def unpack(self, s): return super().unpack(s[...,:self.physical_dim])
    def features(self, s):
        R=self.unpack(s)[1];heading=R[..., :2,0]
        heading=heading/heading.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        return torch.cat((super().features(s),s[...,self.physical_dim:],heading),-1)
    def targets(self, s, s1):
        return torch.cat((super().targets(s,s1),s1[...,self.physical_dim:]-s[...,self.physical_dim:]),-1)
    def integrate(self, s, d):
        physical = super().integrate(s, d[...,:12+2*self.num_joints])
        return torch.cat((physical,s[...,self.physical_dim:]+d[...,12+2*self.num_joints:]),-1)


class NativeVelocityTask:
    def __init__(self, env):
        self.env=env; self.device=env.device; self.dt=env.step_dt
        robot=env.scene['robot']; model=env.sim.mj_model
        self.action_dim=len(robot.joint_names); self.context_dim=self.action_dim+4
        self.feet=env.scene['feet_ground_contact'].data.found.shape[-1]
        self.codec=VelocityCodec(self.action_dim,self.feet).to(self.device)
        self.kin=TreeKinematics(model,robot,self.device)
        self.imu_linear=self.kin.site_ids.index(int(model.sensor_objid[model.sensor('robot/imu_lin_vel').id]))
        self.imu_angular=self.kin.site_ids.index(int(model.sensor_objid[model.sensor('robot/imu_ang_vel').id]))
        self.default_q=robot.data.default_joint_pos[0].clone()
        self.default_qd=robot.data.default_joint_vel[0].clone()
        self.limits=robot.data.soft_joint_pos_limits[0].clone()
        self.obs_dim=env.obs_buf['actor'].shape[-1]; self.goal_dim=3
        self.critic_dim=env.obs_buf['critic'].shape[-1]
        self.reward_cfg={k:env.reward_manager.get_term_cfg(k) for k in env.reward_manager.active_terms if k!='_model_capture'}
        self.foot_ids=self.reward_cfg['foot_clearance'].params['asset_cfg'].site_ids
        self.signals=PhysicalSignals(); required=[]
        def register(name,shape,units,reader,binary=False,transition=False,frame='world'):
            self.signals.register(PrimitiveSpec(name,int(np.prod(shape)),units,frame,binary,transition),measure=reader)
            required.append(name)
        register('foot_contact',(self.feet,),'probability',lambda e:e.scene['feet_ground_contact'].data.found>0,True)
        register('foot_force',(self.feet,3),'N',lambda e:e.scene['feet_ground_contact'].data.force)
        register('first_contact',(self.feet,),'probability',lambda e:e.scene['feet_ground_contact'].compute_first_contact(e.step_dt),True,True)
        self.collision_name='self_collision' if 'self_collisions' in self.reward_cfg else 'nonfoot_ground_touch'
        self.force_shape=env.scene[self.collision_name].data.force_history.shape[1:]
        register('collision_force_history',self.force_shape,'N',lambda e:e.scene[self.collision_name].data.force_history,
                 frame='contact frame; four physics substeps; native sensor selection')
        self.physical_dependencies=tuple(required); self.primitive_specs=self.signals.learned_specs(required)
        self.analytic_primitives=('body_pose','body_velocity','site_pose','site_velocity','angular_momentum')
        self.com_body=robot.body_names.index(env.cfg.events.get('base_com', NS(params={'asset_cfg':NS(body_names=(env.cfg.viewer.body_name,))})).params['asset_cfg'].body_names[0])
        self.friction_geom=int(robot.find_geoms(env.cfg.events['foot_friction'].params['asset_cfg'].geom_names)[0][0]) if 'foot_friction' in env.cfg.events else 0
        self.native_pose=copy.copy(self.reward_cfg['pose'].func)
        self.native_pose.default_joint_pos=self.default_q[None]
        self.noise=torch.zeros(self.obs_dim,device=self.device); offset=0
        lengths=(3,3,3,self.action_dim,self.action_dim,self.action_dim,3)
        for term,length in zip(env.cfg.observations['actor'].terms.values(),lengths):
            if env.cfg.observations['actor'].enable_corruption and term.noise:
                if term.noise.n_min != -term.noise.n_max: raise ValueError('Expected symmetric native uniform noise')
                self.noise[offset:offset+length]=term.noise.n_max
            offset+=length
        if offset != self.obs_dim: raise ValueError('Unsupported native observation layout')
        action=env.action_manager.get_term('joint_pos')
        self.action_joint_ids=action.target_ids.clone()
        self.action_joint_names=list(action.target_names)
        self.pd_scale=torch.as_tensor(action._scale,device=self.device).expand(env.num_envs,self.action_dim)[0].clone()
        ids=robot.indexing.joint_ids.cpu().numpy()
        limits=torch.tensor(model.jnt_range[ids],device=self.device,dtype=torch.float32)
        self.action_bounds=(limits[self.action_joint_ids]-self.default_q[self.action_joint_ids,None]).abs().amax(-1)/self.pd_scale

    def normalize_context(self, x):
        return torch.cat(((x[...,:1]-.75)/.45,x[...,1:1+self.action_dim]/.015,x[...,-3:]/.03),-1)

    def goal(self,c): return c[...,:3]
    def advance(self,c,state=None,context=None): return c
    def valid_start(self,c,h): return torch.ones(c.shape[0],device=c.device,dtype=torch.bool)
    def add_noise(self,o): return o+(2*torch.rand_like(o)-1)*self.noise

    def geometry(self,s): return self.kin.forward(*self.codec.unpack(s))

    def observation(self,s,previous,context,c,primitives=None):
        p,R,v,w,q,qd=self.codec.unpack(s); Rt=R.transpose(-1,-2)
        sites=self.kin.site_motion((self.imu_linear,self.imu_angular),p,R,v,w,q,qd)
        lr=sites[1][...,0,:,:].transpose(-1,-2); ar=sites[1][...,1,:,:].transpose(-1,-2)
        return torch.cat(((lr@sites[2][...,0,:,None]).squeeze(-1),
                          (ar@sites[3][...,1,:,None]).squeeze(-1),-R[...,2,:],
                          q-self.default_q,qd-self.default_qd,previous,c[...,:3]),-1)

    def data_view(self,s,c,action,previous,primitives,previous_state=None,overrides=None):
        B=len(s); p,R,v,w,q,qd=self.codec.unpack(s); Rt=R.transpose(-1,-2)
        bp,bR,bv,bw=bodies=self.geometry(s); sp,sR,sv,sw=self.kin.sites(bodies)
        fields=dict(joint_pos=q,joint_vel=qd,soft_joint_pos_limits=self.limits[None],
                    gravity_vec_w=s.new_tensor([0.,0.,-1.]).expand(B,-1),
                    root_link_lin_vel_b=(Rt@v[...,None]).squeeze(-1),root_link_ang_vel_b=(Rt@w[...,None]).squeeze(-1),
                    projected_gravity_b=-R[...,2,:],body_link_quat_w=mat_to_quat(bR),body_link_pos_w=bp,
                    body_link_lin_vel_w=bv,body_link_ang_vel_w=bw,site_pos_w=sp,site_lin_vel_w=sv)
        if overrides: fields.update(overrides)
        force=primitives['foot_force'].reshape(B,self.feet,3)
        contact=primitives['foot_contact']
        sensor=NS(data=NS(found=contact,force=force,current_air_time=s[...,self.codec.physical_dim+self.feet:]),
                  compute_first_contact=lambda dt:primitives['first_contact'])
        collision=NS(data=NS(force_history=primitives['collision_force_history'].reshape(B,*self.force_shape)))
        view=NS(num_envs=B,step_dt=self.dt,extras={'log':{}},scene={'robot':NS(data=NS(**fields)),
                'feet_ground_contact':sensor,self.collision_name:collision},
                command_manager=NS(get_command=lambda name:c[...,:3]),action_manager=NS(action=action,prev_action=previous))
        return view,bodies

    def reward(self,s,c,action,previous,primitives=None,previous_state=None,soft=True,overrides=None,terms=False):
        view,bodies=self.data_view(s,c,action,previous,primitives,previous_state,overrides)
        data=view.scene['robot'].data; active=(c[...,:2].norm(dim=-1)+c[...,2].abs()>.05).float()
        values={}
        for name,cfg in self.reward_cfg.items():
            if cfg.weight == 0: continue
            if name=='pose': value=self.native_pose(view,**cfg.params)
            elif name=='foot_slip':
                contact=primitives['foot_contact'] if soft else (primitives['foot_contact']>0).float()
                value=(data.site_lin_vel_w[:,self.foot_ids,:2].square().sum(-1)*contact).sum(-1)*active
            elif name=='foot_swing_height':
                old=previous_state[...,self.codec.physical_dim:self.codec.physical_dim+self.feet]
                height=data.site_pos_w[:,self.foot_ids,2]
                contact=primitives['foot_contact']; first=primitives['first_contact']
                peak=(1-contact)*torch.maximum(old,height)+contact*old
                value=((peak/cfg.params['target_height']-1).square()*first).sum(-1)*active
            elif name=='self_collisions':
                forces=primitives['collision_force_history'].reshape(len(s),*self.force_shape).norm(dim=-1)
                value=torch.sigmoid((forces.amax(1)-cfg.params['force_threshold'])).sum(-1) if soft else (forces>cfg.params['force_threshold']).any(1).sum(-1).float()
            elif name=='angular_momentum':
                bp,bR,bv,bw=bodies
                cp=bp+(bR@self.kin.com[...,None]).squeeze(-1)
                cv=bv+torch.cross(bw,cp-bp,dim=-1)
                center=(cp*self.kin.mass[None,:,None]).sum(-2)/self.kin.mass.sum()
                iw=bR@self.kin.inertia@bR.transpose(-1,-2)
                momentum=((iw@bw[...,None]).squeeze(-1)+torch.cross(cp-center[:,None],cv*self.kin.mass[None,:,None],dim=-1)).sum(-2)
                if overrides and 'angular_momentum' in overrides: momentum=overrides['angular_momentum']
                value=momentum.square().sum(-1)
            else: value=cfg.func(view,**cfg.params)
            values[name]=value*cfg.weight*self.dt
        total=torch.stack(list(values.values())).sum(0)
        return (total,values) if terms else total

    def critic_features(self,s,c,context,primitives=None):
        # Exactly the native privileged observation ordering, without sensor noise.
        obs=self.observation(s,s.new_zeros((*s.shape[:-1],self.action_dim)),context,c)
        # last_action is supplied separately to FastSAC's controlled-state Q.
        site=self.kin.sites(self.geometry(s))[0][...,self.foot_ids,2]
        forces=primitives['foot_force']; forces=torch.sign(forces)*torch.log1p(forces.abs())
        return torch.cat((obs,site,s[...,self.codec.physical_dim+self.feet:],primitives['foot_contact'],forces),-1)

    def termination_probability(self,s,c,primitives=None):
        R=self.codec.unpack(s)[1]
        failure=torch.sigmoid((math.cos(math.radians(70))-R[...,2,2])/.04)
        if 'illegal_contact' in self.env.cfg.terminations:
            force=primitives['collision_force_history'].reshape(*s.shape[:-1],*self.force_shape).norm(dim=-1)
            threshold=self.env.cfg.terminations['illegal_contact'].params['force_threshold']
            probability=torch.sigmoid(force.flatten(-2).amax(-1)-threshold)
            failure=1-(1-failure)*(1-probability)
        return failure

    def controlled_critic_features(self,s,c,context,previous_action,history=None,primitives=None):
        features=self.critic_features(s,c,context,primitives)
        index=9+2*self.action_dim
        return torch.cat((features[...,:index],previous_action,features[...,index+self.action_dim:]),-1)
