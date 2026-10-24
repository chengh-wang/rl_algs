"""G1 example using native MJLab physics/actions/reward/reset managers.

Only task terms and a small transition-capture hook extend ManagerBasedRlEnv.
The simulation step/decimation loop is the installed MJLab implementation.
"""
from pathlib import Path
import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields, RecomputeLevel
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mbdac.utils.rotations import quat_to_mat,mat_to_quat,so3_exp
from .robot import MJCF_PATH,G1_JOINT_NAMES,motor_of,_kp,_kd,DEFAULT_JOINT_POS,action_scale


def get_spec():
    spec=mujoco.MjSpec.from_file(MJCF_PATH)
    root=Path(MJCF_PATH).parent
    spec.assets={m.file:(root/m.file).read_bytes() for m in spec.meshes}
    # Match the supplied reproduction's self-collision setting, while keeping floor collisions.
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity: geom.contype=1; geom.conaffinity=2
    return spec


@requires_model_fields('geom_friction','body_ipos',recompute=RecomputeLevel.set_const)
def startup_context(env,env_ids):
    n=env.num_envs; dev=env.device
    env.context=torch.zeros(n,33,device=dev)
    env.context[:,0]=1 if env.eval_mode else torch.empty(n,device=dev).uniform_(.3,1.2)
    if not env.eval_mode:
        env.context[:,30:33]=torch.rand(n,3,device=dev)*torch.tensor([.05,.10,.10],device=dev)-torch.tensor([.025,.05,.05],device=dev)
    robot=env.scene['robot']
    geom_ids=robot.indexing.geom_ids
    env.sim.model.geom_friction[:,geom_ids,0]=env.context[:,:1]
    # Floor's friction must not override the sampled foot friction (MuJoCo max rule).
    floor=[i for i in range(env.sim.mj_model.ngeom) if env.sim.mj_model.geom_bodyid[i]==0]
    env.sim.model.geom_friction[:,floor,0]=1 if env.eval_mode else .3
    torso=env.sim.mj_model.body('robot/torso_link').id
    env.sim.model.body_ipos[:,torso,:]=torch.as_tensor(env.sim.mj_model.body_ipos[torso],device=dev)+env.context[:,30:33]


def reset_motion(env,env_ids):
    ids=env_ids if env_ids is not None else torch.arange(env.num_envs,device=env.device)
    n=len(ids); task=env.model_task; motion=task.motion
    if env._fixed_frames is not None:
        f=env._fixed_frames[ids].clone(); bins=motion.frame_bin[f]
    else:
        soft=torch.softmax(-25*env.bin_return/10,0)
        env.bin_weights.mul_(.999).add_(.001*soft)
        probs=.5/motion.num_bins+.4*soft+.1*env.bin_weights/env.bin_weights.sum()
        bins=torch.multinomial(probs,n,replacement=True)
        start=motion.bin_start[bins]; end=torch.minimum(start+50,motion.clip_end[motion.bin_clip[bins]])
        f=start+(torch.rand(n,device=env.device)*(end-start).float()).long()
    p,R,v,w,q,qd=(x.clone() for x in motion.ref_state(f))
    if not env.eval_mode:
        p+=torch.rand_like(p)*torch.tensor([.1,.1,.02],device=env.device)-torch.tensor([.05,.05,.01],device=env.device)
        rv=(torch.rand_like(p)*2-1)*torch.tensor([.1,.1,.2],device=env.device); R=so3_exp(rv)@R
        v+=(torch.rand_like(v)*2-1)*torch.tensor([.5,.5,.2],device=env.device)
        w+=(torch.rand_like(w)*2-1)*torch.tensor([.52,.52,.78],device=env.device)
        q+=(torch.rand_like(q)*2-1)*.1
        env.context[ids,1:30]=torch.empty(n,29,device=env.device).uniform_(-.01,.01)
    q=torch.maximum(torch.minimum(q,task.kin.q_hi),task.kin.q_lo)
    root=torch.cat((p+env.scene.env_origins[ids],mat_to_quat(R),v,w),-1)
    robot=env.scene['robot']; robot.write_root_state_to_sim(root,env_ids=ids)
    robot.write_joint_state_to_sim(q,qd,env_ids=ids)
    env.f[ids]=f; env.previous_action[ids]=0; env.start_bin[ids]=bins
    env.disc_return[ids]=0; env.ep_return[ids]=0


def observation(env):
    state=env.physical_state()
    result=env.model_task.observation(state,env.previous_action,env.context,env.f)
    return result if env.eval_mode else env.model_task.add_noise(result)


def goal(env): return env.model_task.goal(env.f)


def capture_failure(env):
    env._physical_next=env.physical_state().clone()
    if env._pending is not None:
        f=env._pending['condition']
        env.f=torch.minimum(f+1,env.model_task.motion.clip_end[env.model_task.motion.clip_id[f]])
    env.last_failure=env.model_task.hard_termination(env._physical_next,env.f)
    return torch.zeros_like(env.last_failure) if env.eval_mode else env.last_failure


def timeout(env):
    env.last_timeout=(env.f>=env.model_task.motion.clip_end[env.model_task.motion.clip_id[env.f]]) | (env.episode_length_buf>=env.max_episode_length)
    return torch.zeros_like(env.last_timeout) if env.eval_mode else env.last_timeout


def reward(env):
    old=env._pending
    if old is None: return torch.zeros(env.num_envs,device=env.device)
    r,env._reward_terms=env.model_task.reward(env._physical_next,env.f,old['a'],old['a_prev'],terms=True)
    env.disc_return+=(.99**old['ep_step'].float())*r; env.ep_return+=r
    return r


def make_cfg(task,num_envs,eval_mode=False,seed=0):
    actuators=[]
    for name in G1_JOINT_NAMES:
        effort,vel,arm=motor_of(name)
        actuators.append(BuiltinPositionActuatorCfg(target_names_expr=(name,),stiffness=_kp(arm),damping=_kd(arm),effort_limit=effort,armature=arm))
    entity=EntityCfg(spec_fn=get_spec,init_state=EntityCfg.InitialStateCfg(pos=(0,0,.76),joint_pos=DEFAULT_JOINT_POS),
                     articulation=EntityArticulationInfoCfg(actuators=tuple(actuators)))
    return ManagerBasedRlEnvCfg(decimation=4,scene=SceneCfg(num_envs=num_envs,env_spacing=0.,entities={'robot':entity},terrain=TerrainEntityCfg(terrain_type='plane')),
        sim=SimulationCfg(njmax=500,nconmax=128,mujoco=MujocoCfg(timestep=.005,integrator='implicitfast')),
        episode_length_s=1e7 if eval_mode else 10.,scale_rewards_by_dt=False,
        actions={'joint_targets':JointPositionActionCfg(entity_name='robot',actuator_names=('.*',),scale=1.,offset=0.,use_default_offset=False)},
        observations={'actor':ObservationGroupCfg(terms={'state':ObservationTermCfg(func=observation)},enable_corruption=False),
                      'goal':ObservationGroupCfg(terms={'reference':ObservationTermCfg(func=goal)},enable_corruption=False)},
        rewards={'tracking':RewardTermCfg(func=reward,weight=1.)},
        terminations={'failure':TerminationTermCfg(func=capture_failure),'timeout':TerminationTermCfg(func=timeout,time_out=True)},
        events={'context':EventTermCfg(func=startup_context,mode='startup'),'motion':EventTermCfg(func=reset_motion,mode='reset')},seed=seed)


class G1TrackingEnv(ManagerBasedRlEnv):
    def __init__(self, task,num_envs,device='cuda:0',eval_mode=False,seed=0):
        self.model_task=task; self.eval_mode=eval_mode; self._pending=None; self._fixed_frames=None
        self.f=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.context=torch.zeros(num_envs,33,device=device); self.context[:,0]=1.
        self.previous_action=torch.zeros(num_envs,29,device=device)
        self.start_bin=torch.zeros_like(self.f); self.disc_return=torch.zeros(num_envs,device=device); self.ep_return=torch.zeros_like(self.disc_return)
        self.bin_return=torch.zeros(task.motion.num_bins,device=device); self.bin_weights=torch.full_like(self.bin_return,1/task.motion.num_bins)
        super().__init__(make_cfg(task,num_envs,eval_mode,seed),device=device)
        assert list(self.scene['robot'].joint_names)==G1_JOINT_NAMES
        self.alpha=torch.tensor([action_scale(n) for n in G1_JOINT_NAMES],device=device)

    def physical_state(self):
        q=self.sim.data.qpos; v=self.sim.data.qvel; R=quat_to_mat(q[:,3:7])
        return torch.cat((q[:,:3]-self.scene.env_origins,q[:,3:7],v[:,:3],(R@v[:,3:6,None]).squeeze(-1),q[:,7:],v[:,6:]),-1)

    def _reset_idx(self,env_ids=None):
        episode_logs={}
        if self._pending is not None and env_ids is not None and len(env_ids):
            T=self.episode_length_buf[env_ids].float()
            term=self.reset_terminated[env_ids]
            boot=torch.where(term,0.,(.99**T)*(self.ep_return[env_ids]/T.clamp_min(1))/.01)
            bins=self.start_bin[env_ids]; G=self.disc_return[env_ids]+boot
            # Aggregate duplicate bins explicitly, avoiding nondeterministic last-writer scatter.
            sums=torch.zeros_like(self.bin_return).scatter_add_(0,bins,G)
            counts=torch.zeros_like(self.bin_return).scatter_add_(0,bins,torch.ones_like(G))
            mask=counts>0; self.bin_return[mask]=.8*self.bin_return[mask]+.2*sums[mask]/counts[mask]
            episode_logs=dict(ep_length=T.mean(),ep_return_norm=(self.ep_return[env_ids]/(T*.1)).mean())
        super()._reset_idx(env_ids)
        self.extras.setdefault('log',{}).update(episode_logs)

    @torch.no_grad()
    def step(self, action):
        self.extras['log']={}
        self._pending={'s':self.physical_state().clone(),'a':action.clone(),'a_prev':self.previous_action.clone(),
                       'context':self.context.clone(),'condition':self.f.clone(),'ep_step':self.episode_length_buf.clone()}
        self.previous_action=action.clone()
        target=self.model_task.default_q+self.context[:,1:30]+self.alpha*action
        obs,r,term,trunc,extras=super().step(target)
        extras['log'].update({'reward/'+k:v.mean() for k,v in self._reward_terms.items()})
        extras['model_transition']={**self._pending,'s1':self._physical_next.clone(),'reward':r.clone(),'terminated':term.clone(),'truncated':trunc.clone()}
        self._pending=None
        return obs,r,term,trunc,extras
