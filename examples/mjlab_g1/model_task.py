"""RSL-style model-task contract wrapping the unchanged reference reward formulas."""
import torch
from mbdac.modules.codecs import ArticulatedCodec
from mbdac.utils.rotations import mat_to_6d
from .mdp import MDP, normalize_xi


class G1ModelTask(MDP):
    primitive_specs=()
    analytic_primitives=('root_position','root_orientation','root_velocity','root_angular_velocity',
                         'joint_position','joint_velocity','body_position','body_orientation','body_velocity','body_angular_velocity')
    action_dim=29
    context_dim=33
    obs_dim=99
    goal_dim=730
    critic_dim=914
    amp_feature_dim=195

    def __init__(self, kin, motion, default_q):
        super().__init__(kin,motion,default_q)
        self.codec=ArticulatedCodec(29).to(default_q.device)

    def normalize_context(self, context): return normalize_xi(context)

    def observation(self, state, previous_action, context, condition, primitives=None):
        return super().obs(self.codec.unpack(state),previous_action,context,condition)

    def reward(self, state, condition, action, previous_action, primitives=None,terms=False,previous_state=None):
        return super().reward(self.codec.unpack(state),condition,action,previous_action,terms=terms)

    def termination_probability(self, state, condition, primitives=None):
        return super().soft_termination(self.codec.unpack(state),condition)

    def hard_termination(self, state, condition):
        return super().termination(self.codec.unpack(state),condition)

    def advance(self, condition, state=None, context=None): return condition+1

    def goal(self, condition): return self.motion.goal(condition,10)

    def valid_start(self, condition, horizon):
        return condition+horizon<=self.motion.clip_end[self.motion.clip_id[condition]]

    def critic_features(self, state, condition, context, primitives=None):
        return torch.cat((super().critic_features(self.codec.unpack(state),condition,self.normalize_context(context)),self.goal(condition)),-1)

    def amp_features(self, state):
        p,R,v,w=self.kin.forward(*self.codec.unpack(state))
        anchor=7; body=[i for i in range(14) if i!=anchor]; Rt=R[...,anchor,:,:].transpose(-1,-2)
        bp=(Rt.unsqueeze(-3) @ (p[...,body,:]-p[...,anchor:anchor+1,:]).unsqueeze(-1)).squeeze(-1)
        bR=mat_to_6d(Rt.unsqueeze(-3) @ R[...,body,:,:])
        own=R[...,body,:,:].transpose(-1,-2)
        bv=(own @ v[...,body,:].unsqueeze(-1)).squeeze(-1)
        bw=(own @ w[...,body,:].unsqueeze(-1)).squeeze(-1)
        return torch.cat(tuple(x.flatten(-2) for x in [bp,bR,bv,bw]),-1)

    def sample_expert_features(self, count):
        f=torch.randint(self.motion.num_frames,(count,),device=self.default_q.device)
        f1=torch.minimum(f+1,self.motion.clip_end[self.motion.clip_id[f]])
        return self.amp_features(self.codec.pack(self.motion.ref_state(f))),self.amp_features(self.codec.pack(self.motion.ref_state(f1)))
