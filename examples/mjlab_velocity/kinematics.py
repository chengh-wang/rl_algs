"""Differentiable tree FK from compiled MJCF; no robot names or dimensions.

Root + scalar hinge joints and fixed links are supported. Nonzero joint anchors
and reference angles are included. Model SDK access happens only at construction.
"""
import numpy as np
import torch
from mbdac.utils.rotations import axis_angle_to_mat, quat_to_mat


class TreeKinematics:
    def __init__(self, model, entity, device, prefix='robot/'):
        ids = entity.indexing.body_ids.cpu().tolist()
        self.ids = ids
        self.names = list(entity.body_names)
        index = {b:i for i,b in enumerate(ids)}
        joints = {prefix+n:i for i,n in enumerate(entity.joint_names)}
        self.parent = []; self.joint = []
        axis=[]; anchor=[]; ref=[]
        for b in ids:
            parent = int(model.body_parentid[b]); self.parent.append(index.get(parent, -1))
            count = int(model.body_jntnum[b]); j = int(model.body_jntadr[b])
            if parent not in index:
                if count != 1 or int(model.jnt_type[j]) != 0:
                    raise ValueError('Expected one free root')
                self.joint.append(-1); axis.append([0.,0.,1.]); anchor.append([0.,0.,0.]); ref.append(0.)
            elif count == 0:
                self.joint.append(-1); axis.append([0.,0.,1.]); anchor.append([0.,0.,0.]); ref.append(0.)
            else:
                if count != 1 or int(model.jnt_type[j]) != 3:
                    raise ValueError('Only scalar hinge joints and fixed links are supported')
                self.joint.append(joints[model.joint(j).name])
                axis.append(model.jnt_axis[j]); anchor.append(model.jnt_pos[j])
                ref.append(model.qpos0[model.jnt_qposadr[j]])
        tensor=lambda x:torch.tensor(np.array(x),device=device,dtype=torch.float32)
        self.offset=tensor(model.body_pos[ids]); self.rotation=quat_to_mat(tensor(model.body_quat[ids]))
        self.axis=tensor(axis); self.anchor=tensor(anchor); self.ref=tensor(ref)
        self.site_ids=entity.indexing.site_ids.cpu().tolist()
        self.site_body=[index[int(model.site_bodyid[s])] for s in self.site_ids]
        self.site_pos=tensor(model.site_pos[self.site_ids])
        self.site_rot=quat_to_mat(tensor(model.site_quat[self.site_ids]))
        self.mass=tensor(model.body_mass[ids]); self.com=tensor(model.body_ipos[ids])
        ir=quat_to_mat(tensor(model.body_iquat[ids]))
        self.inertia=ir@torch.diag_embed(tensor(model.body_inertia[ids]))@ir.transpose(-1,-2)

    def forward(self, p, R, v, w, q, qd, body_indices=None):
        indices=tuple(range(len(self.ids))) if body_indices is None else tuple(body_indices)
        required=set(indices)
        for index in indices:
            while self.parent[index]>=0:
                index=self.parent[index];required.add(index)
        P=[None]*len(self.ids);Rs=P.copy();V=P.copy();W=P.copy()
        for i, (parent,j) in enumerate(zip(self.parent,self.joint)):
            if i not in required: continue
            if parent < 0: pi,ri,vi,wi=p,R,v,w
            else:
                rp=Rs[parent]; ro=rp@self.rotation[i]
                if j >= 0:
                    rot=axis_angle_to_mat(self.axis[i],q[...,j]-self.ref[i])
                    local=self.offset[i]+(self.rotation[i]@(self.anchor[i]-(rot@self.anchor[i,:,None]).squeeze(-1))[...,None]).squeeze(-1)
                    pi=P[parent]+(rp@local[...,None]).squeeze(-1); ri=ro@rot
                    aw=(ro@self.axis[i,:,None]).squeeze(-1)*qd[...,j:j+1]
                    wi=W[parent]+aw
                    joint_world=pi+(ri@self.anchor[i,:,None]).squeeze(-1)
                    vi=V[parent]+torch.cross(W[parent],pi-P[parent],dim=-1)+torch.cross(aw,pi-joint_world,dim=-1)
                else:
                    pi=P[parent]+(rp@self.offset[i,:,None]).squeeze(-1);ri=ro;wi=W[parent]
                    vi=V[parent]+torch.cross(wi,pi-P[parent],dim=-1)
            P[i]=pi;Rs[i]=ri;V[i]=vi;W[i]=wi
        return (torch.stack([P[i] for i in indices],-2),torch.stack([Rs[i] for i in indices],-3),
                torch.stack([V[i] for i in indices],-2),torch.stack([W[i] for i in indices],-2))

    def site_motion(self, site_indices, *state):
        """Prune unobserved branches before FK; same values and input gradients."""
        bodies=tuple(dict.fromkeys(self.site_body[s] for s in site_indices))
        p,R,v,w=self.forward(*state,body_indices=bodies)
        local=[bodies.index(self.site_body[s]) for s in site_indices]
        offset=(R[...,local,:,:]@self.site_pos[list(site_indices),:,None]).squeeze(-1)
        return (p[...,local,:]+offset,R[...,local,:,:]@self.site_rot[list(site_indices)],
                v[...,local,:]+torch.cross(w[...,local,:],offset,dim=-1),w[...,local,:])

    def sites(self, bodies):
        p,R,v,w=bodies
        offset=(R[...,self.site_body,:,:]@self.site_pos[...,None]).squeeze(-1)
        return (p[...,self.site_body,:]+offset,R[...,self.site_body,:,:]@self.site_rot,
                v[...,self.site_body,:]+torch.cross(w[...,self.site_body,:],offset,dim=-1),w[...,self.site_body,:])
