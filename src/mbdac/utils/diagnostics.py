"""Prediction diagnostics in physical units; these are not learned reward targets."""
import torch


def masked_rmse(error,mask):
    while mask.ndim<error.ndim: mask=mask.unsqueeze(-1)
    return ((error.square()*mask).sum()/mask.expand_as(error).sum().clamp_min(1)).sqrt()


@torch.no_grad()
def world_model_diagnostics(world_model,task,replay,window=8,batch=128):
    if replay.size<window: return {}
    w=replay.sample_windows(batch,window); state=w['s'][:,0]; predicted=[]
    for j in range(window):
        state=world_model(state,w['a'][:,j],task.normalize_context(w['context'][:,j])); predicted.append(state)
    predicted=torch.stack(predicted,1); error=task.codec.error(predicted,w['s1'])
    mask=w['mask']
    if 'eval_active' in w: mask=mask*w['eval_active'].float()
    result={}
    if hasattr(task.codec,'num_joints'):
        n=task.codec.num_joints
        groups={'position_m':slice(0,3),'rotation_rad':slice(3,6),'linear_velocity_m_s':slice(6,9),
                'angular_velocity_rad_s':slice(9,12),'joint_position_rad':slice(12,12+n),'joint_velocity_rad_s':slice(12+n,12+2*n)}
    else: groups={'state':slice(None)}
    for name,indices in groups.items():
        for label,index in [('one_step',0),(f'step_{window}',window-1)]:
            valid=mask[:,index].sum()
            result[f'model/{label}_samples']=int(valid)
            if valid>0: result[f'model/{label}_{name}_rmse']=float(masked_rmse(error[:,index,indices],mask[:,index]))
    return result
