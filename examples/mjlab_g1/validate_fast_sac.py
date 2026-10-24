"""Native pre-reset/action/memory contracts for the augmented FastSAC Q state."""
import argparse
import json
from pathlib import Path
import torch
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.runners import ModelBasedRunner
from .train import build_task
from .fast_sac import G1FastSACTask,training_config
from .environment import G1TrackingEnv


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--motion-dir',type=Path,default=Path('logs/g1_wm_fastsac_markov_smoke/motion'))
    p.add_argument('--output',type=Path,default=Path('logs/fast_sac_native_contract.json'))
    args=p.parse_args();dev='cuda:0';torch.manual_seed(31)
    original=build_task(dev,args.motion_dir)
    task=G1FastSACTask(original.kin,original.motion,original.default_q)
    cfg=training_config(task,logger='tensorboard')
    cfg['algorithm'].update(wm_hidden=(64,64),buffer_steps=16,wm_window=4)
    cfg['algorithm']['fast_sac'].update(actor_hidden_dim=64,critic_hidden_dim=64,num_atoms=51,action_std_min=.01,action_std_max=.2)
    native=G1TrackingEnv(task,16,dev,seed=31);env=MjlabModelAdapter(native,task)
    runner=ModelBasedRunner(env,cfg,device=dev);alg=runner.alg
    actual=[];all_terminal=[];all_timeout=[];largest_gap=0.
    try:
        for i in range(12):
            if i==0:
                native.episode_length_buf[:8]=native.max_episode_length-1
                native.sim.data.qpos[:4,2]+=2.;native.sim.forward()
            obs=env.get_observations();raw=obs['actor'].clone();a=alg.act(obs)
            actual.append(alg.history.clone())
            nxt,r,d,extra=env.step(a);tr=extra['model_transition']
            tr.update(collector_id=torch.full((16,),i,device=dev),world_id=torch.arange(16,device=dev))
            alg.process_env_step(nxt,r,d,extra)
            row=alg.replay._row(alg.replay.size-1);stored=alg.replay.data
            assert torch.equal(stored['actor_observation'][row],raw)
            live=~d.bool()
            assert torch.equal(stored['next_actor_observation'][row,live],nxt['actor'][live])
            # Previous-action channels have no observation noise in this task.
            assert torch.equal(stored['next_actor_observation'][row,:,64:93],a)
            target=task.default_q+tr['context'][:,1:30]+native.alpha*a
            assert torch.allclose(native.action_manager.action[live],target[live],atol=1e-6)
            if i==0:
                assert tr['terminated'][:4].all() and tr['truncated'][4:8].all()
                largest_gap=float((tr['s1'][:4]-native.physical_state()[:4]).abs().max())
                assert largest_gap>1.
            all_terminal.append(tr['terminated']);all_timeout.append(tr['truncated'])
        start,h=alg._sample_replay_starts(256,10)
        expected=torch.stack(actual)[start['collector_id'],start['world_id']]
        history_error=float((alg._history(h)-expected).abs().max())
        assert history_error==0
        tr,h0,h1,c1=alg._real_batch(256)
        assert torch.equal(h1[:,:-1],h0[:,1:])
        assert torch.equal(h1[:,-1],tr['next_actor_observation'])
        q=alg._q_features(tr['s1'],c1,tr['context'],tr['a'],h1)
        assert q.shape[1]==task.critic_dim+29+10*99
        result=dict(native_manager_based_env=True,num_envs=16,steps=12,q_state_dim=q.shape[1],
            q_has_previous_action_and_actual_policy_memory=True,reference_clock_in_q=True,
            actual_history_max_abs=history_error,pre_reset_physical_gap=largest_gap,
            true_termination_count=int(torch.stack(all_terminal).sum()),time_limit_count=int(torch.stack(all_timeout).sum()),
            raw_to_native_action_mapping_passed=True,physical_next_observation_action_channels_passed=True,
            actor_norm_count=int(alg.actor.obs_norm.count),q_norm_count=int(alg.critic.norm.count))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:runner.close();env.close()


if __name__=='__main__':main()
