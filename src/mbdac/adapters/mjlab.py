"""MJLab bridge for native ManagerBasedRlEnv environments."""
from tensordict import TensorDict


class MjlabModelAdapter:
    def __init__(self, env, model_task):
        self.env=env; self.model_task=model_task
        self.num_envs,self.device=env.num_envs,env.device
        self.num_actions=env.action_manager.total_action_dim
        self.max_episode_length=env.max_episode_length
        env.reset()

    @property
    def cfg(self): return self.env.cfg
    @property
    def episode_length_buf(self): return self.env.episode_length_buf
    @property
    def unwrapped(self): return self.env
    def get_observations(self):
        return TensorDict(self.env.obs_buf,batch_size=[self.num_envs])
    def step(self, actions):
        obs,reward,terminated,truncated,extras=self.env.step(actions)
        if 'model_transition' not in extras: raise KeyError('Native task must capture pre-reset model_transition')
        extras['time_outs']=truncated
        return TensorDict(obs,batch_size=[self.num_envs]),reward,(terminated|truncated).long(),extras
    def close(self): self.env.close()
