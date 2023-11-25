import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class LeggedRobotHistory(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.obs_history_length = self.cfg.env.obs_history_length
        self.num_actor_observation = self.cfg.env.num_actor_observation
        self.num_history_observations = self.obs_history_length * self.num_actor_observation
        self.obs_history = torch.zeros((self.num_envs, self.num_history_observations), 
                                       dtype=torch.float32, 
                                       device=self.sim_device,
                                       requires_grad=False)
        
    def get_observations(self):
        return self.obs_history
    
    def step(self, action):
        super().step(action)
        self.privileged_obs_buf = self.obs_buf.clone()
        self.obs_history = torch.cat((self.obs_history[:, self.num_actor_observation:], self.obs_buf.clone()), dim=-1)
        return self.obs_history, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0.0

    def reset(self):
        obs, privileged_obs = super().reset()
        return obs, privileged_obs