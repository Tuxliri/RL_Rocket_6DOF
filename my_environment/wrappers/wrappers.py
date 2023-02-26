__all__ = [
    "GaudetStateObs",
    "RewardAnnealing",
    "EpisodeAnalyzer",
    "RemoveMassFromObs",
    "VerticalAttitudeReward",
]

import gym
from gym.spaces import Box
from matplotlib import pyplot as plt
from gym import Env

#from my_environment.envs.rocket_env import Rocket6DOF
import numpy as np
import wandb
import pandas as pd

pd.options.plotting.backend = "plotly"

class RewardAnnealing(gym.Wrapper):
    def __init__(self, env: gym.Env, thrust_penalty : float = 0.01) -> None:
        super().__init__(env)
        self.xi = self.reward_coefficients.get("xi", thrust_penalty)

    def step(self, action):
        obs, __, done, info = super().step(action)
        
        old_rewards_dict = info["rewards_dict"]
        new_rewards = [
            "attitude_constraint",
            "goal_conditions",
            'final_position',
            'final_velocity']
        rewards_dict = {key: old_rewards_dict[key] for key in new_rewards}

        rewards_dict["thrust_penalty"] = -self.xi*(action[2]+1)

        reward = sum(rewards_dict.values())

        info["rewards_dict"] = rewards_dict

        return obs, reward, done, info

class EpisodeAnalyzer(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
        #assert isinstance(env.unwrapped, Rocket6DOF)
        self.rewards_info = []
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        
        sim_time = self.env.unwrapped.SIM.t

        self.rewards_info.append({**info["rewards_dict"], 'time': sim_time})

        if done:
            fig = self.env.unwrapped.get_trajectory_plotly()
            states_dataframe = self.env.unwrapped.states_to_dataframe()
            actions_dataframe = self.env.unwrapped.actions_to_dataframe()
            
            rewards_dataframe = pd.DataFrame(self.rewards_info)
            if self.env.unwrapped.shaping_type == 'velocity':
                shaper_dataframe = self.env.unwrapped.vtarg_to_dataframe()
                shaper_name = "ep_history/atarg"
            elif self.env.unwrapped.shaping_type == 'acceleration':
                shaper_dataframe = self.env.unwrapped.atarg_to_dataframe()
                shaper_name = "ep_history/atarg"

            names = self.env.unwrapped.state_names
            values = np.abs(states_dataframe.iloc[-1,:])
            final_errors_dict = {'final_errors/'+ n : v for n,v in zip(names, values)}

            if wandb.run is not None:
                
                wandb.log(
                    {
                        "ep_history/states": states_dataframe.plot(),
                        "ep_history/actions": actions_dataframe.plot(),
                        shaper_name: shaper_dataframe.plot(),
                        "ep_history/rewards": rewards_dataframe.drop('time',axis=1).plot(),
                        "plots3d/atarg_trajectory": self.env.unwrapped.get_atarg_plotly(),
                        "plots3d/trajectory": fig,
                        "ep_statistic/used_mass" : states_dataframe.iloc[0,-1] - states_dataframe.iloc[-1,-1],
                        **final_errors_dict,
                    }
                )
            

            else:
                fig.show()

            self.rewards_info = []

        return obs, rew, done, info

# Reward shaping wrappers
class RemoveMassFromObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(13,))
    def observation(self, obs):
        return obs[0:13]

class VerticalAttitudeReward(gym.Wrapper):
    def __init__(self, env: Env, threshold_height=1e-3, weight=-0.5) -> None:
        super().__init__(env)
        self.threshold_height=threshold_height
        self.reward_weight=weight
    
    def step(self, action):
        obs, rew, done, info = super().step(action)
        state = self.env.unwrapped.get_state() 
        x=state[0]

        
        if x<self.threshold_height and info["rewards_dict"]["final_velocity"]>0:
            q = state[6:10]

            # Compute the angular deviation from vertical attitudes
            vertical_attitude_rew = np.clip(
                2*np.degrees(np.arccos(q[0]))*self.reward_weight,
                a_min=-10,
                a_max=+10,
            )
            
            rew+=vertical_attitude_rew
            
            info["rewards_dict"]["vertical_attitude_reward"]=vertical_attitude_rew

        info["rewards_dict"].setdefault("vertical_attitude_reward",0)
        return obs, rew, done, info