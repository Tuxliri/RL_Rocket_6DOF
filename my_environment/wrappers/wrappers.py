__all__ = [
    "GaudetStateObs",
    "RewardAnnealing",
    "RecordVideoFigure",
    "EpisodeAnalyzer6DOF",
    "EpisodeAnalyzer",
]

import gym
from gym.spaces import Box
from matplotlib import pyplot as plt

from my_environment.envs.rocket_env import Rocket, Rocket6DOF
import numpy as np
import wandb
import pandas as pd

pd.options.plotting.backend = "plotly"

class GaudetStateObs(gym.ObservationWrapper): #TODO: adapt to 6DOF environment
    def __init__(self, env: Rocket) -> None:
        super().__init__(env)
        self.observation_space = Box(low=-1, high=1, shape=(4,))

    def observation(self, observation):
        x, y, th = observation[0:3]
        vx, vy, vth = observation[3:6]

        r = np.array([x, y])
        v = np.array([vx, vy])

        v_targ, t_go = self.env.unwrapped.compute_vtarg(r, v)
        vx_targ, vy_targ = v_targ

        return np.float32([vx - vx_targ, vy - vy_targ, t_go, y])

class RewardAnnealing(gym.Wrapper):
    def __init__(self, env: gym.Env, thrust_penalty : float = 0.01) -> None:
        super().__init__(env)
        self.xi = self.reward_coefficients.get("xi", thrust_penalty)

    def step(self, action):
        obs, __, done, info = super().step(action)
        
        old_rewards_dict = info["rewards_dict"]
        new_rewards = ["attitude_constraint", "rew_goal"]
        rewards_dict = {key: old_rewards_dict[key] for key in new_rewards}

        rewards_dict["thrust_penalty"] = -self.xi*(action[2]+1)

        reward = sum(rewards_dict.values())

        info["rewards_dict"] = rewards_dict

        return obs, reward, done, info

class EpisodeAnalyzer(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
        assert isinstance(env.unwrapped, Rocket6DOF)
        self.rewards_info = []

    def step(self, action):
        obs, rew, done, info = super().step(action)
        
        self.rewards_info.append(info["rewards_dict"])

        if done:
            fig = self.env.unwrapped.get_trajectory_plotly()
            states_dataframe = self.env.unwrapped.states_to_dataframe()
            actions_dataframe = self.env.unwrapped.actions_to_dataframe()
            vtarg_dataframe = self.env.unwrapped.vtarg_to_dataframe()
            fig_rew = pd.DataFrame(self.rewards_info).plot()
            plt.close()

            names = self.env.unwrapped.state_names
            values = np.abs(states_dataframe.iloc[-1,:])
            final_errors_dict = {'final_errors/'+ n : v for n,v in zip(names, values)}

            if wandb.run is not None:
                wandb.log(
                    {
                        "ep_history/states": states_dataframe.plot(),
                        "ep_history/actions": actions_dataframe.plot(),
                        "ep_history/vtarg": vtarg_dataframe.plot(),
                        "ep_history/rewards": fig_rew,
                        "plots3d/vtarg_trajectory": self.env.unwrapped.get_vtarg_trajectory(),
                        "plots3d/trajectory": fig,
                        "ep_statistic/landing_success": info["rewards_dict"]["rew_goal"],
                        "ep_statistic/used_mass" : states_dataframe.iloc[0,-1] - states_dataframe.iloc[-1,-1],
                        **final_errors_dict
                    }
                )
            

            else:
                fig.show()

            self.rewards_info = []

        return obs, rew, done, info
