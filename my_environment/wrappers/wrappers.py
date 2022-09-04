__all__ = [
    "DiscreteActions3DOF",
    "GaudetStateObs",
    "RewardAnnealing",
    "RecordVideoFigure",
    "EpisodeAnalyzer6DOF",
    "EpisodeAnalyzer",
]

import os
from typing import Callable
import gym
from gym.spaces import Discrete, Box
from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt

from my_environment.envs.rocket_env import Rocket, Rocket6DOF
import numpy as np
from gym import logger
import wandb
import pandas as pd
pd.options.plotting.backend = "plotly"

class DiscreteActions3DOF(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont=[[0, -1], [-1, +1], [0, +1], [+1, +1]]):
        super().__init__(env)
        # Create an action table for all possible
        # combinations of the values of thrust and
        # gimbaling action = [delta, thrust]

        self.disc_to_cont = disc_to_cont
        self._action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return np.asarray(self.disc_to_cont[act])

    def get_keys_to_action(self):
        import pygame

        mapping = {
            (pygame.K_LEFT,): 1,
            (pygame.K_LEFT,pygame.K_UP,): 1,
            (pygame.K_RIGHT,): 3,
            (pygame.K_RIGHT,pygame.K_UP,): 3,
            (pygame.K_UP,): 2,
            (pygame.K_MODE,): 0,
        }
        return mapping


class GaudetStateObs(gym.ObservationWrapper):
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

        rewards_dict["thrust_penalty"] = -self.xi*(action[2]+1 if isinstance(self.env.unwrapped,Rocket6DOF) else action[1]+1)

        reward = sum(rewards_dict.values())

        info["rewards_dict"] = rewards_dict

        return obs, reward, done, info

class RecordVideoFigure(RecordVideo):
    def __init__(
        self,
        env,
        video_folder: str,
        image_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        assert isinstance(env.unwrapped, Rocket)
        
        super().__init__(
            env, video_folder, episode_trigger, step_trigger, video_length, name_prefix
        )

        self.image_folder = os.path.abspath(image_folder)
        # Create output folder if needed
        if os.path.isdir(self.image_folder):
            logger.warn(
                f"Overwriting existing images at {self.image_folder} folder (try specifying a different `image_folder` for the `RecordVideoFigure` wrapper if this is not desired)"
            )
        os.makedirs(self.image_folder, exist_ok=True)
        self.rewards_info = []

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        if not self.is_vector_env:
            self.rewards_info.append(infos["rewards_dict"])

        if self.episode_trigger(self.episode_id):
            if not self.is_vector_env:
                if dones:
                    states_dataframe = self.env.unwrapped.states_to_dataframe()
                    actions_dataframe = self.env.unwrapped.actions_to_dataframe()
                    vtarg_dataframe = self.env.unwrapped.vtarg_to_dataframe()
                    fig_rew = pd.DataFrame(self.rewards_info).plot()
                    plt.close()

                    names = self.env.unwrapped.state_names
                    values = np.abs(states_dataframe.iloc[-1,:] - [0,0,np.pi/2,0,0,0,0])
                    final_errors = {'final_errors/'+ n : v for n,v in zip(names, values)}

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "states": states_dataframe.plot(),
                                "actions": actions_dataframe.plot(),
                                "vtarg": vtarg_dataframe.plot(),
                                "rewards": fig_rew,
                                "landing_success": infos["rewards_dict"]["rew_goal"],
                                "used_mass" : states_dataframe.iloc[0,6] - states_dataframe.iloc[-1,6],
                                **final_errors
                            }
                        )
                
            elif dones[0]:
                states_dataframe = self.env.env_method('states_to_dataframe')[0]
                actions_dataframe = self.env.env_method('actions_to_dataframe')[0]
                vtarg_dataframe = self.env.env_method('vtarg_to_dataframe')[0]
                fig_rew = pd.DataFrame(self.env.env_method('rewards_info')).plot()
                plt.close()

                names = self.env.get_attr('state_names')[0]

                values = np.abs(states_dataframe.iloc[-1,:] - [0,0,np.pi/2,0,0,0,0])
                final_errors = {'final_errors/'+ n : v for n,v in zip(names, values)}

                if wandb.run is not None:
                    wandb.log(
                        {
                            "states": states_dataframe.plot(),
                            "actions": actions_dataframe.plot(),
                            "vtarg": vtarg_dataframe.plot(),
                            "rewards": fig_rew,
                            "used_mass" : states_dataframe.iloc[0,6] - states_dataframe.iloc[-1,6],
                            **final_errors
                        }
                    )
                
            pass

        return observations, rewards, dones, infos

    def reset(self, **kwargs):
        self.rewards_info = []
        return super().reset(**kwargs)

    def save_figure(self, figure: plt.Figure, prefix):
        figure_name = f"{prefix}-step-{self.step_id}"
        if self.episode_trigger:
            figure_name = f"{prefix}-step-{self.episode_id}"

        base_path = os.path.join(self.image_folder, figure_name)

        figure.savefig(base_path)

        return None


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


class EpisodeAnalyzer6DOF(RecordVideo):
    def __init__(self,
        env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video"
        ):
        super().__init__(env, video_folder, episode_trigger, step_trigger, video_length, name_prefix)

        assert isinstance(env.unwrapped, Rocket6DOF)
        self.rewards_info = []

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        if dones and self.episode_trigger(self.episode_id):
            if not self.is_vector_env:
                states_dataframe = self.env.unwrapped.states_to_dataframe()
                actions_dataframe = self.env.unwrapped.actions_to_dataframe()
                vtarg_dataframe = self.env.unwrapped.vtarg_to_dataframe()
                fig_rew = pd.DataFrame(self.rewards_info).plot()
                plt.close()

                names = self.env.unwrapped.state_names
                values = np.abs(states_dataframe.iloc[-1,:])
                final_errors = {'final_errors/'+ n : v for n,v in zip(names, values)}

                if wandb.run is not None:
                    wandb.log(
                        {
                            "states": states_dataframe.plot(),
                            "actions": actions_dataframe.plot(),
                            "vtarg": vtarg_dataframe.plot(),
                            "plots3d/trajectory": self.env.unwrapped.get_trajectory_plotly(),
                            "plots3d/vtarg_trajectory": self.env.unwrapped.get_vtarg_trajectory(),
                            "rewards": fig_rew,
                            "landing_success": infos["rewards_dict"]["rew_goal"],
                            "used_mass" : states_dataframe.iloc[0,-1] - states_dataframe.iloc[-1,-1],
                            **final_errors
                        }
                    )
            else:
                raise NotImplementedError
                
        return observations, rewards, dones, infos

    """
    def step(self, action):
        super().step(action)
        if not self.is_vector_env:
            self.rewards_info.append(infos["rewards_dict"])

        if self.episode_trigger(self.episode_id):
            if not self.is_vector_env:
                if dones:
                    states_dataframe = self.env.unwrapped.states_to_dataframe()
                    actions_dataframe = self.env.unwrapped.actions_to_dataframe()
                    vtarg_dataframe = self.env.unwrapped.vtarg_to_dataframe()
                    fig_rew = pd.DataFrame(self.rewards_info).plot()
                    plt.close()

                    names = self.env.unwrapped.state_names
                    values = np.abs(states_dataframe.iloc[-1,:])
                    final_errors = {'final_errors/'+ n : v for n,v in zip(names, values)}

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "states": states_dataframe.plot(),
                                "actions": actions_dataframe.plot(),
                                "vtarg": vtarg_dataframe.plot(),
                                "plots3d/trajectory": self.env.unwrapped.get_trajectory_plotly(),
                                "plots3d/vtarg_trajectory": self.env.unwrapped.get_vtarg_trajectory(),
                                "rewards": fig_rew,
                                "landing_success": infos["rewards_dict"]["rew_goal"],
                                "used_mass" : states_dataframe.iloc[0,-1] - states_dataframe.iloc[-1,-1],
                                **final_errors
                            }
                        )
                    
                    self.episode_id = 0
            elif dones[0]:
                print("EVAL_ENV IS A VECTOR ENVIRONMENT")
                raise NotImplementedError
                states_dataframe = self.env.env_method('states_to_dataframe')[0]
                actions_dataframe = self.env.env_method('actions_to_dataframe')[0]
                vtarg_dataframe = self.env.env_method('vtarg_to_dataframe')[0]
                fig_rew = pd.DataFrame(self.env.env_method('rewards_info')).plot()
                plt.close()

                names = self.env.get_attr('state_names')[0]

                values = np.abs(states_dataframe.iloc[-1,:] - [0,0,np.pi/2,0,0,0,0])
                final_errors = {'final_errors/'+ n : v for n,v in zip(names, values)}

                if wandb.run is not None:
                    wandb.log(
                        {
                            "states": states_dataframe.plot(),
                            "actions": actions_dataframe.plot(),
                            "vtarg": vtarg_dataframe.plot(),
                            "rewards": fig_rew,
                            "used_mass" : states_dataframe.iloc[0,6] - states_dataframe.iloc[-1,6],
                            **final_errors
                        }
                    )
            pass

        return observations, rewards, dones, infos
        """
