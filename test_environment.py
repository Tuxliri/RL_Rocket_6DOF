import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from my_environment.wrappers.wrappers import RewardAnnealing
import gym
from gym.utils.play import PlayPlot
from gym.utils.play import play
from my_environment.wrappers import DiscreteActions3DOF
import pygame
from gym.wrappers import Monitor, TimeLimit
env_id = 'my_environment/Falcon3DOF-v0'

config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(2e6),
    "timestep" : 0.05,
    "max_time" : 100,
    "RANDOM_SEED" : 42,
    "initial_conditions" : [-1600, 2000, np.pi*3/4, 180, -90, 0, 50e3],
    "initial_conditions_range" : [5,50,0,0,0,0,1e3],
    "reward_coefficients" : {
                            "alfa" : -0.01, 
                            "beta" : 0,
                            "delta" : -5,
                            "eta" : 0.2,
                            "gamma" : -10,
                            "kappa" : 10,
                            "xi" : 0.004,
                            "waypoint" : 30,
                            "landing_radius" : 30
                            },
}
config["max_ep_timesteps"] = int(config["max_time"]/config["timestep"])


def make_env(config):
    env = gym.make(
    config["env_id"],
    IC=config["initial_conditions"],
    ICRange=config["initial_conditions_range"],
    timestep=config["timestep"],
    seed=config["RANDOM_SEED"]
    )
    # Anneal the reward (remove v_targ following reward)
    #env = RewardAnnealing(env)

    # Define a new custom action space with only three actions:
    # - no thrust
    # - max thrust gimbaled right
    # - max thrust gimbaled left
    # - max thrust downwards
    # env = DiscreteActions3DOF(env)
    env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])
    return env

env=make_env(config)
    
# env = gym.make(env_id)
policy = PPO('MlpPolicy', env)

# Plot the reward received
def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]

plotter = PlayPlot(callback, 30 * 5, ["reward"])
 
play(env, callback=plotter.callback)#,fps=30)