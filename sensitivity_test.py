import gym
import my_environment
from my_environment.wrappers.wrappers import DiscreteActions3DOF, RecordVideoFigure, RewardAnnealing
from gym.wrappers import TimeLimit
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.ppo import PPO

config = {
        "env_id" : "my_environment/Falcon3DOF-v0",
        "policy_type": "MlpPolicy",
        "total_timesteps": int(1e5),
        "timestep" : 0.05,
        "max_time" : 40,
        "RANDOM_SEED" : 42,
        "initial_conditions" : [100, 1000, np.pi/2, 0, -100, 0],
        "initial_conditions_range" : [5,50,0,0,0,0],
        "reward_coefficients" : {
                                "alfa" : -0.01, 
                                "beta" : -1e-8,
                                "eta" : 2,
                                "gamma" : -10,
                                "delta" : -5
                                }
    }

config["max_ep_timesteps"] = int(config["max_time"]/config["timestep"])

env = gym.make(
    config["env_id"],
    IC=config["initial_conditions"],
    ICRange=config["initial_conditions_range"],
    timestep=config["timestep"],
    seed=config["RANDOM_SEED"]
    )

# Anneal the reward (remove v_targ following reward)
env = RewardAnnealing(env)

# Define a new custom action space with only three actions:
# - no thrust
# - max thrust gimbaled right
# - max thrust gimbaled left
# - max thrust downwards
env = DiscreteActions3DOF(env)
env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])

model = PPO.load('model.zip')

evaluate_policy(model, env, render=True)