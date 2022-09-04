from datetime import datetime
from genericpath import exists

import os
import sys
import gym
import wandb
import numpy as np
import stable_baselines3

from stable_baselines3 import A2C, DQN, PPO, TD3
from tensorboard import program
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

import my_environment
from my_environment.wrappers.wrappers import DiscreteActions3DOF, RecordVideoFigure, RewardAnnealing
from gym.wrappers import TimeLimit

config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(5e3),
    "timestep" : 0.05,
    "max_time" : 150,
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
config["eval_freq"] = int(config["total_timesteps"]/20)

def make_env():
    env = gym.make(
    config["env_id"],
    IC=config["initial_conditions"],
    ICRange=config["initial_conditions_range"],
    timestep=config["timestep"],
    seed=config["RANDOM_SEED"],
    reward_coeff=config["reward_coefficients"]
    )
    
    # Define a new custom action space with only three actions:
    # - no thrust
    # - max thrust gimbaled right
    # - max thrust gimbaled left
    # - max thrust downwards

    # env = DiscreteActions3DOF(env)
    env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])
    env = Monitor(
        env,
        allow_early_resets=True,
        filename="logs_PPO",
        )
    return env

if __name__ == "__main__":

    # Choose the folder to store tensorboard logs
    TENSORBOARD_LOGS_DIR = "./logs"
    

    run = wandb.init(
        project="test_runs",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )   


    env = make_env()
    
    model = PPO(
        config["policy_type"],
        env,
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
        seed=config["RANDOM_SEED"],
        ent_coef=0.01,
        )

    def make_eval_env():
        training_env = make_env()
        return RecordVideoFigure(training_env, video_folder=f"videos/{run.id}",
        image_folder=f"images/{run.id}", episode_trigger= lambda x: x%5==0 )

  
    eval_env = DummyVecEnv([make_eval_env])
    
    callbacksList = [
        EvalCallback(
            eval_env,
            eval_freq = config["max_ep_timesteps"]/20,
            n_eval_episodes = 5,
            render=False,
            deterministic=True,
            ),
        WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
            gradient_save_freq=10000
            ),
        ]

    

    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacksList
    )

    def make_env(config):
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
        # env = DiscreteActions3DOF(env)
        env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])
        return env

    env=make_env(config)

    def make_eval_env(training_env=env):
        return RecordVideoFigure(training_env, video_folder=f"videos/{run.id}",
                image_folder=f"images/{run.id}", episode_trigger= lambda x: x%5==0 )

    callbacksList = [
    EvalCallback(
        eval_env,
        eval_freq = 1e3,
        n_eval_episodes = 5,
        render=False,
        deterministic=True,
        ),
    WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
        gradient_save_freq=10000
        ),
    ]     
    
    model.set_env(env)
    model.learn(
        total_timesteps=2*config["total_timesteps"],
        callback=callbacksList
    )

    run.finish()
