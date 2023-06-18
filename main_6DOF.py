import os

import gym
import wandb

from gym.wrappers import TimeLimit

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from my_environment.wrappers import *
from my_environment.envs import Rocket6DOF_fins
from wandb.integration.sb3 import WandbCallback

def load_config():
    import yaml
    from yaml.loader import SafeLoader

    with open("config.yaml") as f:
        config=yaml.load(f,Loader=SafeLoader)
        sb3_config = config["sb3_config"]
        env_config = config["env_config"]

    return sb3_config, env_config

sb3_config, env_config, = load_config()

MAX_EPISODE_STEPS = int(sb3_config["max_time"]/env_config["timestep"])

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward=-1, max_reward=100):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        import numpy as np
        return np.clip(reward, self.min_reward, self.max_reward)

def make_env():
    env = RemoveMassFromObs(
        Rocket6DOF_fins(**env_config)
        )
    env = TimeLimit(
        env,
        max_episode_steps=MAX_EPISODE_STEPS
        )
    env = Monitor(env)    
    
    return env


def make_eval_env():
    training_env = RemoveMassFromObs(
        Rocket6DOF_fins(**env_config)
        )

    return Monitor(EpisodeAnalyzer(training_env))
        
def start_training():

    # Check if the system has a display, if not start a virtual framebuffer
    have_display = bool(os.environ.get('DISPLAY', None))
    if not have_display:
        from pyvista.utilities.xvfb import start_xvfb
        start_xvfb()

    run = wandb.init(
        config={**env_config, **sb3_config},
        project='RL_rocket_6DOF' if sb3_config["total_timesteps"]>1e5 else 'test_runs',
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        #monitor_gym=True,  # auto-upload the videos of agents playing the game
    )   

    env = make_env()

    model = PPO(
        sb3_config["policy_type"],
        env,
        tensorboard_log=f"runs/{run.id}",
        verbose=2,
        seed=env_config["seed"],
        policy_kwargs={'net_arch': [128, 64]},
        batch_size=512,
        n_steps=16384,
        )
    
    eval_env =  make_eval_env()

    callbacksList = [
        EvalCallback(
            eval_env,
            eval_freq = int(100e3),
            n_eval_episodes = 15,
            render=False,
            deterministic=True,
            verbose=2,
            log_path='evaluation_logs',
            best_model_save_path=f"best_models/{run.id}"
            ),
        WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
            ),
        ]

    # Train the model
    model.learn(
        total_timesteps=sb3_config["total_timesteps"],
        callback=callbacksList
    )
    
    run.finish()

    return None

if __name__=="__main__":
    start_training()