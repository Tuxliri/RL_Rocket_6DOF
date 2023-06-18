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
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

def load_config():
    import yaml
    from yaml.loader import SafeLoader

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
        sb3_config = config["sb3_config"]
        env_config = config["env_config"]

    return sb3_config, env_config

sb3_config, env_config = load_config()

MAX_EPISODE_STEPS = int(sb3_config["max_time"] / env_config["timestep"])


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward=-1, max_reward=100):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        import numpy as np
        return np.clip(reward, self.min_reward, self.max_reward)


@ray.remote
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


@ray.remote
def make_eval_env():
    training_env = RemoveMassFromObs(
        Rocket6DOF_fins(**env_config)
    )

    return Monitor(EpisodeAnalyzer(training_env))


class EvalAndWandbCallbacks(EvalCallback, WandbCallback):
    def __init__(self, eval_env):
        super().__init__(
            eval_env,
            eval_freq=int(100e3),
            n_eval_episodes=15,
            render=False,
            deterministic=True,
            verbose=2,
            log_path='evaluation_logs',
            best_model_save_path=f"best_models/{wandb.run.id}"
        )

    def _init_callback(self):
        WandbCallback._init_callback(self)
        EvalCallback._init_callback(self)


def start_training():
    ray.init(num_cpus=4)

    # Check if the system has a display, if not start a virtual framebuffer
    have_display = bool(os.environ.get('DISPLAY', None))
    if not have_display:
        from pyvista.utilities.xvfb import start_xvfb
        start_xvfb()

    run = wandb.init(
        config={**env_config, **sb3_config},
        project='RL_rocket_6DOF' if sb3_config["total_timesteps"] > 1e5 else 'test_runs',
        sync_tensorboard=True,
    )

    config = {
        "env": make_env.remote(),
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [128, 64]
        },
        "batch_size": 512,
        "num_sgd_iter": 10,
        "lr": 0.001,
        "rollout_fragment_length": 16384,
        "train_batch_size": 32768,
        "callbacks": EvalAndWandbCallbacks(make_eval_env.remote()),
        "evaluation_interval": 100,
        "evaluation_num_episodes": 15,
        "evaluation_config": {
            "env_config": env_config,
            "num_workers": 4
        },
        "framework": "torch"
    }

    results = ray.tune.run(
        PPOTrainer,
        config=config,
        local_dir=".",
        stop={"timesteps_total": sb3_config["total_timesteps"]}
    )

    run.finish()


if __name__ == "__main__":
    start_training()
