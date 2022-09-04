"""
Script to test functionality of the 6DOF environment
"""
from my_environment.envs import Rocket6DOF
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import RecordVideo
from stable_baselines3.common.env_checker import check_env
    
# Import the initial conditions from the setup file
from configuration_file import env_config

kwargs = env_config

# Instantiate the environment
env = Rocket6DOF(**kwargs)
# Check for the environment compatibility with gym and sb3
check_env(env, skip_render_check=False)
env.close()

del env
env = Rocket6DOF(**kwargs)

# Test usage with stable_baselines_3 model
model = PPO('MlpPolicy', env, verbose=1)

# Use a separate environement for evaluation
eval_env = RecordVideo(env = Rocket6DOF(**kwargs),video_folder='6DOF_videos',video_length=500)

import time

start_time = time.time()

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5,render=True)
finish_time = time.time()

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print(f"time to record the episodes: {finish_time-start_time}")


