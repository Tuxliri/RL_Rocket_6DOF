"""
Script to test functionality of the 6DOF environment
"""

from genericpath import isfile
from my_environment.envs import Rocket6DOF
from gym.wrappers import RecordVideo

# Import the initial conditions from the setup file
from gym.wrappers import RecordVideo

import pandas as pd
import os

pd.options.plotting.backend = "plotly"

from stable_baselines3.ppo.ppo import PPO

from main_6DOF import load_config

sb3_config, env_config, = load_config()
# Instantiate the environment
env = Rocket6DOF(**env_config)

# env=RecordVideo(env,video_folder="video_6DOF")

def get_action(action_type = 'null'):
    # [delta_y, delta_z, thrust]
    if action_type == 'null':
        return [0.0, 0.0, -1]

    elif action_type == 'constant':
        return [1.0, 1.0, -0.5]
        
    elif os.path.isfile('model_brisk-donkey-8.zip'):
        model = PPO.load('model_brisk-donkey-8.zip')
        action, __, = model.predict(obs)
        return action

# Initialize the environment
done = False
obs = env.reset()
env.render(mode="human")


landing_attempts = 1
succesful_landings = 0

while landing_attempts <= 10:
    action = get_action()
    obs, rew, done, info = env.step(action)
    env.render(mode="human")
    
    if done:
        landing_attempts += 1
        if info['is_succesful'] is True:
            succesful_landings +=1
        print(info["landing_conditions"])
        env.reset()
        env.render(mode="human")

print(f"The success rate is:{succesful_landings/landing_attempts*100}%")

env.close()
