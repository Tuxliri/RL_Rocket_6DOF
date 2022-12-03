"""
Script to test functionality of the 6DOF environment
"""

from my_environment.envs import Rocket6DOF

# Import the initial conditions from the setup file

import pandas as pd
import os

pd.options.plotting.backend = "plotly"

from stable_baselines3.ppo.ppo import PPO

from main_6DOF import load_config
from wrappers import RemoveMassFromObs

sb3_config, env_config, = load_config()
# Instantiate the environment
env = RemoveMassFromObs(Rocket6DOF(**env_config))

# env=RecordVideo(env,video_folder="video_6DOF")

def get_action(action_type = 'null'):
    # [delta_y, delta_z, thrust]
    if action_type == 'null':
        return [0.0, 0.0, -1]

    elif action_type == 'constant':
        return [1.0, 1.0, -0.5]
        
    elif action_type == 'model':
        assert os.path.isfile('current_model.zip'), 'Model file doesn\'t exist'
        model = PPO.load('current_model.zip')
        action, __, = model.predict(obs)
        return action

# Initialize the environment
done = False
obs = env.reset()
env.render(mode="human")


landing_attempts = 1
succesful_landings = 0

while landing_attempts <= 10:
    action = get_action('model')
    obs, rew, done, info = env.step(action)
    env.render(mode="human")
    
    if done:
    #     landing_attempts += 1
    #     if info['is_succesful'] is True:
            # succesful_landings +=1
        # print(info["landing_conditions"])
        env.reset()
        env.render(mode="human")

# print(f"The success rate is:{succesful_landings/landing_attempts*100}%")

env.close()
