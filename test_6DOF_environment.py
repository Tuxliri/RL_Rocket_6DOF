"""
Script to test functionality of the 6DOF environment
"""

from my_environment.envs import Rocket6DOF
from gym.wrappers import RecordVideo

# Import the initial conditions from the setup file
from gym.wrappers import RecordVideo

import pandas as pd

pd.options.plotting.backend = "plotly"

from stable_baselines3.ppo.ppo import PPO

import yaml
from yaml.loader import SafeLoader

with open("config.yaml") as f:
    config=yaml.load(f,Loader=SafeLoader)
    sb3_config = config["sb3_config"]
    env_config = config["env_config"]

# Instantiate the environment
env = Rocket6DOF(**env_config)

# env=RecordVideo(env,video_folder="video_6DOF")

# [delta_y, delta_z, thrust]
null_action = [0.0, 0.0, -1]
non_null_action = [1.0, 1.0, -0.5]
model = PPO.load('model_brisk-donkey-8.zip')

# Initialize the environment
done = False
obs = env.reset()
env.render(mode="human")

while not done:
    action, __, = model.predict(obs)
    obs, rew, done, info = env.step(action)
    env.render(mode="human")
    if done:
        fig = env.get_vtarg_plotly()
        env.reset()
        env.render(mode="human")

vtarg_dataframe = env.vtarg_to_dataframe()

fig.show()

env.close()
