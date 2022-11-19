"""
Script to test the reward shaping
"""
from my_environment.envs import Rocket6DOF
from gym.utils.play import play, PlayPlot

# Import the initial conditions from the setup file
import yaml
from yaml.loader import SafeLoader

with open("config.yaml") as f:
    config=yaml.load(f,Loader=SafeLoader)
    sb3_config = config["sb3_config"]
    env_config = config["env_config"]
    
# Instantiate the environment
kwargs = env_config
env = Rocket6DOF(**kwargs)

# Define a callback to plot the reward
def callback(obs_t, obs_tp1, action, rew, done, info):
        return [rew,]

plotter = PlayPlot(callback, 30 * 20, ["reward"])


play(env,callback=plotter.callback,fps=1/env_config["timestep"])
