"""
Script to test the reward shaping
"""
from my_environment.envs import Rocket6DOF
from gym.utils.play import play, PlayPlot

# Import the initial conditions from the setup file
from configuration_file import env_config

env_config["IC"] = [500, 100, 100, -50, 0, 0, 1, 0, 0, 0, 0, 0, 0, 45e3]
env_config["ICRange"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# Instantiate the environment
kwargs = env_config
env = Rocket6DOF(**kwargs)

# Define a callback to plot the reward
def callback(obs_t, obs_tp1, action, rew, done, info):
        return [rew,]

plotter = PlayPlot(callback, 30 * 20, ["reward"])


play(env,callback=plotter.callback)
