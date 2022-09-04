"""
Script to test functionality of the 6DOF environment
"""

from my_environment.envs import Rocket6DOF
from gym.wrappers import RecordVideo

# Import the initial conditions from the setup file
from configuration_file import env_config
from gym.wrappers import RecordVideo

# Instantiate the environment
kwargs = env_config
env = Rocket6DOF(**kwargs)

# env=RecordVideo(env,video_folder="video_6DOF")

# [delta_y, delta_z, thrust]
null_action = [0.0, 0.0, -1]
non_null_action = [1.0, 1.0, -0.5]

# Initialize the environment
done = False
obs = env.reset()
env.render(mode="human")

while not done:
    obs, rew, done, info = env.step(null_action)
    env.render(mode="human")

fig = env.get_trajectory_plotly()
env.close()

fig.show()
