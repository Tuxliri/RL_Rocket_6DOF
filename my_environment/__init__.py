from gym.envs.registration import register

from .envs import Rocket6DOF, Rocket6DOF_Fins

# Register the environment
register(
    id='my_environment/Falcon6DOF-v1',
    entry_point='my_environment.envs:Rocket6DOF_Fins'
)
