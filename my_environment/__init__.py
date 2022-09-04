from gym.envs.registration import register

# Register the environment
register(
    id='my_environment/Falcon3DOF-v0',
    entry_point='my_environment.envs:Rocket'
)

register(
    id='my_environment/Falcon6DOF-v0',
    entry_point='my_environment.envs:Rocket6DOF'
)
