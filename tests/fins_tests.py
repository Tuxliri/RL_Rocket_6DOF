from ..my_environment.utils.simulator import Simulator6DOF
import numpy as np
import logging
from my_environment.envs import Rocket6DOF_Fins

np.set_printoptions(precision=3, suppress=True)

def drop_rocket_straight(
        constant_control_action,
        simulation_time = 60
        ):
    
    """
    :param constant_control_action 7-element array
    :param simulation_time real number specifying the simulation time
    """
    initial_conditions = [100,100,100,0,3,4,1,1,0,0,0,0,0,50e3]
    timestep = 0.1
    rocket_sim = Simulator6DOF(initial_conditions,dt=timestep)

    t = 0
    num_steps = int(simulation_time/timestep)
    state = initial_conditions

    for __ in range(num_steps):
        state = rocket_sim.step(constant_control_action)
        t += timestep
    
    return state[0:3]


def validate_fins_control_direction(test_command = "pitch"):
    """
    :param constant_control_action 7-element array
    :param simulation_time real number specifying the simulation time
    """
    fins_angle = np.deg2rad(1)

    gimbal_y, gimbal_z, thrust = 0,0,0

    if test_command == "pitch": 
        """
        (x-z) test plane. This control command should generate
        a positive force in the z-axis direction (in body axis)
        """
        beta_1, beta_2, beta_3, beta_4 = fins_angle, fins_angle, 0, 0

    constant_control_action = [
        gimbal_y, gimbal_z, thrust,
        beta_1, beta_2, beta_3, beta_4,
        ]

    simulation_time = 60


    initial_conditions = [
        100,0,0,
        0,0,0,
        1,0,0,0,
        0,0,0,
        50e3
        ]

    timestep = 0.1
    rocket_sim = Simulator6DOF(initial_conditions,dt=timestep)

    t = 0
    num_steps = int(simulation_time/timestep)
    state_list = [initial_conditions]

    for __ in range(num_steps):
        state = rocket_sim.step(constant_control_action)
        t += timestep
        state_list.append(state)

    print(state[0:3])
    return state[0:3]

    if final_y_position > 0:
        return True

def validate_fins_environment(test_command = "pitch"):
    """
    :param constant_control_action 7-element array
    :param simulation_time real number specifying the simulation time
    """
    fins_angle = np.deg2rad(20)

    gimbal_y, gimbal_z, thrust = 0,0,0

    if test_command == "pitch": 
        """
        (x-z) test plane. This control command should generate
        a positive torque in the z-axis direction (in body axis)
        """
        beta_1, beta_2, beta_3, beta_4 = fins_angle, fins_angle, 0, 0

    elif test_command == "yaw":
        """
        (x-y) test plane. This control command should generate
        a positive torque in the y-axis direction (in body axis)
        """
        beta_1, beta_2, beta_3, beta_4 = 0, 0, fins_angle, fins_angle

    constant_control_action = [
        gimbal_y, gimbal_z, thrust,
        beta_1, beta_2, beta_3, beta_4,
        ]

    initial_conditions = [
        3000,0,0,
        -100,0,0,
        1,0,0,0,
        0,0,0,
        50e3
        ]

    timestep = 0.1

    rocket_env = Rocket6DOF_Fins(initial_conditions,timestep=timestep)
    
    obs = rocket_env.reset()
    done = False
    
    normalized_action = constant_control_action
    normalized_action[3:7] = constant_control_action[3:7]/rocket_env.max_fins_gimbal

    while not done:
        
        obs, rew, done, info = rocket_env.step(normalized_action)
        
        logging.info(f"Euler angles [deg]: {info['euler_angles']}")
        #rocket_env.render(mode="human")
    

    
if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    validate_fins_environment()