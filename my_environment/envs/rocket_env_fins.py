from .rocket_env import Rocket6DOF
from ..utils.simulator_fins import Simulator6DOF_fins
from gym.spaces import Box
import numpy as np

class Rocket6DOF_fins(Rocket6DOF):
    
    def __init__(
            self,
            IC=[500, 100, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 50e3],
            ICRange=[50, 10, 10, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1e3],
            timestep=0.1,
            seed=42,
            reward_shaping_type='acceleration',
            reward_coeff={
                "alfa": -0.01,
                "beta": -1e-8,
                "eta": 2,
                "gamma": -10,
                "delta": -5,
                "kappa": 10,
                "w_r_f" : 1,
                "w_v_f" : 5,
                "max_r_f": 100,
                "max_v_f": 100,
            },
            trajectory_limits={"attitude_limit": [85, 85, 360]},
            landing_params={
                "landing_radius": 30,
                "maximum_velocity": 15,
                "landing_attitude_limit": [10 , 10, 360,],  # [Yaw, Pitch, Roll],
                "omega_lim": [0.2, 0.2, 0.2],
            },
        ) -> None:
        super().__init__(IC, ICRange, timestep, seed, reward_shaping_type, reward_coeff, trajectory_limits, landing_params)

        # Append fins action names
        self.action_names.extend(
            ["beta_fin_1", "beta_fin_2", "beta_fin_3", "beta_fin_4"]
        )

        # Grid fins bounds
        self.max_fins_gimbal = np.deg2rad(20)

        # Redefine action space
        self.action_space = Box(low=-1, high=1, shape=(7,))

        # Reinitialize the action
        self.action = np.zeros(7)

        self.SIM: Simulator6DOF_fins = None

    def _denormalize_action(self, action):
        thruster_action = super()._denormalize_action(action)

        fins_action = np.array(action[3:8]) * self.max_fins_gimbal
        denormalized_action = np.concatenate((thruster_action, fins_action))
        assert denormalized_action.shape == (7,)
        
        return denormalized_action

    def reset(self):
        obs = super().reset()
        self.SIM = Simulator6DOF_fins(self.initial_condition, self.timestep)
        return obs
    
    def step(self, normalized_action):
        obs, rew, done, info = super().step(normalized_action)

        info['euler_angles'] = self.rotation_obj.as_euler("zyx",degrees=True)
        
        return obs, rew, done, info 

if __name__ == "__main__":
    env = Rocket6DOF_fins()
    obs = env.reset()
    action = [0.0, 0.0, -1,
              0.0, 0.0, 0.0, 0.0]
    obs, rew, done, info = env.step(action)
    print("Environment Rocket6DOF ran correctly!")
