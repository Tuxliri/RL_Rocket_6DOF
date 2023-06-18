import logging
from .simulator import Simulator6DOF
import numpy as np
from numpy.typing import ArrayLike

class Simulator6DOF_fins(Simulator6DOF):

    def __init__(self, IC: np.ndarray, dt=0.5) -> None:
        super().__init__(IC, dt)
        self.x_fins = 20  # Fins longitudinal distance from CoM [m]

        # Fins aerodynamic parameters
        self.S_fin = 1.5  # Fin surface [m**2]
        self.C_f_bar = 6  # Fins nominal force coefficient []

        self.r_fins = [  # Fins position vectors
            [self.x_fins, +self.base_radius, 0],
            [self.x_fins, -self.base_radius, 0],
            [self.x_fins, 0, +self.base_radius],
            [self.x_fins, 0, +self.base_radius],
        ]
        self.fins_forces : list = []

    def _compute_forces_inertial_rf(self, attitude_quaternion, control_vector, velocity_I, rho):
        inertial_forces_array =  super()._compute_forces_inertial_rf(attitude_quaternion, control_vector, velocity_I, rho)
        __, total__fins_force = self._get_all_fins_forces(velocity_I,attitude_quaternion,control_vector,rho,)

        return inertial_forces_array + total__fins_force
    
    def _get_body_torques(self, thrust_vector, velocity_inertial, attitude_quaternion, rho):
        body_torque_vector = super()._get_body_torques(thrust_vector, velocity_inertial, attitude_quaternion, rho)
        fins_torque = self._compute_fins_torque(velocity_inertial,rho)
        return body_torque_vector + fins_torque
    
    def _get_all_fins_forces(
        self,
        velocity,
        attitude_quaternion,
        control_vector,
        rho : float = 0.0,
        omega : float = 0.0,  # we assume that omega is small
    ) -> ArrayLike:

        # Extract fins control angles
        control_angles = control_vector[3:7]

        # Compute air velocity
        ROT_MAT_I_TO_B = self._rot_mat_body_to_inertial(attitude_quaternion).transpose()
        velocity_body_frame = np.dot(ROT_MAT_I_TO_B, velocity)

        self.fins_forces = []
        v_air_at_fin = velocity_body_frame  # + np.cross(omega_b, fin_pos_vec)
        q = 0.5 * rho * np.linalg.norm(v_air_at_fin)**2

        for i, _ in enumerate(self.r_fins):
            commanded_angle = control_angles[i]
            angle_of_attack = 0
            C_f = 0

            if i == 0 or i == 1:
                # equivalent to Pitch plane (x-z)
                alfa = np.arctan2(v_air_at_fin[2], v_air_at_fin[0])
                angle_of_attack = commanded_angle - alfa
                F_fin = np.zeros(3)
                F_fin[0] = -np.sin(commanded_angle)
                F_fin[2] = np.cos(commanded_angle)
            elif i == 2 or i == 3:
                # equivalent to Yaw plane (x-y)
                beta = np.arcsin(v_air_at_fin[1] / np.linalg.norm(v_air_at_fin))
                angle_of_attack = -commanded_angle - beta
                F_fin = np.zeros(3)
                F_fin[0] = np.sin(commanded_angle)
                F_fin[1] = np.cos(commanded_angle)

            C_f = self.C_f_bar * np.sin(angle_of_attack)
            F_fin *= q * self.S_fin * C_f
            self.fins_forces.append(F_fin)

        total_force = np.sum(self.fins_forces, axis=0)

        msg = f"""
            [FORCE COMPUTATION]
            Velocity: {velocity}
            Air density: {rho}
            Overall fins force: {total_force}
        """
        logging.debug(msg)

        return self.fins_forces, total_force

    def _compute_fins_torque(
        self,
        velocity,
        rho=0,
    ) -> ArrayLike:

        self.fins_torque = np.zeros(3)
        for fin_pos_vec, fin_force in zip(self.r_fins, self.fins_forces):
            self.fins_torque += self._cross_product(fin_pos_vec, fin_force)

        msg = f"""
            [TORQUE COMPUTATION]
            Velocity: {velocity}
            Air density: {rho}
            Overall fins torque: {self.fins_torque}
        """
        logging.debug(msg)

        return self.fins_torque

if __name__ == "__main__":

    initial_conditions = [100,100,100,0,3,4,1,1,0,0,0,0,0,50e3]
    u = [0,0,1,
         1,1,1,1]

    SIM = Simulator6DOF_fins(initial_conditions)

    SIM.step(u)

    print("The simulator run successfully")