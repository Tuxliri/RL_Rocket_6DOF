# In this file the dynamics are simulated using
# different kind of simulators. A 3DOF simulator,
# a linearized 3DOF and a 6DOF simulink simulator
import numpy as np
from scipy.spatial.transform.rotation import Rotation
from scipy.integrate import solve_ivp


class Simulator6DOF():
    def __init__(self, IC : np.ndarray, dt=0.5) -> None:
        super(Simulator6DOF, self).__init__()

        self.timestep = dt
        self.t = 0
        self.state = IC                         # state is in GLOBAL COORDINATES
                                                # state[0] : x axis position
                                                # state[1] : y axis position
                                                # state[2] : z axis position
                                                # state[3] : vx axis velocity
                                                # state[4] : vy axis velocity
                                                # state[5] : vz axis velocity
                                                # state[6] : q0 quaternion component (scalar part)
                                                # state[7] : q1 quaternion component (vector part)
                                                # state[8] : q2 quaternion component (vector part)
                                                # state[9] : q3 quaternion component (vector part)
                                                # state[10]: omega_1 rotational velocity component
                                                # state[11]: omega_2 rotational velocity component
                                                # state[12]: omega_3 rotational velocity component
                                                # state[13] : rocket mass

        self.states = [IC]
        self.actions = [[0,0,0]]                # action[0] : delta_y thrust gimbal angle
                                                # action[1] : delta_z thrust gimbal angle
                                                # action[2] : thrust magnitude

        self.times = [0]


        # Define environment properties
        self.g0 = 9.81

        # Define rocket properties
        self.m = IC[13]                         # rocket initial mass [kg]
        self.length = 40                        # rocket body length [m]
        self.base_radius = 3.66/2               # rocket base radius
        self.J = np.diag([                      # inertia moment [kg*m^2]
                .5*self.m*self.base_radius**2,
                1/12*self.m*(self.length**2+3*self.base_radius**2),
                1/12*self.m*(self.length**2+3*self.base_radius**2),
                ])     
        self.Jinv = np.linalg.inv(self.J)
        self.Isp = 360                          # Specific impulse [s]
        
        # Aerodynamic parameters
        self.Ca_matrix = np.diag(               # Aerodynamic coefficients matrix [-]
            [0.82,0.82,0.82]
            )
        self.S_ref = np.pi*self.base_radius**2  # Reference aerodynamic surface [m**2]
        self.H_r = 7160                         # Scale height factor [m]
        self.rho_0 = 1.225

        # Geometric properties
        self.r_T_B = [-15, 0, 0]
        self.r_cp_B = [5,0,0]

    def step(self, u):
        """
        Method stepping the environment of a timestep dt
        :param      u: control vector with [delta_y,delta_z,thrust] [rad,rad,N]
        :returns:   state: array containing the state,
                    solution.status: flag indicating status os SciPy ivp solver
        """    
        
        def _height_event(t, y):
            return y[0]

        # RK integration
        _height_event.terminal = True

        solution = solve_ivp(
            fun=lambda t, y: self.RHS(t, y, u),
            t_span=[self.t, self.t+self.timestep],
            y0=self.state,
            events=_height_event
        )

        self.state = np.array([var[-1] for var in solution.y])

        self.t = round(self.t+self.timestep,3)
        
        self.times.append(self.t)

        # Normalize the quaternions in the state
        self.state[6:10] = self._normalize_quaternion(self.state[6:10])
        

        # Keep track of all states and actions
        self.states.append(self.state)
        self.actions.append(u)

        return self.state, solution.status

    def RHS(self, t, state, u):
        """ 
        Function computing the derivatives of the state vector
        in inertial coordinates
        """
        # extract dynamics variables
        r_inertial = state[0:3]
        v_inertial = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        mass = state[13]

        # Implement getting it from the height (y)
        rho = self.rho_0*np.exp(-r_inertial[0]/self.H_r)

        
        g_I = [-self.g0,0,0]

        # Translational dynamics
        F_I = self._compute_forces_inertial_rf(q,u,v_inertial,rho)

        dr = v_inertial
        dv = 1/mass*F_I + g_I

        # Rotational dynamics
        OMEGA = self._get_omega_matrix(omega)
        body_torques = self._get_body_torques(u,v_inertial,q,rho)
        
        dq = 0.5*OMEGA.dot(q)
        dom = self.Jinv.dot(body_torques-np.cross(omega,np.dot(self.J,omega)))

        # Mass depletion
        thrust_magnitude = u[2]
        dm = -thrust_magnitude/(self.g0*self.Isp)
        
        return np.concatenate([dr, dv, dq, dom, [dm]])


    def _normalize_quaternion(self,q):
        return q/np.linalg.norm(q)

    def _compute_forces_inertial_rf(self, attitude_quaternion, control_vector, velocity_I,rho):
        
        R_B_to_I = self._rot_mat_body_to_inertial(attitude_quaternion)
     
        T_body_frame = self._get_thrust_body_frame(control_vector)
        A_body_frame = self._get_aero_force_body(velocity_I,attitude_quaternion,rho)
        
        inertial_force_vector = R_B_to_I.dot(T_body_frame+A_body_frame)

        return inertial_force_vector

    def _get_thrust_body_frame(self, control_vector):

        thrust = control_vector[2]

        ROT_MAT = self._rot_mat_thrust_to_body(
            delta_y=control_vector[0],
            delta_z=control_vector[1])
        T_body_frame = ROT_MAT@[thrust,0.,0.]
        return T_body_frame

    def get_thrust_vector_inertial(self):
        u = self.actions[-1]
        T_body_frame = self._get_thrust_body_frame(u)

        current_state = self.states[-1]
        attitude_quaternion = current_state[6:10]
        R_B_to_I = self._rot_mat_body_to_inertial(attitude_quaternion)

        #Get the thrust vector in the inertial reference frame
        return R_B_to_I.dot(T_body_frame)


    def _rot_mat_body_to_inertial(self, attitude_quaternion):
        """
        We follow the convention that the attitude quaternion has components
        q := [cos(xi/2), sin(xi/2)*rot_axis] = [q0,q1,q2,q3] (LEADING SCALAR CONVENTION)
        """
        q0,q1,q2,q3 = attitude_quaternion

        # As the Rotation.from_quat uses the TRAILING SCALAR CONVENTION
        # we need to shift this term as the last
        rotation = Rotation.from_quat([q1,q2,q3,q0])
        return rotation.as_matrix()

    
    def _rot_mat_thrust_to_body(self, delta_y : float, delta_z : float) -> np.ndarray:
        
        cosy = np.cos(delta_y)
        cosz = np.cos(delta_z)
        siny = np.sin(delta_y)
        sinz = np.sin(delta_z)
        return np.array(
            [
                [cosy*cosz,-siny,-cosy*sinz],
                [siny*cosz, cosy,-siny*sinz],
                [sinz, 0, cosz]
            ]
        )

    def _get_aero_force_body(self, velocity, quaternion, rho=0,):
        ROT_MAT_I_TO_B = self._rot_mat_body_to_inertial(quaternion).transpose()
        velocity_body_frame = ROT_MAT_I_TO_B@velocity
        return -.5*rho*np.linalg.norm(velocity)*self.S_ref*self.Ca_matrix @ velocity_body_frame

    def _get_omega_matrix(self, omega):
        wx,wy,wz = omega

        return np.array([
            [0,-wx,-wy,-wz],
            [wx,0,wz,-wy],
            [wy,-wz,0,wx],
            [wz,wy,-wx,0]
        ])


    def _get_body_torques(self, thrust_vector, velocity,attitude_quaternion,rho):
       
        T_body_frame = self._get_thrust_body_frame(thrust_vector)
        A_body_frame = self._get_aero_force_body(velocity, attitude_quaternion,rho)
        
        def cross_product(a,b):
            return np.array([
                a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0]
            ])

        return cross_product(self.r_T_B,T_body_frame) + cross_product(self.r_cp_B, A_body_frame)