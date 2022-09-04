# In this file the dynamics are simulated using
# different kind of simulators. A 3DOF simulator,
# a linearized 3DOF and a 6DOF simulink simulator
import numpy as np
from scipy.spatial.transform.rotation import Rotation
from scipy.integrate import solve_ivp
from math import fmod

from matplotlib import pyplot as plt


class Simulator3DOF():
    def __init__(self, IC, dt=0.5, dynamics='std3DOF', mass = 50e3) -> None:
        super(Simulator3DOF, self).__init__()

        self.dynamics = dynamics
        self.timestep = dt
        self.t = 0
        self.state = IC                     # state is in GLOBAL COORDINATES
                                            # state[0] : x axis position
                                            # state[1] : y axis position
                                            # state[2] : attitude angle
                                            # state[3] : x axis velocity
                                            # state[4] : y axis velocity
                                            # state[5] : angular velocity
                                            # state[6] : rocket mass

        self.states = [IC]
        self.actions = [[0,0]]
        self.derivatives = []
        self.times = [0]


        # Define height treshold
        # Define environment properties
        self.g0 = 9.81

        # Define rocket properties
        self.m = mass                       # rocket initial mass [kg]
        self.Cdalfa = 2                     # drag coefficient [-]
        self.Cnalfa = 1                     # normal force coefficient [-]
        self.I = 6.04e6                     # inertia moment [kg*m^2]
        self.Isp = 360                      # Specific impulse [s]
        self.dryMass = 25.6e3               # dry mass of the stage [kg]

        # Geometric properties NB: REDEFINE THEM FROM THE TIP OF THE BOOSTER!! (or change torque equation in RHS)
        self.x_CG = 10                      # Center of gravity [m]
        self.x_CP = 20                      # Center of pressure [m]
        self.Sref = 10.5                      # Reference surface [m^2]
        self.x_PVP = 0                      # Thrust gimbal point [m]
        self.x_T = 40

        return None

    def step(self, u):

        if self.dynamics == 'std3DOF':
            def _height_event(t, y):
                return y[1]

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

            self.state[2] = self._wrapTo2Pi(self.state[2])

        else:
            raise NotImplementedError()

        # Keep track of all states
        self.states.append(self.state)
        self.actions.append(u)

        return self.state, solution.status, self.t

    def RHS(self, t, state, u):
        """ 
        Function computing the derivatives of the state vector
        in inertial coordinates
        """
        # extract dynamics variables
        x, y, phi, vx, vz, om, mass = state
        # Get control variables
        delta = u[0]
        T = u[1]

        # Implement getting it from the height (y)
        rho = 1.225  # *exp(-y/H) scaling due to height

        alfa = 0
        #alfa = self._computeAoA(y)

        # Compute aerodynamic coefficients
        Cn = self.Cnalfa*alfa
        Cd = self.Cdalfa*alfa   # ADD Cd0

        Cd = 0.3

        # Compute aero forces
        v2 = vx**2 + vz**2
        Q = 0.5*rho*v2

        A = Cd*Q*self.Sref

        N = Cn*Q*self.Sref

        g = self.g0

        # Compute state derivatives
        ax = (T*np.cos(delta+phi) - N*np.sin(phi) - A*np.cos(phi))/mass
        ay = (T*np.sin(delta+phi) + N*np.cos(phi) - A*np.cos(phi))/mass - g
        dom = (N*(self.x_CG - self.x_CP) - T*np.sin(delta)
               * (self.x_T - self.x_CG))/self.I
        dm=-T/(self.Isp*self.g0)

        dstate = np.array([vx, vz, om, ax, ay, dom, dm])

        return dstate

    def _computeAoA(self, state):  # CHECK
        if self.dynamics == 'std3DOF':
            phi = state[2]
            vx = state[3]
            vy = state[4]

            gamma = np.arctan2(vy, vx)

            if not(vx == 0 and vy == 0):
                alfa = phi - gamma
            else:
                alfa = 0

        else:
            raise NotImplementedError

        return self._normalize(alfa)

    def _wrapTo2Pi(self, angle):
        """
        Wrap the angle between 0 and 2 * pi.

        Args:
            angle (float): angle to wrap.

        Returns:
            The wrapped angle.

        """
        pi_2 = 2. * np.pi

        return fmod(fmod(angle, pi_2) + pi_2, pi_2)

    def _wrapToPi(self, angle):
        """
        Wrap the angle between 0 and pi.

        Args:
            angle (float): angle to wrap.

        Returns:
            The wrapped angle.

        """

        return fmod(fmod(angle, np.pi) + np.pi, np.pi)

class Simulator6DOF():
    def __init__(self, IC : np.ndarray, dt=0.5) -> None:
        super(Simulator6DOF, self).__init__()

        self.timestep = dt
        self.t = 0
        self.state = IC                     # state is in GLOBAL COORDINATES
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
        self.actions = [[0,0,0]]            # action[0] : delta_y thrust gimbal angle
                                            # action[1] : delta_z thrust gimbal angle
                                            # action[2] : thrust magnitude

        self.times = [0]


        # Define environment properties
        self.g0 = 9.81

        # Define rocket properties
        self.m = IC[13]                     # rocket initial mass [kg]
        self.Cdalfa = 2                     # drag coefficient [-]
        self.Cnalfa = 1                     # normal force coefficient [-]
        self.J = np.diag(                   # inertia moment [kg*m^2]
            [75350.25, 6037675.13, 6037675.13]
            )         
        self.Jinv = np.linalg.inv(self.J)
        self.Isp = 360                      # Specific impulse [s]

        # Geometric properties
        self.r_T_B = [-15, 0, 0]
        self.r_cp_B = [5,0,0]
        return None

    def step(self, u):

        
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
        

        # Keep track of all states
        self.states.append(self.state)
        self.actions.append(u)

        return self.state, solution.status, self.t

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
        rho = 1.225  # *exp(-y/H) scaling due to height
        
        g = self.g0  # Simplification of: g=self.g0*(earth_radius/(earth_radius+x))**2
        g_I = [-self.g0,0,0]

        # Translational dynamics
        F_I = self._compute_forces_inertial_rf(q,u,v_inertial)

        dr = v_inertial
        dv = 1/mass*F_I + g_I

        # Rotational dynamics
        OMEGA = self._get_omega_matrix(omega)
        body_torques = self._get_body_torques(u,v_inertial)

        dq = 0.5*OMEGA.dot(q)
        dom = self.Jinv.dot(body_torques-np.cross(omega,np.dot(self.J,omega)))

        # Mass depletion
        thrust_magnitude = u[2]
        dm = -thrust_magnitude/(self.g0*self.Isp)
        
        return np.concatenate([dr, dv, dq, dom, [dm]])


    def _normalize_quaternion(self,q):
        return q/np.linalg.norm(q)

    def _compute_forces_inertial_rf(self, attitude_quaternion, control_vector, velocity_I):
        
        R_B_to_I = self._rot_mat_body_to_inertial(attitude_quaternion)
     
        T_body_frame = self._get_thrust_body_frame(control_vector)
        A_body_frame = self._get_aero_force_body(velocity_I)
   
        inertial_force_vector = R_B_to_I.dot(T_body_frame+A_body_frame)

        return inertial_force_vector

    def _get_thrust_body_frame(self, control_vector):
        delta_y = control_vector[0]
        delta_z = control_vector[1]
        thrust = control_vector[2]

        ROT_MAT = self._rot_mat_thrust_to_body(delta_y,delta_z)
        T_body_frame = ROT_MAT@[thrust,0.,0.]
        return T_body_frame

    def get_thrust_vector_inertial(self):
        u = self.actions[-1]
        T_body_frame = self._get_thrust_body_frame(u)
        T_body_frame = -T_body_frame/np.linalg.norm(T_body_frame)*30

        current_state = self.states[-1]
        attitude_quaternion = current_state[6:10]
        R_B_to_I = self._rot_mat_body_to_inertial(attitude_quaternion)

        #Get the thrust vector in the inertial reference frame
        T_inertial_frame = R_B_to_I.dot(T_body_frame)

        # Get the hinge point of the thrust vector
        r_thrust_inertial_frame = R_B_to_I.dot(2*np.array(self.r_T_B))+current_state[0:3]

        return T_inertial_frame, r_thrust_inertial_frame

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
        return np.array(
            [
                [np.cos(delta_y)*np.cos(delta_z),-np.sin(delta_y),-np.cos(delta_y)*np.sin(delta_z)],
                [np.sin(delta_y)*np.cos(delta_z), np.cos(delta_y),-np.sin(delta_y)*np.sin(delta_z)],
                [np.sin(delta_z), 0, np.cos(delta_z)]
            ]
        )

    def _get_aero_force_body(self, velocity):
        return np.array([0,0,0]) 

    def _get_omega_matrix(self, omega):
        wx,wy,wz = omega

        return np.array([
            [0,-wx,-wy,-wz],
            [wx,0,wz,-wy],
            [wy,-wz,0,wx],
            [wz,wy,-wx,0]
        ])


    def _get_body_torques(self, thrust_vector, velocity):
       
        T_body_frame = self._get_thrust_body_frame(thrust_vector)
        A_body_frame = self._get_aero_force_body(velocity)

        return np.cross(self.r_T_B,T_body_frame) + np.cross(self.r_cp_B, A_body_frame)