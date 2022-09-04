__all__ = ["Rocket6DOF"]

# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import numpy as np
import pyvista as pv
from gym import Env, spaces
from my_environment.utils.simulator import Simulator6DOF
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.spatial.transform.rotation import Rotation as R


class Rocket6DOF(Env):

    """Simple environment simulating a 6DOF rocket"""

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        IC=[500, 100, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 50e3],
        ICRange=[50, 10, 10, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1e3],
        timestep=0.1,
        seed=42,
        reward_coeff={
            "alfa": -0.01,
            "beta": -1e-8,
            "eta": 2,
            "gamma": -10,
            "delta": -5,
            "kappa": 10,
            "xi": 0.004,
        },
        trajectory_limits={"attitude_limit": [1.5, 1.5, 2 * np.pi]},
        landing_params={
            "waypoint": 50,
            "landing_radius": 30,
            "maximum_velocity": 10,
            "landing_attitude_limit": [0.2, 0.2, 2 * np.pi],  # [Yaw, Pitch, Roll],
            "omega_lim": [0.2, 0.2, 0.2],
        },
    ) -> None:

        super(Rocket6DOF, self).__init__()

        self.state_names = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "q0",
            "q1",
            "q2",
            "q3",
            "omega1",
            "omega2",
            "omega3",
            "mass",
        ]
        self.action_names = ["gimbal_y", "gimbal_z", "thrust"]

        # Initial conditions mean values and +- range
        self.ICMean = np.float32(IC)
        self.ICRange = np.float32(ICRange)  # +- range
        self.timestep = timestep
        self.metadata["render_fps"] = 1 / timestep
        self.reward_coefficients = reward_coeff

        # Initial condition space
        self.init_space = spaces.Box(
            low=self.ICMean - self.ICRange / 2,
            high=self.ICMean + self.ICRange / 2,
        )

        self.seed(seed)

        # Actuators bounds
        self.max_gimbal = np.deg2rad(20)  # [rad]
        self.max_thrust = 981e3  # [N]

        # State normalizer and bounds
        t_free_fall = (
            -self.ICMean[3] + np.sqrt(self.ICMean[3] ** 2 + 2 * 9.81 * self.ICMean[0])
        ) / 9.81
        inertia = 6.04e6
        lever_arm = 15.0

        omega_max = (
            self.max_thrust
            * np.sin(self.max_gimbal)
            * lever_arm
            / (inertia)
            * t_free_fall
            / 5.0
        )
        v_max = 2 * 9.81 * t_free_fall

        self.state_normalizer = np.maximum(
            np.array(
                [
                    1.2 * abs(self.ICMean[0]),
                    1.5 * abs(self.ICMean[1]),
                    1.5 * abs(self.ICMean[2]),
                    v_max,
                    v_max,
                    v_max,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    omega_max,
                    omega_max,
                    omega_max,
                    self.ICMean[13] + self.ICRange[13],
                ]
            ),
            1,
        )

        # Set environment bounds
        position_bounds_high = 0.9 * np.maximum(self.state_normalizer[0:3], 100)
        position_bounds_low = -0.9 * np.maximum(self.state_normalizer[1:3], 100)
        position_bounds_low = np.insert(position_bounds_low, 0, -30)
        self.position_bounds_space = spaces.Box(
            low=position_bounds_low, high=position_bounds_high, dtype=np.float32
        )

        # Define observation space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,))

        # TODO: remove this check as when using different observation
        # than the state of the system it would result in a raised error
        assert (
            self.observation_space.shape == self.init_space.shape
        ), f"The observation space has shape {self.observation_space.shape}\
                but the init_space has shape {self.init_space.shape}"

        # Two valued vector in the range -1,+1, for the
        # gimbal angle and the thrust command. It will then be
        # rescaled to the appropriate ranges in the dynamics
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))

        # Environment state variable and simulator object
        self.state = None
        self.infos = []
        self.SIM: Simulator6DOF = None
        self.prev_rotation_obj: R = None
        self.rotation_obj: R = None
        self.action = np.array([0.0, 0.0, 0.0])
        self.vtarg_history = []

        # Trajectory constratints
        self.attitude_traj_limit = trajectory_limits["attitude_limit"]

        # Landing parameters
        self.target_r = landing_params["landing_radius"]
        self.maximum_v = landing_params["maximum_velocity"]
        self.landing_target = [0, 0, 0]
        self.landing_attitude_limit = landing_params["landing_attitude_limit"]

        # self.q_lim = raise NotImplementedError
        self.omega_lim = np.array([0.2, 0.2, 0.2])

        self.waypoint = landing_params["waypoint"]

        # Renderer variables (pyvista)
        self.rocket_body_mesh = None
        self.landing_pad_mesh = None
        self.plotter = None

    def reset(self):
        """Function defining the reset method of gym
        It returns an initial observation drawn randomly
        from the uniform distribution of the ICs
        """

        self.vtarg_history = []
        self.initial_condition = self.init_space.sample()
        self.initial_condition[6:10]=self.initial_condition[6:10]/np.linalg.norm(self.initial_condition[6:10])
        self.state = self.initial_condition

        # Create a rotation object representing the attitude of the system
        # assert (
        #     np.linalg.norm(self.state[6:10]) == 1.0
        # ), f"The quaternion doesn't have unit norm! It's components are {self.state[6:10]}"
        

        self.rotation_obj = R.from_quat(self._scipy_quat_convention(self.state[6:10]))
        if self.prev_rotation_obj is None:
            self.prev_rotation_obj = self.rotation_obj
        # instantiate the simulator object
        self.SIM = Simulator6DOF(self.initial_condition, self.timestep)

        return self._get_obs()

    def step(self, normalized_action):

        self.action = self._denormalize_action(normalized_action)

        self.state, isterminal, __ = self.SIM.step(self.action)
        state = self.state.astype(np.float32)

        # Create a rotation object representing the attitude of the system
        self.prev_rotation_obj = self.rotation_obj
        self.rotation_obj = R.from_quat(self._scipy_quat_convention(state[6:10]))

        # Done if the rocket is at ground or outside bounds
        done = bool(isterminal) or self._check_bounds_violation(state)

        reward, rewards_dict = self._compute_reward(state, self.action)

        info = {
            "rewards_dict": rewards_dict,
            "is_done": done,
            "state_history": self.SIM.states,
            "action_history": self.SIM.actions,
            "timesteps": self.SIM.times,
        }

        info["bounds_violation"] = self._check_bounds_violation(state)

        if info["bounds_violation"]:
            reward += -50

        return self._get_obs(), reward, done, info

    def render(self, mode: str = "rgb_array"):

        assert (
            mode is not None
        )  # The renderer will not call this function with no-rendering.

        if self.plotter is None:
            # In this section the plotter is setup
            args = {}
            if mode == "rgb_array":
                args["off_screen"] = True

            # Creating scene and loading the mesh
            self.plotter = pv.Plotter(args)
            self._add_meshes_to_plotter()

            self.plotter.show(
                auto_close=False,
                interactive=False,
                # interactive_update=True,
            )

            # Set desired camera position
            cpos = [(783.93, -265.23, -1118.80),
                    (262.5, 35.91, 35.91),
                    (0.9150, 0.1524, 0.3734)]
            self.plotter.camera_position = cpos
                

        # Move the rocket towards its new location
        previous_loc = self.rocket_body_mesh.center
        current_loc = self.state[0:3]

        self.rocket_body_mesh.translate(current_loc - previous_loc, inplace=True)

        # Rotate the rocket to the new attitude
        step_rot_vector = self._get_step_rot_vec()
        norm_step_rot = np.linalg.norm(step_rot_vector)  # This gives the rotation angle in [rad]

        if norm_step_rot > 0:
            self.rocket_body_mesh.rotate_vector(
                vector=step_rot_vector / norm_step_rot,
                angle=np.rad2deg(norm_step_rot),
                inplace=True,
                point=current_loc,
            )

        # Redraw the thrust vector
        self.plotter.remove_actor('thrust_vector')
        
        thrust_vector, thrust_vec_location, = self.SIM.get_thrust_vector_inertial()
        arrow_kwargs = {'name': 'thrust_vector'}
        
        # self.plotter.add_arrows(
        #     cent=thrust_vec_location,
        #     direction=thrust_vector,
        #     **arrow_kwargs
        #     )

        self.plotter.update()

        if mode == "rgb_array":
            return self.plotter.image

    def _add_meshes_to_plotter(self):
        current_loc = self.state[0:3]

        self.rocket_body_mesh = pv.Cylinder(
            center=current_loc,
            direction=self.rotation_obj.apply([1, 0, 0]),
            radius=3.66 / 2,
            height=50,
        )

        self.landing_pad_mesh = pv.Circle(radius=self.target_r)
        self.landing_pad_mesh.rotate_y(angle=90)
        # current_vel=self.state[3:6]

        # self.velocity_mesh = pv.Arrow(
        #         start=current_loc,
        #         direction=current_vel,
        #         # scale='auto'
        #         )
        thrust_vector, thrust_vec_location, = self.SIM.get_thrust_vector_inertial()
        arrow_kwargs = {'name': 'thrust_vector'}

        self.plotter.add_arrows(
            cent=thrust_vec_location,
            direction=thrust_vector,
            **arrow_kwargs
            )

        self.plotter.add_mesh(self.rocket_body_mesh,show_scalar_bar=False,color="orange")
        self.plotter.add_mesh(self.landing_pad_mesh,color="red")

        self.plotter.show_axes_all()
        self.plotter.show_grid()

    def close(self) -> None:
        super().close()

        pv.close_all()
        return None

    def _compute_reward(self, state, action):
        reward = 0

        r = state[0:3]
        v = state[3:6]

        v_targ, __ = self._compute_vtarg(r, v)

        thrust = action[2]

        # Coefficients
        coeff = self.reward_coefficients

        # Compute each reward term
        rewards_dict = {
            "velocity_tracking": coeff["alfa"] * np.linalg.norm(v - v_targ),
            "thrust_penalty": coeff["beta"] * thrust,
            "eta": coeff["eta"],
            "attitude_constraint": self._check_attitude_limits(),
            # "attitude_hint" : coeff["delta"]*np.maximum(0,abs(zeta)-zeta_mgn),
            "rew_goal": self._reward_goal(state),
        }

        reward = sum(rewards_dict.values())

        return reward, rewards_dict

    def _check_attitude_limits(self):
        gamma = self.reward_coefficients["gamma"]
        attitude_euler_angles = self.rotation_obj.as_euler("zyx")
        return gamma * np.any(np.abs(attitude_euler_angles) > self.attitude_traj_limit)

    def _reward_goal(self, obs):
        k = self.reward_coefficients["kappa"]
        return k * self._check_landing(obs)

    def get_trajectory_plotly(self):
        trajectory_dataframe = self.states_to_dataframe()
        return self._trajectory_plot_from_df(trajectory_dataframe)

    def get_attitude_trajectory(self):
        trajectory_dataframe = self.states_to_dataframe()
        return self._attitude_traj_from_df(trajectory_dataframe)

    def _attitude_traj_from_df(self, trajectory_df: DataFrame):
        import plotly.express as px

        fig = px.line(trajectory_df[["q0", "q1", "q2", "q3"]])

        return fig

    def _trajectory_plot_from_df(self, trajectory_df: DataFrame):
        import plotly.express as px

        fig = px.line_3d(trajectory_df[["x", "y", "z"]], x="x", y="y", z="z")

        # Set camera location
        camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.5 * 1.25, y=1.25, z=0 * 1.25),
        )

        # fig.update_layout(scene_camera=camera)
        # x_f, y_f, z_f = self.landing_target

        # Add landing pad location and velocity vectors
        z = np.linspace(-self.target_r,self.target_r,100)
        y = np.linspace(-self.target_r,self.target_r,100)
        
        zv,yv = np.meshgrid(z,y)
        xv=1.*(zv**2+yv**2<self.target_r**2)

        fig.add_surface(x=xv,y=yv,z=zv,surfacecolor=xv,showscale=False)
        
        # Add velocity vector
        fig.add_cone(
            x=trajectory_df["x"],
            y=trajectory_df["y"],
            z=trajectory_df["z"],
            u=trajectory_df["vx"],
            v=trajectory_df["vy"],
            w=trajectory_df["vz"],
            sizeref=3,
        )

        fig.update_layout(scene_aspectmode='data')
        return fig

    def _vtarg_plot_figure(self, trajectory_df: DataFrame):
        import plotly.express as px
        
        # Create vtarg dataframe
        vtarg_df = self.vtarg_to_dataframe()

        fig = px.line_3d(trajectory_df[["x", "y", "z"]], x="x", y="y", z="z")

        # Set camera location
        camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.5 * 1.25, y=1.25, z=0 * 1.25),
        )

        fig.update_layout(scene_camera=camera)
        x_f, y_f, z_f = self.landing_target

        # Add landing pad location and velocity vector
        fig.add_scatter3d(x=[x_f], y=[y_f], z=[z_f])
        fig.add_cone(
            x=trajectory_df["x"],
            y=trajectory_df["y"],
            z=trajectory_df["z"],
            u=vtarg_df["v_x"], # TODO: CHANGE TO vtarg
            v=vtarg_df["v_y"],
            w=vtarg_df["v_z"],
            sizeref=3,
        )

        fig.update_layout(scene_aspectmode='data')

        return fig

    def get_vtarg_trajectory(self):
        trajectory_dataframe = self.states_to_dataframe()
        return self._vtarg_plot_figure(trajectory_dataframe)

    def _plotly_fig2array(self, plotly_fig):
        # convert Plotly fig to  an array
        import io

        from PIL import Image

        fig_bytes = plotly_fig.to_image(format="png", width=800, height=800)
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf)
        return np.asarray(img)

    def _normalize_obs(self, obs):
        return (obs / self.state_normalizer).astype("float32")

    def _denormalize_obs(self, obs):
        return obs * self.state_normalizer

    def _denormalize_action(self, action: ArrayLike):
        """Denormalize the action as we've bounded it
        between [-1,+1]. The first element of the
        array action is the gimbal angle  while the
        second is the throttle"""

        gimbal_y = action[0] * self.max_gimbal
        gimbal_z = action[1] * self.max_gimbal

        thrust = (action[2] + 1) / 2.0 * self.max_thrust

        # TODO : Add lower bound on thrust with self.minThrust
        return np.float32([gimbal_y, gimbal_z, thrust])

    def _get_obs(self):
        return self._normalize_obs(self.state)

    def _compute_vtarg(self, r, v):
        tau_1 = 20
        tau_2 = 100
        initial_conditions = self.SIM.states[0]

        v_0 = np.linalg.norm(initial_conditions[3:6])

        rx = r[0]

        if rx > self.waypoint:
            r_hat = r - [self.waypoint, 0, 0]
            v_hat = v - [-2, 0, 0]
            tau = tau_1

        else:
            r_hat = [rx + 1, 0, 0]
            v_hat = v - [-1, 0, 0]
            tau = tau_2

        t_go = np.linalg.norm(r_hat) / np.linalg.norm(v_hat)
        v_targ = (
            -v_0
            * (np.array(r_hat) / max(1e-3, np.linalg.norm(r_hat)))
            * (1 - np.exp(-t_go / tau))
        )

        self.vtarg_history.append(v_targ)

        return v_targ, t_go

    def states_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.SIM.states, columns=self.state_names)

    def actions_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.SIM.actions, columns=self.action_names)

    def vtarg_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.vtarg_history, columns=["v_x", "v_y", "v_z"])

    def used_mass(self):
        initial_mass = self.SIM.states[0][-1]
        final_mass = self.SIM.states[-1][-1]
        return initial_mass - final_mass

    def _check_bounds_violation(self, state: ArrayLike):
        r = np.float32(state[0:3])
        return not bool(self.position_bounds_space.contains(r))

    def _check_landing(self, state):

        r = np.linalg.norm(state[0:3])
        v = np.linalg.norm(state[3:6])
        q = state[6:10]
        omega = state[10:13]

        attitude_euler_angles = self.rotation_obj.as_euler("zyx")

        assert q.shape == (4,), omega.shape == (3,)

        landing_conditions = {
            "zero_height": state[0] <= 1e-3,
            "velocity_limit": v < self.maximum_v,
            "landing_radius": r < self.target_r,
            "attitude_limit": np.any(
                abs(attitude_euler_angles) < self.landing_attitude_limit
            ),
            "omega_limit": np.any(abs(omega) < self.omega_lim),
        }

        return all(landing_conditions.values())

    def seed(self, seed: int = 42):
        self.init_space.seed(seed)
        return super().seed(seed)

    def _get_normalizer(self):
        return self.state_normalizer

    def _rotate_x_to_z(self, vector: ArrayLike):
        ROT_MAT = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

        return ROT_MAT @ vector

    def _scipy_quat_convention(self, leading_scalar_quaternion):
        # return TRAILING SCALAR CONVENTION
        return np.roll(leading_scalar_quaternion, -1)

    def _get_step_rot_vec(self) -> np.ndarray:
        """
        Compute the incremental rotation vector
        by which to rotate the body at each time step
        """
        step_rotation = self.rotation_obj * self.prev_rotation_obj.inv()
        return step_rotation.as_rotvec()

    def get_keys_to_action(self):
        import pygame

        mapping = {
            (
                # pygame.K_RIGHT,
                pygame.K_UP,
            ): [0, 0, +1.0],
            (pygame.K_DOWN,): [0, 0, -1.0],
        }
        return mapping
