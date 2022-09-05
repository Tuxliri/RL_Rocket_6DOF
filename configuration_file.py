from numpy import pi

env_config = {
    "timestep": 0.1,
    "seed": 42,
    "IC": [
        2000, -1600, 300,
        -50, -90, 180,
        0.8660254, 0, 0, -0.5,
        0, 0, 0,
        45e3
        ],
    #"ICRange": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ICRange": [
        50, 10, 10,
        10, 10, 10,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        1e3
        ],
    "reward_coeff": {
        "alfa": -0.01,
        "beta": -1e-7,
        "delta": -5,
        "eta": 0.05,
        "gamma": -10,
        "kappa": 10,
        "xi": 0.004,
    },
    "trajectory_limits": {"attitude_limit": [1.5, 1.5, 2 * pi]},
    "landing_params": {
        "waypoint": 50,
        "landing_radius": 30,
        "maximum_velocity": 10,
        "landing_attitude_limit": [
            10 / 180 * pi,
            10 / 180 * pi,
            2 * pi,
        ],  # [Yaw, Pitch, Roll] in RAD,
        # rotations order zyx
        # VISUALIZATION:
        # https://bit.ly/3CoEdvH
        "omega_lim": [0.2, 0.2, 0.2],
    },
}

TOTAL_TIMESTEP = int(1e6)
MAX_TIME = 150

sb3_config = {
    "env_id": "my_environment/Falcon6DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": TOTAL_TIMESTEP,
    "max_time": MAX_TIME,
    "max_ep_timesteps": int(MAX_TIME / env_config["timestep"]),
    "eval_freq": int(TOTAL_TIMESTEP/20),
}
