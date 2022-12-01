from asyncore import write
from cmath import acos
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3 import PPO
from main_6DOF import make_env, make_eval_env
import csv

from main_6DOF import load_config

(
    sb3_config,
    env_config,
) = load_config()

# open the file in the write mode
f = open("results_montecarlo.csv", "w")
writer = csv.writer(f)

header = [
    "final_position_error",
    "final_velocity_error",
    "attitude_error",
    "angular_velocity_error",
    "used mass",
]

# write the header
writer.writerow(header)


def test_callback(locals, globals):
    if locals["dones"][0]:
        terminal_state = locals["info"]["state_history"][-1]
        position_error = np.linalg.norm(terminal_state[0:3])
        velocity_error = np.linalg.norm(terminal_state[3:6])
        attitude_error = 0.5 * np.rad2deg(np.arccos(terminal_state[6]))
        angular_velocity_error = np.linalg.norm(terminal_state[10:13])
        used_mass = terminal_state[13]

        writer.writerow(
            [
                position_error,
                velocity_error,
                attitude_error,
                angular_velocity_error,
                used_mass,
            ]
        )
    return None


env = make_env()

# model = PPO.load('best_model_5gts65de', env)
model = PPO.load("best_model_2bo71j9m", env)
evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=30,
    render=False,
    callback=test_callback,
)

# close the file
f.close()

import pandas as pd

result_dataframe = pd.read_csv("results_montecarlo.csv")

for label in header:
    print(
        "The "
        + label
        + " has mean:{mu} and standard deviation: {sigma}".format(
            mu=result_dataframe[label].mean(),
            sigma=result_dataframe[label].std(),
        )
    )
