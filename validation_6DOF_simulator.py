from pathlib import Path, WindowsPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_environment.utils.simulator import Simulator6DOF


def plot_norm(
    df: pd.DataFrame,
    logy=True,
):
    return df.apply(np.linalg.norm, axis=1,).plot(
        logy=logy,
    )


initial_conditions = [2000, -1600, 0, -90, 180, 0, 0.866, 0, 0, -0.5, 0, 0, 0, 41e3]
u = [
    np.deg2rad(5),
    np.deg2rad(5),
    98e3,
]  # control vector [delta_y,delta_z,thrust] [rad,rad,N]

TIMESTEP = 0.1
RKT = Simulator6DOF(initial_conditions, dt=TIMESTEP)

timevector = np.linspace(0, 20, int(20 / TIMESTEP) + 1)

state = np.array(initial_conditions)
states, times = [], []
for t in timevector:
    times.append(round(t, 1))
    states.append(list(state))
    state, solution_state = RKT.step(u)
    if solution_state:
        break

state_names = [
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

simulator_states_df = pd.DataFrame(data=states, index=times, columns=state_names)


# Load validated trajectory data using read_csv
from pandas import read_csv

state_names.insert(0, state_names.pop(-1))
validated_trajectory = read_csv("states_table.csv", names=state_names, header=0)

validated_trajectory.index = pd.to_timedelta(validated_trajectory.index)
simulator_states_df.index = pd.to_timedelta(times, unit="S")

error = validated_trajectory.sub(simulator_states_df)  # /initial_conditions

# Plot the norm of the error for each state component
fig, ax = plt.subplots()
r = ["x", "y", "z"]
v = ["vx", "vy", "vz"]
omega = ["omega1", "omega2", "omega3"]
q = ["q0", "q1", "q2", "q3"]
mass = ["mass"]

state_labels = [r,v,omega,q,mass]

RELATIVE_ERROR = False

for state in state_labels:
    if RELATIVE_ERROR:
        mean = validated_trajectory[state].apply(np.linalg.norm,axis=1).mean()
        plot_norm(error[state]/mean)
    else:
        plot_norm(error[state])

import matplotlib as mpl
import matplotlib.dates as mdates

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")
mpl.rcParams["font.serif"] = "Times"

if RELATIVE_ERROR:
    legend = [
        R"$||\bm{r}_{err}||$",
        R"$||\bm{v}_{err}||$",
        R"$||\bm{\omega}_{err}||$",
        R"$||\bm{q}_{err}||$",
        R"$m_{err}$",
    ]
    plt.ylabel(r'$\epsilon_{rel}$')
    plot_name = "validation_error_plot_rel.png"
else:
    legend = [
        R"$||\bm{r}_{err}||\;[m]$",
        R"$||\bm{v}_{err}||\;[m/s]$",
        R"$||\bm{\omega}_{err}||\;[rad/s]$",
        R"$||\bm{q}_{err}||$\;[-]",
        R"$m_{err}\;[kg]$",
    ]
    plt.ylabel(r'$\epsilon_{abs}$')
    plot_name = "validation_error_plot_abs.png"


ax.legend(legend)

ax.grid(True)
# mFmt=mdates.DateFormatter('%H:%M:%S')
# ax.xaxis.set_major_formatter(mFmt)
fig.autofmt_xdate()

plt.xlabel("time [h:m:s.ms]")
# plt.show()

base_path = Path("../RL_rocket_thesis_doc/figures/")
plt.savefig(base_path / plot_name, dpi=1200)
