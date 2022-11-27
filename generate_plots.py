"""
Program used to generate high-quality plots from a training run
The plots generated are the following: 
    - 3D trajectory plot with velocity vectors
    - Target acceleration vector plot
    - Mean and std bars of episodic reward, evaluation reward
    - 
"""
import os
from pathlib import Path

import numpy as np
import plotly.express as px
import yaml
from plotly import io

import wandb as wb


def trajectory_plots(plot, camera_distance=2.5, ticksuffix = 'm/s'):
    camera = dict(
        up=dict(x=1, y=0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(zip(['x','y','z'],
            [camera_distance, camera_distance, camera_distance]))
        )

    plot.update_layout(scene_camera=camera)
    plot.data[2].colorbar.ticksuffix = ' m/s'
    plot.update_layout(showlegend=False)

    return plot

def get_plots_from_step(
    run : wb.run,
    step : float,
    keys : dict,
    ) -> dict:
    """
    Get as input the names of the plots to
    retriev and outputs a dictionary of 
    {name: Figure,} pairs
    """
    history = run.scan_history(
        keys=keys,
        min_step=step,
        max_step=step+1)
    plots_paths = [row[key]['path'] for row in history for key in keys]
    plots_json = [run.file(path) for path in plots_paths]
    
    plots_figures = {key : io.read_json(json.download(replace=True)) for key,json in zip(keys,plots_json)}
    return plots_figures

api = wb.Api()

runs_base_path = "tuxliri/RL_rocket_6DOF/"
figures_base_path = Path("../RL_rocket_thesis_doc/figures/")

with open('plots_config.yaml', 'r') as file:
    runs_type = yaml.safe_load(file)

for name in runs_type:
    plots_config = runs_type[name]
    run = api.run(
        runs_base_path+plots_config['id']
        )
    config = run.config

    # Extract names of the desired plots
    plot_objects = plots_config['plot_objects']

    figures_dict = get_plots_from_step(
        run,
        step=plots_config['step_good_episode'],
        keys=list(plot_objects.keys())
    )
    
    for obj in plot_objects:
        plot_config = plot_objects[obj]

        updated_figure = trajectory_plots(
            figures_dict[obj],
            camera_distance=plot_config['camera_distance'],
            ticksuffix=' m/s²'
            )

        image_path = figures_base_path / Path(name) / Path(obj+'.pdf')
        image_path.parent.mkdir(parents=True, exist_ok=True)


        updated_figure.write_image(
            image_path,
            width=1080,
            height=1080
        )
        