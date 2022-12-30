# RL_rocket
Repository for the development of my master thesis on control of launch vehicles descent and landing through reinforcement learning actors.

## 6DOF problem
In this repository a full 6DOF rocket landing environment is developed, with realistic dynamics equation modeled on a rigid body assumption.

### Continuous action space
The environment employes a continuous action space, with the engine allowed to throttle between `maxThrust` and `minThrust`. The thrust was normalized to lie in the range `[-1, +1]` as best practice for convergence of the algorithms suggest. The engine is gimbaled by two angles $\delta_y$ and $\delta_z$ around two hinge points, respectively moving the engine around the z and y axis.

# Thesis TOC

1. Overview of aerospace control techniques with AI
2. Advantages/Disadvantages of reinforcement learning
3. RL problem structure and algorithm used
4. Environments setup
    - 3DOF environment
    - 6DOF environment
    - Software architecture implementation
5. Different reward functions definitions
6. Results in 3DOF
7. Results in 6DOF

# Docker
To run the algorithm in a Docker container follow these steps:
1. Clone the repository

2. Build the docker image `docker build -t rl_rocket_docker .`
3. Get your Wandb API key from [wandb.ai/authorize](https://www.wandb.ai/authorize)
4. Start the docker container passing the API key as an environmental variable (paste it in place of `$YOUR_API_KEY$`)
`docker run -e WANDB_API_KEY=$YOUR_API_KEY$ -it rl_rocket_docker`
