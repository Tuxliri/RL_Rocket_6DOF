# RL_rocket
Repository for the development of my master thesis on control of launch vehicles descent and landing through reinforcement learning actors.

## 1DOF Toy problem
To understand better the algorithms and its working and to get an idea of the amount of time required to train the model a simplified environment is developed using as a reference the 3DOF environment. This is implemented easily by using the class `gym.Wrapper`. This environment returns as observation a vector containing the height and vertical speed of the rocket. The action is just the thrust as it is unnecessary to control gimbaling of the engine.

### Continuous action space
The initial attempt to control the 1DOF rocket was executed by testing PPO with a continuous action space, with the engine allowed to throttle between `maxThrust` and `minThrust`. The thrust was normalized to lie in the range `[-1, +1]` as best practice for convergence of the algorithms suggest. The initial results with the default hyperparameters for the PPO policy are shown hereafter.

[ADD IMAGES FROM TENSORBOARD]