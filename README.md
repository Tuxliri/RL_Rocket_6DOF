# RL_rocket
Repository for the development of my master thesis on control of launch vehicles descent and landing through reinforcement learning actors.

## 6DOF problem
In this repository a full 6DOF rocket landing environment is developed, with realistic dynamics equation modeled on a rigid body assumption.

### Continuous action space
The environment employes a continuous action space, with the engine allowed to throttle between `maxThrust` and `minThrust`. The thrust was normalized to lie in the range `[-1, +1]` as best practice for convergence of the algorithms suggest. The engine is gimbaled by two angles $\delta_y$ and $\delta_z$ around two hinge points, respectively moving the engine around the z and y axis.