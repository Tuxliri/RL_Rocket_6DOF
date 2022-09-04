from my_environment.utils.simulator import Simulator6DOF

initial_conditions = [100,100,100,0,3,4,1,1,0,0,0,0,0,50e3]
u = [0,0,1]

RKT = Simulator6DOF(initial_conditions)

RKT.step(u)