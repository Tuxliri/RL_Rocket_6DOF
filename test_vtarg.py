from signal import pause
from turtle import pos
import numpy as np
import matplotlib.pyplot as plt

def main():
    initial_conditions =  [-1600, 2000, np.pi*3/4, 180, -90, 0, 50e3]

    r_0 = initial_conditions[0:2]
    v_0 = initial_conditions[3:5]
    r_f = [0,0]
    v_f = [0,0]

    positions = np.linspace(start=r_0, stop=r_f, num=200)
    velocities = np.linspace(start=v_0, stop=v_f, num=200)

    trajectory = zip(positions, velocities)
    vtargs = []
    tgos = []
    

    for r,v in trajectory:
        vtarg, tgo = _compute_vtarg(r,v,IC=initial_conditions)
        vtargs.append(vtarg)
        tgos.append(tgo)
    # Compute the velocity quiver plot
    fig, ax = plt.subplots()
    x_pos = [pos[0] for pos in positions]
    y_pos = [pos[1] for pos in positions]
    U = [v[0] for v in vtargs]
    V = [v[1] for v in vtargs]

    # Let's add some color
    n=-2
    vmags = [np.sqrt(u**2+v**2) for u,v in zip(U,V)]

    ax.quiver(x_pos, y_pos, U, V, tgos, alpha=0.8, units='xy',width=10)

    fig1,ax1=plt.subplots()
    ax1.plot(vmags)
    
    plt.show()
    
    pause()

   
def _compute_vtarg(r, v, IC):
    tau_1 = 100
    tau_2 = 100
    initial_conditions = IC

    v_0 = np.linalg.norm(initial_conditions[3:5])

    # if r[1]>15:
    #     r_hat = r-[0,15]
    #     v_hat = v-[0,-2]
    #     tau = tau_1

    #else:
    #    r_hat = [0,15]
    #    v_hat = v-[0,-1]
    #    tau = tau_2
    r_hat = r
    v_hat = v
    tau = 20

    t_go = np.linalg.norm(r_hat)/np.linalg.norm(v_hat)
    v_targ = -v_0*(r_hat/np.linalg.norm(r_hat))*(1-np.exp(-t_go/tau))
    
    return v_targ, t_go

if __name__ == '__main__':
    main()