
import numpy as np

def lennard_jones_force(r, epsilon=1.0, sigma=1.0):
    return 48*epsilon*((sigma**12)/(r**13) - 0.5*(sigma**6)/(r**7))

def simulate(n_particles=50, steps=100, box=10.0):
    pos = np.random.rand(n_particles,2)*box
    vel = np.random.randn(n_particles,2)*0.1

    traj = []

    for _ in range(steps):
        forces = np.zeros_like(pos)

        for i in range(n_particles):
            for j in range(i+1, n_particles):
                r_vec = pos[i]-pos[j]
                r = np.linalg.norm(r_vec)+1e-5
                f = lennard_jones_force(r)
                forces[i] += f*r_vec/r
                forces[j] -= f*r_vec/r

        vel += forces*0.01
        pos += vel*0.01

        traj.append(pos.copy())

    return np.array(traj)
