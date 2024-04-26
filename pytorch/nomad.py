# Copyright (C) John Atkinson 2017-2024
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nomad_solver import NomadSolver

torch.set_printoptions(linewidth=200)

device = 'cuda'

# dx/dt = y.^2 - x
# dy/dt = x.^2 - y
k = torch.Tensor(
    [[2, 2, 2, 1, 1,  1, 0,  0, 0 ],
     [2, 1, 0, 2, 1,  0, 2,  1, 0 ]]
)

a = torch.Tensor(
    [[0, 0, 0, 0, 0, -1, 1,  0, 0],
     [0, 0, 1, 0, 0,  0, 0, -1, 0]]
)

# lorenz system a and k
# dx/dt = 10*y - 10*x
# dy/dt = 28*x - x*z - y
# # dz/dt = x*y - 3*z
# a = torch.Tensor(
#     [[ 0, 0,  0, -10, 0, 10,  0, 0 ],
#      [ 0, 0, -1,  28, 0, -1,  0, 0 ],
#      [ 0, 1,  0,   0, 0,  0, -3, 0 ]]
# )

# k = torch.Tensor(
#     [[ 1, 1, 1, 1, 0, 0, 0, 0 ],
#      [ 1, 1, 0, 0, 1, 1, 0, 0 ],
#      [ 1, 0, 1, 0, 1, 0, 1, 0 ]]
# )

# define some initial conditions
# (N, 1, batch_size)
initial_conditions = torch.rand((2, 1, 20))
for i in range(initial_conditions.shape[-1]):
    initial_conditions[0, 0, i] = 0 - np.sin(i*0.01)
    initial_conditions[1, 0, i] = 0.0 + np.cos(i*0.01)
    #initial_conditions[2, 0, i] = 0.0 + torch.sin(initial_conditions[1, 0, i])
initial_conditions = initial_conditions.to(device)

# solve the diff eq
solver = NomadSolver(a, k, n=50, device=device)
solver = solver.to(device)
x_traj, t, exec_times = solver(initial_conditions, tstep=0.001, print_time_interval=500)

# get some stats
avg_step_time = np.mean(exec_times)
print(f'Average exec time was {avg_step_time}')
print(f'Total exec time was {np.sum(exec_times)}')

print(len(x_traj))

# convert back to numpy so we can plot
t = np.array(t)
x_traj = torch.cat(x_traj, dim=1)
x_traj = x_traj.detach().cpu().numpy()

# NOTE: 2d
fig = plt.figure(figsize=(20, 10))
plt.plot(t, x_traj[0, :])
plt.plot(t, x_traj[1, :])
plt.legend(['x1', 'x2'])
plt.show()

# NOTE: 3d plots
# x = x_traj[0, ...]
# y = x_traj[1, ...]
# z = x_traj[2, ...]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
# for i in range(x.shape[-1]):
#     ax.plot(x[:, i], y[:, i], z[:, i])

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title('3D Line Plot')

# plt.show()
