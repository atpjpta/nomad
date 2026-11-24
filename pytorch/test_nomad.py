# Copyright (C) John Atkinson 2017-2024
import numpy as np
import unittest
from scipy.integrate import solve_ivp

import torch
from nomad_solver import NomadSolver

# Define the system of differential equations
def system1(t, z):
    x, y = z
    xdot = y**2 - x
    ydot = x**2 - y
    return [xdot, ydot]

# Function to solve the system of differential equations using RK45
def solve_differential_equations_scipy(sys, initial_conditions, tspan, t_eval):
    sol = solve_ivp(sys, tspan, initial_conditions, t_eval=t_eval, method='RK45')
    return sol

# Function to solve the system of differential equations using DOP853
def solve_differential_equations_nomad(initial_conditions, t):
    # dx/dt = y.^2 - x
    # dy/dt = x.^2 - y
    k = torch.Tensor(
        [[2, 2, 2, 1, 1, 1, 0, 0, 0 ],
        [2, 1, 0, 2, 1, 0, 2, 1, 0 ]]
    )

    a = torch.Tensor(
        [[0, 0, 0, 0, 0, -1, 1,  0, 0],
        [0, 0, 1, 0, 0,  0, 0, -1, 0]]
    )

    # lorenz system a and k
    # dx/dt = 10*y - 10*x
    # dy/dt = 28*x - x*z - y
    # dz/dt = x*y - 3*z
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
    # shape needs to be [N, 1, batch_size]
    # here, batch_size means the number of different ICs we want to evaluate the diff eq at
    initial_conditions = torch.Tensor(initial_conditions)
    initial_conditions = initial_conditions[:, None, None]
    initial_conditions = initial_conditions.cuda()

    # solve the diff eq
    solver = NomadSolver(a, k)
    solver = solver.cuda()

    x_traj, t, exec_times = solver(initial_conditions, t=t, print_time_interval=500)

    return x_traj, t

class TestDifferentialEquationSolvers(unittest.TestCase):
    def test_mean_error(self):
        # Define the coefficients and powers for a system of polynomial equations
        coefficients = [[-1, 1], [1, -1]]
        powers = [[[-1, 0], [0, -1]], [[0, -1], [-1, 0]]]

        # Initial conditions
        initial_conditions = [0.5, 1]  # Example initial conditions for a system with 2 variables

        # Time points for which to solve the system
        tspan = [0, 1]
        t_eval = np.arange(tspan[0], tspan[1], step=0.001)

        # Solve the system using RK45
        y_rk45 = solve_differential_equations_scipy(system1, initial_conditions, tspan, t_eval)

        # solve the system with nomad at the same times RK45 used
        t = y_rk45.t
        y_nomad, t_nomad = solve_differential_equations_nomad(initial_conditions, t)
        y_nomad = torch.cat(y_nomad, dim=1)
        y_nomad = y_nomad.detach().cpu().squeeze().numpy()

        # Compute mean error between the two solutions
        mean_error = np.mean(np.abs(y_rk45.y - y_nomad))

        # Define a tolerance level for mean error
        tolerance = 1e-3

        # Assert that the mean error is within the tolerance
        self.assertLess(mean_error, tolerance)

if __name__ == '__main__':
    unittest.main()