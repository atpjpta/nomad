# Copyright (C) John Atkinson 2023
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_next_nomad_matrices(
    a: torch.Tensor,
    ahat: torch.Tensor,
    k: torch.Tensor,
    khat: torch.Tensor,
    device='cuda'):

    # get number of rows (i.e. the number of dimensions for this system
    num_dims = k.shape[0]

    # get number of columns of k and khat
    k_cols = k.shape[1]
    khat_cols = k.shape[1]

    # a_li is a vector of indices, [alpha ; i], that is created for fast indexing
    # indexing of khat and k. khat is indexed by alpha, and k is indexed by i.
    # The indices will retrieve elements of khat and k in such a way that
    # mimics sliding k across khat, starting with the last element of khat
    # lined up with the first element of k. The entire slide requires the
    # kCols*khatCols elements to be accessed, which can also be viewed as N+M-1
    # "entries," as shown below:
    #
    # DIAGRAM OF SLIDE HERE
    #
    a_li = torch.zeros((2, k_cols*khat_cols))

    # initialize the last end index to -1, simplifies the logic
    # when constructing a_li
    idx_end = 0
    for idx in range(0, k_cols + khat_cols - 1):
        if idx < k_cols:
            # create entry of a_li per description above
            # this is the beginning of the slide described above
            khat_idxs = torch.arange(start=khat_cols-idx-1, end=khat_cols, step=1)
            khat_idxs = khat_idxs.unsqueeze(0)
            k_idxs = torch.arange(start=0, end=idx+1, step=1)
            k_idxs = k_idxs.unsqueeze(0)
            # print('less than k')
            # print(khat_idxs)
            # print(k_idxs)
            entry = torch.cat(
                [
                    khat_idxs,
                    k_idxs
                ],
                dim=0
            )
        elif idx < khat_cols:
            # create the entry of a_li per description above
            # this is the middle of the slide described above
            print('I dont think this ever hits')
            khat_idxs = torch.arange(start=khat_cols-idx, end=khat_cols-(idx-k_cols), step=1)
            khat_idxs = khat_idxs.unsqueeze(0)
            k_idxs = torch.arange(start=0, end=k_cols, step=1)
            k_idxs = k_idxs.unsqueeze(0)
            # print('less than khat')
            # print(khat_idxs)
            # print(khat_idxs.shape)
            # print(k_idxs)
            # print(k_idxs.shape)
            entry = torch.cat(
                [
                    khat_idxs,
                    k_idxs
                ],
                dim=0
            )
        elif idx > khat_cols-1:
            # create the entry of a_li per description above
            # this is basically the tail end of the slide, as we are finishing
            khat_idxs = torch.arange(start=0, end=k_cols+khat_cols-idx-1, step=1)
            khat_idxs = khat_idxs.unsqueeze(0)
            k_idxs = torch.arange(start=idx-khat_cols+1, end=k_cols, step=1)
            k_idxs = k_idxs.unsqueeze(0)
            #print('greater than khat')
            # print(khat_idxs)
            # print(khat_idxs.shape)
            # print(k_idxs)
            # print(k_idxs.shape)
            entry = torch.cat(
                [
                    khat_idxs,
                    k_idxs
                ],
                dim=0
            )

        # after we compute the next entry for a_li, we figure out where to insert it
        # we start at the last index, accounting for elements added on previous iterations
        idx_s = idx_end
        idx_end = idx_s + entry.shape[1]

        # add the entry to a_li
        a_li[:, idx_s:idx_end] = entry

    a_li = a_li.long()

    # compute the maximum number of terms in new khat/ahat matrices (actual number of terms is
    # likely much less)
    num_new_cols = a_li.shape[1]

    # create a row vector to represent the kronecker delta we will apply to row j of the sliding k matrix
    d = torch.ones((1, a_li.shape[1])).to(device)

    # preallocate for ahat and khat. cell matrices consume a ton of memory, so
    # an array structure is used for both ahatNew and khatNew. Since
    # coefficients must be calculated numRows times (one for each
    # k_(beta,i) - kroneckerDelta_(beta,j) ), every numColsNew columns of
    # ahatNew and khatNew correspond to a slide with a different value of j.
    ahat_new = torch.zeros((num_dims, a_li.shape[1]*num_dims)).cuda()
    khat_new = torch.zeros((num_dims, a_li.shape[1]*num_dims)).cuda()

    # loop over the number of dimensions to figure out the new a and k matrices
    for j in range(1, num_dims):
        # get the a_(j,i) and k_(alpha,j) values
        a_i_j = a[j, a_li[1, :]].repeat((num_dims, 1))
        kh_al_j = khat[j, a_li[0, :]].repeat((num_dims, 1))

        # compute the a_hat component contained in j:
        # the starting index for this value of j is
        idx_start = (j-1)*num_new_cols
        idx_end = j * num_new_cols

        ahat_new[:, idx_start:idx_end] = ahat[:, a_li[0, :]] * a_i_j * kh_al_j

        delta = torch.cat([torch.zeros((j-1, 1)), torch.ones((1, 1)), torch.zeros(num_dims-j, 1)], dim=0).to(device)
        delta = torch.matmul(delta, d)
        khat_new[:, idx_start:idx_end] = khat[:, a_li[0, :]] + k[:, a_li[1, :]] - delta

    # remove any columns of ahat_new that are all zeros, and the corresponding columns of khat_new
    # find the nonzero columns of a
    zeros_per_column = torch.sum((ahat_new == 0), dim=0)
    non_zero_cols = zeros_per_column != a.shape[0]
    ahat_new = ahat_new[:, non_zero_cols]
    khat_new = khat_new[:, non_zero_cols]
    khat, old_idx = torch.unique(khat_new, dim=1, return_inverse=True)

    # sum the coefficients of delete entries and store in correct location corresponding to wherever
    # the entries of khat_new were moved to
    ahat = torch.zeros(khat.shape).cuda()
    for i in range(0, len(old_idx)):
        ahat[:, old_idx[i]] = ahat[:, old_idx[i]] + ahat_new[:, i]

    # flip to match the original structure defined in the math! should have no
    # effect on the algorithm functionality
    khat = torch.flip(khat, dims=(0,))
    ahat = torch.flip(ahat, dims=(0,))

    return ahat, khat

# dx/dt = y.^2 - x
# dy/dt = x.^2 - y
# k = torch.Tensor(
#     [[2, 2, 2, 1, 1, 1, 0, 0, 0 ],
#      [2, 1, 0, 2, 1, 0, 2, 1, 0 ]]
# )

# a = torch.Tensor(
#     [[0, 0, 0, 0, 0, -1, 1,  0, 0],
#      [0, 0, 1, 0, 0,  0, 0, -1, 0]]
# )

# lorenz system a and k
# dx/dt = 10*y - 10*x
# dy/dt = 28*x - x*z - y
# dz/dt = x*y - 3*z
a = torch.Tensor(
    [[ 0, 0,  0, -10, 0, 10,  0, 0 ],
     [ 0, 0, -1,  28, 0, -1,  0, 0 ],
     [ 0, 1,  0,   0, 0,  0, -3, 0 ]]
)

k = torch.Tensor(
    [[ 1, 1, 1, 1, 0, 0, 0, 0 ],
     [ 1, 1, 0, 0, 1, 1, 0, 0 ],
     [ 1, 0, 1, 0, 1, 0, 1, 0 ]]
)

# find the nonzero columns of a
zeros_per_column = torch.sum((a == 0), dim=0)
non_zero_cols = zeros_per_column != a.shape[0]
a = a[:, non_zero_cols]
k = k[:, non_zero_cols]

# move to cuda
a = a.double().cuda()
k = k.double().cuda()

# take 50 derivatives
n = 50

# create taylor series with the nomad algorithm
# we will do this in steps so it can all be computed efficiently

# step 1: convert our A and K matrices into a series of matrices
# representing the coefficients and exponents of the n-th derivative.
# we will compute the ahat and khat matrices for our chosen N up front and store them
a_hats = {
    1: a
}
k_hats = {
    1: k
}

for i in range(2, n):
    # nomad itself returns the ahat's and khat's corresponding to the next
    # derivative. However, for constructing a taylor series, one must
    # divide by n! for the n-th term in the series. As n grows, this
    # term grows so large that it falls well outside the range of accurate
    # floating point (i.e. double) representation.
    #
    # Furthermore, as n grows, elements of ahat become very large as well,
    # generally at the same rate of factorial growth. This leads to the
    # same issue as n!, the numbers fall outside the range of non-sparse
    # double representation.
    #
    # A workaround to this is dividing the ahat of the n-th derivative
    # by n on each iteration, before computing the (n+1)-th ahat. By doing
    # this, the n! is computed gradually and offsets the growth of
    # individual elements in the n-th ahat. Hopefully, this change is
    # enough to keep higher order ahat elements within the range of
    # nonsparse double representation, and thus improving numerical
    # accuracy.
    #
    # Divide ahat by i before computation, for reasons explained above.
    ahat, khat = compute_next_nomad_matrices( a, a_hats[i-1]/i, k, k_hats[i-1])

    # because of the trick we are doing above where we divide a hat by i progressively
    # to compute the n! in the denominator of the taylor series, elements of ahat
    # typically tend to 0 eventually. so, we catch the condition where all coefficients have
    # converged to 0, and break early in those cases to save on further computations since
    # we know all coefficients past this term are also guaranteed to be 0.
    # if more precision is needed, then we need to use a bigger dtype like float64
    if ahat.shape[1] == 0:
        break

    a_hats[i] = ahat
    k_hats[i] = khat

# we unsqueeze all the coefficient and exponent matrices so that we can quickly compute
# derivatives for batches of initial conditions at once (assumes that last dimension is the batch dimension)
for i in range(1, len(a_hats.keys())+1):
    a_hats[i] = a_hats[i].unsqueeze(-1)
    k_hats[i] = k_hats[i].unsqueeze(-1)

# apply one term of a/k coefficient/exponent matrices to x using the nomad algorithm
def eval_term(a, k, x):
    x = x.repeat((1, k.shape[1], 1))
    x = torch.pow(x, k)
    x = torch.prod(x, dim=0)
    x = x.repeat((a.shape[0], 1, 1))
    x = torch.sum(a * x, dim=1, keepdims=True)
    return x

# start to construct the function chain for computing a nomad step
def step(x, t, a_hats, k_hats):
    for i in range(1, len(a_hats.keys())+1):
        x = x + eval_term(a_hats[i], k_hats[i], x)*(t**i)
    return x

initial_conditions = torch.rand((3, 1, 100))
for i in range(initial_conditions.shape[-1]):
    initial_conditions[0, 0, i] = 0 - np.sin(i*0.01)
    initial_conditions[1, 0, i] = 0.0 + np.cos(i*0.01)
    initial_conditions[2, 0, i] = 0.0 + torch.sin(initial_conditions[1, 0, i])
initial_conditions = initial_conditions.cuda()

# compute nomad trajectories
tstep = 0.0001
t_start = 0
t_end = 1
num_steps = int((t_end - t_start) / tstep)

# some variables to keep track of all the results
t = []
x_traj = []
t.append(0)
x_t = initial_conditions
x_traj.append(x_t)

# solve the differential equations for each initial condition
exec_times = []
import time
for i in range(0, num_steps):
    start_time = time.time()
    x_t = step(x_t, tstep, a_hats, k_hats)
    t.append(t[-1]+tstep)

    x_traj.append(x_t)
    end_time = time.time()
    exec_times.append(end_time - start_time)

    if i % 500 == 0:
        print(f'On t {t[-1]}, took {exec_times[-1]} seconds')

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
# fig = plt.figure(figsize=(20, 10))
# plt.plot(t, x_traj[0, :])
# plt.plot(t, x_traj[1, :])
# plt.legend(['x1', 'x2'])
# plt.show()

# NOTE: 3d plots
x = x_traj[0, ...]
y = x_traj[1, ...]
z = x_traj[2, ...]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
for i in range(x.shape[-1]):
    ax.plot(x[:, i], y[:, i], z[:, i])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D Line Plot')

plt.show()
