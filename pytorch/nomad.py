# Copyright (C) John Atkinson 2023
import torch


def compute_next_nomad_matricest(
    a: torch.Tensor,
    ahat: torch.Tensor,
    k: torch.Tensor,
    khat: torch.Tensor):

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

    print(f'{k_cols=}')
    print(f'{khat_cols=}')

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

    print(a_li)

    # compute the maximum number of terms in new khat/ahat matrices (actual number of terms is
    # likely much less)
    num_new_cols = a_li.shape[1]

    # create a row vector to represent the kronecker delta we will apply to row j of the sliding k matrix
    d = torch.ones((1, a_li.shape[1]))

    # preallocate for ahat and khat. cell matrices consume a ton of memory, so
    # an array structure is used for both ahatNew and khatNew. Since
    # coefficients must be calculated numRows times (one for each
    # k_(beta,i) - kroneckerDelta_(beta,j) ), every numColsNew columns of
    # ahatNew and khatNew correspond to a slide with a different value of j.
    ahat_new = torch.zeros((num_dims, a_li.shape[1]*num_dims))
    khat_new = torch.zeros((num_dims, a_li.shape[1]*num_dims))

    # loop over the number of dimensions to figure out the new a and k matrices
    for j in range(1, num_dims):
        # get the a_(j,i) and k_(alpha,j) values
        a_i_j = torch.repeat(a[j, a_li[2,:]], (num_dims, 1))
        kh_al_j = torch.repeat(khat[j, a_li[1,:]], (num_dims, 1))

        # compute the a_hat component contained in j:
        # the starting index for this value of j is
        idx_start = (j-1)*num_new_cols
        idx_end = j * num_new_cols

        ahat_new[:, idx_start:idx_end] = ahat[:, a_li[0, :]] * a_i_j * kh_al_j

        delta = torch.cat([torch.zeros((j-1, 1)), torch.ones((1, 1)), torch.zeros(num_dims-j, 1)], dim=1)
        delta = torch.matmul(delta, d)
        khat_new[:, idx_start:idx_end] = khat[:, a_li[0, :]] + k[:, a_li[1, :]] - delta

    # remove any columns of ahat_new that are all zeros, and the corresponding columns of khat_new
    # find the nonzero columns of a
    non_zero_cols = torch.sum((ahat_new == 0), dim=-1)
    non_zero_cols = (non_zero_cols != 0)
    ahat_new = ahat_new[non_zero_cols]
    khat_new = khat_new[non_zero_cols]

    khat, old_idx = torch.unique(khat_new, dim=1, return_inverse=True)

    # sum the coefficients of delete entries and store in correct location corresponding to wherever
    # the entries of khat_new were moved to
    ahat = torch.zeros(khat.shape)
    for i in range(0, len(old_idx)):
        ahat[:, old_idx[i]] = ahat[:, old_idx[i]] + ahat_new[:, i]

    # flip to match the original structure defined in the math! should have no
    # effect on the algorithm functionality
    khat = khat[:, ::-1]
    ahat = ahat[:, ::-1]

    return ahat, khat

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

# find the nonzero columns of a
non_zero_cols = torch.sum((a == 0), dim=-1)
non_zero_cols = (non_zero_cols != 0)
a = a[non_zero_cols]
k = k[non_zero_cols]

# move to cuda
a = a.cuda()
k = k.cuda()

# take 50 derivatives
n = 50

# time step
step = 0.05

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

def eval_term(a, k, x):
    x_repeated = torch.repeat(x, (1, k.shape[1]))
    x_repeated = torch.pow(x_repeated, k)
    tmp = torch.prod(x_repeated, dim=0)

    a_multiplicant = torch.repeat(tmp, (a.shape[0], 1))
    val = torch.sum(a * a_multiplicant, dim=1)

    return val

# start to construct the function chain for computing a nomad step
step = lambda x, t: x + eval_term(a_hats[1], k_hats[1], x)*(t**1)

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
    ahat, khat = compute_next_nomad_matricest( a, a_hats[i-1]/i, k, k_hats[i-1])
    a_hats[i] = ahat
    k_hats[i] = khat

    step = lambda x, t: step(x, t) + eval_term(a_hats[i], k_hats[i], x)*(t**i)

initial_conditions = torch.ones((2, 1))
initial_conditions = initial_conditions.cuda()

# compute nomad trajectories
tstep = 0.05

x_traj = []
x_t = initial_conditions
for i in range(0, 1, tstep):
    x_t = step(x_t, tstep)
    x_traj.append(x_t)

    print(x_t)

print(len(x_t))


