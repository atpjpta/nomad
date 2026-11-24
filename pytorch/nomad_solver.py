# Copyright (C) John Atkinson 2017-2024
import torch
import time
import typing

class NomadSolver(torch.nn.Module):
    """
    Class implementing the NOMAD Differential Equation Solver.
    See the NOMAD_derivation.pdf document at the root of this repo for details.
    All variable names follow the derivation.
    """
    def __init__(
        self,
        a: torch.Tensor,
        k: torch.Tensor,
        n: int=50,
        coefficient_keep_threshold: float=0,
        device='cuda') -> None:
        super().__init__()

        self.device = device
        self.coefficient_keep_threshold = coefficient_keep_threshold

        # find the nonzero columns of a
        zeros_per_column = torch.sum((a == 0), dim=0)
        non_zero_cols = zeros_per_column != a.shape[0]
        a = a[:, non_zero_cols]
        k = k[:, non_zero_cols]

        # move to cuda
        a = a.double().to(device)
        k = k.double().to(device)

        # create taylor series with the nomad algorithm
        # we will do this in steps so it can all be computed efficiently

        # step 1: convert our A and K matrices into a series of matrices
        # representing the coefficients and exponents of the n-th derivative.
        # we will compute the ahat and khat matrices for our chosen N up front and store them
        # NOTE: have to use strings so these can be stored as parameter dicts
        self.a_hats = {
            str(0): a
        }
        self.k_hats = {
            str(0): k
        }

        for i in range(1, n):
            # The previous table represents the i-th derivative divided by i!.
            # Linearity lets us divide by the new order before differentiation.
            i_sub1_key = str(i-1)
            i_key = str(i)
            ahat, khat = self.compute_next_nomad_matrices(
                a,
                self.a_hats[i_sub1_key] / (i + 1),
                k,
                self.k_hats[i_sub1_key],
                self.coefficient_keep_threshold,
                self.device
            )

            print(f'on {i}')
            print(ahat.shape)
            print(khat.shape)
            print()

            # An empty derivative table contributes no further Taylor terms.
            if ahat.shape[1] == 0:
                break

            self.a_hats[i_key] = ahat
            self.k_hats[i_key] = khat

        # we unsqueeze all the coefficient and exponent matrices so that we can quickly compute
        # derivatives for batches of initial conditions at once (assumes that last dimension is the batch dimension)
        for i in range(0, len(self.a_hats.keys())):
            i_key = str(i)
            self.a_hats[i_key] = self.a_hats[i_key].unsqueeze(-1)
            self.k_hats[i_key] = self.k_hats[i_key].unsqueeze(-1)

        self.a_hats = torch.nn.ParameterDict(self.a_hats)
        self.k_hats = torch.nn.ParameterDict(self.k_hats)

    @staticmethod
    def compute_next_nomad_matrices(
        a: torch.Tensor,
        ahat: torch.Tensor,
        k: torch.Tensor,
        khat: torch.Tensor,
        coefficient_keep_threshold,
        device):

        # get number of rows (i.e. the number of dimensions for this system
        num_dims = k.shape[0]

        # get number of columns of k and khat
        k_cols = k.shape[1]
        khat_cols = khat.shape[1]

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
        if k_cols == 0 or khat_cols == 0:
            return ahat[:, :0], khat[:, :0]
        a_li = torch.empty((2, k_cols*khat_cols), dtype=torch.long, device=k.device)
        end = 0
        for offset in range(khat_cols - 1, -k_cols, -1):
            first = max(0, -offset)
            stop = min(k_cols, khat_cols - offset)
            field_indices = torch.arange(first, stop, device=k.device)
            start, end = end, end + stop - first
            a_li[0, start:end] = field_indices + offset
            a_li[1, start:end] = field_indices

        # compute the maximum number of terms in new khat/ahat matrices (actual number of terms is
        # likely much less)
        num_new_cols = a_li.shape[1]

        # preallocate for ahat and khat. cell matrices consume a ton of memory, so
        # an array structure is used for both ahatNew and khatNew. Since
        # coefficients must be calculated numRows times (one for each
        # k_(beta,i) - kroneckerDelta_(beta,j) ), every numColsNew columns of
        # ahatNew and khatNew correspond to a slide with a different value of j.
        ahat_new = ahat.new_zeros((num_dims, num_new_cols*num_dims))
        khat_new = khat.new_zeros((num_dims, num_new_cols*num_dims))

        # loop over the number of dimensions to figure out the new a and k matrices
        for j in range(0, num_dims):
            # get the a_(j,i) and k_(alpha,j) values
            a_i_j = a[j, a_li[1, :]].repeat((num_dims, 1))
            kh_al_j = khat[j, a_li[0, :]].repeat((num_dims, 1))

            # compute the a_hat component contained in j:
            # the starting index for this value of j is
            idx_start = j*num_new_cols
            idx_end = (j+1) * num_new_cols

            ahat_new[:, idx_start:idx_end] = ahat[:, a_li[0, :]] * a_i_j * kh_al_j

            khat_new[:, idx_start:idx_end] = khat[:, a_li[0, :]] + k[:, a_li[1, :]]
            khat_new[j, idx_start:idx_end] -= 1

        # remove any columns of ahat_new that are all zeros, and the corresponding columns of khat_new
        # find the nonzero columns of a
        zeros_per_column = torch.sum((torch.abs(ahat_new) <= coefficient_keep_threshold), dim=0)
        non_zero_cols = zeros_per_column != a.shape[0]
        ahat_new = ahat_new[:, non_zero_cols]
        khat_new = khat_new[:, non_zero_cols]
        if khat_new.shape[1] == 0:
            return ahat_new, khat_new
        khat, old_idx = torch.unique(khat_new, dim=1, return_inverse=True)

        # sum the coefficients of delete entries and store in correct location corresponding to wherever
        # the entries of khat_new were moved to
        ahat = ahat_new.new_zeros(khat.shape)
        for i in range(0, len(old_idx)):
            ahat[:, old_idx[i]] = ahat[:, old_idx[i]] + ahat_new[:, i]

        # flip to match the original structure defined in the math! should have no
        # effect on the algorithm functionality
        khat = torch.flip(khat, dims=(1,))
        ahat = torch.flip(ahat, dims=(1,))

        return ahat, khat


    # apply one term of a/k coefficient/exponent matrices to x using the nomad algorithm
    @staticmethod
    def eval_term(
        a: torch.Tensor,
        k: torch.Tensor,
        x: torch.Tensor) -> torch.Tensor:
        x = x.repeat((1, k.shape[1], 1))
        x = torch.pow(x, k)
        x = torch.prod(x, dim=0)
        x = x.repeat((a.shape[0], 1, 1))
        x = torch.sum(a * x, dim=1, keepdims=True)
        return x

    # start to construct the function chain for computing a nomad step
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor) -> torch.Tensor:
        result = x
        for i in range(0, len(self.a_hats.keys())):
            i_key = str(i)
            result = result + self.eval_term(self.a_hats[i_key], self.k_hats[i_key], x)*(t**(i+1))
        return result

    def forward(
        self,
        initial_conditions: torch.Tensor,
        tstep: float=0.001,
        t_start: float=0,
        t_end: float=1,
        t: typing.List=None,
        print_time_interval: int=None):
        # some variables to keep track of all the results
        x_traj = []
        x_t = initial_conditions
        x_traj.append(x_t)
        exec_times = []

        # compute nomad trajectories
        # if t is not specified, we use evenly spaced steps from the start to end time
        # if t is specified, we evaluate at the specified times
        if t is None:
            t = []
            t.append(0)
            x_traj.append(x_t)
            num_steps = int((t_end - t_start) / tstep)
            for i in range(0, num_steps):
                t.append(t[-1]+tstep)
        else:
            num_steps = len(t)

        for i in range(1, num_steps):
            start_time = time.time()
            tstep = t[i] - t[i-1]
            x_t = self.step(x_t, tstep)

            x_traj.append(x_t)
            end_time = time.time()
            exec_times.append(end_time - start_time)

            if print_time_interval is not None and i % print_time_interval == 0:
                print(f'On t {t[-1]}, took {exec_times[-1]} seconds')

        return x_traj, t, exec_times
