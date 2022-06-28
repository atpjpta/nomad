% Copyright (C) John Atkinson 2017

function [ coefs, x, p, k, grad_p, hessian_p ] = polynomial( dims, order )
% Inputs:
%        dims - number of variables in p
%       order - order of polynomial p
%
% Outputs:
%       coefs - <(order+1)^dims> coefficients for each term of polynomial
%           x - <dims> symbolic variables in polynomial p
%           p - a multivariate polynomial of order <order> in <dims> variables
%      grad_p - 1st partial derivitive vector (gradient) of polynomial p w.r.t. x
%   hessian_p - 2nd partial derivative matrix (hessian) of polynomial p w.r.t. x

    % number of terms in polynomial is:
    numTerms = (order+1)^dims

    % generate <dims> variables
    x = sym( 'x%d', [ dims, 1 ], 'real' )

    % generate <(order+1)^dims> coefficients
    coefs = sym( 'a%d', [ 1, numTerms ], 'real' )

    % generate k, a <dims> by <numTerms> matrix
    % k_j_i is the exponent of x_j for the i-th term of the sum
    k = kExps( dims, order )

    % create a multivariate polynomial of order <order> in <dims> variables
    % with general coefficients a_i for the i-th term
    p = sum( coefs.*prod( repmat( x, 1, numTerms ).^k ) )

    % get gradient of polynomial w.r.t. x
    grad_p = jacobian( p, x );

    % get hessian of polynomial w.r.t. x
    hessian_p = hessian( p, x );
end

function k = kExps( dims, order )
% calculate term-exponent matrix by recursion:
% let M = dims and let N = order, then
%           _                                                                             _
%          | N (N+1)^(M-1) times, N-1 (N+1)^M(-1) times, ..., 1 (N+1)^(M-1), 0 (N+1)^(M-1) |
% k(M,N) = |            k(M-1,N),              k(M-1,N), ...,      k(M-1,N),      k(M-1,N) |
%           -                                                                             -
%
    if dims == 1
        k = order:-1:0;
        return;
    else
        kLast = kExps( dims-1, order );
        k = zeros( dims, (order+1)^dims );

        step = (order+1)^(dims-1);

        for i = 1:1:(order+1)
            k( :, (1+(i-1)*step):(i*step) ) = ...
                [ (order+1-i)*ones(1,step) ; kLast ];
        end
    end
end