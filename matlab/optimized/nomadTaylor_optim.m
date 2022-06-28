% Copyright (C) John Atkinson 2017

function [sol, evalTimes, conversionTime] = nomadTaylor_optim( a, k, n )

evalTimes = nan( 1, n );

% variables for function
ic = sym( 'x%d', [ size( k, 1 ), 1 ], 'real' );
syms t

% for first derivative, ahat and khat are original a and k
ahat = a; % ahat = a/(1!), comment for completeness. Explained below
khat = k;

tic;

% start with first order taylor series
sol = ic + evalEq( ahat, khat, ic )*t;

evalTimes(1) = toc; % time first deriv computation

for i = 2:1:n
    tic; % time each iteration

    % nomad itself returns the ahat's and khat's corresponding to the next
    % derivative. However, for constructing a taylor series, one must
    % divide by n! for the n-th term in the series. As n grows, this
    % term grows so large that it falls well outside the range of accurate
    % floating point (i.e. double) representation.
    %
    % Furthermore, as n grows, elements of ahat become very large as well,
    % generally at the same rate of factorial growth. This leads to the
    % same issue as n!, the numbers fall outside the range of non-sparse
    % double representation.
    %
    % A workaround to this is dividing the ahat of the n-th derivative
    % by n on each iteration, before computing the (n+1)-th ahat. By doing
    % this, the n! is computed gradually and offsets the growth of
    % individual elements in the n-th ahat. Hopefully, this change is
    % enough to keep higher order ahat elements within the range of
    % nonsparse double representation, and thus improving numerical
    % accuracy.

    % Divide ahat by i before computation, for reasons explained above.
    [ahat, khat] = nomad_optim( a, ahat./i, k, khat );

    % No division by n!, due to the optimization explained above.
    sol = sol + evalEq( ahat, khat, ic )*(t^i);

    evalTimes(i) = toc;
end

tic

% convert solution to function and return
sol = matlabFunction( sol, 'Vars', {t, ic} );

conversionTime = toc;

end

function val = evalEq( a, k, ic )
    tmp = prod( repmat( ic, 1, size(k,2) ).^k );
    val = sum( a.*repmat( tmp, [ size(a,1), 1 ] ), 2 );
end