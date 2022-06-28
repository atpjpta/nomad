% Copyright (C) John Atkinson 2017

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