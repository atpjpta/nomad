% Copyright (C) John Atkinson 2017

function sol = nomadTaylor( a, k, n, ic )

ahat = a;
khat = k;

syms t
% first order taylor series
sol = ic + evalEq( ahat, khat, ic )*(t);

for i = 2:1:n
    i
    [ahat, khat] = nomad( a, ahat, k, khat );
    sol = sol + evalEq( ahat, khat, ic )*((t)^i)/factorial(i);
end

sol = matlabFunction( sol );

end

function val = evalEq( a, k, ic )
    tmp = prod( repmat( ic, 1, size(k,2) ).^k );
    val = sum( a.*repmat( tmp, [ size(a,1), 1 ] ), 2 );
end