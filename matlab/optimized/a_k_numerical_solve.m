clear all
clc

% dx/dt = y.^2 - x;
% dy/dt = x.^2 - y;

k = [ 2 2 2 1 1  1 0  0 0 ;
      2 1 0 2 1  0 2  1 0 ];

a = [ 0 0 0 0 0 -1 1  0 0 ;
      0 0 1 0 0  0 0 -1 0 ];

% for first derivative, ahat and khat are original a and k
ahat = a; % ahat = a/(1!), comment for completeness. Explained below
khat = k;


% strip out columns of a and k where a is a M by 1 zero vector
nonZeroCols = logical( sum( a ) );
a = a( :, nonZeroCols );
k = k( :, nonZeroCols );

ahat
khat

[ahat, khat] = nomad_optim( a, ahat./1, k, khat );

ahat
khat

[ahat, khat] = nomad_optim( a, ahat./2, k, khat );

ahat
khat