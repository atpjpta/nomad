% Copyright (C) John Atkinson 2017

clear all
close all
%clc

% nomad stiff test
% simple stiff ode

% dx/dt = -250*x
a = [ -250 0 ];

k = [ 1 0 ];

% strip out columns of a and k where a is a M by 1 zero vector
nonZeroCols = logical( sum( a ) );
a = a( :, nonZeroCols );
k = k( :, nonZeroCols );

% 50 derivatives!
n = 1000;

% create taylor series with nomad algorithm
sol = nomadTaylor_optim( a, k, n );

dxdt = @(t,x)(-250*x(1));

% time step
step = 0.1;

tspan = [ 0 1 ];
ic = [ 1 ];

[t,x] = ode45( dxdt, tspan, ic );

% store nomad sol here
nomad_tseries = nan( 1, length( t ) );
nomad_tseries(1) = ic;

for i = 2:1:length( t )
    %t(i) - t(i-1)
    nomad_tseries(i) = sol( t(i) - t(i-1), nomad_tseries(i-1) );
end

plot( t, x, 'linewidth', 2 );
hold on
plot( t, nomad_tseries, 'r--' );
legend( 'ode45', 'nomad' );