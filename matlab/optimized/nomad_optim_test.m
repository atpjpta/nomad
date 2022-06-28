% Copyright (C) John Atkinson 2017

clear all
clc

%% SEE NOMAD_TEST FOR INPUT PARSER
% remove any columns of a that are all zeros, and the corresponding
% column of k

% simple stiff ode
% dx/dt = -250*x
a = [ -250 0 ];

k = [ 1 0 ];

% lorenz system a and k
% dx/dt = 10*y - 10*x
% dy/dt = 28*x - x*z - y
% dz/dt = x*y - 3*z
% a = [ 0 0  0 -10 0 10  0 0 ; ...
%       0 0 -1  28 0 -1  0 0 ; ...
%       0 1  0   0 0  0 -3 0 ];
%
% k = [ 1 1 1 1 0 0 0 0 ; ...
%       1 1 0 0 1 1 0 0 ; ...
%       1 0 1 0 1 0 1 0 ];

% dx/dt = y.^2 - x;
% dy/dt = x.^2 - y;
% k = [ 2 2 2 1 1  1 0  0 0 ;
%       2 1 0 2 1  0 2  1 0 ];
%
% a = [ 0 0 0 0 0 -1 1  0 0 ;
%       0 0 1 0 0  0 0 -1 0 ];

% strip out columns of a and k where a is a M by 1 zero vector
nonZeroCols = logical( sum( a ) );
a = a( :, nonZeroCols );
k = k( :, nonZeroCols );

% 50 derivatives!
n = 50;

% time step
step = 0.05;

% create taylor series with nomad algorithm
sol = nomadTaylor_optim( a, k, n );

%% test taylor series
close all

ic = [ 1 ; 1 ; 1 ];

%dxdt = @( t, x )( [ x(2)^2 - x(1) ; x(1)^2 - x(2) ] );

% Comparing to ODE45
% dx/dt = 10*y - 10*x
% dy/dt = 28*x - x*z - y
% dz/dt = x*y - 3*z
dxdt = @( t, x )( [ 10*x(2) - 10*x(1) ; 28*x(1) - x(1)*x(3) - x(2) ; x(1)*x(2) - 3*x(3) ] );

ode_series = cell( 1, size( ic, 2 ) ); % store time series for each ic
ode_t = cell( 1, size( ic, 2 ) ); % store t returned by ode45

% store times to compute each time series
ode_compTimes = cell( 1, size( ic, 2 ) );
leg = cell( 1, size( ic, 2 ) );

for i = 1:1:length( ode_series )

    tic;
    [ ode_t{i}, ode_series{i} ] = ode45( dxdt, [ 0 10 ], ic(:, i) );
    ode_series{i} = ode_series{i}';
    ode_compTimes{i} = toc;

    figure(2)
    plot3( ode_series{i}(1,:), ode_series{i}(2,:), ode_series{i}(3,:) )
    hold on
    leg{i} = sprintf( '%.2f, ', ic(:,i) );
end

length( ode_t{1} )

figure(2)
title( 'ode45 results' );
xlabel( 'x' );
ylabel( 'y' );
legend( leg, 'Location', 'NorthEastOutside' );
grid on
hold on

for i = 1:1:length( ode_series )
	plot3( ode_series{i}(1,1), ode_series{i}(2,1), ode_series{i}(3,1), 'ko', 'LineWidth', 2 );
    hold on
end

time_series = cell( 1, size( ic, 2 ) ); % store time series for each ic

% store times to compute each time series
compTimes = cell( 1, size( ic, 2 ) );
leg = cell( 1, size( ic, 2 ) );

for i = 1:1:length( time_series )
    t = ode_t{i}; % use time steps from ode45 solver
    time_series{i} = nan( size( ic, 1 ), length( t ) );
    time_series{i}(:,1) = ic(:,i);

    tic;

    for j = 2:1:length( t )
        tstep = t(j) - t(j-1)
        time_series{i}(:,j) = sol( tstep, time_series{i}(:,j-1) );
    end
    compTimes{i} = toc;

    figure(1)
    plot3( time_series{i}(1,:), time_series{i}(2,:), time_series{i}(3,:)  )
    hold on
    leg{i} = sprintf( '%.2f, ', ic(:,i) );
end

figure(1)
title( 'nomad results' );
xlabel( 'x' );
ylabel( 'y' );
zlabel( 'z' );
legend( leg, 'Location', 'NorthEastOutside' );
grid on
hold on

for i = 1:1:length( time_series )
	plot3( time_series{i}(1,1), time_series{i}(2,1), time_series{i}(3,1), 'ko', 'LineWidth', 2 );
end

% difference in result comparison plot
figure
difference = cell( 1, size( ic, 2 ) );
dif_leg = cell( 1, 2*length(leg) );
for i = 1:1:length( difference )
    t = ode_t{i};
    difference{i} = time_series{i} - ode_series{i};
    plot( t, difference{i}(1,:) );
    hold on
    plot( t, difference{i}(2,:) );
    plot( t, difference{i}(3,:) );

    dif_leg{3*(i-1)+1} = sprintf( 'x: %.2f, ', ic(:,i) );
    dif_leg{3*(i-1)+2} = sprintf( 'y: %.2f, ', ic(:,i) );
    dif_leg{3*(i-1)+3} = sprintf( 'z: %.2f, ', ic(:,i) );

end

title( 'difference in nomad and ode45' );
xlabel( 'x' );
ylabel( 'y' );
legend( dif_leg, 'Location', 'NorthEastOutside' );
grid on

figure
for i = 1:1:size( ic, 2 )
    t = ode_t{i};
    plot( t, time_series{i}(1,:) );
    hold on
    plot( t, time_series{i}(2,:) );
    plot( t, time_series{i}(3,:) );

end
title( 'nomad t vs x, y' );
xlabel( 'x' );
ylabel( 'y' );
legend( dif_leg, 'Location', 'NorthEastOutside' );
grid on

figure
for i = 1:1:size( ic, 2 )
    t = ode_t{i};
    plot( t, ode_series{i}(1,:) );
    hold on
    plot( t, ode_series{i}(2,:) );
    plot( t, ode_series{i}(3,:) );

end
title( 'ode t vs x, y' );
xlabel( 'x' );
ylabel( 'y' );
legend( dif_leg, 'Location', 'NorthEastOutside' );
grid on

% time to compute comparison plot
time_bars = zeros( size( ic, 2 ), 2 );

for i = 1:1:size( ic, 2 )
    time_bars( i, 1 ) = compTimes{i};
    time_bars( i, 2 ) = ode_compTimes{i};
end

figure
bar( time_bars )
xlabel( 'ic set' );
ylabel( 'total time to compute series' );
%legend( leg, 'Location', 'NorthEastOutside' );
