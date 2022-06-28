% Copyright (C) John Atkinson 2017

clear all
close all
clc

%% plotting kExps
dims = 3;
order = 3;

k = kExps( dims, order );

n = 1:1:size( k, 2 );
o = ones( size( n ) );

colors = { 'r', 'b', 'm', 'c', 'g', 'y', 'k' };

figure
for j = 1:1:dims
    stem3( n, j*o, k(j,:), strcat( colors{j} ) );
    hold on
end

%% 3d of khat

numTerms = size( k, 2 );
[alpha, i] = meshgrid( 1:1:numTerms, 1:1:numTerms );

subRows = ceil( sqrt( dims ) );
subCols = ceil( sqrt( dims ) );
figure

for j = 1:1:dims
    khat = zeros( numTerms, numTerms );
    leg = cell( 1, length( 0:1:2*order ) );

    for idx1 = 1:1:numTerms
        for idx2 = 1:1:numTerms
            khat(idx1,idx2) = k(j,idx1) + k(j,idx2);
        end
    end

    % figure
    subplot( subRows, subCols, j );
    for b = min(min(khat)):1:max(max(khat))
        % color code 3d stem by value of new k
        vals = ones(size(khat))*b;
        vals(vals~=khat) = NaN;
        stem3( alpha, i, vals, colors{mod(b,length(colors))+1} );

        % no need to display twice
        if j == 1
            leg{b+1} = sprintf( 'khat = %d', b );
        end

        hold on
    end

    xlabel( 'alpha' );
    ylabel( 'i' );
    zlabel( 'k(j,alpha) + k(j,i)' );
    title( sprintf( 'j = %d', j ) );

    % no need to display twice
    if j == 1
        legend( leg, 'Location', 'NorthWestOutside' );
    end

    %view(2)
    %axis( [ 1 (order+1)^(dims-1) 1 (order+1)^(dims-1) ] );
end