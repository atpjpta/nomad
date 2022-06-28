% Copyright (C) John Atkinson 2017

clear all
close all
clc

%% Will extrapolate to function later.
% NO_MD_nD: N-th Order M-Dimensional n-th Derivative
% (looks like nomad.. cool name?)

% 1) function must take inputs of M-dimensional variables and M-dimensional
%    system of equations (both as cells of strings!!!) and the dimension M
%    and order N as arguments

% variables
vars = cell( 1, 2 );
vars{1} = 'x';
vars{2} = 'y';
vars{3} = 'z';

% system of equations
inSys = cell( 1, 2 );
inSys{1} = 'dx/dt = 10*y - 10*x';  %y^2 - x';
inSys{2} = 'dy/dt = 28*x - x*z - y'; %x^2 - y';
inSys{3} = 'dz/dt = x*y - 3*z';
M = 3; % dimensions
N = 1; % highest order is 2

noSpace = '\S'; % strip whitespace
sys = cell(1,2);
for idx = 1:1:length( inSys )
    inSys{idx} = inSys{idx}( regexp( inSys{idx}, noSpace ) );

    % store only dynamics in sys2, strip off dvar/dt
    sys{idx} = regexp( inSys{idx}, '=', 'split' );
    sys{idx} = sys{idx}{2};
end

k = kExps( M, N ); % get k matrix for an (M,N) poly
% get size of k
[numDims, numColumns] = size( k );

% initialize coefficient matrix "a" to 0s
a = zeros( numDims, numColumns );

%% NOTE: Equation parser needs a lot of work! Significant trouble with
% parsing cross terms, however, works good enough for simple systems as
% shown above
for idx = 1:1:numColumns % iteratate over columns of k
    % construct search terms for system of equations based on k-matrix
    term = '(';
    for j = 1:1:numDims % iterate over rows of k

        if k(j,idx) > 1
            % if exponent is greater than 1, add the corresponding variable
            % with that exponent to the search term
            term = strcat( term, vars{j}, '\^', num2str( k(j,idx) ) );
        elseif k(j,idx) == 1
            % if exponent is 1, only add the corresponding variable to the
            % search term
            term = strcat( term, vars{j}, '([^\^]|$)' );
        end

        % if the search term already contains a variable, and there are
        % more terms to add with non-zero exponents, append a
        % multiplication symbol
        if ~strcmp( term, '(' ) && j < numDims
            if k(j+1,idx) > 0
                term = strcat( term, '\*' );
            end
        end

        % on last row
        if j == numDims
            if strcmp( term, '(' )
                % if nothing else was added, it is the constant term. Thus,
                % the regexp is as follows:
                term = '((\d+/\d+)|(\d+\.{0,1}\d+)|(\d+))$';
            else
                % end subexpression grouping
                term = strcat( term, ')');
            end
        end
    end

    % iterate over system of equations, checking for the current search
    % term. If found, a(j,i) is updated with the coefficient of the search
    % term.
    for j = 1:1:numDims

        % by design, the last column of k corresponds to constant terms.
        % Thus, it is as simple as grabbing the coefficient like so, then
        % continuing.
        if idx == numColumns
            match = regexp( sys{j}, term, 'match' );

            if ~isempty( match )
                a(j,idx) = str2double( match );
            end

            continue;
        end

        match = regexp( sys{j}, term );

        % if no match, proceed to check the next equation
        if isempty( match )
            continue;
        end

        fprintf( 'Term: %s\n\tEq: %s\n\tMatch: %s\n\n', term, sys{j}, sys{j}(match) );

        if match == 1
            % if the match is the VERY first term of the equation, then
            % clearly the first character in the equation is a
            % variable. Thus, it must have a coefficient of one.
            a(j,idx) = 1;
        else
            % get the operator that precedes the match
            op = sys{j}(match-1);

            if strcmp( op, '+' )
                % if the operator preceding the match is a plus sign, then
                % its coefficient must be a positive 1.
                a(j,idx) = 1;
            elseif strcmp( op, '-' )
                % if the operator preceding the match is a minus sign, then
                % its coefficient must be a negative 1.
                a(j,idx) = -1;
            elseif strcmp( op, '*' )
                % if the operator preceding the match is a multiplication
                % sign, then its coefficient must be some number other than
                % +/- 1.

                % Thus, get all text preceding this operator.
                tmp = sys{j}(1:(match-2));

                % pattern to match a fraction or decimal coefficient
                d = '((\d+/\d+)|(\d+\.{0,1}\d+)|(\d+))$';

                % get coefficient index
                coefIdx = regexp( tmp, d );

                % determine sign of coefficient
                if coefIdx > 1
                    coefSign = tmp( coefIdx-1 );
                else
                    % if coefficient index is 1, then there cannot be a
                    % negative sign in front of it. Thus, it must be
                    % positive.
                    coefSign = '+';
                end

                % get the coefficient itself
                coef = tmp( coefIdx:end );

                % check for division operator
                fracIdx = regexp( coef, '/' );

                % If coefficient is fractional, handle elegantly. Else,
                % simply convert coefficient to double.
                if ~isempty( fracIdx )
                    coefNum = str2double( coef( 1:(fracIdx-1) ) );
                    coefDenom = str2double( coef( (fracIdx+1):end ) );
                    coef = coefNum/coefDenom;
                else
                    coef = str2double( coef );
                end

                % handle cases where coefficient is negative
                if strcmp( coefSign, '-' )
                    coef = -coef;
                end

                % a(j,i) is at the end of the text
                % preceding this operator, and was computed above.
                a(j,idx) = coef;
            else
               error( 'Character preceding the match is not an operator.\n' );
            end
        end
    end
end

%% Magic time
% remove any columns of a that are all zeros, and the corresponding
% column of k

% lorenz system a and k
a = [ 0 0  0 -10 0 10  0 0 ; ...
      0 0 -1  28 0 -1  0 0 ; ...
      0 1  0   0 0  0 -3 0 ];

k = [ 1 1 1 1 0 0 0 0 ; ...
      1 1 0 0 1 1 0 0 ; ...
      1 0 1 0 1 0 1 0 ];

nonZeroCols = logical( sum( a ) );
a = a( :, nonZeroCols )
k = k( :, nonZeroCols )

x = sym( 'x%d', [ size(a,1), 1 ], 'real' );

ics = [ 0.1  1 0.1 ; 0.7 0.2 1 ];

step = linspace( 0, 0.05, 3 );

colors = { 'b', 'r' };

for j = 1:1:size( ics, 1 )
tic
ic = ics( j, : )';
for yy = 1:1:200
    yy
    t0 = 0;
    n = 50;
    sol = nomadTaylor( a, k, n, ic );
    xs = sol( step );
    plot3( xs(1,:), xs(2,:), xs(3,:), colors{j} );
    hold on
    ic = [xs(1,end); xs(2,end); xs(3,end) ];
end
c(j) = toc
end

% make sure it parsed right
% tmp = prod( repmat( x, 1, size(k,2) ).^k );
% p1 = sum( a.*[tmp;tmp], 2 );
%
%
% % start with 5 derivs
% n = 1000;
% da = cell( 1, n);
% dk = cell( 1, n );
% p = cell( 1, n );
%
% da{1} = a;
% dk{1} = k;
% p{1} = p1;
%
% for i = 2:1:n
%     % for 2nd deriv, ahat = a and khat = k
%     tic
%     [ da{i}, dk{i} ] = nomad( da{1}, da{i-1}, dk{1}, dk{i-1} );
%     t = toc;
%     fprintf( 'n = %d, took %.4f seconds\n', i, t );
%
%     % check result
%     %tmp = prod( repmat( x, 1, size(dk{i},2) ).^dk{i} );
%     %p{i} = sum( da{i}.*[tmp;tmp], 2 );
%
%
% end
%
% create a multivariate polynomial of order <order> in <dims> variables
% with general coefficients a_i for the i-th term
% tmp = prod( repmat( x, 1, size(k,2) ).^k );
% p = sum( a.*[tmp;tmp], 2 )
% sys












