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

% system of equations
inSys = cell( 1, 2 );
inSys{1} = 'dx/dt = y^2 - x';
inSys{2} = 'dy/dt = x^2 - y';

M = 2; % dimensions
N = 2; % highest order is 2

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

%% check correct operation of coefficient parser
x = sym( 'x%d', [ M, 1 ], 'real' );
numTerms = (N+1)^M;

% create a multivariate polynomial of order <order> in <dims> variables
% with general coefficients a_i for the i-th term
% tmp = prod( repmat( x, 1, numTerms ).^k );
% p = sum( a.*[tmp;tmp], 2 )
% sys

%% Magic time
zeroCols = zeros( 1, numColumns );
for idx = 1:1:numColumns
    if a(:,idx) == 0
        zeroCols(idx) = 1;
    end
end

a = a(:,~zeroCols);
k = k(:,~zeroCols);

% on first iteration, khat = k and ahat = a
khat = k;
ahat = a;

% get number of rows, constant for a given system (# of dimensions)
numDims = size( k, 1 );
% get # of columns of k and khat
kCols = size( k, 2 );
khatCols = size( khat, 2 );

% ali is a vector of indices, [ alpha ; i ], that is created for fast
% indexing of khat and k. khat is indexed by alpha, and k is indexed by i.
% The indices will retrieve elements of khat and k in such a way that
% mimics sliding k across khat, starting with the last element of khat
% lined up with the first element of k. The entire slide requires the
% kCols*khatCols elements to be accessed, which can also be viewed as N+M-1
% "entries," as shown below:
%
% DIAGRAM OF SLIDE HERE
%
ali = zeros( 2, kCols*khatCols );

% initialize last end index to 0, greatly simplifies logic when
% constructing ali
idxE = 0;
for idx = 1:1:(kCols+khatCols-1)
    if idx < kCols
        % create entry (See diagram above)
        entry = [ (khatCols-idx+1):khatCols ; ...
                  1:idx ];
        % index start, account for elements added on previous iteration
        idxS = idxE+1;
        % index end, account for length of new entry
        idxE = idxS+size(entry,2)-1;
        % add entry to ali
        ali( :, idxS:idxE ) = entry;
    elseif idx <= khatCols
        % create entry (See diagram above)
        entry = [ (khatCols-idx+1):(khatCols-mod(idx,kCols)) ; ...
                  1:kCols ];
        % index start, account for elements added on previous iteration
        idxS = idxE+1;
        % index end, account for length of new entry
        idxE = idxS+size(entry,2)-1;
        % add entry to ali
        ali( :, idxS:idxE ) = entry;
    elseif idx > khatCols
        % create entry (See diagram above)
        entry = [ 1:(2*khatCols-idx) ; ...
                  (idx+1-kCols):kCols ];
        % index start, account for elements added on previous iteration
        idxS = idxE+1;
        % index end, account for length of new entry
        idxE = idxS+size(entry,2)-1;
        % add entry to ali
        ali( :, idxS:idxE ) = entry;
    end
end

% maximum possible number of terms in new khat/ahat (actual number of terms
% is likely much less!)
numColsNew = size( ali, 2 );

% kronecker delta applied to row j of sliding k
d = ones( 1, size( ali, 2 ) );

% preallocate for ahat and khat. cell matrices consume a ton of memory, so
% an array structure is used for both ahatNew and khatNew. Since
% coefficients must be calculated numRows times (one for each
% k_(beta,i) - kroneckerDelta_(beta,j) ), every numColsNew columns of
% ahatNew and khatNew correspond to a slide with a different value of j.
ahatNew = zeros( numDims, size( ali, 2 )*numDims );
khatNew = zeros( numDims, size( ali, 2 )*numDims );

% no reason to loop... can do a crazy one liner with some matrix math
for j = 1:1:numDims
    % get a_(j,i) and k_(alpha,j) values
    a_i_j = repmat( a( j, ali(2,:) ), [2, 1] );
    kh_al_j = repmat( khat( j, ali(1,:) ), [2, 1] );

    % compute a_hat component contained in j:
    % starting index for this value of j
    idxS = 1 + (j-1)*numColsNew;
    % ending index for this value of j
    idxE = j*numColsNew;
    % ahatNew_(L,alpha) = ahat_(L,alpha)*a_(j,i)*khat_(j,alpha)
    ahatNew( :, idxS:idxE ) = ahat(:,ali(1,:)).*a_i_j.*kh_al_j;

    % khatNew_(beta,alpha) = khat_(beta,alpha) + k_(beta,i) - delta(beta,j)
    khatNew( :, idxS:idxE ) = khat(:,ali(1,:)) + k(:,ali(2,:)) - ...
                              [zeros(j-1, 1); 1; zeros(numDims-j, 1)]*d;
end

% remove any columns of ahatNew that are all zeros, and the corresponding
% column of khatNew
nonZeroCols = logical( sum( ahatNew ) );
ahatNew = ahatNew( :, nonZeroCols );
khatNew = khatNew( :, nonZeroCols );

%
[khat, newIdx, oldIdx] = unique( khatNew', 'rows' );
khat = khat';
newIdx = newIdx';
oldIdx = oldIdx';

ahat = zeros( size(khat) );
for i = 1:1:length( oldIdx )
    ahat( :, oldIdx(i) ) = ahat( :, oldIdx(i) ) + ahatNew( :, i );
end

% create a multivariate polynomial of order <order> in <dims> variables
% with general coefficients a_i for the i-th term
% tmp = prod( repmat( x, 1, size(k,2) ).^k );
% p = sum( a.*[tmp;tmp], 2 )
% sys












