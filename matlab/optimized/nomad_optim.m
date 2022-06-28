% Copyright (C) John Atkinson 2017

function [ ahat, khat ] = nomad_optim( a, ahat, k, khat )

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

% DEBUG
%         idx
%         kCols
%         (khatCols-idx+1):(khatCols-(idx-kCols))
%                   1:kCols

        entry = [ (khatCols-idx+1):(khatCols-(idx-kCols)) ; ...
                  1:kCols ];
        % index start, account for elements added on previous iteration
        idxS = idxE+1;
        % index end, account for length of new entry
        idxE = idxS+size(entry,2)-1;
        % add entry to ali
        ali( :, idxS:idxE ) = entry;
    elseif idx > khatCols
        % create entry (See diagram above)

% DEBUG
%         [1:(kCols+khatCols-idx);...
%         (idx+1-khatCols):kCols]

        entry = [ 1:(kCols+khatCols-idx) ; ...
                  (idx+1-khatCols):kCols ];
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
    a_i_j = repmat( a( j, ali(2,:) ), [numDims, 1] );
    kh_al_j = repmat( khat( j, ali(1,:) ), [numDims, 1] );

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

% remove duplicate entries from khat
[khat, ~, oldIdx] = unique( khatNew', 'rows' );
khat = khat';
oldIdx = oldIdx';

% sum coefficients of duplicate entries and store in correct location
% corresponding to whereever the entry in khatNew was moved to
ahat = zeros( size(khat) );
for i = 1:1:length( oldIdx )
    ahat( :, oldIdx(i) ) = ahat( :, oldIdx(i) ) + ahatNew( :, i );
end

% flip to match original structure defined in the math! should have no
% effect on the algorithms functionality
khat = fliplr( khat );
ahat = fliplr( ahat );

end













