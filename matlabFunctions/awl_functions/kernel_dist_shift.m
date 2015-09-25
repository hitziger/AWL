function [distance,perm] = kernel_dist_shift(D1,D2)

% calculates the distance between two sets of kernels, which is shift-,
% sign-, and order-invariant (through use of cross-correlation, absolute
% value, and cycling through all permutations)
%
% D1: matrix, contains kernels as columns
% D2: matrix, contains kernels as columns, same dimensions as D1
%
% distance: calculated distance, scalar
% perm : matching permutation of second set of kernels


if size(D1) ~= size(D2)
    error('Dimensions of input matrix have to match')
end

K = size(D1,2);
dist_mat = zeros(K,K);
for k1=1:K
    for k2=1:K
        dist_mat(k1,k2) = sqrt(1-max(abs(xcorr(D1(:,k1),D2(:,k2)))));
    end
end

P = perms(1:K);
dist_vect = zeros(1,size(P,1));
for p = 1:size(P,1)
    dist_vect(p) = mean(diag(dist_mat(:,P(p,:))));
end
[distance,p] = min(dist_vect);
perm = P(p,:);