function [match_values,perms] = waveform_matching(D,Dapprox)

% matches set of extended waveforms to set of calculated waveforms, 
% compensating for different order and different latencies
% 
% INPUT:
% D: (matrix) contains extended waveforms (normalized) in columns
% Dapprox: (matrix) contains calculated waveforms in columns, possibly in
%   different order than D
%
% OUTPUT: 
% match_values: (vector) contains errors w.r.t. original waveforms
% perms: (vector) permutation describing matching between D, Dapprox


K=size(D,2);
next = size(D,1);
n = size(Dapprox,1);
S = next-n+1;
Dexp = zeros(n,S*K);
for k=1:K
    for s=1:S
        Dexp(:,(k-1)*S+s) = D(s:s+n-1,k);
        Dexp(:,(k-1)*S+s) = Dexp(:,(k-1)*S+s) / norm(Dexp(:,(k-1)*S+s));
    end
end

errors = zeros(size(Dexp,2),size(Dapprox,2));
for k1 = 1:size(Dexp,2)
    for k2 = 1:size(Dapprox,2)
        errors(k1,k2) = norm(Dexp(:,k1) - Dapprox(:,k2));
    end
end


match_values = zeros(1,K);
perms = zeros(1,K);
for k=1:K
    [m,ind] = min(errors); 
    [val,k2] = min(m);
    k1 = ceil(ind(k2)/S);
    match_values(k1) = val;
    perms(k1) = k2;
    errors((k1-1)*S+1:k1*S,:) = Inf;
    errors(:,k2) = Inf;    
end
