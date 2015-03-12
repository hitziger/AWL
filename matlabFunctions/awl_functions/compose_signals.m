function [Xrec] = compose_signals(D,A,Shifts,nLeft)

% compose signals given a shiftable waveform representation
%
% INPUT:
% D: (matrix) contains (extended) waveforms in columns
% A: (matrix) contains coefficients of waveforms (rows) per trials (cols)
% Shifts: (matrix) contains waveform latencies, same dimensions as A
% nLeft: no of sampling points of extension (each side) of waveforms
%
% OUTPUT: 
% Xrec: reconstructed signals

% parameters
K = size(D,2);
nExt = size(D,1);
M = size(A,2);
n = nExt-2*nLeft;
          
% calculate reconstruction
Xrec = zeros(n,M);
Index = nLeft+1-Shifts;
for j=1:M
    for k=1:K
        if A(k,j)
            Xrec(:,j) = Xrec(:,j) + D(Index(k,j):Index(k,j)-1+n,k)*A(k,j);
        end
    end
end
 