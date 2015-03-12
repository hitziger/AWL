function [X,shift,amp] = create_examples(D,t,param,PLOT)

% creates trials (X) as linear combinations of shifted waveforms (D)
%
% INPUT
% D: (matrix) containing waveforms in columns
% t: (vector) time axis of signals to be generated
% param: struct with different parameters
%   - sigmaAmp : standard deviation of waveforms' amplitudes across trials
%   - sigmaT   : standard deviation of waveforms' latencies across trials
%   - M        : number of trials to be generated
%   - positive : if true, negative amplitudes are discarded
% PLOT : if true, plot first trials
%
% OUTPUT
% X: (matrix) contains generated trials in columns
% shift: (matrix) contains latency of each waveform per trial
% amp: (matrix) contains amplitude of each waveform per trial

% parse parameters
if isfield(param,'sigmaAmp')
    sigmaAmp = param.sigmaAmp;
else
    sigmaAmp = 0;
end
if isfield(param,'sigmaT')
    sigmaT = param.sigmaT;
else
    sigmaT = 0;
end
if isfield(param,'M')
    M = param.M;
else
    error('missing field M');
end
if isfield(param,'positive')
    positive = param.positive;
else 
    positive = false;
end


% calculate some secondary parameters
fSamp = 1/(t(2)-t(1));

K = size(D,2);
n = length(t);
nLeft = (size(D,1)-n)/2;

% generate latencies
lat_t = sigmaT*randn(K,M);
lat_n = round(lat_t*fSamp);
for k=1:K
    for m=1:M
        while (lat_n(k,m)>nLeft || lat_n(k,m)<-nLeft)
            lat_n(k,m) = floor(fSamp*sigmaT*randn);
        end
    end
end

% generate amplitudes
amp = 1+sigmaAmp*randn(K,M);
if (positive)
    for k=1:K
        for m=1:M
            while amp(k,m)<0
                amp(k,m) = 1+sigmaAmp*randn;
            end
        end
    end
end

% generate trials
X = zeros(n,M);
for j=1:M
    for i=1:K
        atom = D(nLeft+1+lat_n(i,j):end-nLeft+lat_n(i,j),i);
        X(:,j) = X(:,j) + amp(i,j) * atom;
    end
end
shift = -lat_n;

% plot  first trials if specified
if nargin>3
    if PLOT
        figure;
        for i=1:10
            subplot(2,5,i);
            plot(t,X(:,i));
            xlim([t(1) t(end)]);
            ylim([min(min(X(:,1:10))) max(max(X(:,1:10)))])
            title(['tr. example ' num2str(i) '/' num2str(M) ]);
        end
    end
end

