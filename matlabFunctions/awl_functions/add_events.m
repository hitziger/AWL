function X = add_events(t,X,param) 
% adds spurious events (real Gabor wavelets) to set of signals
%
% INPUT: 
% t: (vector) time axis of signals in X
% X: (matrix) contains signals as columns
% param: (struct) specifies events:
%   - n: no of affected signals
%   - freq: [f1 f2], min and max frequency of Gabor atoms
%   - energy: [e1 e2], min and max energy of Gabor atoms
%   - events_per_trial; (vector) no. of Gabor atoms for each signal
%
% OUTPUT:
% X: (matrix) signals + spurious events

% parse paramters
if ~(isfield(param,'n'))
    error('n must be set')
end
if ~(isfield(param,'freq'))
    error('freq must be set')
end
if ~(isfield(param,'energy'))
    error('energy must be set')
end
if ~(isfield(param,'events_per_trial'))
    error('events_per_trial must be set')
end
M = size(X,2);
selection = randperm(M);
selection = selection(1:param.n);

par = struct;
par.window_type = 'gauss';
for j=selection
    for k=1:param.events_per_trial(j)
        par.window_sigma = param.sigma(1) + rand*(param.sigma(2) - param.sigma(1));
        par.window_offset = param.offset(1) + rand*(param.offset(2) - param.offset(1));
        par.freq = param.freq(1) + rand*(param.freq(2) - param.freq(1));
        par.sig = 'sine';
        par.amp = 1;
        spurious_atom = define_atom(t,par)';
        energy = param.energy(1) + rand*(param.energy(2) - param.energy(1));
        spurious_atom = energy/norm(spurious_atom) * spurious_atom;
        X(:,j) = X(:,j) + spurious_atom;
    end
end



