function [D,text,t,nleft]= generate_dictionary(f_samp,tborder,par)


%% parameters
rng('default')
if nargin<2
    tborder = 0;
end

K = 3;
t0 = 0;
t1 = 5;
sig = {'sine','spike','saw'};
offset = [0,3.5,0];
%freq = [6.33 0.0105 18.37]; old version to generate synthetic experiment
freq = [6.37 0.0105 2.21];
energy = [1 1 1];
window_type = {'gauss','none','none'};
window_sigma = [0.35 0.02 0];
window_offset = [2 0 0];
window_cutoff = zeros(1,3);

if nargin>2
    if isfield(par,'freq')
        freq = par.freq;
    end
end

% secondary parameters
dt=1/f_samp;
[nleft,t,text] = set_up_t(t0,t1,dt,tborder);

%% define dictionary atoms
param = struct;
D = zeros(length(text),K);
for i=1:K
    param.sig = sig{i};
    param.window_type = window_type{i};
    param.window_sigma = window_sigma(i);
    param.window_offset = window_offset(i);
    param.window_cutoff = window_cutoff(i);
    param.offset = offset(i);
    param.freq = freq(i);
    param.amp = energy(i);
    D(:,i) = define_atom(text,param);
    %param.window_cutoff = param.window_cutoff + 2*sigma_t;
end
% normalize ?
%for i=1:3
%    D(:,i) = D(:,i)/norm(D(:,i));
%end




