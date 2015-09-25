% processes contiguous dataset with C-AWL, for one kernel with adaptive
% duration (dilation-invariant)
%
% learned spike representation is saved to .mat, and can be visualized with 
% script 'plot_results'
%
% ESTIMATED EXECUTION TIME: < 10 minutes 


% load AWL toolbox
run('../../../load_AWL_toolbox');

%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq = 1250;

% load continuous data
datadir = '../../data/';
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X = cast(X,'double');
t = linspace(0,1/sfreq*(length(X)-1),length(X));

% load markers (for spike initialization)
file = ['markers_isolated_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);


%%%%% calculate average spike from epochs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% spike parameters
t_left = -0.05;     % [seconds] time until negative peak
t_dist = 0.2;       % [seconds] duration of negative spike wave
t_right = 2+t_left; % [seconds] time after negative peak

% select spike with maximal energy
ind = (1:length(t))';
sel = ind(t>(t(markers(1))+t_left) & t < (t(markers(1))+t_right));
t_spike = t(sel) - t(markers(1));
d = zeros(size(t_spike));
sel = sel - markers(1);
ind = 0;
val = 0;
for i=1:length(markers)
    current_val = norm(X(sel+markers(i)));
    if (current_val > val)
        ind = i;
        val = current_val;
    end        
end
d = X(sel+markers(ind));

% plot template spike (used to initialize algorithm)
PLOT = 1;
if PLOT
    figure;
    plot(t_spike,d)
end

%%%%% spike learning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par=struct;
par.D = d;                          % kernel initialization
par.ndist=ceil(sfreq*t_dist);       % samples between detected kernels (corr. to 0.2 seconds)
par.ncenter = sum(t_spike<=0);      % samples until negative peak, for alignment
par.iter=5;                         % total number of iterations
par.maxstretch=8;                   % maximal relative stretch between dilations
par.nstretch=61;                    % number of initial dilations
par.nfactor = 11;                   % factor for refining dilations (multi-resolution approach)
par.alpha=0.1;                      % detection threshold 
par.verbose=true;

% run C-AWL for one kernel with adaptive duration (AD)
res=mexADSpike(X,par);

% save results
savedir = 'results/';
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'res_ADSpike'],'res','t_spike','t','sfreq','X');

