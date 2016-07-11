% processes contiguous dataset with multiclass spike learning (MC-Spike)
%
% all representations are saved to a .mat file and can be visualized with 
% script 'plot_results'
%
% ESTIMATED EXECUTION TIME: < 10 minutes 

% load AWL toolbox
run('../../../load_AWL_toolbox');

% set directories for saving results
datadir = '../../data/';
savedir = 'results/';

%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq = 1250;

% load continuous data
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X = cast(X,'double');
t = linspace(0,1/sfreq*(length(X)-1),length(X));

% load markers of isolated spikes
file = ['markers_isolated_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);


%%%%% select isolated spike with max energy for initialization %%%%%%%%%%%%

% spike parameters
t_left = -0.05;     % [seconds] time until negative peak
t_dist = 0.2;       % [seconds] duration of negative spike wave (use this for distance between detected spikes)
t_right = 1.5+t_left; % [seconds] time after negative peak

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
    figure
    plot(t_spike,d)
    title('spike used for initialization')
    xlabel('time [s]')
end


%%%%% learn hierarchical spike representations %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set parameters
par=struct;
par.D=d;                           % initial spike
par.ndist = floor(t_dist*sfreq);   % samples between spike detections
par.ncenter = sum(t_spike<=0);     % samples until max peak (for alignment)
par.iter=10;                       % iterations per representation
par.maxK=5;                        % maximal spikes of representation
par.alpha=0.1;                     % detection threshold (WARNING: calculation becomes slow if alpha < 0.1 !!!)
par.verbose=true;

res=mexMCSpike(X,par);
t_spike = t_spike - t_left;


%%%%% save %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'res_MCSpike'],'res','t_spike');

