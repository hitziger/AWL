% ATTENTION: ESTIMATED EXECUTION TIME: on laptop ~10h

% compare detection accuracies of MC-Spike, AD-Spike, and template matching
%
% - detection on real dataset (epileptiform spikes in LFP-recording)
% - different noise levels (white Gaussian) are added
% - the use of two different templates is compared:
%       - "good" template: average over spikes in noiseless data
%       - noisy template: one example spike directly from noisy data
%
% results are saved to a .mat file and can be visualized with 
% script 'plot_results'
%

% load AWL toolbox
run('../../load_AWL_toolbox');



%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq = 1250;

% load continuous data
datadir = '../data/';
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X = cast(X,'double');
t = linspace(0,1/sfreq*(length(X)-1),length(X));

% load epoch markers (for spike initialization)
file = ['markers_isolated_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);

% load true spike detections for comparison
file = ['markers_all_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
t=t';
http://start.fedoraproject.org/

%%%%% calculate average spike from epochs (for "good" template) %%%%%%%%%%%

% learning settings
t_left = -0.05;     % [seconds] time until max peak
t_dist = 0.2;       % [seconds] duration of peak
t_right = 2+t_left; % [seconds] time after max peak

% average over epochs and normalize
ind = (1:length(t))';
sel = ind(t>(t(markers(1))+t_left) & t < (t(markers(1))+t_right));
t_spike = t(sel) - t(markers(1));
d = zeros(size(t_spike));
sel = sel - markers(1);
for i=1:length(markers)
    d = d + X(sel+markers(i));    
end
d = d/length(markers);
d = d/norm(d);

% plot template spike (used to initialize algorithm)
PLOT = 1;
if PLOT
    figure;
    plot(t_spike,d)
    xlabel('time [s]')
    title('"good" spike template')
end

%%%%% Latency of prominent spike, for initialization with noisy template %%

% determine minimal amplitude in noiseless signal X
[~,minInd] = min(X);

% determine indices of spike occurrence, used later initialize template
% directly on noisy data
ind = (1:length(t))';
init_indices = ind(t>(t(minInd)+t_left) & t < (t(minInd)+t_right));


%%%%% Parameters for all methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ndist = floor(t_dist*sfreq);    % minimal spike-to-spike distance
ncenter = sum(t_spike<=0);      % samples until max peak (for alignment)
ndetects = length(markers_all); % total number of spikes in data 


%%%%% Define white Gaussian noise for different SNRs %%%%%%%%%%%%%%%%%%%%%%
SNR_DB = -40:2:10;
rng(0)
noise = randn(size(X));
noise = noise/norm(noise);
noise_amps = norm(X)./10.^(SNR_DB/20);

% plot examples
figure
for i = 1:2    
    subplot(2,1,i)
    length(noise_amps);
    j = (i-1) * (length(noise_amps)-1) + 1;
    plot(t,X+noise_amps(j)*noise,'r');
    hold on 
    plot(t,X,'b');
    axis tight;
    xlabel('time [s]')
    legend(['SNR: ' num2str(SNR_DB(j))],'original data')
end

%%%%% Struct for results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ninit = 2;              % number of initializations
MC_maxK = 5;            % max representation size for MC-Spike
nmethods = MC_maxK+2;   % number of methods, MC-Spike, AD-Spike, t.m.
nSNR = length(SNR_DB);  % number of noise-levels
detections = zeros(nSNR,ninit,nmethods,ndetects);
%%

%%%%% Detection with MC-Spike %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

% paramter struct
par=struct;
par.ndist = ndist;         % min. spike-to-spike distance
par.ncenter = ncenter;     % samples until max peak, for alignment
par.ndetects = ndetects;   % # of spikes to be detected
par.iter=10;               % iterations per representation
par.maxK=MC_maxK;          % maximal representation size
par.verbose=true;

for i = 22:length(noise_amps)
    X_noisy = X+noise_amps(i)*noise;
    for j=1:2
        if j==1
            fprintf('***** MC-Spike, SNR %g *****\n', SNR_DB(i))
            par.D = d;
        else
            fprintf('***** MC-Spike, SNR %g, noisy init *****\n', SNR_DB(i))
            par.D = X_noisy(init_indices);
            par.D = par.D/norm(par.D);
        end
        res=mexMCSpike(X_noisy,par);
        for k=1:MC_maxK
            detections(i,j,k,:) = res.latencies{k};
        end
        
    end
end


%%%%% Detection with AD-Spike %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paramter struct
par=struct;
par.ndist = ndist;          % min. spike-to-spike distance
par.ncenter = ncenter;      % samples until max peak, for alignment
par.ndetects = ndetects;    % # of spikes to be detected
par.iter=3;                 % iterations 
par.maxstretch = 8;         % maximal relative stretch factor
par.nstretch = 21;        % number of discrete stretches
par.nfactor = 21;           % factor to increase stretch resolution 
par.verbose=true;


for i = 1:length(noise_amps)    
    X_noisy = X+noise_amps(i)*noise;
    for j=1:2
        if j==1
            fprintf('***** AD-Spike, SNR %g *****\n', SNR_DB(i))
            par.D = d;
        else
            fprintf('***** AD-Spike, SNR %g, noisy init *****\n', SNR_DB(i))
            par.D = X_noisy(init_indices);
            par.D = par.D/norm(par.D);
        end
        res=mexADSpike(X_noisy,par);
        detections(i,j,MC_maxK+1,:) = res.latencies;
    end
end

%%%%%  template matching %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par = struct;
par.ndist=ndist;
par.ndetects = ndetects;
par.verbose=false;
for i = 1:length(noise_amps)
    X_noisy = X+noise_amps(i)*noise;    
    for j=1:2        
        if j==1
            fprintf('***** Template Matching, SNR %g *****\n', SNR_DB(i))
            par.D = d;
        else
            fprintf('***** Template Matching, SNR %g, noisy init *****\n', SNR_DB(i))
            par.D = X_noisy(init_indices);
            par.D = par.D/norm(par.D);
        end
        res=mexSpikeTemplateMatching(X_noisy,par);
        detections(i,j,MC_maxK+2,:) = res.latencies;
    end
end


%%%%% save results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
detections = detections + ncenter;
savedir = 'results/';
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'detected_spike_locations'],'detections','SNR_DB');



