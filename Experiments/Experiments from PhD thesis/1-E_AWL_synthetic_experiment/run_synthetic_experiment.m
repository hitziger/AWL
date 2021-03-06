% IMPORTANT: this script uses the external function fastica from the 
% fastICA package (http://research.ics.aalto.fi/ica/fastica/), make sure to
% add its folder to the search path
%
% generate synthetic data and process with PCA, ICA, and E-AWL
%
% generated data:
% - D_ext: extended dictionary matr., contains three waveforms (columns)
% - X:  trial matrix, contains the signals (columns), 
%   each a linear combination of shifts of the waveforms
% - X_train = X + noise
% four settings:
% 1) high SNR, constant latencies, increasing SD's of amplitudes
% 2) high SNR, small SD of amplitudes, increasing SD's of latencies
% 3) small SD of amplitudes and latencies, increasing SNR (white noise)
% 4) small SD of amplitudes and latencies, increasing SNR (struct. noiset)
% two error measures:
% 1) RMS error between original and learned waveforms (after matching and
% aligning)
% 2) RMS error between original and learned amplitudes
%
% used functions (included): 
% generate_dictionary: generates extended dictionary D_ext
%   -> uses set_up_t, define_atom, which uses make_spike
% create_examples: generates trials using D_ext
% sign_correction_dict: corrects signs of "negatively used" waveforms
% waveform_matching: matches calculated waveforms with original ones,
%   compensates for arbitrary order of waveforms and different latencies
% compose_signals: calculates reconstructed signals from AWL representation
% mexEAWL: performs epoched AWL 


% load AWL toolbox
run('../../../load_AWL_toolbox');

%%%%% SETTING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% general parameters 
fsamp = 100;            % sampling rate in Hz
tborder = 5;            % maximal shift in seconds (time window is [0 5])
nsignals = 200;         % number of signals

% parameters for processing
ncases = 4;                 % see the four settings described above
nsimulations = 20;          % number of simulations per case 1)-4)
SNR_def = 10;               % default SNR for 1) and 2)
sigma_amp_def = 0.3;        % default SD of amplitudes, for 2)-4)
sigma_lat_def = 0.01;       % default SD of latencies, for 3) and 4)
sigma_amp_range = [-1.5,0]; % range of amp SDs (log10), for 1)
sigma_lat_range = [-3,0];   % range of lat SDs (log10), for 2)
SNR_range = [-15,15];       % range of lat SNRs, for 3) and 4)
pca_preprocessing = false;  % preprocess ICA with PCA

% parameters for AWL
nattempts = 3;              % no of repetitions to prevent bad local min
parAWL = struct;
parAWL.iter = 100;          % no of iterations
parAWL.randomize = false;   % randomize order of training examples
parAWL.verbose = false;      % verbose mode
parAWL.clean = true;        % clean unsued atoms
parAWL.align = true;        % align atoms w.r.t. mean shift
parAWL.initMethod = 'Gaussian';  % initialization method 
parAWL.posAlpha = true;     % positivity constraint on coefficients


%%%%% GENERATING DICTIONARY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate dictionary (D_ext contains longer waveforms, for shifting)
% waveforms are defined in function generate_dictionary
[D_ext,t_ext,t,nleft] = generate_dictionary(fsamp, tborder);
D = D_ext(nleft+1:end-nleft,:);
PLOT_DICT = false;
if (PLOT_DICT)
    figure;
    K = size(D_ext,2);
    for i=1:K
        subplot(K,1,i)
        plot(t_ext,D_ext(:,i));
        ylim([min(min(D_ext)) max(max(D_ext))]);
        hold on
        plot([t(1) t(1)],[min(min(D_ext)) max(max(D_ext))],'r');
        plot([t(end) t(end)],[min(min(D_ext)) max(max(D_ext))],'r');
        title(['Atom' num2str(i)]);
        xlabel('time [s]');
    end
end
K = size(D,2);


%%%%% SET UP STUFF FOR LEARNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nmethods = 5;
ncrits = 2;
results = zeros(nmethods,ncrits,ncases,nsimulations);
errors = zeros(1,nattempts);                     
match_temp = zeros(1,nattempts);
res_temp = zeros(1,nattempts); 
gauss_noise = randn(1,length(t)*nsignals);
gauss_noise = gauss_noise - mean(gauss_noise);
gauss_noise = gauss_noise / sqrt(var(gauss_noise));
gauss_noise = reshape(gauss_noise, [length(t),nsignals]);
parTrials = struct;
parTrials.M = nsignals;
parTrials.positive = true;

% dimensionality reductions for ICA
dimReduct = [nsignals, 10, K];

%% study varying amplitudes
nleft_orig = nleft;
nleft_long = 20;
nleft_short = nleft_long;
nleft = nleft_short;
time = cputime;
case_id = 1;
rng(1)
sigma_amps = logspace(sigma_amp_range(1),sigma_amp_range(2),nsimulations);
for i = 1:length(sigma_amps)
    parTrials.sigmaAmp = sigma_amps(i);
    parTrials.sigmaT = 0;
    [X,~,~] = create_examples(D_ext,t,parTrials,false);
    sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));
    sigma_noise = sigma_X/10^(SNR_def/20);
    X_train = X + sigma_noise*gauss_noise;
    norm_X = sqrt(sum(sum(X.^2)));    

    % PCA
    method_id = 1;
    [U,S,V] = svd(X_train);    
    D_pca = U(:,1:K);
    A_pca = S(1:K,:)*V';
    [D_pca,A_pca] = sign_correction_dict(D_pca,A_pca);
    [matches_pca,~] = waveform_matching(D_ext,D_pca);
    results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_pca*A_pca).^2)))/norm_X;
    results(method_id,2,case_id,i) = mean(matches_pca);
    
    % ICA 
    for j=1:3
        method_id = method_id+1;
        [D_ica, A_ica, ~] = fastica(X_train','lastEig',dimReduct(j),'numOfIC',K,'stabilization','on');
        A_ica = A_ica';         
        D_ica = D_ica';     
        [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
        for k=1:K
            normk = norm(D_ica(:,k));
            D_ica(:,k) = D_ica(:,k)/normk;
            A_ica(k,:) = A_ica(k,:) * normk;
        end      
        [matches_ica,~] = waveform_matching(D_ext,D_ica);
        results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_ica*A_ica).^2)))/norm_X;
        results(method_id,2,case_id,i) = mean(matches_ica);
    end
    
    % AWL
    method_id = method_id+1;
    for l=1:nattempts
        D_awl = [];
        for k=1:K              
            display(['learning ' num2str(k) ' atoms'])
            new = randn(size(X,1)+2*nleft,1);
            parAWL.D=[D_awl,new];
            [D_awl,code,~]=mexEAWL(X_train,parAWL); 
        end
        Xrec = compose_signals(D_awl,code.A,code.Delta,nleft);
        res = X_train - Xrec;
        errors(l) = sum(sum(res.^2));
        D_awl = D_awl(nleft+1:end-nleft,:);
        [matches_awl,~] = waveform_matching(D_ext,D_awl);               
        match_temp(l) = mean(matches_awl);
        res_temp(l) = sqrt(sum(sum((X-Xrec).^2)))/norm_X;
        if 0
            figure
            for k=1:K
                subplot(3,3,k)
                plot(t,D_pca(:,k))
                subplot(3,3,k+K)
                plot(t,D_ica(:,k))
                subplot(3,3,k+2*K)
                plot(t,D_awl(:,k))
            end
            drawnow
            sigma_amps(i)
            pause
        end
    end
    [~,ind]=min(errors);
    results(method_id,1,case_id,i) = res_temp(ind);
    results(method_id,2,case_id,i) = match_temp(ind);
end
elapsed(1) = cputime - time;


%% now study varying latencies
%nleft = nleft_orig;
rng(1)
nleft = nleft_long;
case_id = 2;
sigma_lats = logspace(sigma_lat_range(1),sigma_lat_range(2),nsimulations);
for i = 1:length(sigma_lats)
    parTrials.sigmaAmp = sigma_amp_def;
    parTrials.sigmaT = sigma_lats(i);
    [X,~,~] = create_examples(D_ext,t,parTrials,false);
    sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));
    sigma_noise = sigma_X/10^(SNR_def/20);
    X_train = X + sigma_noise*gauss_noise; 
    norm_X = sqrt(sum(sum(X.^2)));

    % PCA
    method_id = 1;
    [U,S,V] = svd(X_train);    
    D_pca = U(:,1:K);
    A_pca = S(1:K,:)*V';
    [D_pca,A_pca] = sign_correction_dict(D_pca,A_pca);
    [matches_pca,~] = waveform_matching(D_ext,D_pca);
    results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_pca*A_pca).^2)))/norm_X;
    results(method_id,2,case_id,i) = mean(matches_pca);
    
    % ICA   
    for j=1:3
        method_id = method_id+1;
        [D_ica, A_ica, ~] = fastica(X_train','lastEig',dimReduct(j),'numOfIC',K,'stabilization','on');
        A_ica = A_ica';     
        D_ica = D_ica';     
        [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
        for k=1:K
            normk = norm(D_ica(:,k));
            D_ica(:,k) = D_ica(:,k)/normk;
            A_ica(k,:) = A_ica(k,:) * normk;
        end      
        [matches_ica,~] = waveform_matching(D_ext,D_ica);
        results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_ica*A_ica).^2)))/norm_X;     
        results(method_id,2,case_id,i) = mean(matches_ica);          
    end  
      
    % AWL
    method_id = method_id+1;
    for l=1:nattempts
        D_awl = [];
        for k=1:K              
            display(['learning ' num2str(k) ' atoms'])
            new = randn(size(X,1)+2*nleft,1);
            parAWL.D=[D_awl,new];
            [D_awl,code,~]=mexEAWL(X_train,parAWL); 
        end
        Xrec = compose_signals(D_awl,code.A,code.Delta,nleft);
        res = X_train - Xrec;
        errors(l) = sum(sum(res.^2));
        D_awl = D_awl(nleft+1:end-nleft,:);
        [matches_awl,~] = waveform_matching(D_ext,D_awl);               
        match_temp(l) = mean(matches_awl); 
        res_temp(l) = sqrt(sum(sum((X-Xrec).^2)))/norm_X;
        if 0
            figure
            for k=1:K
                subplot(3,3,k)
                plot(t,D_pca(:,k))
                subplot(3,3,k+K)
                plot(t,D_ica(:,k))
                subplot(3,3,k+2*K)
                plot(t,D_awl(:,k))
            end
            drawnow
            sigma_lats(i)
            pause
        end
    end
    [~,ind]=min(errors);
    results(method_id,1,case_id,i) = res_temp(ind);
    results(method_id,2,case_id,i) = match_temp(ind);
end
    
elapsed(2) = cputime - time;

%% generate default data for noise studies
nleft = nleft_short;
parTrials.sigmaAmp = sigma_amp_def;
parTrials.sigmaT = sigma_lat_def;
[X,~,~] = create_examples(D_ext,t,parTrials,false);
norm_X = sqrt(sum(sum(X.^2)));
sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));

SNR = linspace(SNR_range(1),SNR_range(2),nsimulations);
sigma_noise = sigma_X./10.^(SNR/20);

% change nleft for this purpose
%nleft = nleft_short;

%% white noise

case_id = 3;
rng(1)

for i = 1:length(SNR)
    X_train = X + sigma_noise(i)*gauss_noise;
    
    % PCA
    method_id = 1;
    [U,S,V] = svd(X_train);    
    D_pca = U(:,1:K);
    A_pca = S(1:K,:)*V';
    [D_pca,A_pca] = sign_correction_dict(D_pca,A_pca);
    [matches_pca,~] = waveform_matching(D_ext,D_pca);
    results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_pca*A_pca).^2)))/norm_X;
    results(method_id,2,case_id,i) = mean(matches_pca);
    
    % ICA
    for j=1:3
        method_id = method_id+1;
        [D_ica, A_ica, ~] = fastica(X_train','lastEig',dimReduct(j),'numOfIC',K,'stabilization','on');
        A_ica = A_ica';     
        D_ica = D_ica';     
        [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
        for k=1:K
            normk = norm(D_ica(:,k));
            D_ica(:,k) = D_ica(:,k)/normk;
            A_ica(k,:) = A_ica(k,:) * normk;
        end      
        [matches_ica,~] = waveform_matching(D_ext,D_ica);
        results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_ica*A_ica).^2)))/norm_X;
        results(method_id,2,case_id,i) = mean(matches_ica);
    end      
     
    % AWL
    method_id = method_id+1;
    for l=1:nattempts
        D_awl = [];
        for k=1:K              
            display(['learning ' num2str(k) ' atoms'])
            new = randn(size(X,1)+2*nleft,1);
            parAWL.D=[D_awl,new];
            [D_awl,code,~]=mexEAWL(X_train,parAWL); 
        end
        Xrec = compose_signals(D_awl,code.A,code.Delta,nleft);
        res = X_train - Xrec;
        errors(l) = sum(sum(res.^2));
        D_awl = D_awl(nleft+1:end-nleft,:);
        [matches_awl,~] = waveform_matching(D_ext,D_awl);               
        match_temp(l) = mean(matches_awl);
        res_temp(l) = sqrt(sum(sum((X-Xrec).^2)))/norm_X;
    end
    [~,ind]=min(errors);
    results(method_id,1,case_id,i) = res_temp(ind);
    results(method_id,2,case_id,i) = match_temp(ind);
end
    
    
elapsed(3) = cputime - time;
%% now study structured noise 

case_id = 4;
rng(1)
% create random events
parEv = struct;
parEv.n=size(X,2);
max_events_per_trial = 3;
events_per_trial=floor((max_events_per_trial+1)*rand(1,parEv.n));
parEv.events_per_trial = events_per_trial;
parEv.offset = [t(1) t(end)];
parEv.energy = [0, 1];
parEv.freq = [0.1 30];
parEv.sigma = [0.01 0.2];
events = zeros(size(X));
events = add_events(t,events,parEv);
mean_events = mean(reshape(events,[1 size(events,1)*size(events,2)]));
events = events - mean_events;
sigma_events = sqrt(sum(sum(events.^2))/(size(events,1)*size(events,2)-1));
events_white = events / sigma_events;

for i = 1:length(SNR)
    events = sigma_noise(i) * events_white + mean_events;
    X_train = X + events;

     % PCA
    method_id = 1;
    [U,S,V] = svd(X_train);    
    D_pca = U(:,1:K);
    A_pca = S(1:K,:)*V';
    [D_pca,A_pca] = sign_correction_dict(D_pca,A_pca);
    [matches_pca,~] = waveform_matching(D_ext,D_pca);
    results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_pca*A_pca).^2)))/norm_X;
    results(method_id,2,case_id,i) = mean(matches_pca);
    
    % ICA
    for j=1:3
        method_id = method_id+1;
        [D_ica, A_ica, ~] = fastica(X_train','lastEig',dimReduct(j),'numOfIC',K,'stabilization','on');
        A_ica = A_ica';     
        D_ica = D_ica';     
        [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
        for k=1:K
            normk = norm(D_ica(:,k));
            D_ica(:,k) = D_ica(:,k)/normk;
            A_ica(k,:) = A_ica(k,:) * normk;
        end      
        [matches_ica,~] = waveform_matching(D_ext,D_ica);
        results(method_id,1,case_id,i) = sqrt(sum(sum((X-D_ica*A_ica).^2)))/norm_X;
        results(method_id,2,case_id,i) = mean(matches_ica);
    end      
     
    % AWL
    method_id = method_id+1;
    for l=1:nattempts
        D_awl = [];
        for k=1:K              
            display(['learning ' num2str(k) ' atoms'])
            new = randn(size(X,1)+2*nleft,1);
            parAWL.D=[D_awl,new];
            [D_awl,code,~]=mexEAWL(X_train,parAWL); 
        end
        Xrec = compose_signals(D_awl,code.A,code.Delta,nleft);
        res = X_train - Xrec;
        errors(l) = sum(sum(res.^2));
        D_awl = D_awl(nleft+1:end-nleft,:);
        [matches_awl,~] = waveform_matching(D_ext,D_awl);               
        match_temp(l) = mean(matches_awl);
        res_temp(l) = sqrt(sum(sum((X-Xrec).^2)))/norm_X;
    end
    [~,ind]=min(errors);
    results(method_id,1,case_id,i) = res_temp(ind);
    results(method_id,2,case_id,i) = match_temp(ind);
end    
elapsed(4) = cputime - time;

%% one case with default values
rng(1)

parTrials.sigmaAmp = sigma_amp_def;
parTrials.sigmaT = 2*sigma_lat_def;
[X,lats,amps] = create_examples(D_ext,t,parTrials,false);
sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));  
norm_amps = sqrt(sum(sum(amps.^2)));

SNR_events = 2.2;    
sigma_events = sigma_X./10.^(SNR_events/20);
events = sigma_events * events_white + mean_events;
X_events = X + events;

SNR_gauss = -1;
sigma_gauss = sigma_X./10.^(SNR_gauss/20);    
noise = events + sigma_gauss*gauss_noise;
X_train = X + noise;

sigma_noise = sqrt(var(reshape(noise,[1 size(noise,1)*size(noise,2)])));
SNR_total = 20*log10(sigma_X/sigma_noise);
    
% PCA
[U,S,V] = svd(X_train);    
D_pca = U(:,1:K);
A_pca = S(1:K,:)*V';
[D_pca,A_pca] = sign_correction_dict(D_pca,A_pca);
[matches_pca,perms_pca] = waveform_matching(D_ext,D_pca);
A_pca = A_pca(perms_pca,:);
D_pca = D_pca(:,perms_pca);

% ICA
for j=1:3
    [D_ica, A_ica, ~] = fastica(X_train','lastEig',dimReduct(j),'numOfIC',K,'stabilization','on');
    A_ica = A_ica';     
    D_ica = D_ica';     
    [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
    for k=1:K
        normk = norm(D_ica(:,k));
        D_ica(:,k) = D_ica(:,k)/normk;
        A_ica(k,:) = A_ica(k,:) * normk;
    end      
    [matches_ica,perms_ica] = waveform_matching(D_ext,D_ica);
    eval(['A_ica' num2str(dimReduct(j))  '= A_ica(perms_ica,:);'])
    eval(['D_ica' num2str(dimReduct(j))  '= D_ica(:,perms_ica);'])    
end      

% AWL
D_awl_temp = cell(1,nattempts);
A_awl_temp = cell(1,nattempts);
T_awl_temp = cell(1,nattempts);
for l=1:nattempts
    D_awl = [];
    for k=1:K              
        display(['learning ' num2str(k) ' atoms'])
        new = randn(size(X,1)+2*nleft,1);
        parAWL.D=[D_awl,new];
        [D_awl,code,~]=mexEAWL(X_train,parAWL); 
    end
    Xrec = compose_signals(D_awl,code.A,code.Delta,nleft);
    res = X_train - Xrec;
    errors(l) = sum(sum(res.^2));
    A_awl = code.A;
    T_awl = code.Delta;
    D_awl = D_awl(nleft+1:end-nleft,:);
    [~,perms_awl] = waveform_matching(D_ext,D_awl);    
    A_awl = A_awl(perms_awl,:);
    T_awl = T_awl(perms_awl,:);
    D_awl = D_awl(:,perms_awl);
    D_awl_temp{l} = D_awl;
    A_awl_temp{l} = A_awl;
    T_awl_temp{l} = T_awl;
end
[~,ind]=min(errors);
D_awl = D_awl_temp{ind};
A_awl = A_awl_temp{ind};
T_awl = T_awl_temp{ind};


elapsed(5) = cputime - time;
%% save results

savedir = 'results/';
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'results_synthetic_experiment.mat']);
disp('finished script run_synthetic_experiment')
disp(['total time elapsed: ' num2str(elapsed(5)) ' seconds'])




