% IMPORTANT: this script uses the external function fastica.m from the 
% fastICA package (http://research.ics.aalto.fi/ica/fastica/), make sure to
% add its folder to the search path
%
% generate synthetic data and process with MoTIF, ICA, and E-AWL
%
% generated data:
% - D_ext: dictionary matrix, contains three kernels to generate trials X,
% (kernels in D_ext are slightly longer than trials so they can be shifted
% to generate the trials)
% - X:  trial matrix, contains the signals (columns), 
%   each a linear combination of shifts of the kernels
% - X_train = X + noise
% four settings:
% 1) high SNR, small SD of amplitudes and latencies, varying no. of kernels
% 2) high SNR, constant latencies, increasing SDs of amplitudes
% 3) high SNR, small SD of amplitudes, increasing SDs of latencies
% 4) small SD of amplitudes and latencies, increasing SNR (pink noise)
% 
% error measured shift-, sign-, and order-invariant distance epsilon
% between original and calculated kernels
%
% used functions (included): 
% generate_dictionary: generates extended dictionary D_ext
%   -> uses set_up_t, define_atom, which uses make_spike
% create_examples: generates trials using D_ext
% sign_correction_dict: corrects signs of "negatively used" waveforms
% kernel_dist_shift: calculates shift-invariant distance between
%       dictionaries (epsilon)
% compose_signals: calculates reconstructed signals from AWL representation
% mexEAWL: performs epoched AWL
%
% external functions: fastica.m (see top)


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
sigma_lat_def = 0.01;        % default SD of latencies, for 3) and 4)
sigma_amp_range = [-1.5,0]; % range of amp SDs (log10), for 1)
sigma_lat_range = [-3,0];   % range of lat SDs (log10), for 2)
SNR_range = [-15,15];       % range of lat SNRs, for 3) and 4)

% parameters for AWL
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
[D_ext,t_ext,t,next] = generate_dictionary(fsamp, tborder);
D = D_ext(next+1:end-next,:);
K = size(D,2);
for k=1:K
    D(:,k) = D(:,k)/norm(D(:,k));
end
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



%%%%% SET UP STUFF FOR LEARNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nmethods = 4;
kernel_error = zeros(nmethods,ncases,nsimulations);
pink_noise = zeros(length(t),nsignals);
for i=1:nsignals
    pink_noise(:,i) = pinknoise(length(t));
end
parTrials = struct;
parTrials.M = nsignals;
parTrials.positive = true;


%% study increasing dictionary size
nlatencies = 20;
time = cputime;
case_id = 1;
rng(1)
for Kvar = 1:K
    parTrials.sigmaAmp = sigma_amp_def;
    parTrials.sigmaT = sigma_lat_def;
    [X,~,~] = create_examples(D_ext(:,1:Kvar),t,parTrials,false);
    sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));
    sigma_noise = sigma_X/10^(SNR_def/20);
    X_train = X + sigma_noise*pink_noise;
    norm_X = sqrt(sum(sum(X.^2)));    

    % MoTIF
    method_id = 1;
    % make atoms shorter than signals, to allow for jitter
    l_atoms = size(X_train,1)-2*nlatencies;
    [~,D_short,A_motif] = MoTIF_learning(X_train,l_atoms,Kvar,1,0);
    D_motif = zeros(size(X_train,1),Kvar);
    % extend the shorter MoTIF atoms, normalize them
    for k=1:Kvar
        D_motif(nlatencies+1:l_atoms+nlatencies,k) =  D_short(:,k);
        norm_k = norm(D_motif(:,k));
        D_motif(:,k) = D_motif(:,k) / norm_k;
        A_motif(k,:) = A_motif(k,:) / norm_k;
    end
    D_motif = D_motif(1:size(X_train,1),:);
    [D_motif,A_motif] = sign_correction_dict(D_motif,A_motif);
    kernel_error(method_id,case_id,Kvar) = kernel_dist_shift(D(:,1:Kvar),D_motif);    
    
    % ICA     
    method_id = method_id+1;
    [D_ica, A_ica, ~] = fastica(X_train','lastEig',Kvar,'numOfIC',Kvar,'stabilization','on');
    A_ica = A_ica';         
    D_ica = D_ica';     
    [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
    for k=1:Kvar
        normk = norm(D_ica(:,k));
        D_ica(:,k) = D_ica(:,k)/normk;
        A_ica(k,:) = A_ica(k,:) * normk;
    end      
    kernel_error(method_id,case_id,Kvar) = kernel_dist_shift(D(:,1:Kvar),D_ica);
    
    % AWL
    method_id = method_id+1;
    D_awl = [];
    for k=1:Kvar              
        display(['learning ' num2str(k) ' atoms'])
        new = randn(size(X,1)+2*nlatencies,1);
        parAWL.D=[D_awl,new];
        [D_awl,code,~]=mexEAWL(X_train,parAWL); 
    end
    Xrec = compose_signals(D_awl,code.A,code.Delta,nlatencies);
    res = X_train - Xrec;
    D_awl = D_awl(nlatencies+1:end-nlatencies,:);
    for k=1:Kvar
        D_awl(:,k)=D_awl(:,k)/norm(D_awl(:,k));
    end
    kernel_error(method_id,case_id,Kvar) = kernel_dist_shift(D(:,1:Kvar),D_awl);
    
    % AWL-I
    method_id = method_id+1;
    D_awli = [zeros(nlatencies,Kvar); D_ica; zeros(nlatencies,Kvar)];          
    display(['learning ' num2str(Kvar) ' atoms, ICA init'])
    parAWL.D=D_awli;
    [D_awli,code,~]=mexEAWL(X_train,parAWL); 
    Xrec = compose_signals(D_awli,code.A,code.Delta,nlatencies);
    D_awli = D_awli(nlatencies+1:end-nlatencies,:);
    for k=1:Kvar
        D_awli(:,k)=D_awli(:,k)/norm(D_awli(:,k));
    end
    kernel_error(method_id,case_id,Kvar) = kernel_dist_shift(D(:,1:Kvar),D_awli);
    
    % plots
    if 0
        figure
        for k=1:Kvar
            subplot(4,Kvar,k);
            plot(D_motif(:,k));
            subplot(4,Kvar,Kvar+k);
            plot(D_ica(:,k));
            subplot(4,Kvar,2*Kvar+k);
            plot(D_awl(:,k));
            subplot(4,Kvar,3*Kvar+k);
            plot(D_awli(:,k));
        end
    end        
end
elapsed(1) = cputime - time;


%% study varying amplitudes
time = cputime;
case_id = 2;
rng(1)
sigma_amps = logspace(sigma_amp_range(1),sigma_amp_range(2),nsimulations);
for i = 1:length(sigma_amps)
    parTrials.sigmaAmp = sigma_amps(i);
    parTrials.sigmaT = 0;
    [X,~,~] = create_examples(D_ext,t,parTrials,false);
    sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));
    sigma_noise = sigma_X/10^(SNR_def/20);
    X_train = X + sigma_noise*pink_noise;
    norm_X = sqrt(sum(sum(X.^2)));    


    % MoTIF
    method_id = 1;
    % make atoms shorter than signals, to allow for jitter
    l_atoms = size(X_train,1)-2*nlatencies;
    [~,D_short,A_motif] = MoTIF_learning(X_train,l_atoms,K,1,0);
    D_motif = zeros(size(X_train,1),K);
    % extend the shorter MoTIF atoms, normalize them
    for k=1:K
        D_motif(nlatencies+1:l_atoms+nlatencies,k) =  D_short(:,k);        
        norm_k = norm(D_motif(:,k));
        D_motif(:,k) = D_motif(:,k) / norm_k;
        A_motif(k,:) = A_motif(k,:) / norm_k;
    end
    D_motif = D_motif(1:size(X_train,1),:);
    [D_motif,A_motif] = sign_correction_dict(D_motif,A_motif);
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_motif);
    
    
    % ICA     
    method_id = method_id+1;
    [D_ica, A_ica, ~] = fastica(X_train','lastEig',K,'numOfIC',K,'stabilization','on');
    A_ica = A_ica';         
    D_ica = D_ica';     
    [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
    for k=1:K
        normk = norm(D_ica(:,k));
        D_ica(:,k) = D_ica(:,k)/normk;
        A_ica(k,:) = A_ica(k,:) * normk;
    end      
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_ica);
    
    % AWL
    method_id = method_id+1;
    D_awl = [];
    for k=1:K              
        display(['learning ' num2str(k) ' atoms'])
        new = randn(size(X,1)+2*nlatencies,1);
        parAWL.D=[D_awl,new];
        [D_awl,code,~]=mexEAWL(X_train,parAWL); 
    end
    Xrec = compose_signals(D_awl,code.A,code.Delta,nlatencies);
    res = X_train - Xrec;
    D_awl = D_awl(nlatencies+1:end-nlatencies,:);
    for k=1:K
        D_awl(:,k)=D_awl(:,k)/norm(D_awl(:,k));
    end
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_awl);
    
    % AWL-I
    method_id = method_id+1;
    D_awli = [zeros(nlatencies,K); D_ica; zeros(nlatencies,K)];          
    display(['learning ' num2str(K) ' atoms, ICA init'])
    parAWL.D=D_awli;
    [D_awli,code,~]=mexEAWL(X_train,parAWL); 
    Xrec = compose_signals(D_awli,code.A,code.Delta,nlatencies);
    D_awli = D_awli(nlatencies+1:end-nlatencies,:);
    for k=1:K
        D_awli(:,k)=D_awli(:,k)/norm(D_awli(:,k));
    end
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_awli);
    
    % plots
    if 0
        figure
        for k=1:K
            subplot(4,K,k);
            plot(D_motif(:,k));
            subplot(4,K,K+k);
            plot(D_ica(:,k));
            subplot(4,K,2*K+k);
            plot(D_awl(:,k));
            subplot(4,K,3*K+k);
            plot(D_awli(:,k));
        end
        pause
    end
        
end
elapsed(2) = cputime - time;


%% now study varying latencies
rng(1)
case_id = 3;
sigma_lats = logspace(sigma_lat_range(1),sigma_lat_range(2),nsimulations);
for i = 1:length(sigma_lats)
    parTrials.sigmaAmp = sigma_amp_def;
    parTrials.sigmaT = sigma_lats(i);
    [X,~,~] = create_examples(D_ext,t,parTrials,false);
    sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));
    sigma_noise = sigma_X/10^(SNR_def/20);
    X_train = X + sigma_noise*pink_noise;
    norm_X = sqrt(sum(sum(X.^2)));
    
    % MoTIF
    method_id = 1;
    % make atoms shorter than signals, to allow for jitter
    l_atoms = size(X_train,1)-2*nlatencies;
    [~,D_short,A_motif] = MoTIF_learning(X_train,l_atoms,K,1,0);
    D_motif = zeros(size(X_train,1),K);
    % extend the shorter MoTIF atoms, normalize them
    for k=1:K
        D_motif(nlatencies+1:l_atoms+nlatencies,k) =  D_short(:,k);
        norm_k = norm(D_motif(:,k));
        D_motif(:,k) = D_motif(:,k) / norm_k;
        A_motif(k,:) = A_motif(k,:) / norm_k;
    end
    D_motif = D_motif(1:size(X_train,1),:);
    [D_motif,A_motif] = sign_correction_dict(D_motif,A_motif);
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_motif);
     
    % ICA     
    method_id = method_id+1;
    [D_ica, A_ica, ~] = fastica(X_train','lastEig',K,'numOfIC',K,'stabilization','on');
    A_ica = A_ica';         
    D_ica = D_ica';     
    [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
    for k=1:K
        normk = norm(D_ica(:,k));
        D_ica(:,k) = D_ica(:,k)/normk;
        A_ica(k,:) = A_ica(k,:) * normk;
    end      
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_ica);
 
    % AWL
    method_id = method_id+1;
    D_awl = [];
    for k=1:K              
        display(['learning ' num2str(k) ' atoms'])
        new = randn(size(X,1)+2*nlatencies,1);
        parAWL.D=[D_awl,new];
        [D_awl,code,~]=mexEAWL(X_train,parAWL); 
    end
    Xrec = compose_signals(D_awl,code.A,code.Delta,nlatencies);
    res = X_train - Xrec;
    D_awl = D_awl(nlatencies+1:end-nlatencies,:);
    for k=1:K
        D_awl(:,k)=D_awl(:,k)/norm(D_awl(:,k));
    end
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_awl);
    
    % AWL-I
    method_id = method_id+1;
    D_awli = [zeros(nlatencies,K); D_ica; zeros(nlatencies,K)];          
    display(['learning ' num2str(K) ' atoms, ICA init'])
    parAWL.D=D_awli;
    [D_awli,code,~]=mexEAWL(X_train,parAWL); 
    Xrec = compose_signals(D_awli,code.A,code.Delta,nlatencies);
    D_awli = D_awli(nlatencies+1:end-nlatencies,:);
    for k=1:K
        D_awli(:,k)=D_awli(:,k)/norm(D_awli(:,k));
    end
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_awli);
    
    % plots
    if 0
        figure
        for k=1:K
            subplot(4,K,k);
            plot(D_motif(:,k));
            subplot(4,K,K+k);
            plot(D_ica(:,k));
            subplot(4,K,2*K+k);
            plot(D_awl(:,k));
            subplot(4,K,3*K+k);
            plot(D_awli(:,k));
        end
        pause
    end
        
end
    
elapsed(3) = cputime - time;

%% generate default data for noise studies
parTrials.sigmaAmp = sigma_amp_def;
parTrials.sigmaT = sigma_lat_def;
[X,~,~] = create_examples(D_ext,t,parTrials,false);
norm_X = sqrt(sum(sum(X.^2)));
sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));

SNR = linspace(SNR_range(1),SNR_range(2),nsimulations);
sigma_noise = sigma_X./10.^(SNR/20);

%% pink noise

case_id = 4;
rng(1)

for i = 1:length(SNR)
    X_train = X + sigma_noise(i)*pink_noise;
    
    % MoTIF
    method_id = 1;
    % make atoms shorter than signals, to allow for jitter
    l_atoms = size(X_train,1)-2*nlatencies;
    [~,D_short,A_motif] = MoTIF_learning(X_train,l_atoms,K,1,0);
    D_motif = zeros(size(X_train,1),K);
    % extend the shorter MoTIF atoms, normalize them
    for k=1:K
        D_motif(nlatencies+1:l_atoms+nlatencies,k) =  D_short(:,k);
        norm_k = norm(D_motif(:,k));
        D_motif(:,k) = D_motif(:,k) / norm_k;
        A_motif(k,:) = A_motif(k,:) / norm_k;
    end
    D_motif = D_motif(1:size(X_train,1),:);
    [D_motif,A_motif] = sign_correction_dict(D_motif,A_motif);
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_motif);
    
    % ICA     
    method_id = method_id+1;
    [D_ica, A_ica, ~] = fastica(X_train','lastEig',K,'numOfIC',K,'stabilization','on');
    A_ica = A_ica';         
    D_ica = D_ica';     
    [D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
    for k=1:K
        normk = norm(D_ica(:,k));
        D_ica(:,k) = D_ica(:,k)/normk;
        A_ica(k,:) = A_ica(k,:) * normk;
    end      
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_ica);
 
    % AWL
    method_id = method_id+1;
    D_awl = [];
    for k=1:K              
        display(['learning ' num2str(k) ' atoms'])
        new = randn(size(X,1)+2*nlatencies,1);
        parAWL.D=[D_awl,new];
        [D_awl,code,~]=mexEAWL(X_train,parAWL); 
    end
    Xrec = compose_signals(D_awl,code.A,code.Delta,nlatencies);
    res = X_train - Xrec;
    D_awl = D_awl(nlatencies+1:end-nlatencies,:);
    for k=1:K
        D_awl(:,k)=D_awl(:,k)/norm(D_awl(:,k));
    end
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_awl);
    
    % AWL-I
    method_id = method_id+1;
    D_awli = [zeros(nlatencies,K); D_ica; zeros(nlatencies,K)];          
    display(['learning ' num2str(K) ' atoms, ICA init'])
    parAWL.D=D_awli;
    [D_awli,code,~]=mexEAWL(X_train,parAWL); 
    Xrec = compose_signals(D_awli,code.A,code.Delta,nlatencies);
    D_awli = D_awli(nlatencies+1:end-nlatencies,:);
    for k=1:K
        D_awli(:,k)=D_awli(:,k)/norm(D_awli(:,k));
    end
    kernel_error(method_id,case_id,i) = kernel_dist_shift(D,D_awli);
end      
    
elapsed(4) = cputime - time;


%% qualitative comparison, SNR: 5 dB
rng(1)

parTrials.sigmaAmp = sigma_amp_def;
parTrials.sigmaT = 2*sigma_lat_def;
[X,lats,amps] = create_examples(D_ext,t,parTrials,false);
sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));  
norm_amps = sqrt(sum(sum(amps.^2)));

SNR_pink = 5;
sigma_pink = sigma_X./10.^(SNR_pink/20);    
%noise = events + sigma_white*white_noise;
noise = sigma_pink*pink_noise;
X_train = X + noise;

sigma_noise = sqrt(var(reshape(noise,[1 size(noise,1)*size(noise,2)])));
SNR_total = 20*log10(sigma_X/sigma_noise);

% MoTIF
% make atoms shorter than signals, to allow for jitter
l_atoms = size(X_train,1)-2*nlatencies;
[~,D_short,A_motif] = MoTIF_learning(X_train,l_atoms,K,1,0);
D_motif = zeros(size(X_train,1),K);
% extend the shorter MoTIF atoms, normalize them
for k=1:K
    D_motif(nlatencies+1:l_atoms+nlatencies,k) =  D_short(:,k);
    norm_k = norm(D_motif(:,k));
    D_motif(:,k) = D_motif(:,k) / norm_k;
    A_motif(k,:) = A_motif(k,:) / norm_k;
end
D_motif = D_motif(1:size(X_train,1),:);
[D_motif,A_motif] = sign_correction_dict(D_motif,A_motif);
[~,perm] = kernel_dist_shift(D,D_motif);
D_motif = D_motif(:,perm);
A_motif = A_motif(perm,:);

% ICA     
[D_ica, A_ica, ~] = fastica(X_train','lastEig',K,'numOfIC',K,'stabilization','on');
A_ica = A_ica';         
D_ica = D_ica';     
[D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);
for k=1:K
    normk = norm(D_ica(:,k));
    D_ica(:,k) = D_ica(:,k)/normk;
    A_ica(k,:) = A_ica(k,:) * normk;
end      
[~,perm] = kernel_dist_shift(D,D_ica);
D_ica = D_ica(:,perm);
A_ica = A_ica(perm,:);

% AWL
D_awl = [];
for k=1:K              
    display(['learning ' num2str(k) ' atoms'])
    new = randn(size(X,1)+2*nlatencies,1);
    parAWL.D=[D_awl,new];
    [D_awl,code,~]=mexEAWL(X_train,parAWL); 
end
D_awl = D_awl(nlatencies+1:end-nlatencies,:);
for k=1:K
    D_awl(:,k)=D_awl(:,k)/norm(D_awl(:,k));
end
[~,perm] = kernel_dist_shift(D,D_awl);    
D_awl = D_awl(:,perm);

% AWL-I
D_awli = [zeros(nlatencies,K); D_ica; zeros(nlatencies,K)];          
display(['learning ' num2str(K) ' atoms, ICA init'])
parAWL.D=D_awli;
[D_awli,code,~]=mexEAWL(X_train,parAWL); 
D_awli = D_awli(nlatencies+1:end-nlatencies,:);
for k=1:K
    D_awli(:,k)=D_awli(:,k)/norm(D_awli(:,k));
end
[~,perm] = kernel_dist_shift(D,D_awli);
D_awli = D_awli(:,perm);


elapsed(5) = cputime - time;



%% qualitative comparison, SNR: -5 dB

rng(1)

parTrials.sigmaAmp = sigma_amp_def;
parTrials.sigmaT = 2*sigma_lat_def;
[X,lats,amps] = create_examples(D_ext,t,parTrials,false);
sigma_X = sqrt(var(reshape(X,[1 size(X,1)*size(X,2)])));  
norm_amps = sqrt(sum(sum(amps.^2)));

SNR_pink = -5;
sigma_pink = sigma_X./10.^(SNR_pink/20);    
noise = sigma_pink*pink_noise;
X_pink = X + noise;

sigma_noise = sqrt(var(reshape(noise,[1 size(noise,1)*size(noise,2)])));
SNR_total = 20*log10(sigma_X/sigma_noise);

% MoTIF
% make atoms shorter than signals, to allow for jitter
l_atoms = size(X_pink,1)-2*nlatencies;
[~,D_short,A_motif2] = MoTIF_learning(X_pink,l_atoms,K,1,0);
D_motif2 = zeros(size(X_pink,1),K);
% extend the shorter MoTIF atoms, normalize them
for k=1:K
    D_motif2(nlatencies+1:l_atoms+nlatencies,k) =  D_short(:,k);
    norm_k = norm(D_motif2(:,k));
    D_motif2(:,k) = D_motif2(:,k) / norm_k;
    A_motif2(k,:) = A_motif2(k,:) / norm_k;
end
D_motif2 = D_motif2(1:size(X_pink,1),:);
[D_motif2,A_motif2] = sign_correction_dict(D_motif2,A_motif2);
[~,perm] = kernel_dist_shift(D,D_motif2);
D_motif2 = D_motif2(:,perm);

% ICA     
[D_ica2, A_ica2, ~] = fastica(X_pink','lastEig',K,'numOfIC',K,'stabilization','on');
A_ica2 = A_ica2';         
D_ica2 = D_ica2';     
[D_ica2,A_ica2] = sign_correction_dict(D_ica2,A_ica2);
for k=1:K
    normk = norm(D_ica2(:,k));
    D_ica2(:,k) = D_ica2(:,k)/normk;
    A_ica2(k,:) = A_ica2(k,:) * normk;
end      
[~,perm] = kernel_dist_shift(D,D_ica2);
D_ica2 = D_ica2(:,perm);

% AWL
D_awl2 = [];
for k=1:K              
    display(['learning ' num2str(k) ' atoms'])
    new = randn(size(X,1)+2*nlatencies,1);
    parAWL.D=[D_awl2,new];
    [D_awl2,code,~]=mexEAWL(X_pink,parAWL); 
end
Xrec = compose_signals(D_awl2,code.A,code.Delta,nlatencies);
res = X_pink - Xrec;
D_awl2 = D_awl2(nlatencies+1:end-nlatencies,:);
for k=1:K
    D_awl2(:,k)=D_awl2(:,k)/norm(D_awl2(:,k));
end
[~,perm] = kernel_dist_shift(D,D_awl2);    
D_awl2 = D_awl2(:,perm);

% AWL-I
D_awli2 = [zeros(nlatencies,K); D_ica; zeros(nlatencies,K)];          
display(['learning ' num2str(K) ' atoms, ICA init'])
parAWL.D=D_awli2;
[D_awli2,code,~]=mexEAWL(X_pink,parAWL); 
Xrec = compose_signals(D_awli2,code.A,code.Delta,nlatencies);
D_awli2 = D_awli2(nlatencies+1:end-nlatencies,:);
for k=1:K
    D_awli2(:,k)=D_awli2(:,k)/norm(D_awli2(:,k));
end
[~,perm] = kernel_dist_shift(D,D_awli2);
D_awli2 = D_awli2(:,perm);

elapsed(6) = cputime - time;

%% save results

savedir = 'results/';
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'results_synthetic_experiment.mat']);
disp('finished script run_synthetic_experiment')
disp(['total time elapsed: ' num2str(elapsed(6)) ' seconds'])




