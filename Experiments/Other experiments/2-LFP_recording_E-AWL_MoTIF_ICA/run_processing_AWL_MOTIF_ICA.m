% IMPORTANT: make sure to include fastica.m in the search path
%
% script processes epoched spike dataset in two ways:
%
% 1) learn hierarchical representations with the Epoched AWL algorithm,
%       with no. of waveforms increasing from K=1, ...,Kmax.
% 2) for fixed K=Kmax calculate MoTIF and ICA
%
% all representations are saved to a .mat file and can be visualized with 
% script 'plot_results'
%
% ESTIMATED EXECUTION TIME: < 2 minutes 


% load AWL toolbox
run('../../../load_AWL_toolbox');

% load dataset
datadir = '../../data/';
file = 'LFP_data_epoched_125_Hz.mat';
filename = [datadir file];
load(filename)

% determine min and max of data and time
min_X = min(min(X));
max_X = max(max(X));
min_t = t(1);
max_t = t(end);

X_train = X;

%% Hierarchical AWL for different sizes K=1,...,Kmax

Kmax = 5;

% set seed for random number generator
rng(1);

% initialize arrays for different AWL representations
D_awl_all = cell(1,Kmax);
code_awl_all = cell(1,Kmax);
X_awl_all = cell(1,Kmax);
A_awl_mean_all = cell(1,Kmax);
error_awl_all = zeros(1,Kmax);

% set parameters for epoched AWL algorithm
param=struct;
param.nlatencies = 0.2*size(X_train,1);  % allowed latencies
param.align = true;                 % align kernels inside algorithm
param.clean = true;                 % clean unused kernels
param.lambda = 0;                   % l_1 regularization
param.iter = 100;
param.posAlpha = true;              % positivity constraint
param.randomize = false;            % randomize order of training examples
param.verbose = false; 
D = [];

% loop over K to learn increasing representations
for K=1:Kmax   
    display(['learning ' num2str(K) ' atoms'])
    new = randn(size(X,1)+2*param.nlatencies,1);
    param.D=[D,new];
    [D,code,~]=mexEAWL(X_train,param);  
    A_mean = mean(abs(code.A),2);
    X_awl = compose_signals(D,code.A,code.Delta,param.nlatencies);
    D_awl_all{K} = D;
    code_awl_all{K} = code;
    A_awl_mean_all{K} = A_mean;
    X_awl_all{K} = X_awl;
    error_awl_all(K) = mean(sum((X_train-X_awl).^2,1))/mean(sum(X_train.^2,1));        
end
    
% define inner part of extended waveforms
sel_inner = param.nlatencies+1:param.nlatencies+size(X_train,1);

%% E-AWL (non hierarchical)

K=Kmax;

display(['E-AWL, non-hierarchical, learning ' num2str(K) ' atoms'])
new = randn(size(X,1)+2*param.nlatencies,Kmax);
param.D=new;
[D,code,~]=mexEAWL(X_train,param);  
A_mean = mean(abs(code.A),2);
X_awl = compose_signals(D,code.A,code.Delta,param.nlatencies);
D_awl_non_hier = D;
code_awl_non_hier = code;
A_awl_mean_non_hier = A_mean;
X_awl_non_hier = X_awl;
error_awl_non_hier = mean(sum((X_train-X_awl).^2,1))/mean(sum(X_train.^2,1)); 



%% perform MoTIF
K = Kmax;

% make atoms shorter than signals, to allow for jitter
l_atoms = ceil(size(X_train,1)*0.6);
[~,D_short,A_motif] = MoTIF_learning(X_train,l_atoms,K,1,0);

D_motif = zeros(size(X_train,1),K);

% extend the shorter MoTIF atoms
for k=1:K
    [~,ind] = min(D_short(:,k));
    nleft = max(0,ceil(size(X_train,1)/2) - ind);
    D_motif(nleft+1:l_atoms+nleft,k) =  D_short(:,k);
    norm_k = norm(D_motif(:,k));
    D_motif(:,k) = D_motif(:,k) / norm_k;
    A_motif(k,:) = A_motif(k,:) / norm_k;
end

D_motif = D_motif(1:size(X_train,1),:);


%% perform ICA
% ICA using fastICA algorithm (http://research.ics.aalto.fi/ica/fastica/),
[D_ica, A_ica, ~] = fastica(X_train','lastEig',K,'numOfIC',K,'stabilization','on');
D_ica = D_ica';
A_ica = A_ica'; 

% correct signs based on average coefficient
[D_ica,A_ica] = sign_correction_dict(D_ica,A_ica);

% normalize waveforms (scale coefficients correspondingly)
for k=1:K
    normk = norm(D_ica(:,k));
    D_ica(:,k) = D_ica(:,k)/normk;
    A_ica(k,:) = A_ica(k,:) * normk;
end 

% reconstruct signals and calculate error w.r.t. original data
X_ica = D_ica * A_ica;
error_ica = mean(sum((X_train-X_ica).^2,1))/mean(sum(X_train.^2,1)); 

% calculate mean coefficients, establish descending order
% (waveforms+coeffs)
A_ica_mean = mean(abs(A_ica),2);
[A_ica_mean, order] = sort(A_ica_mean,1,'descend');
D_ica = D_ica(:,order);

%% E-AWL (non hierarchical)

K=Kmax;

display(['E-AWL, non-hierarchical, learning ' num2str(K) ' atoms'])
Dinit = zeros(size(X,1)+2*param.nlatencies,Kmax);
Dinit(nleft+1: nleft+size(D_ica,1),:) = D_ica;
param.D=Dinit;
[D,code,~]=mexEAWL(X_train,param);  
A_mean = mean(abs(code.A),2);
X_awli = compose_signals(D,code.A,code.Delta,param.nlatencies);
D_awli_non_hier = D;
code_awli_non_hier = code;
A_awli_mean_non_hier = A_mean;
X_awli_non_hier = X_awli;
error_awli_non_hier = mean(sum((X_train-X_awli).^2,1))/mean(sum(X_train.^2,1)); 


%% save results
savedir = 'results/';
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'results_real_EAWL_MOTIF_ICA'])






