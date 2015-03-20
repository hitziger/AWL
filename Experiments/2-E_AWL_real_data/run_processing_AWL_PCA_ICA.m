% IMPORTANT: make sure to include fastica.m in the search path
%
% script processes epoched spike dataset in two ways:
%
% 1) learn hierarchical representations with the Epoched AWL algorithm,
%       with no. of waveforms increasing from K=1, ...,Kmax.
% 2) for fixed K=Kmax calculate PCA and ICA decomposition
%
% all representations are saved to a .mat file and can be visualized with 
% script 'plot_results'
%
% ESTIMATED EXECUTION TIME: < 2 minutes 


% load AWL toolbox
run('../../load_AWL_toolbox');

% load dataset
datadir = '../data/';
file = 'LFP_data_epoched_125_Hz.mat';
filename = [datadir file];
load(filename)

% determine min and max of data and time
min_X = min(min(X));
max_X = max(max(X));
min_t = t(1);
max_t = t(end);

X_train = X;

%% start with AWL for different sizes K=1,...,Kmax

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
param.mode = 1;
param.nLeft = size(X_train,1)-1;
param.align = true;
param.clean = true;
param.lambda = 0;
param.iter = 100;
param.posAlpha = true;
param.randomize = false; % randomize order of training examples
param.verbose = false;  %
param.initMethod = 'Gaussian';
D = [];

% loop over K to learn increasing representations
for K=1:Kmax   
    display(['learning ' num2str(K) ' atoms'])
    new = randn(size(X,1)+2*param.nLeft,1);
    param.D=[D,new];
    [D,code,~]=mexEAWL(X_train,param);  
    A_mean = mean(abs(code.A),2);
    X_awl = compose_signals(D,code.A,code.Delta,param.nLeft);
    D_awl_all{K} = D;
    code_awl_all{K} = code;
    A_awl_mean_all{K} = A_mean;
    X_awl_all{K} = X_awl;
    error_awl_all(K) = mean(sum((X_train-X_awl).^2,1))/mean(sum(X_train.^2,1));        
end
    
% define inner part of extended waveforms
sel_inner = param.nLeft+1:param.nLeft+size(X_train,1);

%% perform PCA
K = Kmax;

% PCA using singular value decomposition
[U,S,V] = svd(X_train);
D_pca = U(:,1:K);
A_pca = S(1:K,:)*V';

% correct signs based on average coefficient
[D_pca,A_pca] = sign_correction_dict(D_pca,A_pca);

% manual sign correction, purely cosmetic for better comparison with ICA
for k = [3 4 5]
    D_pca(:,k) = -D_pca(:,k); 
    A_pca(k,:) = -A_pca(k,:);
end

% reconstruct signals and calculate error w.r.t. original data
X_pca = D_pca * A_pca;
error_pca = mean(sum((X_train-X_pca).^2,1))/mean(sum(X_train.^2,1));

% calculate mean coefficients
A_pca_mean = mean(abs(A_pca),2);
clear U S V

%% perform ICA
% ICA using fastICA algorithm (http://research.ics.aalto.fi/ica/fastica/),
error_ica = zeros(1,3);
dimReduct = [size(X,2) 10 K];
for j=1:3
    [D_ica, A_ica, ~] = fastica(X_train','lastEig',dimReduct(j),'numOfIC',K,'stabilization','on');
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
    error_ica(j) = mean(sum((X_train-X_ica).^2,1))/mean(sum(X_train.^2,1)); 

    % calculate mean coefficients, establish descending order
    % (waveforms+coeffs)
    A_ica_mean = mean(abs(A_ica),2);
    [A_ica_mean, order] = sort(A_ica_mean,1,'descend');
    D_ica = D_ica(:,order);
    
    eval(['A_ica' num2str(dimReduct(j))  '= A_ica;'])
    eval(['D_ica' num2str(dimReduct(j))  '= D_ica;'])    
end

%% save results
savedir = 'results/';
if ~exist(savedir,'dir')
    mkdir(savedir)
end
save([savedir 'results_real_EAWL_PCA_ICA'])






