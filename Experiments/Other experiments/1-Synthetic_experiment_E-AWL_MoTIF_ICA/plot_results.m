% IMPORTANT: first run run_synthetic_experiment.m !!!
%
% visualize results calculated by run_synthetic_experiment, saved in .mat
% file results_synthetic_experiment

% set paths from AWL toolbox
run('../../../load_AWL_toolbox');

% load the results
resdir = 'results/';
load([resdir 'results_synthetic_experiment'])
%%
% parameters
colors = {'-xgreen','-omagenta','-<cyan','-*blue','->red'};

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = true;

% set directory for saving .fig and .pdf 
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    param.indir = savedir;
    param.outdir = savedir;
end

%% show kernel error


x_axis = [sigma_amps;sigma_amps; sigma_lats; SNR; SNR];
titles = {'Increasing number of kernels', 'Variable amplitudes (const. latencies)','Variable latencies','Pink noise'};
x_labels = {'number of kernels K','std. deviation $\sigma_a$', 'std. deviation $\sigma_\delta$ [s]','SNR [dB]'};
y_labels = {'kernel error $\varepsilon$'};
legends = {'MoTIF','ICA','E-AWL','ICA+E-AWL'};
h_comp = figure;

for case_id = 1:ncases    
    subaxis(2,2,case_id, 'SpacingVertical', 0.18,'SpacingHorizontal', 0.08, 'Padding', 0, 'MarginBottom', 0.1, 'MarginLeft', 0.08, 'MarginRight', 0.02, 'MarginTop', 0.1)  
    for method_id = 1:nmethods
        if case_id<2
            plot(1:K,squeeze(kernel_error(method_id,case_id,1:K)),colors{method_id},'MarkerSize',8)
        elseif case_id<4                
            semilogx(x_axis(case_id,:),squeeze(kernel_error(method_id,case_id,:)),colors{method_id},'MarkerSize',8)                   
        else
            plot(x_axis(case_id,:),squeeze(kernel_error(method_id,case_id,:)),colors{method_id},'MarkerSize',8)
        end 
        hold on
        drawnow
    end
    title(titles{case_id},'fontsize',14,'interpreter','latex')
    xlabel(x_labels{case_id},'fontsize',14,'interpreter','latex')
    ylabel(y_labels{1},'fontsize',14,'interpreter','latex')
    axis tight
    ylim([0,1])
    if case_id == 1
        legend(legends,'Location','NorthWest','fontsize',10)
    end    
end
if SAVE_FIGS
    param.width = 8;
    param.height = 4.5;
    filename = 'synth_perf_comp';
    saveas(h_comp,[savedir filename],'fig')
    fig2pdf(filename,param);
end

%% show generated signals

h_sigs = figure;
ntrials = 3;
nrows = 3;
data = {X,X_train,X_pink};
y_labels = {'original trials','noisy (SNR: 5 dB)', 'noisy (SNR: -5dB)'};
display(['average SNR is ' num2str(SNR_total) ' dB.'])

for i=1:nrows
    for j=1:ntrials
        subaxis(nrows,ntrials,(i-1)*ntrials+j,'Spacing', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.01, 'MarginTop', 0.05)  
        plot(t,data{i}(:,j),'k')
        if i==1
            title(['trial ' num2str(j)],'fontsize',14)
        end
        if i==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end
        if j==1
            ylabel(y_labels{i},'fontsize',14);
        else
            set(gca,'YTick',[])
        end
        ylim([-0.7 0.5])
    end
end
if SAVE_FIGS
    param.width = 12;
    param.height = 6;
    filename = 'synth_gen_trials';
    saveas(h_sigs,[savedir filename],'fig')
    fig2pdf(filename,param);
end

%% show original dictionary

h_dicts = figure;
ncols = K;
data = {D};
nrows = length(data);

for i=1:nrows
    for j=1:ncols
        subaxis(nrows,ncols,(i-1)*ncols+j,'Spacing', 0.02, 'MarginBottom', 0.15, 'MarginLeft', 0.03, 'MarginRight', 0.01, 'MarginTop', 0.1)
        plot(t,data{i}(:,j)/norm(data{i}(:,j)),'k')
        if i==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end
        if j~=1
            set(gca,'YTick',[])
        end
        axis tight
        ylim([-.4,.2])
        title(['Kernel ' num2str(j)],'fontsize',14)
    end
end

if SAVE_FIGS
    param.width = 12;
    param.height = nrows*3;
    filename = 'synth_dict';
    saveas(h_dicts,[savedir filename],'fig')
    fig2pdf(filename,param);
end
%% show recovered dictionaries (high SNR 5 db)

% correct sign for better visualization
for k=1:K
    if max(D_motif(:,k))>-min(D_motif(:,k))
        D_motif(:,k) = -D_motif(:,k);
    end
end
h_dicts = figure;
ncols = K;
data = {D_motif,D_ica,D_awl,D_awli};
nrows = length(data);
y_labels = {'MoTIF', 'ICA', 'E-AWL','ICA + E-AWL'};

for i=1:nrows
    for j=1:ncols
        subaxis(nrows,ncols,(i-1)*ncols+j,'Spacing', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.01, 'MarginTop', 0.02)
        plot(t,data{i}(:,j)/norm(data{i}(:,j)),'k')
        if i==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end
        if j==1
            ylabel(y_labels{i},'fontsize',14);
        else
            set(gca,'YTick',[])
        end
        ylim([-.4,.2])
        xlim([0.2,4.8])
    end
end
if SAVE_FIGS
    param.width = 12;
    param.height = 6;
    filename = 'synth_rec_dicts_high_SNR';
    saveas(h_dicts,[savedir filename],'fig')
    fig2pdf(filename,param);
end


%% show recovered dictionaries

% correct sign for better visualization
for k=1:K
    if max(D_motif2(:,k))>-min(D_motif2(:,k))
        D_motif2(:,k) = -D_motif2(:,k);
    end
end

h_dicts = figure;
ncols = K;
data = {D_motif2,D_ica2,D_awl2,D_awli2};
nrows = length(data);
y_labels = {'MoTIF', 'ICA', 'E-AWL','ICA + E-AWL'};

for i=1:nrows
    for j=1:ncols
        subaxis(nrows,ncols,(i-1)*ncols+j,'Spacing', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.01, 'MarginTop', 0.02)
        plot(t,data{i}(:,j)/norm(data{i}(:,j)),'k')
        if i==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end
        if j==1
            ylabel(y_labels{i},'fontsize',14);
        else
            set(gca,'YTick',[])
        end
        ylim([-.4,.2])
        xlim([0.2,4.8])
    end
end
if SAVE_FIGS
    param.width = 12;
    param.height = 6;
    filename = 'synth_rec_dicts_low_SNR';
    saveas(h_dicts,[savedir filename],'fig')
    fig2pdf(filename,param);
end

