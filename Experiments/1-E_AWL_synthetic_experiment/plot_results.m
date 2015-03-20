% IMPORTANT: first run run_synthetic_experiment.m !!!
%
% visualize results calculated by run_synthetic_experiment, saved in .mat
% file results_synthetic_experiment

% set paths from AWL toolbox
run('../../load_AWL_toolbox');

% load the results
resdir = 'results/';
load([resdir 'results_synthetic_experiment'])

% correct the waveform error (better normalization)
results(:,2,:,:) = results(:,2,:,:)/sqrt(2);

% parameters
colors = {'-omagenta','-xgreen','-+cyan','-*blue','->red'};

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = false;

% set directory for saving .fig and .pdf 
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    param.indir = savedir;
    param.outdir = savedir;
end



%% show results, only waveform and residual error

x_axis = [sigma_amps; sigma_lats; SNR; SNR];
x_labels = {'amplitude SD $\sigma_a$', 'latency SD $\sigma_\delta$ [s]','SNR w.r.t. white Gaussian noise [dB]', 'SNR w.r.t. structured noise [dB]'};
y_labels = {'res. error $\varepsilon_X$', 'wf. error $\varepsilon_d$'};
legends = {'PCA','ICA200','ICA10','ICA3','AWL'};
h_comp = figure;

for case_id = 1:ncases
    for crit_id = 1:ncrits
        subaxis(ncases,ncrits,ncrits*(case_id-1)+crit_id, 'Spacing', 0, 'Padding', 0.04, 'MarginBottom', 0.03, 'MarginLeft', 0.01, 'MarginRight', 0.01, 'MarginTop', 0.01)  
        for method_id = 1:nmethods            
            if case_id<3                
                semilogx(x_axis(case_id,:),squeeze(results(method_id,crit_id,case_id,:)),colors{method_id},'MarkerSize',8)                   
            else
                plot(x_axis(case_id,:),squeeze(results(method_id,crit_id,case_id,:)),colors{method_id},'MarkerSize',8)
            end 
            hold on
            drawnow
        end
        xlabel(x_labels{case_id},'fontsize',14,'interpreter','latex')
        ylabel(y_labels{crit_id},'fontsize',14,'interpreter','latex')
        axis tight
        ylim([0,1.5])
        if crit_id == 1 && case_id == 1
            legend(legends,'Location','NorthWest','fontsize',10)
        end
    end
end
if SAVE_FIGS
    param.width = 12;
    param.height = 12;
    filename = 'synth_perf_comp';
    saveas(h_comp,[savedir filename],'fig')
    fig2pdf(filename,param);
end

%% show generated signals

h_sigs = figure;
ntrials = 4;
nrows = 3;
data = {X,X_events,X_train};
y_labels = {'original trials','+struct. noise', '+struct./white noise'};
display(['average SNR is ' num2str(SNR_total) ' dB.'])

for i=1:nrows
    for j=1:ntrials
        subaxis(nrows,ntrials,(i-1)*ntrials+j,'Spacing', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.01, 'MarginTop', 0.02)  
        plot(t,data{i}(:,j),'k')
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
        subaxis(nrows,ncols,(i-1)*ncols+j,'Spacing', 0.02, 'MarginBottom', 0.15, 'MarginLeft', 0.01, 'MarginRight', 0.01, 'MarginTop', 0.02)
        plot(t,data{i}(:,j)/norm(data{i}(:,j)),'k')
        if i==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end
        if j~=1
            set(gca,'YTick',[])
        end
        ylim([-.4,.2])
    end
end

if SAVE_FIGS
    param.width = 12;
    param.height = nrows*3;
    filename = 'synth_dict';
    saveas(h_dicts,[savedir filename],'fig')
    fig2pdf(filename,param);
end
%% show recovered dictionaries

h_dicts = figure;
ncols = K;
data = {D_pca,D_ica200,D_ica10,D_ica3,D_awl};
nrows = length(data);
y_labels = {'PCA', 'ICA200', 'ICA10', 'ICA3', 'AWL'};

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
    end
end
if SAVE_FIGS
    param.width = 12;
    param.height = 6;
    filename = 'synth_rec_dicts';
    saveas(h_dicts,[savedir filename],'fig')
    fig2pdf(filename,param);
end

