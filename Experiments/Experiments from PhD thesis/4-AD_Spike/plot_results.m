% IMPORTANT: first execute run_ADSpike.m !!!
% IMPORTANT: for comparison with MC_Spike first run experiment in folder
%            3-MC_Spike

% - visualize results obtained with AD-Spike
% - save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../../load_AWL_toolbox');

% load data
filename = 'results/res_ADSpike';
load(filename)

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = false;

% set saving directory
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    param.indir = savedir;
    param.outdir = savedir;
end


%% plot AD-Spike representation 

% plot parameters
nrows = 2;
ncols = 2;
plot_id = 1;
h_AD = figure;
markersize = 4;

% margins for subaxis
sv = 0.08;
sh = 0.05;
mb = 0.07;
ml = 0.1;
mr = 0.1;
mt = 0.03;

% plot spike form
factor = mean(res.coefficients);
D_scaled = factor * res.D;
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
plot(t_spike,D_scaled,'b','Linewidth',1.5);
xlim([t_spike(1) 0.1])
ylim([-800 200])        
xlabel('time [s]')
ylabel('\muV')

% plot spike form
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
plot(t_spike,D_scaled,'b','Linewidth',1.5);
xlim([t_spike(1) 1.5])
ylim([-800 200])        
xlabel('time [s]')
ylabel('\muV')
set(gca,'YAxisLocation','right')

% plot coefficients in time
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
coeffs = (res.coefficients/sqrt(sfreq));
plot(t(res.latencies),coeffs,'b+','MarkerSize',markersize);
xlabel('time [s]')
ylabel('correlation coefficients')   
xlim([t(1) t(end)]);

    
% plot dilations in time
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
dilations = res.stretchList(res.stretchInd);
plot(t(res.latencies),dilations,'b+','MarkerSize',markersize);
xlabel('time [s]')
ylabel('spike dilations')
axis tight
ylim([0 2.5])
xlim([t(1) t(end)]);
set(gca,'YAxisLocation','right')

% save
if SAVE_FIGS
    param.width = 10;
    param.height = 6;
    filename = 'spikesAD';
    saveas(h_AD,[savedir filename],'fig')
    fig2pdf(filename,param);
end


%% plot performance comparison MC-Spike vs. AD-Spike 

% IMPORTANT: first run experiment in folder 3-MC_Spike!

% load data
maxK = 8;
filename = 'res_MCSpike';
MC = load(['../3-MC_Spike/results/' filename]);

% define colors 
colors = get(0,'DefaultAxesColorOrder');
colors(8,:) = [1 0.5 0.2];
set(0,'DefaultAxesColorOrder',colors)

% set plot parameters and margins
nrows = 1;
ncols = 3;
plot_id = 1;
sv = 0.07;
sh = 0.07;
mb = 0.12;
ml = 0.06;
mr = 0.02;
mt = 0.03;

% plot fits
h_res = figure;
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
for k=1:8    
    bar(k,mean(MC.res.fits{k}),'FaceColor',colors(k,:))
    hold on 
end
val = mean(res.fits);
h_plot = plot(0:9,val*ones(size(0:9)),'k','LineWidth',2);
legend(h_plot,'AD-Spike','Location','North');

set(gca,'XTick',1:8)
set(gca,'XTickLabel',1:8)
ylim([0 1])
ylabel('average spike fit')
xlabel('number of classes (MC-Spike)')


% plot residual
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
legend_entries = cell(9,1);
for k=1:8
    bar(k,MC.res.residual(k),'FaceColor',colors(k,:));
    legend_entries{k} = ['MC-Spike, K=' num2str(k)];
    hold on 
end
val = norm(res.R)/norm(X);
h_plot = plot(0:9,val*ones(size(0:9)),'k','LineWidth',2);
legend_entries{9} = 'AD-Spike';
legend(legend_entries{:},'Location','North');
set(gca,'XTick',1:8)
set(gca,'XTickLabel',1:8)
ylim([0 1])
ylabel('residual energy')
xlabel('number of classes (MC-Spike)')

% plot number of detected spikes 
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
for k=1:8
    bar(k,length(MC.res.latencies{k}),'FaceColor',colors(k,:))
    hold on 
end
val = length(res.latencies);
h_plot(1) = plot(0:9,val*ones(size(0:9)),'k','LineWidth',2);
legend(h_plot,'AD-Spike','Location','North');
h_plot(2) = plot(0:9,520*ones(size(0:9)),'r','LineWidth',2);
legend(h_plot,'AD-Spike','total number of spikes');
set(gca,'XTick',1:8)
set(gca,'XTickLabel',1:8)
ylim([510 522])
ylabel('detected spikes')
xlabel('number of classes (MC-Spike)')

% save
if SAVE_FIGS
    savedir =  'figures/';
    param.indir = savedir;
    param.outdir = savedir;
    param.width = 10;
    param.height = 4;
    filename = 'spikesAD_comp';
    saveas(h_res,[savedir filename],'fig')
    fig2pdf(filename,param);
end



