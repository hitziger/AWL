% visualize results 
% save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../../load_AWL_toolbox');
datadir = '../../data/';
freq = 1250;

% load continuous data
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X = cast(X,'double');

% load data
filename = 'results/res_MCSpike';
load(filename)

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = true;

% set saving directory
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    param.indir = savedir;
    param.outdir = savedir;
end

% make time axis
sfreq = round((length(t_spike)-1) / (t_spike(end)-t_spike(1)));
t = 0:1/sfreq:3200;

% define 8 colors and markers for plotting
colors = get(0,'DefaultAxesColorOrder');
colors(8,:) = [1 0.5 0.2];
set(0,'DefaultAxesColorOrder',colors)

%% plot representation for K=5

K=5; 
nrows = 3;
ncols = 1;
plot_id = 1;
h_5 = figure;

% margins for subaxis package
sv = 0.08;
sh = 0.05;
mb = 0.1;
ml = 0.15;
mr = 0.05;
mt = 0.03;

% plot spike forms
D_scaled = zeros(size(res.D{K}));
for k=1:K
    factor = mean(res.coeffs{K}(res.labels{K}==k-1));
    D_scaled(:,k) = factor*res.D{K}(:,k);
end
fontsize = 10;
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
plot(t_spike,D_scaled,'Linewidth',1.5);
xlim([t_spike(1) 1.5])
ylim([-800 200])        
xlabel('time [s]','fontsize',fontsize)
ylabel('\muV','fontsize',fontsize)
legend('spike 1','spike 2','spike 3','spike 4','spike 5','location','SouthEast')
%set(gca,'YAxisLocation','right')

% plot coefficients in time
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
energies = (res.coeffs{K}/sqrt(sfreq));
for k=1:K    
    hold on
    sel = (res.labels{K} == k-1);
    plot(t(res.latencies{K}(sel)),energies(sel),'+','Color',colors(k,:),...
        'MarkerSize',4);
end
xlabel('time [s]','fontsize',fontsize)
ylabel('spike coefficients','fontsize',fontsize)   
xlim([t(1) t(end)]);    
ylim([0,270])

% plot inter-spike distances against coefficients
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
diffs = t(res.latencies{K}(2:end) - res.latencies{K}(1:end-1));
for k=1:K    
    sel = (res.labels{K}(2:end) == k-1);
    semilogx(diffs(sel),energies([false sel]),'+','Color',colors(k,:),...
        'MarkerSize',4);
    hold on        
end
xlabel('time to previous spike [s]','fontsize',fontsize)
ylabel('spike coefficients','fontsize',fontsize)     

% save
if SAVE_FIGS
    param.width = 4;
    param.height = 5;
    filename = 'spikesMC_5';
    saveas(h_5,[savedir filename],'fig')
    fig2pdf(filename,param);
end

%% plot kernels on top of detected spikes
figure
offset = 0;
for k=1:K
    sel = (res.labels{K} == k-1); 
    hold on
    pos = res.latencies{K}(sel);
    
    t_axis = t_spike(t_spike<=0.15);
    for i=1:length(pos)
        plot(t_axis,X(pos(i)+1:pos(i)+length(t_axis))+offset,'k','linewidth',.2);
        hold on
    end
    factor = mean(res.coeffs{K}(res.labels{K}==k-1));
    D_scaled(:,k) = factor*res.D{K}(:,k);
    h=plot(t_axis,D_scaled(1:length(t_axis),k)+offset,'Color',colors(k,:),'linewidth',2)
    xlim([0,0.15])
    xlabel('time [s]')
    ylim([-4800 200])
    offset = offset - 1000;    
end
set(gca,'YTickLabel', []);
set(gca,'YTick', []);

% save
if SAVE_FIGS
    param.width = 2.5;
    param.height = 5;
    filename = 'spikesMC_5_plus_data';
    saveas(h,[savedir filename],'fig')
    fig2pdf(filename,param);
end
    
