% visualize results 
% save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../../load_AWL_toolbox');

% load data
filename = 'results/res_MCSpike';
load(filename)

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = false;

% set saving directory
if SAVE_FIGS
    savedir = 'figures/';
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
set(0,'DefaultAxesColorOrder',colors, ...
      'DefaultAxesLineStyleOrder','-|-o|-*|-x|-s|-d|->|-<')

%% plot hier. representations row-wise (spike shapes + coefficients)

maxK = 8;
nrows = maxK;
ncols = 2;

% margins for subaxis package
sv = 0.02; 
sh = 0.02;
mb = 0.06;
mt = 0.03;

marg_side = 0.03;
marg_middle = 0;
legend_entries = {};
h_MC=figure;
for K=1:maxK    
    % plot spike forms
    D_scaled = zeros(size(res.D{K}));
    for k=1:K
        factor = mean(res.coeffs{K}(res.labels{K}==k-1));
        D_scaled(:,k) = factor*res.D{K}(:,k);
    end
    plot_id = (K-1)*ncols+1;    
    ml = marg_side+0.03;
    mr = marg_middle+marg_side;
    subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
    plot(t_spike,D_scaled,'Linewidth',1.5);
    xlim([t_spike(1) 0.3])
    ylim([-800 200])
    h = get(gca,'Children');
    legend_entries{K} = ['spike ' num2str(K)];
    legend(h(1),['spike ' num2str(K)],'Location','SouthEast');      
    ylabel('$\mu$V','interpreter','latex')        
    if (K==maxK) 
        xlabel('time [s]')
    else
        set(gca, 'XTick', []);
    end   
    if (K==1) 
        title('Spike templates')
    end   
    
    % plot coefficients in time   
    plot_id = (K-1)*ncols+2;         
     ml = marg_side+marg_middle+0.03;
    mr = marg_side;
    subaxis(nrows,ncols,plot_id,'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
    for k=1:K
        hold on
        sel = (res.labels{K} == k-1);
        plot(t(res.latencies{K}(sel)),(res.coeffs{K}(sel)/sqrt(sfreq)),'+','Color',colors(k,:),'MarkerSize',4);
    end
    xlim([t(1) t(end)]);
    ylim([0 250])
    set(gca,'YAxisLocation','right')    
    if (K==maxK) 
        xlabel('time [s]')
    else
        set(gca, 'XTick', []);
    end   
    if (K==1) 
        title('Spike coefficients')
    end   
end

% save
if SAVE_FIGS
    param.width = 10;
    param.height = 10;
    filename = 'spikesMC_8classes';
    saveas(h_MC,[savedir filename],'fig')
    fig2pdf(filename,param);
end
   

%% plot performance measures 
h_res = figure;
nrows = 1;
ncols = 3;
plot_id = 1;

% margins for subaxis package
sv = 0.07;
sh = 0.07;
mb = 0.2;
ml = 0.06;
mr = 0.02;
mt = 0.03;

% plot spike fits
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
for k=1:8    
    bar(k,mean(res.fits{k}),'FaceColor',colors(k,:))
    hold on 
end
set(gca,'XTick',1:8)
set(gca,'XTickLabel',1:8)
ylim([0 1])
ylabel('average spike fit')
xlabel('number of spike classes K')

% plot residual
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
for k=1:8
    bar(k,res.residual(k),'FaceColor',colors(k,:))
    hold on 
end
set(gca,'XTick',1:8)
set(gca,'XTickLabel',1:8)
ylim([0 1])
ylabel('residual energy')
xlabel('number of spike classes K')

% plot number of detected spikes 
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
for k=1:8
    bar(k,length(res.latencies{k}),'FaceColor',colors(k,:))
    hold on 
end
h = plot(0:9,520*ones(size(0:9)),'r','lineWidth',2);
legend(h, 'total number of spikes')
set(gca,'XTick',1:8)
set(gca,'XTickLabel',1:8)
ylim([510 523])
ylabel('detected spikes')
xlabel('number of spike classes K')

% save
if SAVE_FIGS
    param.width = 10;
    param.height = 2;
    filename = 'spikesMC_residual';
    saveas(h_res,[savedir filename],'fig')
    fig2pdf(filename,param);
end


%% analyze spiking rate for K=5 
K=5; 
nrows = 3;
ncols = 2;
plot_id = 1;
h_5 = figure;

% margins for subaxis package
sv = 0.08;
sh = 0.05;
mb = 0.07;
ml = 0.1;
mr = 0.1;
mt = 0.03;

% plot spike forms
D_scaled = zeros(size(res.D{K}));
for k=1:K
    factor = mean(res.coeffs{K}(res.labels{K}==k-1));
    D_scaled(:,k) = factor*res.D{K}(:,k);
end
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
plot(t_spike,D_scaled,'Linewidth',1.5);
xlim([t_spike(1) 0.1])
ylim([-800 200])        
xlabel('time [s]')
ylabel('\muV')

% plot spike forms
D_scaled = zeros(size(res.D{K}));
for k=1:K
    factor = mean(res.coeffs{K}(res.labels{K}==k-1));
    D_scaled(:,k) = factor*res.D{K}(:,k);
end
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
plot(t_spike,D_scaled,'Linewidth',1.5);
xlim([t_spike(1) 1.5])
ylim([-800 200])        
xlabel('time [s]')
ylabel('\muV')
legend('spike 1','spike 2','spike 3','spike 4','spike 5','Location',...
    'SouthEast')
set(gca,'YAxisLocation','right')

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
xlabel('time [s]')
ylabel('spike coefficients')   
xlim([t(1) t(end)]);    
ylim([0,270])
    
% plot distances in time
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
diffs = t(res.latencies{K}(2:end) - res.latencies{K}(1:end-1));

for k=1:K
    sel = (res.labels{K}(2:end) == k-1);
    plot(t(res.latencies{K}(sel)),log(diffs(sel)),'+','Color',...
        colors(k,:),'MarkerSize',4);
    hold on   
end
xlabel('time [s]')
ylabel('-log(LSR)')
axis tight
xlim([t(1) t(end)]);
set(gca,'YAxisLocation','right')
ylim([-1.5,4])


% plot rates vs coeffs
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
for k=1:K    
    sel = (res.labels{K}(2:end) == k-1);
    plot(log(diffs(sel)),energies([false sel]),'+','Color',colors(k,:),...
        'MarkerSize',4);
    hold on        
end
xlabel('-log(LSR)')
ylabel('spike coefficients')     

% save
if SAVE_FIGS
    param.width = 7;
    param.height = 6;
    filename = 'spikesMC_5';
    saveas(h_5,[savedir filename],'fig')
    fig2pdf(filename,param);
end

