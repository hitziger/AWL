% IMPORTANT: first execute run_ADSpike.m !

% - visualize results obtained with AD-Spike
% - save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../../load_AWL_toolbox');

datadir = '../../data/';
freq = 1250;

% load continuous data
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X = cast(X,'double');


% load results
filename = 'results/res_ADSpike';
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


%%%%% plot AD-Spike representation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nrows = 3;
ncols = 1;
plot_id = 1;
h_AD = figure;

% margins for subaxis package
sv = 0.08;
sh = 0.05;
mb = 0.07;
ml = 0.15;
mr = 0.05;
mt = 0.03;

markersize = 2;

% plot spike form
factor = mean(res.coefficients);
D_scaled = factor * res.D;
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
plot(t_spike,D_scaled,'b','Linewidth',2);
xlim([t_spike(1) 1.5])
ylim([-800 200])        
xlabel('time [s]')
ylabel('\muV')

% plot coefficients in time
subaxis(nrows,ncols,plot_id,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
plot_id = plot_id + 1;
coeffs = (res.coefficients/sqrt(sfreq));
plot(t(res.latencies),coeffs,'b+','MarkerSize',markersize);
xlabel('time [s]')
ylabel('spike coefficients')   
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

% save
if SAVE_FIGS
    param.width = 4;
    param.height = 5;
    filename = 'spikesAD';
    saveas(h_AD,[savedir filename],'fig')
    fig2pdf(filename,param);
end

%% plot kernel on top of detected spikes

h2 = figure;

% plot 4 samples of detected spikes with corresponding kernel instantiations on top
L = length(res.coefficients);
pos = res.latencies;
p = [50,200,350,500];
offset = [-1000, -2300,-3000,-4000];
for i=1:length(p)
    plot(t_spike,X(pos(p(i))+1:pos(p(i))+length(t_spike))+offset(i),'k','linewidth',2);
    hold on
    str = res.stretchInd(p(i));
    plot(t_spike,res.coefficients(p(i))*res.Ds(1:length(t_spike),str)+offset(i),'linewidth',2);
end

% plot all detected spikes with kernel on top
factor = mean(res.coefficients);
D_scaled = factor * res.D;

for i=1:length(pos)
    plot(t_spike,X(pos(i)+1:pos(i)+length(t_spike)),'k','linewidth',.2);
    hold on
end
plot(t_spike,D_scaled,'b','Linewidth',3);
xlim([t_spike(1) 0.15])        
xlabel('time [s]')
set(gca,'YAxisLocation','right')
    xlabel('time [s]')
    ylim([-5200 700])
    offset = offset - 1000;    
set(gca,'YTickLabel', []);
set(gca,'YTick', []);
legend('data sample','kernel','location','SouthEast')

% save
if SAVE_FIGS
    param.width = 2.5;
    param.height = 5;
    filename = 'spikesAD_plus_data';
    saveas(h2,[savedir filename],'fig')
    fig2pdf(filename,param);
end
    


