% plot LFP spikes against bandpass filtered CBF, showing that spikes are in
% phase with CBF activity

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = false;

% load AWL toolbox
run('../../../load_AWL_toolbox');

%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq = 1250;

% load continuous data
datadir = '../../data/';
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X=cast(X, 'double');
t = linspace(0,1/sfreq*(length(X)-1),length(X));

file = ['LD_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
Y=cast(Y, 'double');

%%%%% bandpass filter CBF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first bandpass filter CBF around 1 Hz
type = 'band';
band = [0.55 1.35];
dAtt = 0.2;
VERBOSE = false;
b = custom_filter_design(type,band,dAtt,sfreq,VERBOSE);
Y_band = custom_filter(b,Y);

%%%%% show LFP and CBF in same plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% use subaxis to eliminate borders
h=figure; 
sv = 0;
sh = 0;
mb = 0.13;
ml = 0.07;
mr = 0.05;
mt = 0.03;
subaxis(1,1,1,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)

% use double axes for plotting overlapping lfp and cbf
[ax,p1,p2] = plotyy(t,X,t,Y_band,'plot','plot');

ylim(ax(1),[-850,250])
ylim(ax(2),[-2 2])

xlim(ax(1),[t(1),t(end)])
xlim(ax(2),[t(1),t(end)])

set(ax(1),'yColor','r')
set(ax(2),'yColor','k')
set(p1,'Color','r')
set(p2,'Color','k','LineWidth',2)
set(ax(1),'YTick',[-800 -600 -400 -200 0 200])

ylabel(ax(1),'$\mu$V','interpreter','latex')
ylabel(ax(2),'aribtrary units (a. u.)')
 
xlabel('time [s]')
legend('LFP','CBF bandpass filtered (0.8-1.1 Hz)','Location','SouthEast')

% now select time window of increased spike activity 
window = [670 720];     % seconds
xlim(ax(1),window)
xlim(ax(2),window)

%%
%%%%% Save figure as .fig and .pdf %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    param.indir = savedir;
    param.outdir = savedir;
    param.width = 10;
    param.height = 3;
    filename = 'LFP_CBF_window_filtered';
    saveas(h,[savedir filename],'fig')
    fig2pdf(filename,param);
end
  



