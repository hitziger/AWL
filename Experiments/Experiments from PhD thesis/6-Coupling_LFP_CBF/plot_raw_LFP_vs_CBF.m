% plot raw local field potentials (LFP) against cerebral blood flow (CBF)
% show different windows illustrating cases both clear and non-visible
% hemodynamic response

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


%%%%% show LFP and CBF in same plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot entire signal and two different time windows
window1 = [670 720];
window2 = [800 1100];

% first entire signal, use subaxis to eliminate borders
h=figure; 

% use subaxis to eliminate borders
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
[ax,p1,p2] = plotyy(t,X,t,Y,'plot','plot');

ylim(ax(1),[-850,250])
ylim(ax(2),[0 14])

xlim(ax(1),[t(1),t(end)])
xlim(ax(2),[t(1),t(end)])

set(ax(1),'yColor','r')
set(ax(2),'yColor','k')
set(p1,'Color','r')
set(p2,'Color','k')
set(ax(1),'YTick',[-800 -600 -400 -200 0 200])

ylabel(ax(1),'\muV')
ylabel(ax(2),'aribtrary units (a. u.)')
 
xlabel('time [s]')
legend('LFP','CBF','Location','SouthEast')

% optionally make some boxes around windows
do_boxes = true;
if do_boxes
    hold(ax(2))
    eps = 1e-20;
    box = [0.1 13.9];
    plot(ax(2),[window1(1) window1(1)+eps],box, 'k','linewidth',3)
    plot(ax(2),[window1(2) window1(2)+eps],box, 'k','linewidth',3)
    plot(ax(2),[window1(1) window1(2)],[box(1) box(1)], 'k','linewidth',3)
    plot(ax(2),[window1(1) window1(2)],[box(2) box(2)], 'k','linewidth',3)

    plot(ax(2),[window2(1) window2(1)+eps],box, 'k','linewidth',3)
    plot(ax(2),[window2(2) window2(2)+eps],box, 'k','linewidth',3)
    plot(ax(2),[window2(1) window2(2)+eps],[box(1) box(1)], 'k','linewidth',3)
    plot(ax(2),[window2(1) window2(2)+eps],[box(2) box(2)], 'k','linewidth',3)
end

% save plot 
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    param.indir = savedir;
    param.outdir = savedir;
    param.width = 10;
    param.height = 4;
    if do_boxes
        filename = 'LFP_CBF_full_boxes';
    else
        filename = 'LFP_CBF_full';
    end
    saveas(h,[savedir filename],'fig')
    fig2pdf(filename,param);


    % plot and save first window
    xlim(ax(1),window1)
    xlim(ax(2),window1)

    param.width = 10;
    param.height = 3;
    filename = 'LFP_CBF_window1';
    saveas(h,[savedir filename],'fig')
    fig2pdf(filename,param);

    % plot and save second window
    xlim(ax(1),window2)
    xlim(ax(2),window2)


    param.width = 10;
    param.height = 3;
    filename = 'LFP_CBF_window2';
    saveas(h,[savedir filename],'fig')
    fig2pdf(filename,param);
end
  



