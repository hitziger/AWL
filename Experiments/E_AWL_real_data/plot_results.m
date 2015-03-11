% - visualize results 
% - save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../load_AWL_toolbox');

% load data
filename = 'results/results_real_EAWL_PCA_ICA';
load(filename)

% set saving directory
savedir =  'figures/';
param.indir = savedir;
param.outdir = savedir;


%% plot spike samples and average across epochs

% calculate average
av = mean(X,2);

% pick epochs to plot
class_offset=[0 25 50 75 100 125 150];

% set parameters
nclass=length(class_offset);
nplots = nclass+1;
nrows = 2;
ncols = ceil((nplots)/nrows);
xlims = [-1,2];

% plot figure with epochs and average
h_spikes = figure;
for i=1:nclass+1
    subaxis(nrows,ncols,i,'SpacingVert', 0.1, 'SpacingHoriz', 0.02, 'MarginBottom', 0.1, 'MarginLeft', 0.05, 'MarginRight', 0.01, 'MarginTop', 0.1)   
    if i<=nclass
        ex_id = class_offset(i)+1;
        plot(t,X(:,ex_id),'black');
        title(['Epoch ' num2str(ex_id)],'fontsize',14);
    else 
        plot(t,av,'black');
        title('average over epochs','fontsize',14);
    end
    xlim(xlims);    
    ylim([-0.5 0.1]);     
    if mod(i,ncols)~=1
        set(gca, 'YTick', []);
    end
    if nplots - i < ncols
        xlabel('time [s]','fontsize',14)
    else
        set(gca, 'XTick', []);
    end
end

% save .fig and converted .pdf
param.width = 12;
param.height = 4.5;
filename = 'spike_epochs_and_average';
saveas(h_spikes,[savedir filename],'fig')
fig2pdf(filename,param);

%% plot hierarchical AWL representations: waveforms, latency and coefficient distributions

% define a coarse time axis for latency distributions
nfac = 5;
dt = t(2)-t(1);
dt_coarse = nfac * dt;
right_side = 0:dt_coarse:t(end);
t_coarse = [-right_side(end:-1:2) right_side];
distr = zeros(size(t_coarse));

% parameters
h_dicts = figure;
nrows = Kmax;
ncols = Kmax+1;

% get max coefficient over all representations for universal colormap
max_coeff = 0;
for K=1:Kmax
    max_coeff = max(max_coeff,max(max(double(code_awl_all{K}.A))));
end
clims = [0 max_coeff];

% prepare custom colormap 
cmap = colormap('Gray');
cmap = cmap.^(1/2);

% loop over hierarchical representations, one repr. per row 
for K=1:Kmax  
    D = D_awl_all{K}(sel_inner,:);
    Delta = double(code_awl_all{K}.Delta)*dt;
    A = double(code_awl_all{K}.A);
    
    % plot waveforms and their lateny distributions in first K columns 
    for k=1:K        
        subaxis(nrows,ncols,(K-1)*ncols+k, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.05, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)        
        for i=1:length(distr)
            sel = ((t_coarse(i)-.5*dt_coarse) < Delta(k,:)) & (Delta(k,:) <= (t_coarse(i) +  .5*dt_coarse));
            distr(i) = sum(A(k,sel));
        end
        % use double axes for plotting waveforms + latency distributions
        [ax,p1,p2] = plotyy(t,D(:,k),t_coarse,distr,'plot','plot');        
        ylim(ax(1),[-0.6,0.3])
        xlim(ax(1),[-1,2])
        xlim(ax(2),[-1.5,1.5])
        ylim(ax(2),[0,3*max(distr)])        
        set(ax(2),'YTick',[])
        set(ax(2),'yColor','k')
        set(ax(1),'yColor','k')
        set(p1,'Color','k')
        set(p2,'Color','r')
        set(p2,'LineStyle','-')
        set(p2,'LineWidth',1.5)
        set(ax(2),'XTick',[])        
        if K==k
            title(['waveform ' num2str(k)], 'fontsize',14)
            if k==1
                legend('waveform', 'latency distribution','Location','EastOutside')
            end
        end         
        if K==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end        
        if k~=1
            set(gca,'YTick',[])
        end
    end 
    
    % plot coefficients as gray scale images in last column
    subaxis(nrows,ncols,K*ncols, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.05, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)
    %imagesc(log(code_awl_all{K}.A),[-4,0.5])
    imagesc(code_awl_all{K}.A,clims);
    colormap(cmap);
    %if K==1
    %    colorbar('Location','WestOutside')
    %end
    if K==1
        cbar_axes = colorbar('Position',[0.75 0.5 0.03 0.3]);
        set(cbar_axes,'YAxisLocation','left')
    end
    set(gca,'YTick',1:K)
    set(gca,'YTickLabel',1:K)
    set(gca,'YAxisLocation','right')
    ylabel('waveforms','fontsize',14)
    if K==1
        title('coefficients', 'fontsize',14)
    end
    if K==nrows
        xlabel('epochs','fontsize',14)
    else
        set(gca,'XTick',[])
    end
end

% save .fig and converted .pdf
param.width = 12;
param.height = 10;
filename = 'epoched_spikes_awl_representations';
saveas(h_dicts,[savedir filename],'fig')
fig2pdf(filename,param);

%% plot waveform representations of PCA, ICA, and AWL

% combine representations for easier loop
dict = {D_pca,D_ica169,D_ica10,D_ica5,D_awl_all{K}(sel_inner,:)};
A_all = {A_pca,A_ica169,A_ica10,A_ica5,code_awl_all{K}.A};
methods = {'PCA','ICA169','ICA10','ICA5','AWL'};
nmethods = length(methods);

% parameters
h_dicts_comp = figure;
nrows = length(methods);
ncols = K+2;
ncols2 = K;

% get max coefficient over all representations for universal colormap
max_coeff = 0;
for i=1:nmethods
    max_coeff = max(max_coeff,max(max(abs(A_all{i}))));
end
clims = [0 max_coeff];



% loop over rows, one method per row 
for i=1:nrows
    
    % plot waveforms in first K columns
    for k=1:K
        subaxis(nrows,ncols,(i-1)*ncols+k, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)
        plot(t,dict{i}(:,k),'k')
        ylim([-0.5 0.4])
        xlim([-1,2]);        
        if i==1
            title(['waveform ' num2str(k)], 'fontsize',14)
        end                
        if i==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end
        if k==1
            ylabel(methods{i},'fontsize',14);
        else
            set(gca,'YTick',[])
        end
    end 
    
    % plot coefficients as gray scale images in last column
    subaxis(nrows,ncols2,i*ncols2, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)
    imagesc(abs(A_all{i}),clims)
    colormap(cmap)
    
    %cbar_axes = colorbar('Location','NorthOutside')
    cbar_axes = colorbar('Position',[0.76 1-0.025-0.18*i 0.0075 0.15]);
    set(cbar_axes,'YAxisLocation','left')
    if K==1
        %cbar_axes = colorbar('Position',[0.75 0.5 0.03 0.3])
        set(cbar_axes,'YAxisLocation','left')
    end
    set(gca,'YTick',1:K)
    set(gca,'YTickLabel',1:K)
    set(gca,'YAxisLocation','right')
    ylabel('waveforms','fontsize',14)
    if i==1
        title('abs coefficients', 'fontsize',14)
    end
    if i==nrows
        xlabel('epochs','fontsize',14)
    else
        set(gca,'XTick',[])
    end
end

param.width = 12;
param.height = 6;
filename = 'epoched_spikes_AWL_PCA_ICA_VARIOUS_ICA';
saveas(h_dicts_comp,[savedir filename],'fig')
fig2pdf(filename,param);
