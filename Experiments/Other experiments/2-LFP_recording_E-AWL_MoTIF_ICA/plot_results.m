% IMPORTANT: first execute run_processing_AWL_MOTIF_ICA.m !

% - visualize results 
% - save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../../load_AWL_toolbox');

% load data
filename = 'results/results_real_EAWL_MOTIF_ICA';
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
    subaxis(nrows,ncols,i,'SpacingVert', 0.1, 'SpacingHoriz', 0.02, 'MarginBottom', 0.1, 'MarginLeft', 0.06, 'MarginRight', 0.01, 'MarginTop', 0.1)   
    if i<=nclass
        ex_id = class_offset(i)+1;
        plot(t,X(:,ex_id),'black');
        title(['epoch ' num2str(ex_id)],'fontsize',14);
    else 
        plot(t,av,'r','LineWidth',2);
        title('average over epochs','fontsize',14);
    end
    xlim(xlims);    
    ylim([-800 200]);
    if mod(i,ncols)~=1
        set(gca, 'YTick', []);
    else
        ylabel('$\mu$V','fontsize',16,'interpreter','latex')
    end
    if nplots - i < ncols
        xlabel('time [s]','fontsize',14)
    else
        set(gca, 'XTick', []);
    end
end

% save .fig and converted .pdf
if SAVE_FIGS
    param.width = 12;
    param.height = 4.5;
    filename = 'spike_epochs_and_average';
    saveas(h_spikes,[savedir filename],'fig')
    fig2pdf(filename,param);
end
%% plot hierarchical AWL representations: kernels, latency and coefficient distributions

% define a coarse time axis for latency distributions
nfac = 5;
dt = t(2)-t(1);
dt_coarse = nfac * dt;
right_side = 0:dt_coarse:t(end);
t_coarse = [-right_side(end:-1:2) right_side];
distr = zeros(size(t_coarse));

% parameters
h_dicts = figure;
nrows = Kmax+2;
ncols = Kmax+1;

% get max coefficient over all representations for universal colormap
max_coeff = 0;
for K=1:Kmax
    max_coeff = max(max_coeff,max(max(double(code_awl_all{K}.A))));
end
max_coeff = max(max_coeff,max(max(double(code_awl_non_hier.A))));
clims = [0 max_coeff];

% prepare custom colormap 
cmap = colormap('Gray');
cmap = cmap.^(1/2);

% calulate distances and change order 
[error1,perm] = kernel_dist_shift(D_awl_all{K},D_awl_non_hier);
D_awl_non_hier = D_awl_non_hier(:,perm);
code_awl_non_hier.A = code_awl_non_hier.A(perm,:);
code_awl_non_hier.Delta = code_awl_non_hier.Delta(perm,:);
[error2,perm] = kernel_dist_shift(D_awl_all{K},D_awli_non_hier);
D_awli_non_hier = D_awli_non_hier(:,perm);
code_awli_non_hier.A = code_awli_non_hier.A(perm,:);
code_awli_non_hier.Delta = code_awli_non_hier.Delta(perm,:);

% loop over hierarchical representations, one repr. per row 
for K=1:Kmax + 2 
    if K <= Kmax
        D = D_awl_all{K}(sel_inner,:);
        Delta = double(code_awl_all{K}.Delta)*dt;
        A = double(code_awl_all{K}.A);
    elseif K==Kmax + 1
        D = D_awl_non_hier(sel_inner,:);
        Delta = double(code_awl_non_hier.Delta)*dt;
        A = double(code_awl_non_hier.A);
    else 
        D = D_awli_non_hier(sel_inner,:);
        Delta = double(code_awli_non_hier.Delta)*dt;
        A = double(code_awli_non_hier.A);
    end
    
    % plot kernels and their lateny distributions in first K columns 
    for k=1:min(K,Kmax)        
        subaxis(nrows,ncols,(K-1)*ncols+k, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.05, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)        
        for i=1:length(distr)
            sel = ((t_coarse(i)-.5*dt_coarse) < Delta(k,:)) & (Delta(k,:) <= (t_coarse(i) +  .5*dt_coarse));
            distr(i) = sum(A(k,sel));
        end
        % use double axes for plotting kernels + latency distributions
        [ax,p1,p2] = plotyy(t,D(:,k),t_coarse,distr,'plot','plot');        
        ylim(ax(1),[-0.6,0.5])
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
            title(['kernel ' num2str(k)], 'fontsize',14)
            if k==1
                legend('kernel', 'latency distribution','Location','EastOutside')
            end
        end         
        if K==nrows
            xlabel('time [s]','fontsize',14)
        else
            set(gca,'XTick',[])
        end        
        if k~=1
            set(gca,'YTick',[])
        else
            if K<=Kmax
                ylabel('hier. E-AWL', 'fontsize',14);
            elseif K==Kmax+1
                ylabel('E-AWL', 'fontsize',14);
            else
                ylabel('ICA + E-AWL', 'fontsize',14);
            end
        end
    end 
    
    % plot coefficients as gray scale images in last column
    subaxis(nrows,ncols,K*ncols, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.05, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)
    %imagesc(log(code_awl_all{K}.A),[-4,0.5])
    imagesc(A,clims);
    colormap(cmap);
    %if K==1
    %    colorbar('Location','WestOutside')
    %end
    if K==1
        cbar_axes = colorbar('Position',[0.75 0.5 0.03 0.3]);
        set(cbar_axes,'YAxisLocation','left')
        ylabel(cbar_axes,'\mu V','fontsize',14)
    end
    set(gca,'YTick',1:K)
    set(gca,'YTickLabel',1:K)
    set(gca,'YAxisLocation','right')
    ylabel('kernels','fontsize',14)
    if K==1
        title('coefficients', 'fontsize',14)
    end
    if K==nrows
        xlabel('epochs','fontsize',14)
    else
        set(gca,'XTick',[])
    end
end

dist = kernel_dist_shift(D_awl_all{Kmax},D_awl_non_hier);
display(['kernel distance between hierarchical and non-hierarchical: ' ...
    num2str(dist)])

% save .fig and converted .pdf
if SAVE_FIGS
    param.width = 12;
    param.height = 10;
    filename = 'real_dicts_eawl';
    saveas(h_dicts,[savedir filename],'fig')
    fig2pdf(filename,param);
end

%% plot kernel representations of MoTIF, ICA, and E-AWL

K=Kmax;

% combine representations for easier loop
dict = {D_motif,D_ica,D_awl_all{K}(sel_inner,:)};
A_all = {A_motif,A_ica,code_awl_all{K}.A};
methods = {'MoTIF','ICA','hier. E-AWL'};
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
    
    % plot kernels in first K columns
    for k=1:K
        subaxis(nrows,ncols,(i-1)*ncols+k, 'SpacingHoriz', 0.02, 'SpacingVert', 0.02, 'MarginBottom', 0.07, 'MarginLeft', 0.05, 'MarginRight', 0.05, 'MarginTop', 0.05)
        plot(t,dict{i}(:,k),'k')
        ylim([-0.5 0.4])
        xlim([-1,2]);        
        if i==1
            title(['kernel ' num2str(k)], 'fontsize',14)
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
    %if i==1
        cbar_axes = colorbar('Position',[0.76 0.98-0.3*i 0.0075 0.2]);
        set(cbar_axes,'YAxisLocation','left')
        title(cbar_axes,'\mu V','fontsize',14)
    %end
    
    set(gca,'YTick',1:K)
    set(gca,'YTickLabel',1:K)
    set(gca,'YAxisLocation','right')
    ylabel('kernels','fontsize',14)
    if i==1
        title('absolute coefficients', 'fontsize',14)
    end
    if i==nrows
        xlabel('epochs','fontsize',14)
    else
        set(gca,'XTick',[])
    end
end
if SAVE_FIGS
    param.width = 12;
    param.height = 6;
    filename = 'real_dicts_comp';
    saveas(h_dicts_comp,[savedir filename],'fig')
    fig2pdf(filename,param);    
end


%% plot kernel distances
D_awl_hier = D_awl_all{K}(sel_inner,:);
D_awl = D_awl_non_hier(sel_inner,:);
D_awli = D_awli_non_hier(sel_inner,:);
for k=1:Kmax
    D_awl_hier(:,k) = D_awl_hier(:,k) / norm(D_awl_hier(:,k));
    D_awl(:,k) = D_awl(:,k) / norm(D_awl(:,k));
    D_awli(:,k) = D_awli(:,k) / norm(D_awli(:,k));
end
methods = {'MoTIF','ICA','hier. E-AWL','E-AWL','ICA + E-AWL'};
data = {D_motif,D_ica,D_awl_hier,D_awl,D_awli};
N = length(data);
dist_mat = zeros(N,N);

for i=1:N
    for j=1:N
        dist_mat(i,j) = kernel_dist_shift(data{i},data{j});
    end
end
h = figure;
subaxis(1,1,1,'MarginBottom', 0.12, 'MarginLeft', 0.15, 'MarginRight', -0.01, 'MarginTop', 0.13)

imagesc(abs(dist_mat))
colormap(cmap)
colorbar
set(gca,'YTick',1:N)
set(gca,'YTickLabel',methods,'fontsize',16)
set(gca,'XTick',1:N)
set(gca,'XTickLabel',methods,'fontsize',16)
title('Kernel distances $\varepsilon$ between methods','fontsize',16,'interpreter','latex')

if SAVE_FIGS
    param.width = 10;
    param.height = 3;
    filename = 'dist_mat';
    saveas(h,[savedir filename],'fig')
    fig2pdf(filename,param);    
end