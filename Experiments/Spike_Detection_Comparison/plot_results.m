% - plot detection performances
% - save obtained .figs and .pdfs to figures directory

% load AWL toolbox
run('../../load_AWL_toolbox');

% load true spike detections
sfreq = 1250;
datadir = '../data/';
file = ['markers_all_' num2str(sfreq) '_Hz.mat'];
filename = [datadir file];
load(filename);

% load detections
filename = 'results/detected_spike_locations';
load(filename)


%% calculate accuracies 

% initializations
methods = {'MCSpike, K=1','MCSpike, K=5','ADSpike','template matching'};
nmethods = length(methods);
detections_sel = detections(:,:,[1,5,6,7],:);
nSNR = size(detections_sel,1);
ninits = size(detections_sel,2);
accuracies = zeros(nSNR,ninits,nmethods);

% detection tolerance [seconds]
tol = 0.0999; 

for i=1:nSNR
    for j=1:ninits
        for k=1:nmethods
            diffMat = bsxfun(@minus,detections_sel(i,j,k,:),markers_all(:)');
            results = (1/sfreq*abs(diffMat)<tol); 
            matches = sum(results,2);
            if (any(matches>1) || any(matches<0) ) 
                error('sth wrong here')
            end
            n_detected = sum(matches);
            accuracies(i,j,k) = n_detected/length(markers_all);  
        end
    end
end
    
%% make plots

% plot parameters
rows = 2;
cols = 2;
plot_range = {(SNR_DB<=0), (SNR_DB>=-25)};
linespecs = {'r-*','r-d','g--','k-.'}

% margins for subaxis
sv = 0.07;
sh = 0.07;
mb = 0.08;
ml = 0.08;
mr = 0.02;
mt = 0.05;

h_accuracy = figure
for i=1:rows
    for j=1:cols
        subaxis(rows,cols,(i-1)*cols+j,...
        'SpacingVert', sv, 'SpacingHoriz', sh, ...
        'MarginBottom', mb, 'MarginLeft', ml, 'MarginRight', mr, ...
        'MarginTop', mt)
        
        hold all
        for k=1:nmethods
            if k<=2
                plot(SNR_DB(plot_range{j}),accuracies(plot_range{j},i,k),linespecs{k},'LineWidth',1.3)
            else
                plot(SNR_DB(plot_range{j}),accuracies(plot_range{j},i,k),linespecs{k},'LineWidth',2.5)
            end
        end
        
        if i == 1
            if j==1
                legend(methods,'Location','SouthEast')
                title('low SNR')               
            else
                title('high SNR')
            end
        else
            xlabel('SNR [dB]')
        end
        if j == 1
            ylim([0,1])
            if i==1
                ylabel({'detection accuracy';'(good template)'})
            else
                ylabel({'detection accuracy';'(noisy template)'})
            end
        else
            ylim([0.97,1])
        end
    end
end

%%
% save
savedir =  'figures/';
param.indir = savedir;
param.outdir = savedir;
param.width = 8;
param.height = 6;
filename = 'accuracies_noisy';
saveas(h_accuracy,[savedir filename],'fig')
fig2pdf(filename,param);        
        
