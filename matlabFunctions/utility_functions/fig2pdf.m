function fig2pdf(filename,param)
% properly converts a .fig to .pdf

% default parameters
width = 12;         % inches
height = 9;         % inches
indir = [pwd '/'];
outdir = [pwd '/'];
tight = false;

if nargin>1
    if isfield(param,'width')
        width = param.width;
    end
    if isfield(param,'height')
        height = param.height;
    end
    if isfield(param,'indir')
        indir = param.indir;
    end
    if isfield(param,'outdir')
        outdir = param.outdir;
    end
    if isfield(param,'tight')
        tight = param.tight;
    end
end

% open figure and set some properties 
h=open([indir filename '.fig']);
%set(findall(h,'-property','FontSize'),'FontSize',fontsize)
%set(findall(h,'-property','XLimMode'),'XLimMode','manual')
%set(findall(h,'-property','XTickMode'),'XTickMode','manual')
%set(findall(h,'-property','XTickLabelMode'),'XTickLabelMode','manual')
%set(findall(h,'-property','YLimMode'),'YLimMode','manual')
%set(findall(h,'-property','YTickMode'),'YTickMode','manual')
%set(findall(h,'-property','YTickLabelMode'),'YTickLabelMode','manual')

set(gca, 'LooseInset', get(gca, 'TightInset')); % use for tight borders
set(gcf,'PaperPosition',[0 0 width height])
set(gcf,'PaperSize',[width height])
if tight
    tightfig;
end


print(h, '-painters', '-dpdf', '-r600', [outdir filename]);
%saveas(h,[outdir filename],'pdf');
close(h)
