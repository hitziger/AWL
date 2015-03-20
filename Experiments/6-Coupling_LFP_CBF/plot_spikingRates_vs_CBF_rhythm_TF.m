% IMPORTANT: first run Experiments/3-MC_Spike/run_MCSpike.m to obtain spike
% detections
%
% plot spectrogram of CBF
% compare with local spiking rates (LSR) of detected spikes

% optionally save figures (.fig and .pdf) 
SAVE_FIGS = false;

% load AWL toolbox
run('../../load_AWL_toolbox');

%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq = 1250;

% load LFP data
datadir = '../data/';
file = ['LFP_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
X=cast(X, 'double');
t = linspace(0,1/sfreq*(length(X)-1),length(X));

% load LD data
file = ['LD_data_contiguous_' num2str(freq) '_Hz.mat'];
filename = [datadir file];
load(filename);
Y=cast(Y, 'double');

% load learned spike representation
datadir = '../3-MC_Spike/results/';
file = 'res_MCSpike.mat';
filename = [datadir file];
load(filename);
t=t(:);

%%%%% First lowpass filter and downsample CBF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reduce frequency content to below 10 Hz
type = 'low';
band = 10;
dAtt = 0.5;
b = custom_filter_design(type,band,dAtt,sfreq);
Y_temp = custom_filter(b,Y);

% downsample from 1250 Hz to 25 Hz
factor = 50;
Y_low = downsample(Y_temp,factor);
t_low = downsample(t,factor);
sfreq_low = sfreq/factor;

%%%%% Calculate spectrogram of CBF and plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L=2^12;          
factor_fft = 2^1;
fftlen=factor_fft*L;
window = gausswin(L,5);
noverlaps = L-20;
[S,F,T,~] = spectrogram(Y_low,window,noverlaps,fftlen,sfreq_low);

% set borders 
tmin = 0;
tmax = max(T);
fmin = 0;
fmax = max(F);

% plot
h=figure;
fmax = 1.2;
imagesc(T(tmin<T & T<tmax),F(fmin<F & F<fmax),abs(S(fmin<F & F<fmax, tmin<T & T<tmax)),[0 200]);  
ylabel('frequency [Hz]')    
xlabel('time [s]')
colormap('default')
colorbar
set(gca,'YDir','normal');

% save
if SAVE_FIGS
    savedir =  'figures/';
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    filename = 'figures/CBF_spectrogram';  
    saveas(h,filename,'fig')
    print('-painters','-dpdf','-r600',filename) 
end

%%%%% Calculate local spiking rates (LSR), plot on top of CBF spect. %%%%%%
% define local spiking rate
K=5;
lats = res.latencies{K};
coeffs = res.coeffs{K};
labels = res.labels{K};
colors=get(gca,'ColorOrder');
lsr = 1./t(lats(2:end) - lats(1:end-1));
times = t(lats(2:end));

% plot CBF
h=figure;          
imagesc(T(tmin<T & T<tmax),F(fmin<F & F<fmax),...
    abs(S(fmin<F & F<fmax, tmin<T & T<tmax)),[0 200]);  
ylabel('frequency [Hz]')    
xlabel('time [s]')
colormap(gray);
colorbar
set(gca,'YDir','normal');

% plot LSR
hold on
h2 = plot(times,lsr,'*','color','r','Markersize',8);
legend(h2,'LSR of detected spikes')
xlim([T(1),T(end)])
xlabel('time [s]')

% save
if SAVE_FIGS
    filename = 'figures/CBF_spectrogram_plus_LSR';  
    saveas(h,filename,'fig')
    print('-painters','-dpdf','-r600',filename) 
end

