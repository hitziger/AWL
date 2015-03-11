% plot spectrogram of CBF
% compare with local spiking rates (LSR) of detected spikes

% load AWL toolbox
run('../../load_AWL_toolbox');

% set directories for saving results
datadir = '../data/';

%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq = 1250;

% load LFP data
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
datadir = '../MC_Spike/results/';
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
filename = 'figures/CBF_spectrogram';  
saveas(h,filename,'fig')
print('-painters','-dpdf','-r600',filename) 

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
filename = 'figures/CBF_spectrogram_plus_LSR';  
saveas(h,filename,'fig')
print('-painters','-dpdf','-r600',filename) 

