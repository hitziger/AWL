function compare_ffts(x,y,fSamp)

xdft = fft(x);
ydft = fft(y);
freq = linspace(0,fSamp/2,length(x)/2+1);
plot(freq,abs(xdft(1:floor(length(x)/2)+1)));
hold on;
plot(freq,abs(ydft(1:floor(length(x)/2)+1)),'r-.','linewidth',2);
legend('First Signal','Second Signal');
