function b=custom_filter_design(type,band,dAtt,sfreq,VERBOSE)

switch type
    case 'high'
        mags = [0 1];
    case 'low'
        mags = [1 0];
    case 'band'
        mags = [0 1 0];     
    case 'band_out'
        mags = [1 0 1];      
    otherwise
        error('filter type must be high or low');
end
l=length(mags);
devs = 0.1 * ones(1,l);
fcuts = [];
for i=1:length(band);
    fcuts = [fcuts band(i)-dAtt band(i)+dAtt];
end
[n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,sfreq);
n = n + rem(n,2);
b = fir1(n,Wn,ftype,kaiser(n+1,beta));

if nargin>4
    if (VERBOSE)
        [H,f] = freqz(b,1,linspace(0,1.5*max(band)),sfreq);
        figure;
        plot(f,abs(H));
        xlabel('frequency');
        ylabel('magnitude of filter');
    end
end