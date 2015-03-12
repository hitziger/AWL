function Y=custom_filter(b,X,varargin)
        
% parse input
p=inputParser;
p.addRequired('b');
p.addRequired('X',@(x) validateattributes(x, {'numeric'}, {'nonempty'}));
p.addParamValue('sfreq',1,@(x) validateattributes(x, {'numeric'}, {'positive','nonempty'}));
p.addParamValue('forward_only',false,@islogical);
p.addParamValue('VERBOSE',false,@islogical);
p.parse(b,X,varargin{:});
r = p.Results;

if r.forward_only
    Y = filter(b,1,X);
else
    Y = filtfilt(b,1,X);
end

if nargin>2
    if (r.VERBOSE)
        figure;
        compare_ffts(X,Y,sfreq);
        legend('before filter','after filter');
        title('comparison of ffts');
        drawnow;
    end
end
