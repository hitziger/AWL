function atom = define_atom(t,param)

% defines atom on a given time axis, with specified shape and parameters
%
% INPUT:
% t: (vector) time axis 
% param: (struct) containing shape and parameter specifications (see below)
% 
% OUTPUT:
% atom: defined waveform 
%
% USED FUNCTIONS:
% make_spike: defines an epileptiform spike

% default parameters
window = ones(1,length(t));

% parse paramters
if ~(isfield(param,'freq'))
    error('freq must be set')
end
freq = param.freq;
if ~(isfield(param,'amp'))
    error('amp must be set')
end
amp = param.amp;
if isfield(param,'window_type')
    if ~strcmp(param.window_type,'none')
        if isfield (param,'window_sigma')
            sigma = param.window_sigma;
        else 
            sigma = (t(end)-t(1))/4;
        end
        if isfield (param,'window_offset')
            t_offset = param.window_offset;
        else 
            t_offset = t(1)+(t(end)-t(1))/2;
        end
        if strcmp(param.window_type,'gauss')
            window = exp(-(t-t_offset).^2/(2*sigma^2));
        elseif strcmp(param.window_type,'rect')
            window = abs(t-t_offset)<sigma;
        else
            error(['Window type "' param.window_type '" not known'])
        end
    end
end
if isfield(param,'window_cutoff')
    window(t<t(1)+param.window_cutoff)=0;
    window(t>t(end)-param.window_cutoff)=0;
end

arg = 2*pi*freq;
if ~(isfield(param,'sig'))
    error('field sig must be set')
end
if strcmp(param.sig,'sine')
    atom = cos(arg*t) .* window;
elseif strcmp(param.sig,'saw')
    atom = sawtooth(arg*t) .* window;
elseif strcmp(param.sig,'spike')
    atom = make_spike(t) .* window;    
else 
    error('invalid value of field sig')
end

if isfield(param,'offset')
    if param.offset > 0
        nshift = sum(t<param.offset & 0<t);
        atom(nshift+1:end) = atom(1:end-nshift);
        atom(1:nshift) = 0;
    end
end
atom = (amp/norm(atom))*atom;
       
 
