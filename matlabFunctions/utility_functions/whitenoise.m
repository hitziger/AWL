function y = whitenoise(N)

% function: y = whitenoise(N) 
% N - number of samples to be returned in row vector
% y - row vector of white noise samples

% The function generates a sequence of white noise samples. 

% generate white noise
y = randn(1, N);

% ensure unity standard deviation and zero mean value
y = y - mean(y);
yrms = sqrt(mean(y.^2));
y = y/yrms;

end