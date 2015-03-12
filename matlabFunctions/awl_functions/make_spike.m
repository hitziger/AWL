function spike = make_spike(t,verbose)
% create spike as superposition of two gamma functions (negative peak and
% positive slow wave), starting at 0.2 seconds, approx. 2s long
% input t: vector with non-negative values between 0 and at least 2 [s] 


% fast negative peak
offset = 0.2;
k=1.5;
theta=0.05;
spike1 = -gampdf(t-offset,k,theta);

% positive slow wave
k=2;
theta=0.25;
spike2 = gampdf(t-0.2,k,theta);
spike = spike1 + 1.5*spike2;

if nargin>1
    if verbose == true
        figure
        subplot(3,1,1)
        plot(t,spike1)
        subplot(3,1,2)
        plot(t,spike2)
        subplot(3,1,3)
        spike = spike1 + 1.5*spike2;
        plot(t,spike)
    end
end

