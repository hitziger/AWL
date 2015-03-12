function [D,A] = sign_correction_dict(D,A)
% given the dictionary representation D, A, this function checks if each
% waveform (column of D) has positive mean energy (mean over corresponding
% row of A); otherwise sign of the waveform and its energies is reversed 

signs = 2*((sum(A,2)>0) -0.5);
signsMat = repmat(signs',size(D,1),1);
D = D.*signsMat;
signsMat = repmat(signs,1,size(A,2));
A = A.*signsMat;