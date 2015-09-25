function fen = gaussian(N,sigma2)

% fen = gaussian(N,sigma2)
%
% fen est une gaussienne centrée, de variance sigma2, tronquée à N echantillons et normée

milieu = (N+1)/2;
fen = exp(- ((1:N) - milieu).^2/sigma2);
fen = fen/ sqrt(fen*fen');
