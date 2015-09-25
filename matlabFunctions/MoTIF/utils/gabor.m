function fen=gabor(N,f,fs)

% fen = gabor(N,f,fs)
%
% fen est un atome de Gabor de taille N echantillons, de frequence f, de phase nulle, et normé.
% il est fenetré par une fenetre de hanning
% fs : frequence d'echantillonage, par defaut 1

if nargin < 3
    fs = 1;
end
fen = cos(2*pi*(0:N-1)*f/fs);
fen = (fen(:).*hanning(N))';
fen = fen/sqrt(fen*fen');
