function [g_mn , h_g , m_g , n_g] = decomp_plus_hilbert(g)

% [g_mn , h_g , m_g , n_g] = decomp_plus_hilbert(g)
%
% decomposes the vector g in
% - m_g : its mean component (D component)
% - n_g : its Nyquist component
% - g_mn : g - m_g - n_g (centered and de-Nyquisted)
% and computes
% - h_g : its hilbert transform

g = g(:);

% premiere version ... pas de fft
% plus rapide

h_g = imag(hilbert(g));
l_g = length(g);
m_g = repmat(mean(g),l_g,1);
if ((l_g/2)==floor(l_g/2))
  n_g = sum(g(1:2:l_g-1)-g(2:2:l_g));
  n_g = repmat([n_g; -n_g],l_g/2,1)/l_g;
else
  n_g = zeros(l_g,1);
end

g_mn = g - m_g - n_g;


% $$$ % deuxieme version ... tout en fft
% $$$ % plus lente
% $$$ 
% $$$ f_g = fft(g);
% $$$ 
% $$$ l_g = length(g);
% $$$ m_g = real(ifft([f_g(1);zeros(l_g-1,1)]));
% $$$ f_g(1) = 0;
% $$$ 
% $$$ l_g_2 = l_g/2;
% $$$ if ((l_g_2)==floor(l_g_2))
% $$$   n_g = real(ifft([zeros(l_g_2,1); f_g(l_g_2+1); zeros(l_g_2-1,1)]));
% $$$   f_g(l_g_2+1) = 0;
% $$$   g_mn = real(ifft(f_g));
% $$$   f_g(2:l_g_2) = -sqrt(-1)*f_g(2:l_g_2);
% $$$   f_g(l_g_2+2:end) = sqrt(-1)*f_g(l_g_2+2:end);
% $$$   h_g = real(ifft(f_g));
% $$$ else
% $$$   n_g = zeros(l_g,1);
% $$$   g_mn = real(ifft(f_g));
% $$$   f_g(2:ceil(l_g_2)) = -sqrt(-1)*f_g(2:ceil(l_g_2));
% $$$   f_g(ceil(l_g_2)+1:end) = sqrt(-1)*f_g(ceil(l_g_2)+1:end);
% $$$   h_g = real(ifft(f_g));
% $$$ end
