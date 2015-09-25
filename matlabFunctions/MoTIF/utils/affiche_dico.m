function affiche_dico(d)

% affiche_dico(d)
%
% affiche les formes d'onde des atomes du dictionnaire d (une structure : d(1).atom est la premier atom)
nb_atoms = length(d);
sqrt_n_atoms = ceil(sqrt(nb_atoms));
for n=1:nb_atoms
    subplot(sqrt_n_atoms,sqrt_n_atoms,n);
    plot(d(n).atom);
    N = length(d(n).atom);
    set(gca,'Xlim',[1 N],'Ylim',[-1 1]);
end

