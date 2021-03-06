function [dict_est, dict_est_mat,coeffs_mat] = MoTIF_learning(frames,l_atom,n_atoms,constraint_tag,hilbert_tag)

% [dict_est, dict_est_mat,coeffs_mat] = MoTIF_learning(frames,l_atom,n_atoms [,constraint_tag,hilbert_tag])
%
% learns n_atoms using MoTIF algorithm, on the frames, taking
% into account a notion of phase of the atom, using the Hilbert
% transform, according to the chosen hilbert_tag
% 
% frames : a matrix, which columns are the frames
%          or a struct array, where the nth frame is 
%          frames(n).sig
% l_atom : the size of the atoms to learn
% n_atoms : the number of atoms to learn
% constraint_tag =
%   0 : no constraint added on the atoms
%  {1}: the atoms are constrained to be as decorrelated as possible 
% hilbert_tag = 
%  {0}: we just search the atoms that are the most correlated with the 
%       frames
%   1 : we search the atoms such that the selected frames are the most 
%       correlated with a linear combination of the atom and its 
%       hilbert transform (hypothesis on the atom to be centered and 
%       without Nyquist component) 
%   2 : idem than 1, except that the mean and the Nyquist component of 
%       the atom are taken into account, in the sense of the 
%       simplified objective function
% dict_est : a struct array, where the nth atom is dict_est(n).atom
% dict_est_mat : a matrix,which columns are the learnt atoms
% coeffs_mat : a (n_atoms x n_frames) matrix containing the absolute 
%              correlation values of the atoms with the frames

if nargin < 4
  constraint_tag = 1;
  hilbert_tag = 0;
elseif nargin < 5
  hilbert_tag = 0;
end

if isstruct(frames)
  % first look at the size of the frames
  n_frames = length(frames);
  max_l_frame = 0;
  for n=1:n_frames
    if (max_l_frame < frames(n).sig)
      max_l_frame = frames(n).sig;
    end
  end
  % create the matrix of frames
  % n_frames columns
  % max_l_frames rows
  mat_frames = zeros(max_l_frames,n_frames);
  % save the size of the frames, in order not to
  % pick in the zero-padded part of the extended
  % frame after
  max_pos_frames = zeros(1,n_frames);
  for n=1:n_frames
    temp_l_frame = length(frames(n).sig);
    mat_frames(1:temp_l_frame,n) = frames(n).sig(:);
    max_pos_frames(n) = temp_l_frame;
  end
  max_pos_frames = max_pos_frames - l_atom+1;
  
else
  mat_frames = frames;
  [l_frame,n_frames] = size(mat_frames);
  max_pos_frames = repmat(l_frame-l_atom+1,1,n_frames);
end

dict_est_mat = zeros(l_atom,n_atoms);
coeffs_mat = zeros(n_atoms,n_frames);
constraint_mat = [];

% the screen print is only done every n_step_disp step
n_step_disp = 1;

% start of the loop on the atoms
for n=1:n_atoms

  % initialiaze the new atom
  atom = randn(l_atom,1);
  if (hilbert_tag == 1)
    f_atom = fft(atom);
    f_atom(1) = 0;
    if ((l_atom/2)==floor(l_atom/2))
      f_atom(l_atom/2+1) = 0;
    end
    atom = real(ifft(f_atom));
  end
  atom = atom/sqrt(atom'*atom);
  
  % initialize the values of the objetive function
  val_obj = -1;
  val_obj_prec = -2;
  
  % start of the loop for learning one atom
  iter = 0;
  while (val_obj > val_obj_prec)
    %while (any(pos_patches ~= pos_patches_prec) == 1)
    iter = iter + 1;
    val_obj_prec = val_obj;
    
    % research of the start position of the patches
    [pos_patches,val_obj,max_val] = search_pos_patches(mat_frames,max_pos_frames,atom,hilbert_tag);
    
    if ((iter/n_step_disp) == floor(iter/n_step_disp))
      fprintf('MoTIF - atom %i/%i - it. %i - objective function : %f\n',n,n_atoms,iter,val_obj);
    end
    % update of the current atom 
    atom = update_atom(mat_frames,pos_patches,l_atom,constraint_mat,hilbert_tag);
  end

  % add the last atom to the dictionary
  dict_est_mat(:,n) = atom;
  % add the absolut coefficients
  coeffs_mat(n,:) = max_val;
  
  % construct the constraint matrix
  if (constraint_tag == 1)
    switch hilbert_tag
     case 0
      temp_mat = translate_matrix(atom);
      if (n == 1)
	constraint_mat = temp_mat*temp_mat';
      else
	constraint_mat = constraint_mat + temp_mat*temp_mat';
      end
     case 1
      temp_mat = translate_matrix(atom);
      f_temp_mat = fft(temp_mat);
      f_temp_mat = f_temp_mat(2:ceil(l_atom/2),:);
      if (n == 1)
	constraint_mat = f_temp_mat*conj(f_temp_mat).';
      else
	constraint_mat = constraint_mat + f_temp_mat*conj(f_temp_mat).';
      end
     case 2
      temp_mat = translate_matrix(atom);
      f_temp_mat = fft(temp_mat);
      f_temp_mat = f_temp_mat(1:floor(l_atom/2+1),:);
      if (n == 1)
	constraint_mat = f_temp_mat*conj(f_temp_mat).';
      else
	constraint_mat = constraint_mat + f_temp_mat*conj(f_temp_mat).';
      end
     otherwise
      % not implemeted yet
      constraint_mat = [];
    end
  end 
  fprintf('MoTIF - atom %i/%i - last it. (%i) - objective function : %f\n',n,n_atoms,iter,val_obj);
end

% conversion of the dictionary in a struct array
dict_est(1).atom = [];
dict_est = repmat(dict_est(1),1,n_atoms);
for n=1:n_atoms
  dict_est(n).atom = dict_est_mat(:,n);
end
