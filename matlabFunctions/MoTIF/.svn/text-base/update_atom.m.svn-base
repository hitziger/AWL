function atom = update_atom(mat_frames,pos_patches,l_atom,constraint_mat,hilbert_tag)

% atom = update_atom(mat_frames,pos_patches,l_atom,dict,hilbert_tag)
%
% performs the second step of the MoTIF algorithm
% The positions of the patches are fixed, and we do a PCA
%
% mat_frames : a matrix which columns are the training signals
% pos_patches : indexes of the first element of each patch in
%               the frames (a vector with as many elements as 
%               the number of columns in mat_frames) 
% l_atom : the size of the atom to learn
% constraint_mat : the constraint matrix
%   constraint_mat = [] : the atom update is not constrained by
%                         previous atoms 
%   else : the atom is updated, with a constraint on the
%          decorrelation with the previous atoms. The size of the
%          matrix depends on hilbert_tag
%      hilbert_tag = 0 : l_atom by l_atom,and real 
%      hilbert_tag = 1 : l_atom/2-1 by l_atom/2-1, and complex
% hilbert_tag = 
%   0 : we just search the atoms that are the most correlated with the 
%       frames
%   1 : we search the atoms such that the selected frames are the most 
%       correlated with a linear combination of the atom and its 
%       hilbert transform (hypothesis on the atom to be centered and 
%       without Nyquist component) 
%   2 : idem than 1, except that the mean and the Nyquist component of 
%       the atom are taken into account, in the sense of the 
%       simplified objective function
%
% atom : the learnt atom
%
% Caution :
% uses the mdiag.m funtion, available in the lmi toolbox

mask_mat = mask(pos_patches,pos_patches+l_atom-1,size(mat_frames,1));
mat_patches = reshape(mat_frames(find(mask_mat)),l_atom,size(mat_frames,2));

P = ceil(l_atom/2+1);
odd = (P == l_atom/2+1);

switch hilbert_tag
 case 0
  % PCA on the translated frames
  if (prod(size(constraint_mat)) == 0)
    [eig_vec,eig_val] = eig(mat_patches*mat_patches.');
  else
    [eig_vec,eig_val] = eig(mat_patches*mat_patches.',constraint_mat);
  end
  [a,b] = max(abs(diag(eig_val)));
  atom = eig_vec(:,b);
 case 1
  f_F = fft(mat_patches);
  A1 = f_F(2:P-1,:) * conj(f_F(2:P-1,:)).';
  if (prod(size(constraint_mat)) == 0)
    [eig_vec,eig_val] = eig(A1);
  else
    [eig_vec,eig_val] = eig(A1,constraint_mat);
  end
  [a,b] = max(abs(diag(eig_val)));
  if (odd)
    f_atom =[0;eig_vec(:,b);0;flipud(conj(eig_vec(:,b)))];
  else
    f_atom =[0;eig_vec(:,b);flipud(conj(eig_vec(:,b)))];
  end
  atom = real(ifft(f_atom));
  atom = atom / sqrt(atom.'*atom);
 case 2
  f_F = fft(mat_patches);
  A1 = f_F(2:P-1,:) * conj(f_F(2:P-1,:)).';
  z = zeros(1,P-2);
  if (odd)
    A3 = f_F([1; P],:) * conj(f_F([1;P],:)).';
    A4 = [A3(1,1),z,A3(1,2);z',A1,z';A3(2,1),z,A3(2,2)];
  else
    A3 = f_F(1,:) * conj(f_F(1,:)).';
    A4 = [A3(1,1),z;z',A1];
  end
  
  if (prod(size(constraint_mat)) == 0)
    [eig_vec,eig_val] = eig(A4);
  else
    [eig_vec,eig_val] = eig(A4,constraint_mat);
  end
  [a,b] = max(abs(diag(eig_val)));
  if (odd)    
    f_atom =[eig_vec(:,b);flipud(conj(eig_vec(2:end-1,b)))];
  else
    f_atom =[eig_vec(:,b);flipud(conj(eig_vec(2:end,b)))];
  end
  atom = real(ifft(f_atom));
  atom = atom / sqrt(atom.'*atom);
 otherwise
  % this part does the same thing than hilbert_tag = 2, but closer
  % to the true objective function. Unluckily, it is not optimal at
  % this step, and the value is not assured to increase... that's
  % why it is commented
% $$$   f_F = fft(mat_patches);
% $$$   A1 = f_F(2:P-1,:) * conj(f_F(2:P-1,:)).';
% $$$   z = zeros(1,P-2);
% $$$   if (odd)
% $$$     A3 = f_F([1; P],:) * conj(f_F([1;P],:)).';
% $$$   else
% $$$     A3 = f_F(1,:) * conj(f_F(1,:)).';
% $$$   end
% $$$   
% $$$   if (prod(size(constraint_mat)) == 0)
% $$$     [eig_vec_1,eig_val_1] = eig(A1);
% $$$     [eig_vec_3,eig_val_3] = eig(A3);
% $$$   else
% $$$     % not implemented yet
% $$$   end
% $$$   [a_1,b_1] = max(abs(diag(eig_val_1)));
% $$$   [a_3,b_3] = max(abs(diag(eig_val_3)));
% $$$   f_g1 = eig_vec_1(:,b_1);
% $$$   f_g3 = eig_vec_3(:,b_3);
% $$$   
% $$$   if (odd)    
% $$$     h_Inner = [2/(P-2)*f_g1'*f_F(2:P-1,:) ; 1/2*f_g3'*f_F([1;P],:)];
% $$$   else    
% $$$     h_Inner = [2/(P-2)*f_g1'*f_F(2:P-1,:) ; 1/2*f_g3*f_F(1,:)];
% $$$   end
% $$$   if (prod(size(constraint_mat)) == 0)
% $$$     [eig_vec_4,eig_val_4] = eig(real(h_Inner*conj(h_Inner).'));
% $$$   else
% $$$     % not implemented yet
% $$$   end
% $$$   [a_4,b_4] = max(abs(diag(eig_val_4)));
% $$$   coeffs = eig_vec_4(:,b_4);
% $$$   if (odd)    
% $$$     f_atom =[coeffs(2)*f_g3(1);coeffs(1)/sqrt(2)*f_g1;coeffs(2)*f_g3(2);flipud(conj(f_g1))/sqrt(2)];
% $$$   else
% $$$     f_atom =[coeffs(2)*f_g3(1);coeffs(1)/sqrt(2)*f_g1;flipud(conj(f_g1))/sqrt(2)];
% $$$   end
% $$$   atom = real(ifft(f_atom));
% $$$   atom = atom / sqrt(atom.'*atom);
end