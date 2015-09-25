function mask_mat = mask(low_vec,high_vec,n_rows,n_cols)

% mask_mat = mask(low_vec,high_vec [,n_rows,n_cols])
%
% create a mask matrix, with only 1s and 0s.
% on the nth column, the first components are 0s,
% then they are 1s from low_vec(n) to high_vec(n),
% then 0s again
% the size of the columns is n_rows if n_rows is
% specified. If it's not specified, the length
% is the max of high_vec
%
% low_vec and high_vec must have the same length,
% and their components must be positive integers
%
% low_vec and high_vec can be scalars if all the 
% bound indices are the same. If both are scalar, 
% n_rows and n_cols must be specified. It's the 
% only case where n_cols has to be specified.
%
% ex : 
% >> mask([1 2 1 3],[4 2 3 4],4)
% 
% ans =
% 
%      1     0     1     0
%      1     1     1     0
%      1     0     1     1
%      1     0     0     1
%
% >> mask(2,4,5,2)
% 
% ans =
% 
%      0     0
%      1     1
%      1     1
%      1     1
%      0     0

low_vec = low_vec(:)';
high_vec = high_vec(:)';

l_low = length(low_vec);
l_high = length(high_vec);

if (l_low == 1)

  if (l_high == 1)
    low_vec = repmat(low_vec,1,n_cols);
    high_vec = repmat(high_vec,1,n_cols);
  else 
    n_cols = l_high;
    low_vec = repmat(low_vec,1,n_cols);
  end

elseif (l_high == 1)

  n_cols = l_low;
  high_vec = repmat(high_vec,1,n_cols);

elseif nargin > 3

  n_cols = min(l_low,n_cols);

else

  n_cols = l_low;

end

if nargin < 3
  n_rows = max(high_vec);
end

bb = (1:n_rows)';
bb = bb(:,ones(1,n_cols));

if (all(low_vec == 1))
  high = high_vec(ones(n_rows,1),1:n_cols);
  mask_mat = (bb<=high);
elseif (all(high_vec == n_rows))
  low = low_vec(ones(n_rows,1),1:n_cols);
  mask_mat = (bb>=low);
else
  high = high_vec(ones(n_rows,1),1:n_cols);
  low = low_vec(ones(n_rows,1),1:n_cols);
  mask_mat = ((bb<=high)&(bb>=low));
end