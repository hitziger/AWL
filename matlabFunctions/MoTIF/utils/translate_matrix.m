function translated_matrices = translate_matrix(matrix)

% translated_matrices = translate_matrix(matrix)
%
% matrix has M column vectors of size N
% these columns will be translated
% translated_matrices is a matrix of N rows and (2*N-1)*M columns
%
% examples :
% 
% v = [0    translate_matrix(v) = [1 5 3 3 0 0 0 0 0
%      3                           0 1 5 3 3 0 0 0 0
%      3                           0 0 1 5 3 3 0 0 0 
%      5                           0 0 0 1 5 3 3 0 0
%      1]                          0 0 0 0 1 5 3 3 0]
%
% >> mat = randint(3,2,10)
% 
% mat =
% 
%      3     4
%      8     1
%      6     4
%
% >> translate_matrix(mat)
% 
% ans =
% 
%      6     8     3     0     0     4     1     4     0     0
%      0     6     8     3     0     0     4     1     4     0
%      0     0     6     8     3     0     0     4     1     4



[N,M] = size(matrix);

x = (1:N)';
y = (2*N-2):-1:0;
t = x(:,ones(1,2*N-1)) + y(ones(N,1),:);

u = t(:);
v = (3*N-2)*(0:M-1);
w = u(:,ones(1,M)) + v(ones((2*N-1)*N,1),:);
x = reshape(w,N,M*(2*N-1));

m = [zeros(N-1,M); matrix; zeros(N-1,M)];

x(:) = m(x);
translated_matrices = x;