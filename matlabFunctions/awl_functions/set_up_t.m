function [nLeft,t,t_ext] = set_up_t(t0,t1,dt,sigma_t)
% creates regular and extended time axis given corresponding parameters
% 
% INPUT:
% t0: start of regular time window (sec)
% t0: end of regular time window (sec)
% dt: sampling distance (sec)
% sigma_t: length of extensions to left and right (sec)
%
% OUTPUT:
% nLeft: no. of sampling points of each extension
% t: regular time axis
% t_ext: extended time axis


t=t0:dt:t1;
t1=t(end);

temp=t0:-dt:t0-sigma_t;
t_left=temp(end:-1:2);
nLeft=length(t_left);

temp=t1:dt:t1+nLeft*dt;
t_right=temp(2:end);

t_ext=[t_left t t_right];
