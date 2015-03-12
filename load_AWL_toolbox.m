% updates path: adds all folders necessary for experiments

if ~exist('AWL_loaded','var')
    % mex-implementation of AWL,  TODO: change this directory
    addpath([pwd '/mexFunctions/build'])
    
    % functions from this package
    addpath([pwd '/matlabFunctions/awl_functions'])
    addpath([pwd '/matlabFunctions/utility_functions'])

    AWL_loaded = true;
end 