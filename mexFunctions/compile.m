%%%%% OPTIONS PROVIDED BY THE USER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fftw library required
% IMPORTANT: specify STATIC library, (program crashes for
% dynamic library, due to conflict with MATLAB internal fftw library):
% yum install fftw-static

fftw_static_lib='/usr/lib/libfftw3.a';
fftw_header = '/usr/include/';

fftw_static_lib='/home/shitzige/compiles/lib/libfftw3.a';
fftw_header = '/home/shitzige/compiles/include/';


%%%%% COMPILATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% interactive debug mode displays several plots during calculation
debug_mode=0;

% enable fftw, otherwise currently compilation errors
enable_fftw=1;

% directories
source_dir_cawl='src/CAWL/';
source_dir_eawl='src/EAWL/';
out_dir='build/';

% files to be compiled
COMPILE_CAWL = { 'mexMCSpike.cpp',...
            'mexADSpike.cpp',...
            'mexSpikeTemplateMatching.cpp',...               
};

COMPILE_EAWL = {'mexEAWL.cpp'};

% blas library
blas_link='-lmwblas -lmwlapack';
links_lib = blas_link;

DEFS = '-DUSE_BLAS_LIB';
if ~verLessThan('matlab','7.9.0')
   DEFS=[DEFS ' -DNEW_MATLAB'];
end

%flags
link_flags=' -O ';
compile_flags=' -O ';

if enable_fftw
    DEFS=[DEFS ' -DFFT_CONV'];
else
    fftw_lib = '';
end
if debug_mode
    DEFS=[DEFS  ' -DDEBUG_MODE'];
end

get_architecture
if sixtyfourbits
    DEFS=[DEFS ' -largeArrayDims'];
end
if windows
    warning('Compilations not tested on windows, might fail!')
    DEFS=[DEFS ' -DWINDOWS -DREMOVE_'];
end


for k = 1:(length(COMPILE_CAWL) + length(COMPILE_EAWL))
    if k<=length(COMPILE_CAWL)
        str = ['-I' source_dir_cawl ' -I' fftw_header ' ' [source_dir_cawl COMPILE_CAWL{k}] ' ' fftw_static_lib];
    else
        ind = k-length(COMPILE_CAWL);
        str = ['-I' source_dir_eawl ' -I' fftw_header ' ' [source_dir_eawl COMPILE_EAWL{ind}] ' ' fftw_static_lib];
    end
    fprintf('compilation of: %s\n',str);
    if windows
        str = [str ' -outdir ' out_dir, ' ' DEFS ' ' links_lib ' OPTIMFLAGS="' compile_flags '" ']; 
    else
        str = [str ' -outdir ' out_dir, ' ' DEFS ' CXXOPTIMFLAGS="' compile_flags '" LDOPTIMFLAGS="' link_flags '" ' links_lib];
    end
    fprintf(['mex ' str])
    %str = [' -g ' str];
    args = regexp(str, '\s+', 'split');
    args = args(find(~cellfun(@isempty, args)));
    args{:};
    mex(args{:});
end
