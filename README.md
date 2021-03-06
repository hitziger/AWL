This folder contains the data and the MATLAB code to reproduce the 
experiments conducted in the PhD thesis "Modeling the Variability of
Neuroelectrical Activity in the Brain" by Sebastian Hitziger, as well
as other experiments using the Adaptive Waveform Learning (AWL)
algorithms. 

Content of folders:
-mexFunctions: C++ source code of mex functions, need to be compiled
-matlabFunctions: different custom utility functions
-Experiments: contains a subfolder for each experiment in the thesis 
(chronological order)

Requirements:
-Operating system: linux
-C-library fftw3 (http://www.fftw.org/) for fast implementation
-MATLAB package fastICA (http://research.ics.aalto.fi/ica/fastica/), 
used in some of the experiments

Installation (under linux):
-open script mexFunctions/compile.m in MATLAB
-make specifications for fftw library (see script)
-run script mexFunctions/compile.m, matlab functions are written 
to mexFunctions/build/

Running experiments:
-open one of the folders in Experiments/ in MATLAB
-most folders contain a script run_*.m, execute it first
-results are calculated and can be visualized with plot_results.m
-each script calls load_AWL_toolbox.m which updates the search path
-NOTE: all experiments run within 10 minutes on a laptop, except for 
spike detection comparison (~10 hours)
-NOTE: some experiments require previous experiments to be executed first, 
best run them in the order of their numbering



