/*!
 *
 *                File mexMCSpike.cpp
 *
 * Usage (in Matlab): res = mexMCSpike(x,param);
 *
 * Function for detecting spikes in x and learning multiple spike classes.
 *
 * INPUT:
 * x: vector of dimension Nx1
 * param: struct with options (see Param Struct and function parse_parameters)
 *
 * OUTPUT: 
 * res: struct containing spike representation (see function create_output)
 *
 * */


#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <mex.h>
#include <my_mex_sh.h>
#include <my_utils_sh.h>
#include <matrix.h>
#include <matrix2.h>

#ifndef INFINITY 
    #define INFINITY 1e20
#endif
#ifndef MY_PRECISION 
    #define MY_PRECISION 1e-10
#endif

///////////////////////////////////////////////////////////////////////////////
////   STRUCT PARAM_DICT_LEARN                                             ////
///////////////////////////////////////////////////////////////////////////////
////       -stores parameters for learning                                 ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct ParamStruct {
    int             iter;       
    int             ncenter;
    int             ndist;
    int             ndetects;
    T               alpha;
    bool            fft;
    bool            no_update;
    bool            verbose;
};

///////////////////////////////////////////////////////////////////////////////
////   SUBROUTINE ALIGN_PEAK                                               ////
///////////////////////////////////////////////////////////////////////////////
////       -aligns maximal peak of spike template at given sample point    ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void align_peak(
            Vector<T>&  Dk,          // spike template
      const int         ncenter      // align at this sampling point
    ) {

    int peakpos = Dk.argmin();
    int shift = ncenter - peakpos;
    if (shift > 0) {
        for (int l=Dk.l()-1; l>shift-1; --l)
            Dk(l) = Dk(l-shift);
        for (int l=shift-1; l>-1; --l)
            Dk(l) = 0;
    }
    if (shift < 0) {
        for (int l=0; l<Dk.l()+shift; ++l)
            Dk(l) = Dk(l-shift);
        for (int l=Dk.l()+shift; l<Dk.l(); ++l)
            Dk(l) = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION PARSE_PARAMETERS                                           ////
///////////////////////////////////////////////////////////////////////////////
////       -reads signal, spike template and parameters from input         ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void parse_parameters(
    ParamStruct<T>& param,      // stores learning parameters
    Vector<T>&      X,          // signal
    Matrix<T>&      D,          // spike classes
    const int       nlhs,       // # of lhs-arguments
    mxArray         *plhs[],    // pointer to lhs-arguments
    const int       nrhs,       // # of rhs-arguments
    const mxArray   *prhs[]     // pointer to rhs-arguments
    ) {
    
    ///// check number of arguments ///////////////////////////////////////////
    if (nrhs != 2) mexErrMsgTxt("Bad number of input arguments");
    if (nlhs != 1) mexErrMsgTxt("Bad number of output arguments");

    ///// check consistency of arguments //////////////////////////////////////
    if (!mexCheckType<T>(prhs[0])) mexErrMsgTxt("argument 1 should be scalar");
    if (!mxIsStruct(prhs[1]))      mexErrMsgTxt("argument 2 should be struct");
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("argument 1 should be of double precision");

    ///// read signal /////////////////////////////////////////////////////////
    const mwSize* dimsX=mxGetDimensions(prhs[0]);   
    int N = static_cast<int>(dimsX[0]);             // sample points in signal
    if (dimsX[0]<2) mexErrMsgTxt("argument 1 should be a column vector\n");
    if (dimsX[1]!=1) mexErrMsgTxt("argument 1 should be a column vector\n");
    T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0])); 
    X.resize(N);
    X.set(prX);

    ///// read template spike and set up spike matrix /////////////////////////
    unsigned maxK = getScalarStructDef<int>(prhs[1],"maxK",5); // max. number
    mxArray* pr_D = mxGetField(prhs[1],0,"D");
    unsigned n; 
    if (!pr_D) {
        mexWarnMsgTxt("initial spike template D should be provided, " 
                "initializing at maximal energy");
        if (!mxGetField(prhs[1],0,"n")) mexErrMsgTxt("average spike length n "
                "has to be provided when template is missing!");
        n = getScalarStruct<unsigned>(prhs[1],"n"); // sample points per spike 
        D.resize(n,maxK);
        Vector<T> Dk(D,0);
        param.ndist = getScalarStructDef<int>(prhs[1],"ndist",ceil(0.05*n));
        init_first_spike(X,Dk,param.ndist);
    } else {
        T* prD = reinterpret_cast<T*>(mxGetPr(pr_D));
        const mwSize* dimsD=mxGetDimensions(pr_D);
        if (dimsD[0]<2) mexErrMsgTxt("spike template should be a column vector\n");
        if (dimsD[1]!=1) mexErrMsgTxt("spike template should be a column vector\n");
        n = static_cast<unsigned>(dimsD[0]);  // number of sample points of spike
        D.resize(n,maxK);
        Vector<T> Dk(D,0);
        Dk.copy(prD);               
        Dk/=Dk.norm2();
    }

    ///// other parameters ////////////////////////////////////////////////////
    param.iter                  // number of iterations per decomposition 
        = getScalarStructDef<int>(prhs[1],"iter",10);
    param.ncenter               // approximate length of negative peak 
        = getScalarStructDef<int>(prhs[1],"ncenter",ceil(0.05*n));
    if (param.ncenter<=0) mexErrMsgTxt("param.ncenter has to be positive");
    param.ndist                 // approximate length of negative peak 
        = getScalarStructDef<int>(prhs[1],"ndist",ceil(0.05*n));
    if (param.ndist<param.ncenter) mexErrMsgTxt("param.ndist cannot be smaller than param.ncenter");
    if (param.ndist>n) mexErrMsgTxt("param.ndist cannot be greater than no. of sample points of spikes");
    param.ndetects             // if >0, determines number of detections 
        = getScalarStructDef<int>(prhs[1],"ndetects",0);
    param.alpha                  // 0<alpha<1, minimal atom/signal correlation
        = getScalarStructDef<T>(prhs[1],"alpha",0.1);             
    param.verbose = getScalarStructDef<bool>(prhs[1],"verbose",false);
    param.no_update =  getScalarStructDef<bool>(prhs[1],"no_update",false); 
    param.fft =  getScalarStructDef<bool>(prhs[1],"fft",false) ? 
        true : (getScalarStructDef<bool>(prhs[1],"no_fft",false) ? 
                false : 4*log2(N)<n);

} 

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION INIT_FIRST_SPIKE                                           ////
///////////////////////////////////////////////////////////////////////////////
////       -initializes first spike if not provided:                       ////
////         sliding window is used to find maximal energy in signal       ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void init_first_spike(
    const Vector<T>&    X,     // signal 
    Vector<T>&          D,     // spike form (empty on input)
    const int           ndist  // length of negative wave 
    ) {

    // calculate signals energy at the beginning
    const int n=D.l(); // temporarily change D's size to calculate energy
    D.l(ndist);
    D.copy(X.ptr());
    D*=D;
    T max_energy = D.sum();
    D.l(n);

    // now find max energy across all time samples 
    T energy = max_energy;
    int max_ind = 0;
    for (int j=0; j<X.l()-n; ++j) {
        energy -= pow(X(j),2);
        energy += pow(X(j+ndist),2);
        if (energy > max_energy){
            max_ind = j+1;
            max_energy = energy;
        }
    }
    D.zeros();
    D.copy(X.pos(max_ind));
    D/=D.norm2();
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION INIT_SPIKE                                                 ////
///////////////////////////////////////////////////////////////////////////////
////       -initializes new spike:                                         ////
////       -if (rand_init): use random spike occurrence                    ////
////       -else: use least well fitted spike                              ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void init_spike(
    const Vector<T>&        X,                  // signal  
    const Vector<T>&        R,                  // residual
    const Matrix<T>&        D,                  // spike templates
          Vector<T>&        d,                  // new spike template
    const std::vector<int>& tau,                // spike latencies
    const std::vector<T>&   coeffs,             // spike coefficients
    const std::vector<int>& labels,             // spike labels
    const std::vector<T>&   fits,               // spike fit values
    const int               ndist,              // length of negative wave
    const bool              rand_init=false     // random initialization
     ) {
    
    // choose index for spike initialization
    int ind;
    if (rand_init) ind = rand()%tau.size();
    else ind = std::min_element(fits.begin(),fits.end()) - fits.begin();

    // initialize d
    d.zeros();
    d.l(ndist);
    d.copy(R.pos(tau[ind]));
    d.add(coeffs[ind],D.col(labels[ind]));
    d.l(D.m());
#ifdef DEBUG_SPIKE_INIT
        mexFigure();
        mexPrintf("DEBUG: Showing spike initialization!\n");
        mexPlot(d.l(),X.pos(tau[ind]),"b");
        mexHoldOn();
        mexPlot(d.l(),R.pos(tau[ind]),"r");
        mexPlot(d.l(),d.ptr(),"g");
        mexTitle("signal (b), residual (r), initialization (g)");
        mexEvalString("drawnow");
        mexPause();

        mexFigure();
        mexPrintf("DEBUG: Showing whole signal, init at %d!\n", tau[ind]);
        mexPlot(X.l(),X.pos(0),"b");
        mexHoldOn();
        mexPlot(R.l(),R.pos(0),"r");
        Vector<T> temp(R.l());
        temp.copy(X.ptr());
        temp-=R;
        mexPlot(temp.l(),temp.ptr(),"g");
        mexTitle("signal (b), residual (r), approximations (g)");
        mexEvalString("drawnow");
        mexPause();
#endif
    d/=d.norm2();
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION SPARSE_CODING                                              ////
///////////////////////////////////////////////////////////////////////////////
////       -detects spikes and estimates coefficients                      ////
////       -encoding is performed by matching pursuit over all translated  ////
////        spikes; for efficiency a preselection of candidates is made    ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void sparse_coding(  
            Matrix<T>&          DtR,        // correlations spikes/signal
            Vector<T>&          R,          // residual
    const   Matrix<T>&          D,          // spike shapes
    const   CovShMat<T>&        DtD,        // correl. between shifted spikes 
            std::vector<int>    tau[],      // spike latencies (empty)
            std::vector<T>      coeffs[],   // spike coefficients (empty)
            Vector<T>&          DtRval,     // highest correlations (empty)
            Vector<int>&        DtRcomp,    // correlation indices (empty)
    const   T                   alpha,      // relative correlation threshold 
    const   int                 ndist,      // no. of time samples of peak
    const   int                 ndetects    // if >0, determines # of detections
    ) {
                
    ///// Preliminaries //////////////////////////////////////////////////////
    const int K=DtR.n();
    const int S=DtR.m();
    const int n=D.m();
    const int nblock=ndist;

    ///// clear latencies tau and coeffcients ////////////////////////////////
    for (int k=0; k<K; ++k) {
        tau[k].resize(0);
        coeffs[k].resize(0);
    }

    ///// Preselect detections from DtR above threshold //////////////////////
    DtRval.l(DtRval.maxLength());
    T thresh = alpha * DtR.max();
    int pos=-1;
    for (int k=0; k<K; ++k) {
        for (int i=0; i<S; ++i) {
            if (DtR(i,k) >= 0.5*thresh) {
                ++pos;
                DtRcomp[pos]=k*S+i;
                DtRval[pos] = DtR(i,k);
            }
        }
    }

    // sort these correlation values and their corresponding positions
    DtRval.l(++pos);
    quick_decr(DtRval.ptr(),DtRcomp.ptr(),0,DtRval.l());

    // first detection, index, class index, and position index 
    int maxInd = DtRcomp[0];
    int ik = maxInd/S;
    int is = maxInd%S;

    // determine second threshold, corresponding to negative spike wave 
    // this allows to exclude false detections arising from fitting slow
    // positive spike wave to slow fluctuations in the data
    Vector<T> temp1(ndist);
    Vector<T> temp2(ndist);
    temp1.copy(D.col(ik));
    temp1 /= temp1.norm2();
    temp2.copy(R.pos(is));
    T thresh2 = alpha * temp1.dot(temp2);
   
    ///// find all correlations above threshold /////////////////////////////// 
    T maxVal = 0.;
    T coeff = 0.;
    int count = 0;
    for (int i=0;;++i) {
        // first use preselected correlations, then check for any other left
        if (i<DtRval.l()) {
            maxInd = DtRcomp[i];
            maxVal= DtR[maxInd];
            if (maxVal<thresh) continue; 
        } else {
            maxInd = DtR.argmax();
            maxVal = DtR[maxInd];
            if (maxVal<thresh & ndetects == 0) break;
        }

        // determine spike number and position
        ik = maxInd/S;
        is = maxInd%S;
 
        // calculate spike coefficient
        Vector<T> Dk(D,ik);
        coeff = blas_dot<T>(D.m(),Dk.ptr(),1,R.pos(is),1); 

        // block spikes close to detection
        if (nblock>0) {
            for (int j= max(is-nblock,0); j<min(is+nblock+1,S); ++j)
                for (int k=0; k<K; ++k)
                    DtR(j,k)=-INFINITY;
        }

        // use second threshold to exclude detections with low correlation
        // with the fast negative spike wave
        temp1.copy(Dk.ptr());
        temp1 /= temp1.norm2();
        temp2.copy(R.pos(is));
        if (temp1.dot(temp2) <thresh2 & ndetects == 0) continue;

        // update residual
        Dk.addTo(-coeff,R.pos(is));
        
        // store position and coefficient for spike ik 
        tau[ik].push_back(is);
        coeffs[ik].push_back(coeff);
        
        ++count;
        if (0 < ndetects & ndetects <= count) break;

        ///// Update correlations and fit values /////////////////////////////
        // define region where updates are necessary
        const int nleft= min(n-1,is);
        const int nright=min(n-1,S-is-1);
        const int nel = nleft+nright+1;
        const int first = is - nleft;

        // correlations with residual update by using spike-to-spike
        // correlations DtD
        for (int k=0; k<K; ++k) {
            Vector<T> DtRk(DtR.pos(first,k),nel);
            for (int l = -nleft; l<nright; ++l) {
                DtRk(l+nleft)+= -coeff * DtD(k,ik,l);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CALC_RES                                                   ////
///////////////////////////////////////////////////////////////////////////////
////       -calculate residual signal by subtracting all weighted,         ////
////         stretched spike occurrences                                   ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void calc_res(   
          Vector<T>&        R,          // residual signal         
    const Vector<T>&        X,          // original signal
    const Matrix<T>&        D,          // spikes
    const std::vector<int>  tau[],      // spike latencies 
    const std::vector<T>    coeffs[]    // spike coefficients 
    ) {
    
    // start with original signal X
    R.copy(X.ptr());
    
    // loop over all spikes 
    for (unsigned k=0; k<D.n(); ++k) {
        Vector<T> Dk(D,k);
        // loop over all occurrences of Dk
        for (unsigned l=0; l<tau[k].size(); ++l) {
            // subtract occurrence (weighted with coefficient) from residual
            Dk.addTo(-coeffs[k][l],R.ptr()+tau[k][l]); 
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CALC_SPIKE_FITS                                            ////
///////////////////////////////////////////////////////////////////////////////
////       -calculate residual signal by subtracting all weighted,         ////
////         stretched spike occurrences                                   ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void calc_spike_fits(
    const Vector<T>&        R,          // residual signal         
    const Matrix<T>&        D,          // spikes
          std::vector<T>&   fits,       // spike fits (empty)             
    const std::vector<int>& tau,        // spike latencies 
    const std::vector<T>&   coeffs,     // spike coefficients 
    const std::vector<int>& labels,     // spike labels      
          Vector<T>&        ref         // workspace (empty)
    ) {
    
    // loop over all spike occurrences
    ref.resize(D.m());
    fits.resize(tau.size());
    for (int l=0;l<tau.size();++l) {
        const Vector<T> Rk(const_cast<T*>(R.ptr()+tau[l]),D.m());
        ref.copy(R.ptr()+tau[l]);
        ref.add(coeffs[l],D.col(labels[l]));
        fits[l] = 1.-(Rk.norm2()/ref.norm2());
    }

}
        

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CONVERT_OUT                                                ////
///////////////////////////////////////////////////////////////////////////////
////       -convert representation to be ready for output                  ////
////       -old rep: lats and coeffs are stored separately for each spike  ////
////       -new rep: stored together, class labels indicate spike type     ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void convert_out(
          std::vector<int>& tau_out,    // latencies - out format 
          std::vector<T>&   coeffs_out, // coefficients - out format 
          std::vector<int>& labels_out, // labels - out format
    const std::vector<int>  tau[],      // latencies
    const std::vector<T>    coeffs[],   // coefficients  
    const int               K      
    ) {

    int len = 0;
    for (int k=0; k<K; ++k) len += tau[k].size();
    tau_out.resize(len);
    coeffs_out.resize(len);
    labels_out.resize(len);
    int offset = 0;
    int nel = 0;
    for (int k=0; k<K; ++k) {
        nel = tau[k].size();
        std::copy(tau[k].begin(),tau[k].end(),tau_out.begin()+offset);
        std::copy(coeffs[k].begin(),coeffs[k].end(),coeffs_out.begin()+offset);
        std::fill_n(labels_out.begin()+offset,nel,k);
        offset += nel;
    }
    std::vector<int> tau2(tau_out);
    quick_sort(tau_out,coeffs_out);
    quick_sort(tau2,labels_out);
    tau2.clear();
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION SPIKE_UPDATE                                               ////
///////////////////////////////////////////////////////////////////////////////
////       -updates spike shape, using the detected occurrences            ////
////       -handles overlaps by solving linear system (Toeplitz matrix)    ////
////       -centering maximal negative peak at given position              ////
////       -centering spike w.r.t. mean stretch                            ////
////       -normalizing spike at the end                                   ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void spike_update(   
    const Vector<T>&        X,          // original signal
          Vector<T>&        R,          // residual signal
          Matrix<T>&        D,          // spike matrix
          Matrix<T>&        Toep,       // Toeplitz matrix (empty)
          std::vector<int>  tau[],      // spike latencies
          std::vector<T>    coeffs[],   // spike coefficients
    const int               ncenter     // sample where peaks are aligned
    ) {

    const unsigned K=D.n();
    unsigned i,k,l,j,m;
    int overlap_left, overlap_right; 
    for (k=0; k<K; ++k) {
        if (tau[k].size()<2) continue;
        quick_sort(tau[k],coeffs[k]);
        Vector<T> Dk(D,k);
        // add to residual
        for (l=0; l<tau[k].size(); ++l) {
            Dk.addTo(coeffs[k][l],R.ptr()+tau[k][l]); 
        }
        Dk.zeros();

#ifdef DEBUG_SPIKE_UPDATE
        mexFigure();
        mexPrintf("DEBUG: Showing residual!\n");
        mexPlot(R.l(),R.ptr());
        mexTitle("DEBUG: residual");
        mexEvalString("drawnow");
        mexPause();
        mexFigure();
        mexTitle("DEBUG: spike accumulation");
#endif

        for (l=0; l<tau[k].size(); ++l) {
            Dk.add(coeffs[k][l],R.ptr()+tau[k][l]);
#ifdef DEBUG_SPIKE_UPDATE
            mexPlot(Dk.l(),R.pos(tau[k][l]));
            mexPlot(Dk.l(),Dk.ptr());
            mexEvalString("drawnow");
#endif
        }

        // set up toeplitz vector
        Vector<T> toep(Toep,0);
        toep[0]=0;
        for (l=0; l<tau[k].size(); ++l) toep[0]+=pow(coeffs[k][l],2);
        for (j=1; j<Dk.l(); ++j) {
            toep[j]=0;
            for (l=0; l<tau[k].size(); ++l) {
                for (m=l+1; m<tau[k].size(); ++m) {
                    if (tau[k][m]-tau[k][l]>=j) {
                        if (tau[k][m]-tau[k][l]==j) 
                            toep[j]+= coeffs[k][m]*coeffs[k][l];
                        break;
                    }
                }
            }
        } 

        // set up matrix from vector
        for (i=1;i<Dk.l();++i) {
            blas_copy<T>(Dk.l()-i,Toep.ptr(),1,Toep.pos(i,i),1);
            blas_copy<T>(i,Toep.ptr()+1,1,Toep.col(i),-1);
        }
        
        ptrdiff_t ipiv[Dk.l()];
        lapack_gesv<double>(Dk.l(),1,Toep.ptr(),Dk.l(),ipiv,Dk.ptr(),Dk.l());

        // update residual
        for (l=0; l<tau[k].size(); ++l) {
            Dk.addTo(-coeffs[k][l],R.ptr()+tau[k][l]); 
        }
    }

    // align peaks and normalize  
    for (k=0; k<K; ++k) {
        Vector<double> Dk(D,k);
        align_peak(Dk,ncenter);
        // test if norm is vanishing (shouldn't happen)
        if (Dk.norm2()<1e-20) {
            mexPrintf("EXCEPTION: Spike %d has vanishing norm!",k);
            mexErrMsgTxt("Exiting");
        }
        Dk/=Dk.norm2();
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CREATE_OUTPUT                                              ////
///////////////////////////////////////////////////////////////////////////////
////       -writes all spike representations back to output                ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void create_output(
    mxArray                 *plhs[],        // ptr to lhs-arguments
    const Matrix<T>&        D_out,          // learned spike templates 
    const std::vector<int>  tau_out[],      // learned latencies
    const std::vector<T>    coeffs_out[],   // learned coefficients
    const std::vector<int>  labels_out[],   // learned labels
    const std::vector<T>    fits_out[],     // spike fit values
    const Vector<T>&        res,            // residual energies
    const int               maxK            // max representation size
    ) { 

    // D
    mxArray* Dptr = mxCreateCellMatrix(1,maxK);
    int offset = 0;
    for (int K=0; K<maxK; ++K){
        offset = K*(K+1)/2;
        const Matrix<T> DK(const_cast<T*>(D_out.col(offset)),D_out.m(),K+1);
        mxSetCell(Dptr, K, mexMatrix2Mex(DK));
    }

    // tau, coeffs, labels 
    mxArray* Tptr = mxCreateCellMatrix(1,maxK);
    mxArray* Cptr = mxCreateCellMatrix(1,maxK);
    mxArray* Lptr = mxCreateCellMatrix(1,maxK);
    mxArray* Fptr = mxCreateCellMatrix(1,maxK);
    int maxLen = 0;
    for (int K=0; K<maxK; ++K) maxLen = max(maxLen,(int) tau_out[K].size());
    Vector<T> temp(maxLen);
    for (int K=0; K<maxK; ++K){
        temp.resize(tau_out[K].size());
        temp.l(tau_out[K].size());
        // taus
        for (int l=0; l<tau_out[K].size();++l) temp(l) = tau_out[K][l] + 1.;
        mxSetCell(Tptr, K, mexMatrix2Mex(temp));
        // coeffs
        std::copy(coeffs_out[K].begin(),coeffs_out[K].end(),temp.ptr());
        mxSetCell(Cptr, K, mexMatrix2Mex(temp));
        // labels
        std::copy(labels_out[K].begin(),labels_out[K].end(),temp.ptr());
        mxSetCell(Lptr, K, mexMatrix2Mex(temp));
        // fits      
        std::copy(fits_out[K].begin(),fits_out[K].end(),temp.ptr());
        mxSetCell(Fptr, K, mexMatrix2Mex(temp));
    }

    ///// create a structure //////////////////////////////////////////////////
    const char* field_names[] = {"D","latencies","coeffs","labels","fits",
        "residual"};
    const int nfields = sizeof(field_names)/sizeof(field_names[0]);
    plhs[0] = mxCreateStructMatrix(1,1,nfields,field_names); 
    mxArray* in[nfields];
    
    ///// copy data into struct ///////////////////////////////////////////////
    in[0] = Dptr; 
    in[1] = Tptr; 
    in[2] = Cptr; 
    in[3] = Lptr; 
    in[4] = Fptr; 
    in[5] = mexMatrix2Mex(res); 
    for (int i=0; i<nfields; ++i) mxSetField(plhs[0],0,field_names[i],in[i]);
}

      

///////////////////////////////////////////////////////////////////////////////
////   MAIN FUNCTION                                                       ////
///////////////////////////////////////////////////////////////////////////////
////       -main function called by matlab                                 ////
///////////////////////////////////////////////////////////////////////////////
void mexFunction(
    int             nlhs,       // # of lhs-arguments                  
    mxArray         *plhs[],    // pointer to lhs-arguments
    int             nrhs,       // # of rhs-arguments                  
    const mxArray   *prhs[]     // pointer to rhs-arguments
    ) {

    // initialize timers and random number generator
    timer timeMISC, timeCOV, timeCOR, timeSC, timeDU, timeALL;
    timeALL.start();
    timeMISC.start();
    srand(1);

    // parse parameters and set up signal and dictionary
    ParamStruct<double> param;
    Vector<double> X;
    Matrix<double> D;
    parse_parameters(param,X,D,nlhs,plhs,nrhs,prhs);
    const unsigned maxK = D.n();

    if (param.verbose) {
        mexPrintf("***********************************************\n");
        mexPrintf("****Hierarchical Multiclass Spike Learning*****\n");
        mexPrintf("***********************************************\n");
        mexPrintf("Parameters: \n");
        mexPrintf("  -maximal size of spike represenations: %d\n", maxK);
        mexPrintf("  -iterations per spike representation: %d\n", param.iter);
        mexPrintf("  -relative detection threshold alpha: %g\n", param.alpha);
        mexPrintf("  -number of samples to peak: %d\n", param.ncenter);
        mexPrintf("  -number of samples of negative wave: %d\n", param.ndist);
        if (param.ndetects>0)
            mexPrintf("  -number of detections manually set to: %d\n"
                    , param.ndetects);
        if (param.no_update>0) 
            mexPrintf("  -no spike updates, only 1 detection");
        mexPrintf("\n");
    }

    ///// Initializations /////////////////////////////////////////////////////
    Vector<double> R(X.l());
    CovShMat<double> DtD(maxK,D.m());
    CorShMat<double> DtX(R.l(),D.m(),maxK,param.fft);
    Matrix<double> Toep(D.m(),D.m());
    DtX.setX(X);
    std::vector<int> tau[maxK];
    std::vector<double> coeffs[maxK];
    Vector<int> DtRcomp(maxK*DtX.m());
    Vector<double> DtRval(maxK*DtX.m());
    Matrix<double> Dwork(2*D.m(),maxK);
    Vector<double> work(3*D.m());
    double normX = X.norm2();
    
    // Output stuff
    const int Kout = (maxK*(maxK+1))/2;
    std::vector<int>    tau_out[maxK];
    std::vector<double> coeffs_out[maxK];
    std::vector<int>    labels_out[maxK];
    std::vector<double> fits_out[maxK];
    Vector<double>      res_out(maxK);
    Matrix<double>      D_out(D.m(),Kout);

    // initialize residual with X 
    R.copy(X.ptr());

    timeMISC.stop();
    ///// Learn representations for different numbers of spike classes K ////// 
    for (int K=1; K<=maxK; ++K) {
        
        timeMISC.start();
        if (param.verbose) { 
            if (K==1) mexPrintf("\n**********Learning %d spike class***************\n", K);
            else      mexPrintf("\n**********Learning %d spike classes*************\n", K);
            mexEvalString("drawnow");
        }
        
        ///// Add new class to spike matrix ///////////////////////////////////
        D.n(K);
        Vector<double> Dk(D,K-1);
        if (K>1) init_spike(X,R,D,Dk,tau_out[K-2],coeffs_out[K-2],labels_out[K-2],fits_out[K-2],param.ndist);
        Dk /= Dk.norm2();
#ifdef DEBUG_MODE
            mexPrintf("DEBUG: Showing all spikes after initialization!\n");
            for (int k=0; k<K; ++k) {
                mexFigure();
                Vector<double> Dkk(D,k);
                mexPlot(Dkk.l(),Dkk.ptr());
                mexTitle("DEBUG: spike after initialization");
                mexPause();
            }
#endif

        ///// Reserve memory for latencies and coefficients of new spike //////
        tau[K-1].reserve(1000);
        coeffs[K-1].reserve(1000);

        ///// Add column for new spike to correlation matrix DtX///////////////
        DtX.n(K);
        timeMISC.stop();
       
        int nattempts = 0;      // attempts made for reinitialization
        bool restart = false;      // restart after reinitialization
        for (int i=0;;++i) {
            if (param.verbose) {
                mexPrintf("Iteration %d of %d.\n", i, param.iter);
                mexEvalString("drawnow");
            }

            ///// Calculate correlations between spikes ///////////////////////
            timeCOV.start();
            if (param.verbose && i==0) { 
                mexPrintf("Calculating correlations between spikes...");
                mexEvalString("drawnow");
            }
            DtD.set(D);
            if (param.verbose && i==0) { 
                mexPrintf(" - done!\n");
                mexEvalString("drawnow");
            }
            timeCOV.stop();
            timeCOR.start();
            ///// Calculate correlations of spikes with signal ////////////////
            if (param.verbose && i==0) { 
                mexPrintf("Calculating correlations spikes/signal...");
                mexEvalString("drawnow");
            }
            DtX.update(D);
            if (param.verbose && i==0) { 
                mexPrintf(" - done!\n");
                mexEvalString("drawnow");
            }
            timeCOR.stop();
#ifdef DEBUG_MODE
            Vector<double> corr_0(DtX,0);
            mexFigure();
            mexPlot(corr_0.l(),corr_0.ptr());
            mexTitle("correlation values");
            mexPause();
#endif

            ///// Sparse coding ///////////////////////////////////////////////
            timeSC.start();
            if (param.verbose && i==0) { 
                mexPrintf("Sparse coding...");
                mexEvalString("drawnow");
            }
            R.copy(X.ptr());

            sparse_coding(DtX,R,D,DtD,tau,coeffs,DtRval,DtRcomp,param.alpha,param.ndist,param.ndetects);
            if (param.verbose && i==0) { 
                mexPrintf(" - done!\n");
                for (int k=0; k<K; ++k) mexPrintf("Spike %d found %d times.\n",k,tau[k].size());
                mexEvalString("drawnow");
            }
            timeSC.stop();
#ifdef DEBUG_MODE
            mexFigure();
            mexPlot(X.l(),X.ptr());
            mexHoldOn();
            mexPlot(R.l(),R.ptr(),"r");
            Vector<double> diff(R.l());
            diff.copy(X.ptr());
            diff-=R;
            mexPlot(diff.l(),diff.ptr(),"g");
            mexTitle("original (bl), residual (r), approx (g)");
            mexPause();
#endif

            ///// Check if spike needs to be reinitialized ////////////////////
            if (nattempts>-1){
                for (int k=0; k<K; ++k) {
                    if (tau[k].size()<3) {
                        if (nattempts > 3) {
                            if (param.verbose)
                                mexWarnMsgTxt("Too few detections of spike %d, "
                                    "despite multiple reinitializations");
                            nattempts = -1;
                            break;
                        }
                        if (param.verbose) 
                            mexPrintf("Only %d detections of spike %d, "
                                    "reinitializing\n",tau[k].size(),k);
                        convert_out(tau_out[K-1],coeffs_out[K-1],labels_out[K-1],
                                tau,coeffs,K);
                        calc_spike_fits(R,D,fits_out[K-1],tau_out[K-1],
                                coeffs_out[K-1],labels_out[K-1],work);
                        bool rand_init = true;
                        Vector<double> Dkk(D,k);
                        init_spike(X,R,D,Dkk,tau_out[K-1],coeffs_out[K-1],
                                labels_out[K-1],fits_out[K-1],param.ndist,
                                rand_init);
                        Dkk /= Dkk.norm2();
                        ++nattempts;
                        restart = true;
                        break;
                    }
                }
            }
            if (restart) {
                restart = false;
                if (param.verbose) mexPrintf("Restarting iterations\n");
                i=0;
                continue;
            }

            ///// Check if maximal number of iterations reached ///////////////
            if (i>=param.iter) break; 
            if (param.no_update) break;

            ///// Spike updates ///////////////////////////////////////////////
            timeDU.start();
            if (param.verbose && i==0) { 
                mexPrintf("Performing spike updates...");
                mexEvalString("drawnow");
            }
            spike_update(X,R,D,Toep,tau,coeffs,param.ncenter);
            if (param.verbose && i==0) { 
                mexPrintf(" - done!\n");
                mexEvalString("drawnow");
            }
#ifdef DEBUG_MODE
            mexPrintf("DEBUG: Showing all spikes after spike update!\n");
            mexFigure();
            for (int k=0; k<K; ++k) {
                Vector<double> Dkk(D,k);
                mexPlot(Dkk.l(),Dkk.ptr());
                mexTitle("DEBUG: spike after spike update");
                mexPause();
            }
#endif
            timeDU.stop();
            timeMISC.start();
            // check if some spike has vanishing norm (shouldn't happen) 
            for (int k=0; k<K; ++k) {
                Vector<double> Dk(D,k);
                double dummy =Dk.norm2();
                if (Dk.norm2()<1e-20) {
                    mexPrintf("EXCEPTION: Spike %d has vanishing norm!",k);
                    mexErrMsgTxt("Exiting");
                }
            }
            timeMISC.stop();
        }
        
        // save current representation
        convert_out(tau_out[K-1],coeffs_out[K-1],labels_out[K-1],tau,coeffs,K);
        calc_spike_fits(R,D,fits_out[K-1],tau_out[K-1],coeffs_out[K-1],
                labels_out[K-1],work);

        // save spike forms
        int offset = (K*(K-1))/2;
        for (int k=0; k<K; ++k) {
            Vector<double> Dk(D_out.col(offset+k),D_out.m());
            Dk.copy(D.col(k));
        }

        // save residual
        res_out(K-1) = R.norm2()/X.norm2();

        if (param.verbose) {
            mexPrintf("Finished iterations, %g percent of variance explained\n",
                    (X.norm2()-R.norm2())/X.norm2()*100);
            mexPrintf("Spikes detected in total: %d\n", tau_out[K-1].size());
            //mexPrintf("Latencies: ");
            //for (int l=0; l<tau_out[K-1].size(); ++l) mexPrintf("%d ",tau_out[K-1][l]);
            //mexPrintf("\nCoefficients: ");
            //for (int l=0; l<coeffs_out[K-1].size(); ++l) mexPrintf("%g ",coeffs_out[K-1][l]);
            //mexPrintf("\nLabels: ");
            //for (int l=0; l<labels_out[K-1].size(); ++l) mexPrintf("%d ",labels_out[K-1][l]);
            mexEvalString("drawnow");
        }
    }

    // set up the output
    timeMISC.start();
    if (param.verbose) {
        mexPrintf("\nMax. representation size %d reached, creating output",maxK);
        mexEvalString("drawnow");
    } 
    create_output(plhs,D_out,tau_out,coeffs_out,labels_out,fits_out,res_out,
            maxK);
    if (param.verbose) { 
        mexPrintf(" - done!\n");
        mexEvalString("drawnow");
    }

    timeMISC.stop();
    timeALL.stop();
    if (param.verbose) {
        std::cout << std::endl;
        std::cout << "Time elapsed (total):            " << timeALL.elapsedTime() << std::endl;
        std::cout << "  -correlations between spikes:  " << timeCOV.elapsedTime() << std::endl;
        std::cout << "  -correlations spikes/signal:   " << timeCOR.elapsedTime() << std::endl;
        std::cout << "  -sparse coding:                " << timeSC.elapsedTime() << std::endl;
        std::cout << "  -spike updates:                " << timeDU.elapsedTime() << std::endl;
        std::cout << "  -misc:                         " << timeMISC.elapsedTime() << std::endl;
    }
}






