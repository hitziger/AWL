     
/*!
 * \file
 *
 *                File mexADSpike.cpp
 *
 * Usage (in Matlab): res = mexADSpike(x,param);
 * INPUT:
 * x: vector of dimension Nx1
 * param: struct with options (see ParamStruct and function parse_parameters)
 *
 * OUTPUT: 
 * res: struct containing spike representation (see function create_output)
 *
 * */

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

#ifdef DEBUG_MODE
bool debugging_active=true;
#endif


///////////////////////////////////////////////////////////////////////////////
////   STRUCT PARAM_DICT_LEARN                                             ////
///////////////////////////////////////////////////////////////////////////////
////       -stores parameters for training                                 ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct ParamStruct {
    int     iter;           // number of iterations for learning 
    int     ndist;          // time samples blocked around detection
    T       maxstretch;     // maximal stretch extent
    int     nstretch;       // number of discrete stretches
    int     nfactor;        // factor for high resolution stretches
    int     ncenter;        // time sample where peak is centered
    T       alpha;          // relative detection threshold
    int     ndetects;       // number of spikes to detect
    bool    fft;            // use fft for correlation calculation
    bool    verbose;        // display various messages
};

///////////////////////////////////////////////////////////////////////////////
////   SUBROUTINE ALIGN_PEAK                                               ////
///////////////////////////////////////////////////////////////////////////////
////       -aligns maximal peak of spike template at given sample point    ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void align_peak(
    Vector<T>&  D,          // spike template
    int         ncenter     // align at this sampling point
    ) {

    int peakpos = D.argamax();
    int shift = ncenter - peakpos;
    if (shift > 0) {
        for (int l=D.l()-1; l>shift-1; --l)
            D(l) = D(l-shift);
        for (int l=shift-1; l>-1; --l)
            D(l) = 0;
    }
    if (shift < 0) {
        for (int l=0; l<D.l()+shift; ++l)
            D(l) = D(l-shift);
        for (int l=D.l()+shift; l<D.l(); ++l)
            D(l) = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION PARSE_PARAMETERS                                           ////
///////////////////////////////////////////////////////////////////////////////
////       -reads signal, spike template and parameters from input         ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void parse_parameters(
              ParamStruct<T>& param,    // calculation parameters 
              Vector<T>& X,             // input signal
              Vector<T>& D,             // template spike
              Matrix<T>& Dc,            // peaks of stretched spikes
    const int nlhs,                     // length of plhs
              mxArray *plhs[],          // left hand side arguments
    const int nrhs,                     // length of prhs
    const     mxArray *prhs[]           // right hand side arguments
    ) {
    
    ///// check number of arguments ///////////////////////////////////////////
    if (nrhs != 2) mexErrMsgTxt("Two input arguments required!");
    if (nlhs > 1)
        mexErrMsgTxt("Maximally one output argument !");

    ///// check consistency of arguments //////////////////////////////////////
    if (!mexCheckType<T>(prhs[0])) mexErrMsgTxt("argument 1 should be scalar");
    if (!mxIsStruct(prhs[1]))      mexErrMsgTxt("argument 2 should be struct");
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("argument 1 should be of double precision");

    ///// read signal /////////////////////////////////////////////////////////
    const mwSize* dimsX=mxGetDimensions(prhs[0]);   
    int N = static_cast<int>(dimsX[0]);             // sample points per signal
    if (dimsX[0]<2) mexErrMsgTxt("argument 1 should be a column vector\n");
    if (dimsX[1]!=1) mexErrMsgTxt("argument 1 should be a column vector\n");
    T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0])); 
    X.resize(N);
    X.set(prX);
    
    ///// read template spike /////////////////////////////////////////////////
    mxArray* pr_D = mxGetField(prhs[1],0,"D");
    param.ndist                // distance to previous spike 
        = getScalarStruct<int>(prhs[1],"ndist");
    if (param.ndist<=0) mexErrMsgTxt("ndist has to be positive");
    unsigned n; 
    if (!pr_D) {
        mexWarnMsgTxt("initial spike template D should be provided, " 
                "initializing at maximal energy");
        if (!mxGetField(prhs[1],0,"n")) mexErrMsgTxt("average spike length n "
                "has to be provided when template is missing!");
        n = getScalarStruct<unsigned>(prhs[1],"n");
        if (param.ndist>n) mexErrMsgTxt("ndist cannot be larger than n");
        D.resize(n);
        init_spike(X,D,param.ndist);
    } else {
        T* prD = reinterpret_cast<T*>(mxGetPr(pr_D));
        const mwSize* dimsD=mxGetDimensions(pr_D);
        if (dimsD[0]<2) mexErrMsgTxt("spike template should be a column vector\n");
        if (dimsD[1]!=1) mexErrMsgTxt("spike template should be a column vector\n");
        n = static_cast<unsigned>(dimsD[0]);  // number of sample points of spike
        if (param.ndist>n) mexErrMsgTxt("ndist cannot be larger than template length");
        D.resize(n);
        D.copy(prD);               
        D/=D.norm2();
    }

    ///// other parameters ////////////////////////////////////////////////////
    param.verbose = getScalarStructDef<bool>(prhs[1],"verbose",false);
    param.iter                  // number of iterations 
        = getScalarStructDef<int>(prhs[1],"iter",5);
    param.maxstretch            // fraction maximal stretch / minimal stretch
        = getScalarStructDef<T>(prhs[1],"maxstretch",5.);
    param.nstretch              // number of discrete stretches
        = getScalarStructDef<int>(prhs[1],"nstretch",101);
    param.nfactor                // factor for higher stretch resolution in output
        = getScalarStructDef<int>(prhs[1],"nfactor",11);
    if (param.nfactor%2 == 0) {
        param.nfactor += 1;
        if (param.verbose) mexPrintf("made nfactor odd, for convenience\n");
    }
    param.ncenter               // number of sample points left from peak 
        = getScalarStructDef<int>(prhs[1],"ncenter",n/2);
    param.alpha                 // 0<alpha<1
        = getScalarStructDef<T>(prhs[1],"alpha",0.1);             
    param.ndetects               // number of spikes to detect   
        = getScalarStructDef<int>(prhs[1],"ndetects",0);
    int temp = getScalarStructDef<int>(prhs[1],"fft",0);
    if (temp>0)  param.fft = true;          // force fft
    if (temp==0) param.fft = 10*log2(N)<n;  // automatic optimization
    if (temp<0)  param.fft = false;         // prohibit fft

    ///// ensure template spike is aligned correctly //////////////////////////
    align_peak(D,param.ncenter);

    ///// make Dc longer, as it will contain the stretched spikes /////////////
    //unsigned nlong = 1+2*floor(param.maxstretch * n/2.);
    Dc.resize(n,param.nstretch);
} 
 
///////////////////////////////////////////////////////////////////////////////
////   FUNCTION INIT_SPIKE                                                 ////
///////////////////////////////////////////////////////////////////////////////
////       -initializes first spike if not provided:                       ////
////         sliding window is used to find maximal energy in signal       ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void init_spike(
    const Vector<T>& X,     // signal 
    Vector<T>& D,           // spike form (empty)
    const int ndist         // peak to peak distance, used for initialization
    ) {

    // calculate signals energy at the beginning
    const int n=D.l(); // change D's size, use it temporarily as as workspace
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
////   FUNCTION SPARSE_CODING                                              ////
///////////////////////////////////////////////////////////////////////////////
////       -learns latencies, coefficients, and stretches of detected      ////
////         spikes                                                        ////
////       -uses matching pursuit with preselection                        ////
////       -uses only low resolution stretches (Dc) for coding             ////
////       -recalculates stretches and coeffs for high resolution (Ds)     ////
///////////////////////////////////////////////////////////////////////////////
// TODO: verify resorting of correlation values after each step

template <typename T>
void sparse_coding(
    const Matrix<T>&      Ds,           // spikes (high stretch resolution)  
    const Matrix<T>&      Dc,           // spikes (low stretch resolution)
    Matrix<T>&            DctR,         // correlations Dc with signal 
    CovShMat<T>&          DtD,          // cross-correlation between spikes Dc 
    Vector<T>&            R,            // residual (initially original signal)
    const int             ndist,        // t. points blocked around det.
    std::vector<int>&     tau,          // latencies
    std::vector<int>&     stretchInd,   // stretch indices 
    std::vector<T>&       coeffs,       // coefficients
    Vector<T>&            DctRval,      // preselected correlations
    Vector<int>&          DctRcomp,     // preselected corr. indices
    T                     alpha,        // ratio max/min correlation
    Vector<T>&            work,         // work space
    const int             ndetects      // if >0, determines # of detections
    ) {


    ///// Preliminaries //////////////////////////////////////////////////////
    const int Kc=Dc.n();
    const int S=DctR.m();
    const int n=Dc.m();
    
    ///// Clear latencies tau and coeffcients ////////////////////////////////
    tau.resize(0);
    coeffs.resize(0);
    stretchInd.resize(0);

    ///// Preselect detections from DctR with threshold //////////////////////
    T thresh = alpha * DctR.max();
    int pos=-1;
    for (int k=0; k<Kc; ++k) {
        const Vector<T> DctRk(DctR,k);
        for (int i=0; i<S; ++i) {
            if (DctR(i,k) >= 0.5*thresh) {
                ++pos;
                DctRcomp[pos]=k*S+i;
                DctRval[pos]=DctR(i,k);
            }
        }
    }

    ///// order candidates, beginning with largest correlation ///////////////
    DctRval.l(++pos);
    quick_decr(DctRval.ptr(),DctRcomp.ptr(),0,DctRval.l());
    
    ///// iteratively detect largest correlations and store parameters ///////
    int maxInd;
    T maxVal;
    int count = 0;
    for (int i=0;; ++i) {
        if (i<DctRval.l()) {
            maxInd = DctRcomp[i];           // largest correlation index
            maxVal= DctR[maxInd];
            if (maxVal<thresh) continue;    // continue if value too small
        } else {
            maxInd = DctR.argmax();
            maxVal = DctR[maxInd];
            if (maxVal<thresh & ndetects == 0) break;
        }
        int id_stretch=maxInd/S;            // stretch index
        int is=maxInd%S;                    // get position of detection
 

        ///// correct correlations that change with current detection ////////
        // define surroundings of detection
        const int nleft= min(n-1,is);
        const int nright=min(n-1,S-is-1);
        const int nel = nleft+nright+1;
        const int first = is - nleft;
        int nleft_test=0;
        for (int k=0; k<Kc; ++k) {
            Vector<T> DctRk(DctR.pos(first,k),nel);
            for (int l = -nleft; l<nright; ++l) {
                DctRk(l+nleft)+= -maxVal * DtD(k,id_stretch,l);
            }
        }

        ///// block correlations with indices close to current detection /////
        if (ndist>0) {
            int j= max((int) is-ndist,0);
            for (; j<min(is+ndist+1,S); ++j) {
                for (int k=0; k<Kc; ++k) {
                    DctR(j,k)=-INFINITY;
                }
            }
        }
        const int Ks=Ds.n();  
        const int res_factor = Ks/Kc;
        int stretch_Ds = id_stretch * res_factor+res_factor/2;

        // recalculate exact coefficient and stretch        
        T val = 0.;
        const int offset = id_stretch * res_factor; 
        for (int k=offset; k<offset+res_factor; ++k) {
            val = blas_dot<T>(Ds.m(),const_cast<T*>(Ds.col(k)),1,R.pos(is),1);
            if (val>maxVal) {
                stretch_Ds = k;
                maxVal = val;
            }
        }
        // update residual
        Vector<T> Dsk(Ds,stretch_Ds);
        Dsk.addTo(-maxVal,R.pos(is));
 
        tau.push_back(is);                  // store latency
        coeffs.push_back(maxVal);           // store coefficient
        stretchInd.push_back(stretch_Ds);   // store stretch index
        ++count;
        if (0 < ndetects & ndetects <= count) break;

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
    const Matrix<T>&        Ds,         // stretched spikes
    const std::vector<int>& tau,        // latencies for each stretch
    const std::vector<int>& stretchInd, // latencies for each stretch
    const std::vector<T>&   coeffs      // coefficients for each stretch
    ) {

    // start with original signal X
    R.copy(X.ptr());
    // loop over all detections 
    for (unsigned l=0; l<tau.size(); ++l) {
        int k = stretchInd[l];
        Vector<T> Dsk(Ds,k);
        // subtract occurrence (weighted with coefficient) from residual
        Dsk.addTo(-coeffs[l],R.pos(tau[l])); 
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION SPIKE_UPDATE                                               ////
///////////////////////////////////////////////////////////////////////////////
////       -updates spike shape, using the detected occurrences            ////
////       -currently using only non-overlapping occurrences               ////
////       -centering maximal negative peak at given position              ////
////       -centering spike w.r.t. mean stretch                            ////
////       -normalizing spike at the end                                   ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void spike_update(   
    const Vector<T>&        X,          // original signal
          Vector<T>&        R,          // residual signal
          Vector<T>&        D,          // spike shape
    const Matrix<T>&        Ds,         // stretched spikes 
          std::vector<int>& tau,        // latencies of occurrences
          std::vector<int>& stretchInd, // stretch indices of occurrences
          std::vector<T>&   coeffs,     // coefficients of occurrences
    const Vector<T>&        stretchList,// stretches corresponding to indices
          int               ncenter,    // sample point where peak is aligned
          bool              verbose
    ) {

    const int nstretch=stretchList.l();

    ///// Calculate geometric mean of stretches //////////////////////////////

    // direct calculation of geometrical mean too instable, try with log
    T sum = 0.;
    T factor = 0.;
    for (int j=0; j<stretchInd.size(); ++j) {
        sum += log(stretchList[stretchInd[j]]);
        factor += 1.;
        //sum += coeffs[j]*log(stretchList[stretchInd[j]]);
        //factor += coeffs[j];
    }
    sum /= factor;
    T mean_stretch = exp(sum); 

    // TODO: delete or verbose this output
    if (verbose) mexPrintf("  -Spike update, mean stretch is %g\n",mean_stretch);


    ///// Perform update, using all non-overlapping spike occurrences ////////
    D.zeros();
    unsigned count=0; 
    for (int l=0; l<tau.size(); ++l) {
        // check that detection does not overlap
        if (l>0)
            if (tau[l]-tau[l-1]<D.l()*stretchList[stretchInd[l-1]]) 
                continue;
        if (l<tau.size()-1)
            if (tau[l+1]-tau[l]<D.l()*stretchList[stretchInd[l]]) 
                continue;
        ++count;
        // add occurrence of spike Dsk back to residual
        const Vector<T> Dsk(Ds,stretchInd[l]);
        Dsk.addTo(coeffs[l],R.ptr()+tau[l]); 

        // update spike D, performing mean-corrected stretching 
        T spacing = stretchList[stretchInd[l]]/mean_stretch;
        T pos1 = tau[l]+ncenter*((T) Ds.m()/D.l()-spacing); // start of spike
        extract_and_add_to_D(D,R,coeffs[l],spacing,pos1);
    }

    if (verbose) mexPrintf("  -%d non-overlapping spikes used for update\n", count);
    if (count == 0) mexErrMsgTxt("No isolated spikes found for update, "
            "aborting!");
    if (count < 10) mexWarnMsgTxt("WARNING: less than 10 isolated spikes found "
            "for update!");
    
    ///// Realignment of main peak ///////////////////////////////////////////
    align_peak(D,ncenter);
    
    // normalization (maybe unnecessary, later normalizations of Ds, Dc) 
    D/=D.norm2();
}

      
///////////////////////////////////////////////////////////////////////////////
////   SUBROUTINE EXTRACT_AND_ADD_TO_D                                     ////
///////////////////////////////////////////////////////////////////////////////
////       -extracts spike occurrence from residual and adds it to spike D,////
////         given its starting position and dilation                      ////
////       -uses linear interpolation to compensate for dilation           ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void extract_and_add_to_D(
          Vector<T>& D,         // spike shape 
    const Vector<T>& R,         // residual signal
    const T          coeff,     // coefficient of spike occurrence
    const T          spacing,   // dilation of occurrence
          T          pos        // starting position of occurrence
    ) {

    // loop over time samples of D while increasing position pos in residual
    for (int j=0; j<D.l(); ++j) {
        int left = floor(pos);
        // do linear interpolation (verify that we are inside residual)
        if  (left>=0 && left+1<R.l()) {
            D[j] += coeff*spacing*((pos-left) * R[left+1] + (1-(pos-left)) * R[left]);
        }
        pos+= spacing;
    } 
}

///////////////////////////////////////////////////////////////////////////////
////   SUBROUTINE SET_UP_DSK                                               ////
///////////////////////////////////////////////////////////////////////////////
////       -defines stretched spike with given dilation from template D    ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void set_up_Dsk(
          Vector<T>& Dsk,       // stretched spike 
    const Vector<T>& D,         // template spike form
    const T          spacing    // dilation of spike
    ) {

    int ncenter = D.argamax();            
    T pos = ncenter*(1- spacing*Dsk.l()/D.l());    
    Dsk.zeros();
    for (int j=0; j<Dsk.l(); ++j) {
        int left = floor(pos);
        // linear interpolation
        if (left<0 || left+1>= D.l()) Dsk[j]=0;
        else Dsk[j] = (pos-left) * D[left+1] + (1-(pos-left)) * D[left];
        pos+=spacing; 
    }
    Dsk/=Dsk.norm2();
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION SET_UP_DS                                                  ////
///////////////////////////////////////////////////////////////////////////////
////       -defines all stretched spikes from template D, given stretchList////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void set_up_Ds(
          Matrix<T>& Ds,            // stretched spikes 
    const Vector<T>& D,             // template spike form
    const Vector<T>& stretchList    // list of dilations
    ) {

    T spacing;
    int nstretch = stretchList.l();
    Ds.n(nstretch);
    for (int i=0; i<nstretch; ++i) {
        spacing = 1./stretchList[i];
        Vector<T> Dsk(Ds.col(i),Ds.m());
        set_up_Dsk(Dsk,D,spacing);
    }
}
///////////////////////////////////////////////////////////////////////////////
////   FUNCTION SET_UP_DC                                                  ////
///////////////////////////////////////////////////////////////////////////////
////       -sets up compact dictionary (low dilation resolution)           ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void set_up_Dc(
          Matrix<T>& Dc,    // stretched spikes (low dilation resolution)
    const Matrix<T>& Ds     // stretched spikes (high dilation resolution)
    ) {

    const int res_factor = Ds.n()/Dc.n();
    int Kc=Dc.n();
    Dc.zeros();
    int offset = res_factor/2;
    for (int k=0; k<Kc; ++k) {
        Vector<T> Dck(Dc,k);
        Dck.copy(Ds.col(offset + k*res_factor));
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CALC_SPIKE_FITS                                            ////
///////////////////////////////////////////////////////////////////////////////
////       -calculate fit values for spike occurrences                     ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void calc_spike_fits(
    const Vector<T>&        R,          // residual signal         
    const Matrix<T>&        Ds,         // spikes
          std::vector<T>&   fits,       // spike fits (empty)             
    const std::vector<int>& tau,        // spike latencies 
    const std::vector<T>&   coeffs,     // spike coefficients 
    const std::vector<int>& stretchInd  // stretch indices 
    ) {
    
    // loop over all spike occurrences
    Vector<T> ref(Ds.m());
    fits.resize(tau.size());
    for (int l=0;l<tau.size();++l) {
        const Vector<T> Rk(const_cast<T*>(R.pos(tau[l])),Ds.m());
        ref.copy(R.pos(tau[l]));
        ref.add(coeffs[l],Ds.col(stretchInd[l]));
        fits[l] = 1.-(Rk.norm2()/ref.norm2());
    }
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CREATE_OUTPUT                                              ////
///////////////////////////////////////////////////////////////////////////////
////       -writes learned spike shape, latencies, coefficients,           ////
////         and stretches back to output                                  ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void create_output(
        mxArray             *plhs[],        // rhs-arguments
  const Vector<T>&          D,              // spike shape
  const std::vector<int>&   tau,            // latencies
  const std::vector<T>&     coeffs,         // coefficients
  const std::vector<int>&   stretchInd,     // stretch indices
  const std::vector<T>&     fits,           // spike fits
  const Vector<T>&          stretchList,    // stretch factors
  const Matrix<T>&          Ds,             // all spike dilations
  const Vector<T>&          R               // residual signal
        ) { 

    ///// first put everything in proper matrix forms /////////////////////////
    unsigned nDetections = tau.size();

    Matrix<T> Tau(nDetections,1);
    for (int j=0; j<tau.size(); ++j) Tau(j,0)=(T) tau[j]+1;

    Matrix<T> A(nDetections,1);
    std::copy(coeffs.begin(),coeffs.end(),A.col(0));

    Matrix<T> StretchInd(nDetections,1);
    for (int j=0; j<nDetections; ++j) StretchInd(j,0)= stretchInd[j]+1;
    
    Matrix<T> Fits(nDetections,1);
    std::copy(fits.begin(),fits.end(),Fits.col(0));
    
    Matrix<T> StretchList(const_cast<T*>(stretchList.ptr()),stretchList.l(),1);

    Matrix<T> Dout(const_cast<T*>(D.ptr()),D.l(),1);
    Matrix<T> Rout(const_cast<T*>(R.ptr()),R.l(),1);
    

    ///// create a structure //////////////////////////////////////////////////
    const char* field_names[] = {"D","latencies","coefficients","stretchInd",
        "fits","stretchList","Ds","R"};
    const int nfields = sizeof(field_names)/sizeof(field_names[0]);
    plhs[0] = mxCreateStructMatrix(1,1,nfields,field_names); 
    mxArray* in[nfields];

    ///// copy D to output ////////////////////////////////////////////////////
    in[0] = mexMatrix2Mex(Dout);
    in[1] = mexMatrix2Mex(Tau);
    in[2] = mexMatrix2Mex(A);
    in[3] = mexMatrix2Mex(StretchInd);
    in[4] = mexMatrix2Mex(Fits);
    in[5] = mexMatrix2Mex(StretchList);
    in[6] = mexMatrix2Mex(Ds);
    in[7] = mexMatrix2Mex(Rout);
    
    for (int i=0; i<nfields; ++i) mxSetField(plhs[0],0,field_names[i],in[i]);
}


///////////////////////////////////////////////////////////////////////////////
////   FUNCTION MEX_AD_SPIKE                                               ////
///////////////////////////////////////////////////////////////////////////////
////       -main function                                                  ////
////       -detects all spike occurrences in input signal                  ////
////       -determines their latencies, dilations, and coefficients        ////
////       -learns template spike shape                                    ////
///////////////////////////////////////////////////////////////////////////////
void mexFunction(
          int     nlhs,     // length of plhs 
          mxArray *plhs[],  // left hand side arguments, provided in matlab
          int     nrhs,     // length of prhs
    const mxArray *prhs[]   // right hand side arguments, provided in matlab
    ) {
    
#ifdef DEBUG_MODE
    mexPrintf("*******************************************************************\n");
    mexPrintf("******************** D E B U G G I N G*****************************\n");
    mexPrintf("*******************************************************************\n");
#endif

    ///// Define timers for performance measure ///////////////////////////////
    timer timeSETUP, timeSHIFT, timeCORR, timeSTRETCH, timeSC, timeSPIKE, timeMISC,
          timeALL;
    timeALL.start();
    timeSETUP.start();

    ///// Parse parameters and set up signal and template spike shape /////////
    ParamStruct<double> param;
    Vector<double> X;           // original signal                       
    Vector<double> D;           // template spike shape
    Matrix<double> Dc;          // negative waves of stretched spikes
    parse_parameters(param,X,D,Dc,nlhs,plhs,nrhs,prhs);

#ifdef DEBUG_MODE
    mexPrintf("DEBUG: Showing input signal!\n");
    mexFigure();
    mexPlot(X.l(), X.ptr());
    mexTitle("DEBUG: input signal");
    mexPause();
    mexPrintf("DEBUG: Showing provided spike template!\n");
    mexFigure();
    mexPlot(D.l(), D.ptr());
    mexTitle("DEBUG: provided spike template");
    mexPause();
#endif

    if (param.verbose) {
        mexPrintf("\n");
        mexPrintf("*******************************************************************\n");
        mexPrintf("******************Stretchable Spike Learning***********************\n");
        mexPrintf("*******************************************************************\n");
        mexPrintf("Parameters: \n");
        mexPrintf("  -number of iterations: %d\n", param.iter);
        mexPrintf("  -maximal stretch factor: %g\n", param.maxstretch);
        mexPrintf("  -stretched copies for detection: %d\n", param.nstretch);
        mexPrintf("  -stretch resolution factor for output: %d\n", param.nfactor);
        mexPrintf("  -centering peak at sample point: %d\n",param.ncenter);
        if (param.ndetects>0)
            mexPrintf("  -number of detections manually set to: %d\n", param.ndetects);
#ifdef FFT_CONV
        if (param.fft) mexPrintf("  -calculating correlations with fft\n");
#else
        if (param.fft) mexPrintf("  -WARNING: can use fft only if compiled with -DFFT_CONV\n");
#endif
        mexEvalString("drawnow");
    }

    ///// Initializations ////////////////////////////////////////////////////
    Vector<double> stretchList(param.nstretch * param.nfactor); 
    Vector<double> R(X.l());                              // residual signal
    R.copy(X.ptr());
    Matrix<double> Ds(Dc.m(),stretchList.l());            // stretched spikes
    CorShMat<double> DctX(R.l(),Dc.m(),Dc.n(),param.fft); // correlations
    CovShMat<double> DtD(Dc.n(),Dc.m());           
    DctX.setX(X);
    std::vector<int> tau;                   // spike latencies
    std::vector<int> tau2;                  // copy of tau
    std::vector<int> stretchInd;            // spike's dilation indicies
    std::vector<double> coeffs;             // spike coefficients 
    std::vector<double> fits;               // goodness of fit 
    Vector<int> DtRcomp(Dc.n()*DctX.m());   // used in sparse coding
    Vector<double> DtRval(Dc.n()*DctX.m()); // used in sparse coding 
    Vector<double> work;
    
    ///// Set up stretchList /////////////////////////////////////////////////
    stretchList[0]=1./sqrt(param.maxstretch);
    double factor = exp(log(param.maxstretch)/(double) (stretchList.l()-1));
    for (int i=1; i<stretchList.l(); ++i)
        stretchList[i]=stretchList[i-1]*factor;

    if (param.verbose) mexPrintf("  -min/max stretch: %g, %g\n",
            stretchList[0], stretchList[stretchList.l()-1]);
    timeSETUP.stop();

    ///// Iterate over sparse coding and spike update ////////////////////////
    if (param.verbose) 
        mexPrintf("\n**********Starting iterations********\n");
    for (int i=0;; ++i) {
        if (param.verbose) {
            mexPrintf("Iteration %d of %d\n", i, param.iter);
            mexEvalString("drawnow");
        }

        ///// Setting up stretched spikes ////////////////////////////////////
        timeSTRETCH.start();
        set_up_Ds(Ds,D,stretchList);
        set_up_Dc(Dc,Ds);

        // calculate cross-correlations between stretched spikes /////////////
        DtD.set(Dc);

        if (param.verbose) {
            mexPrintf("  -Stretched spikes set up\n");
            mexEvalString("drawnow");
        }

#ifdef DEBUG_MODE
        if (debugging_active) {
            mexPrintf("DEBUG: Showing stretched spikes\n");
            mexFigure();
            for (int j=0; j<Ds.n(); ++j) {
                mexPlot(Ds.m(), Ds.col(j)); 
                mexHoldOn();
            }
            mexTitle("DEBUG: Stretched spike forms");
            mexPause();
            mexPrintf("DEBUG: Stretched spikes (low stretch resolution)\n");
            mexFigure();
            for (int j=0; j<Dc.n(); ++j) {
                mexPlot(Dc.m(), Dc.col(j)); 
                mexHoldOn();
            }
            mexTitle("DEBUG: Stretched spikes (low stretch resolution)\n");

            mexPause();
            mexContinue(debugging_active);
        }
#endif
        timeSTRETCH.stop();

        ///// Calculating correlations ///////////////////////////////////////
        timeCORR.start();
        DctX.update(Dc);
        timeCORR.stop();
        if (param.verbose) {
            mexPrintf("  -Correlations calculated\n");
            mexEvalString("drawnow");
        }

        ///// Sparse coding ///////////////////////////////////////////////////
        timeSC.start();
        R.copy(X.ptr());
        sparse_coding(Ds,Dc,DctX,DtD,R,param.ndist,tau,stretchInd,coeffs,DtRval,
                DtRcomp,param.alpha,work,param.ndetects);
        timeSC.stop();
        if (param.verbose) {
            mexPrintf("  -Sparse coding done\n");
            mexEvalString("drawnow");
        }
        ///// Sort detections in temporal order /////////////////////////////////// 
        tau2 = tau;
        quick_sort(tau,coeffs);
        quick_sort(tau2,stretchInd);

        ///// Calculate residual //////////////////////////////////////////////
        timeSHIFT.start();
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

        if (param.verbose) {
            mexPrintf("  -Spike found %d times\n", tau.size());
            mexEvalString("drawnow");
        }
        timeSHIFT.stop();

        ///// Check if maximal number of iterations reached ///////////////////
        if (i>=param.iter) {
            if (param.verbose) {
                mexPrintf("Finished iterations, %g percent of variance explained\n\n",
                        (X.norm2()-R.norm2())/X.norm2()*100);
                mexEvalString("drawnow");
            }
            break;
        }

        ///// Spike update ////////////////////////////////////////////////////
        timeSPIKE.start();
#ifdef DEBUG_MODE
        if (debugging_active) {
            mexPrintf("DEBUG: Showing spike before update!\n");
            mexFigure();
            mexPlot(D.l(),D.ptr());
            mexPrintf("mean of spike is: %g\n",D.mean());
            mexTitle("DEBUG: spike before update");
            mexHoldOn();
            mexPause();
        }
#endif
        spike_update(X,R,D,Ds,tau,stretchInd,coeffs,stretchList,param.ncenter,
                param.verbose);
#ifdef DEBUG_MODE
        if (debugging_active) {
            mexPrintf("DEBUG: Showing spike after update!\n");
            mexPlot(D.l(),D.ptr(),"r");
            mexTitle("DEBUG: before (blue) and after (red) update");
            mexPause();
        }
#endif
        timeSPIKE.stop();
        if (param.verbose) {
            mexPrintf("  -Spike update done\n");
            mexEvalString("drawnow");
        }
    }
    ///// Calculate fit values of spikes /////////////////////////////////////
    calc_spike_fits(R,Ds,fits,tau,coeffs,stretchInd);
    
    ///// Set up the output ////////////////////////////////////////////////// 
    if (param.verbose) {
        mexPrintf("Creating output\n");
        mexEvalString("drawnow");
    }
    timeMISC.start();
    if (nlhs==1) create_output(plhs,D,tau,coeffs,stretchInd,fits,stretchList,Ds,R);
    timeMISC.stop();
    timeALL.stop();
    if (param.verbose) {
        mexPrintf("Elapsed time in seconds\n");
        mexPrintf("  -total:               %g\n",timeALL.elapsedTime());
        mexPrintf("  -preparation:         %g\n",timeSETUP.elapsedTime());
        mexPrintf("  -spike covariances:   %g\n",timeSTRETCH.elapsedTime());
        mexPrintf("  -correlations:        %g\n",timeCORR.elapsedTime());
        mexPrintf("  -sparse coding:       %g\n",timeSC.elapsedTime());
        mexPrintf("  -spike update:        %g\n",timeSPIKE.elapsedTime());
        mexPrintf("  -miscellaneous:       %g\n",timeMISC.elapsedTime());
        mexPrintf("******************Finished Spike Learning**************************\n");
        mexEvalString("drawnow");
    }
}

