/*!
 * \file
 *
 *                File mexSpikeTemplateMatching.h
 * 
 * Usage (in Matlab): res = mexSpikeTemplateMatching(x,param);
 *
 * Function for detecting spikes in a signal x, based on cross-correlation
 * values with a template, returns spike positions pos
 * 
 * INPUT:
 * x: vector of dimension Nx1
 * param: struct with template and options (see function parse_parameters)
 *
 * OUTPUT: 
 * res: struct, contains 2 vectors of dimension nx1 (with n<N), 
 *      containing latencies and correlations at the detections
 *
 * */


#ifndef FFT_CONV
    #define FFT_CONV
#endif

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
////   STRUCT PARAM_STRUCT                                                 ////
///////////////////////////////////////////////////////////////////////////////
////       -stores input parameters                                        ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct ParamStruct {
    int             ndist;          // minimal peak-to-peak distance for detection
    int             ndetects;       // number of spikes to be detected
    T               alpha;          // correlation threshold for detection
    bool            fft;            // fft based cross correlation 
    bool            verbose;
};


///////////////////////////////////////////////////////////////////////////////
////   FUNCTION PARSE_PARAMETERS                                           ////
///////////////////////////////////////////////////////////////////////////////
////       -reads signal, spike template and parameters from input         ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void parse_parameters(
        ParamStruct<T>& param, 
        Vector<T>&      X,        
        Vector<T>&      d,
        const int       nlhs, 
        mxArray         *plhs[],
        const int       nrhs, 
        const mxArray   *prhs[]
) {
    
    ///// check number of arguments ///////////////////////////////////////////
    if (nrhs != 2) mexErrMsgTxt("Bad number of input arguments");
    if (nlhs != 1)
        mexErrMsgTxt("Bad number of output arguments");

    ///// check consistency of arguments //////////////////////////////////////
    if (!mexCheckType<T>(prhs[0])) mexErrMsgTxt("argument 1 should have scalar values");
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
    X.copy(prX);

    ///// read template spike /////////////////////////////////////////////////
    mxArray* pr_d = mxGetField(prhs[1],0,"D");
    unsigned n; 
    if (!pr_d) {
        mexWarnMsgTxt("initial spike template D should be provided, " 
                "initializing at maximal energy");
        if (!mxGetField(prhs[1],0,"n")) mexErrMsgTxt("average spike length n "
                "has to be provided when template is missing!");
        n = getScalarStruct<unsigned>(prhs[1],"n"); // sample points per spike 
        d.resize(n);
        init_first_spike(X,d);
    } else {
        T* prD = reinterpret_cast<T*>(mxGetPr(pr_d));
        const mwSize* dimsD=mxGetDimensions(pr_d);
        if (dimsD[0]<2) mexErrMsgTxt("spike template should be a column vector\n");
        if (dimsD[1]!=1) mexErrMsgTxt("spike template should be a column vector\n");
        n = static_cast<unsigned>(dimsD[0]);  // number of sample points of spike
        d.resize(n);
        d.copy(prD);               
        d/=d.norm2();
    }

    ///// other parameters ////////////////////////////////////////////////////
    param.ndist                 // minimal peak-to-peak distance between spikes
        = getScalarStructDef<int>(prhs[1],"ndist",ceil(0.05*n));
    if (param.ndist<0) mexErrMsgTxt("param.ndist cannot be nonnegative");
    if (param.ndist>N) mexErrMsgTxt("param.ndist cannot be greater than number of signal samples");
    param.ndetects             // if >0, determines number of detections 
        = getScalarStructDef<int>(prhs[1],"ndetects",0);
    param.alpha                 // 0<alpha<1, ratio of max/min correlation for detection 
        = getScalarStructDef<T>(prhs[1],"alpha",0.1);             
    param.verbose = getScalarStructDef<bool>(prhs[1],"verbose",false);
    param.fft =  getScalarStructDef<bool>(prhs[1],"fft",false) ? 
        true : (getScalarStructDef<bool>(prhs[1],"no_fft",false) ? 
                false : 4*log2(N)<n);

} 

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION CREATE_OUTPUT                                              ////
///////////////////////////////////////////////////////////////////////////////
////       -writes all spike detections and coefficients to output         ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void create_output(
        mxArray                 *plhs[],
        const std::vector<int>  tau,
        const std::vector<T>    coeffs
    ) { 

    // tau, coeffs 
    Vector<T> Tau(tau.size());
    Vector<T> Coeffs(coeffs.size());
    std::copy(tau.begin(),tau.end(),Tau.ptr());
    std::copy(coeffs.begin(),coeffs.end(),Coeffs.ptr());
   
    ///// create a structure //////////////////////////////////////////////////
    const char* field_names[] = {"latencies","coeffs"};
    const int nfields = sizeof(field_names)/sizeof(field_names[0]);
    plhs[0] = mxCreateStructMatrix(1,1,nfields,field_names); 
    mxArray* in[nfields];

    ///// copy data into struct ///////////////////////////////////////////////
    in[0] = mexMatrix2Mex(Tau); 
    in[1] = mexMatrix2Mex(Coeffs); 
    for (int i=0; i<nfields; ++i) mxSetField(plhs[0],0,field_names[i],in[i]);
}

///////////////////////////////////////////////////////////////////////////////
////   FUNCTION INIT_FIRST_SPIKE                                           ////
///////////////////////////////////////////////////////////////////////////////
////       -initializes first spike if not provided:                       ////
////         sliding window is used to find maximal energy in signal       ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void init_first_spike(const Vector<T>& X, Vector<T>& D) {

    const int n=D.l();
    Vector<T> work(X.l()-n+1);
    const T* pX = X.ptr()+n;
    D.copy(pX);
    D*=D;
    work[0] = D.sum();
    for (int j=0; j<work.l()-1; ++j) {
        int ind = j%n;
        work[j+1]=work[j]-D[ind];
        D[ind]=pow(pX[j],2);
        work[j+1]+=D[ind];
    }
    D.zeros();
    D.copy(X.ptr()+work.argmax()+n,n);
    D/=D.norm2();
}

      
///////////////////////////////////////////////////////////////////////////////
////   MAIN FUNCTION                                                       ////
///////////////////////////////////////////////////////////////////////////////
////       -this function is called by matlab                              ////
////       -detects spikes from in provided signal, given a template       ////
///////////////////////////////////////////////////////////////////////////////

void mexFunction(
        int             nlhs, 
        mxArray         *plhs[], 
        int             nrhs, 
        const mxArray   *prhs[]
        ) {
    

    ///// Parse parameters and set up signal and dictionary ///////////////////
    ParamStruct<double> param;
    Vector<double> X;
    Vector<double> D;
    parse_parameters(param,X,D,nlhs,plhs,nrhs,prhs);

    if (param.verbose) {
        mexPrintf("***********************************************\n");
        mexPrintf("****Spike Template Matching *******************\n");
        mexPrintf("***********************************************\n");
        mexPrintf("Parameters: \n");
        if (param.ndetects>0) {
            mexPrintf("  -number of detections manually set to: %d\n", param.ndetects);
        } else {
            mexPrintf("  -ratio min/max detected correlation: %g\n", param.alpha);
        }
        mexPrintf("  -peak-to-peak distance between spike detections: %d\n", param.ndist);
        mexPrintf("\n");
        mexEvalString("drawnow");
    }

    ///// Initializations /////////////////////////////////////////////////////
    CorShMat<double>    DtX(X.l(),D.l(),1,param.fft);
    Vector<int>         DtXind(DtX.m());
    Vector<double>      DtXval(DtX.m());
    std::vector<int>    tau;
    std::vector<double> coeffs;
    if (param.ndetects>0) {
        tau.reserve(param.ndetects);
        coeffs.reserve(param.ndetects);
    } else {
        tau.reserve(1000);
        coeffs.reserve(1000);
    }
    
    ///// Perform correlation-based detection /////////////////////////////////

    // calculate cross correlations between signal and template
    DtX.setX(X);
    DtX.update(D);

    // sort correlation values in decreasing order
    DtXval.copy(DtX.ptr());
    for (int i=0; i<DtXind.l(); ++i) DtXind(i)=i;
    quick_decr(DtXval.ptr(),DtXind.ptr(),0,DtXval.l());

    for (int i=0; i<DtX.m(); ++i) {
        int ind = DtXind(i);
        double val = DtX[ind];
        if ( val == -INFINITY ) continue;
        if ( (val < param.alpha) & (param.ndetects == 0) ) break;

        // store detection
        tau.push_back(ind);
        coeffs.push_back(val);

        // stop if given limit reached
        if (0 < param.ndetects & param.ndetects <= tau.size()) break;

        // block spikes close to detection
        if (param.ndist>0) {
            for (int j= max(ind - (int) param.ndist,0); j<min(ind + (int) param.ndist, (int) DtX.m()); ++j)
                DtX[j]=-INFINITY;
        }
    }

    ///// Write back to output ////////////////////////////////////////////////
    quick_sort(tau,coeffs);
    create_output(plhs,tau,coeffs);

}






