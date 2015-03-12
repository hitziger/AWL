/*!
 *
 *                File my_mex_sh.h
 *
 * Custom routines to facilitate work with mex files 
 *
 * Misc. subroutines: 
 * - conversion between arrays, vectors, matrices (matrix.h) and mex-arrays
 * - type checking
 *
 * Interactive (calls MATLAB-functions):
 * - tools for matlab plots
 * - interactive debugging
 *
 * */

#ifndef MY_MEX_SH_H
#define MY_MEX_SH_H

#include <algorithm>
#include <assert_custom.h>
#include <mex.h>
#include <matrix.h>




///////////////////////////////////////////////////////////////////////////////
////   SUBROUTINES                                                         ////
///////////////////////////////////////////////////////////////////////////////


// copies array of length len
template <typename T>
inline void _copy_array(const int len, const T* in, T* out) { 
    blas_copy<T>(*const_cast<int*>(&len),const_cast<T*>(in),1,out,1);
}

// converts 2d array to mex
inline mxArray* mexArray2Mex(const int m, const int n, const double* X) {
    mxArray* res = mxCreateDoubleMatrix(m,n,mxREAL);
    _copy_array(m*n,X,mxGetPr(res));
    return res;
}

// converts 1d array to mex                               
inline mxArray* mexArray2Mex(const int l, const double* X) {
    return mexArray2Mex(1,l,X);
}

// converts a Vector to mex matrix
inline mxArray* mexMatrix2Mex(const Vector<double>& v) {
    return mexArray2Mex(v.l(), v.ptr());
}

// converts a Matrix to mex matrix
mxArray* mexMatrix2Mex(const Matrix<double>& m) {
    return mexArray2Mex(m.m(),m.n(),m.ptr());
}

// converts a scalar to mex matrix
template <typename T>
mxArray* mexScalar2Mat(const T val) {
    double v=(double) val;
    return mexArray2Mex(1,1,&v);
}

// check the type of an array
template <typename T>
bool mexCheckType(const mxArray* array);

// check the type of an array (double)
template <>
inline bool mexCheckType<double>(const mxArray* array) {
    return mxGetClassID(array) == mxDOUBLE_CLASS && !mxIsComplex(array);
}

// check the type of an array (float)
template <> 
inline bool mexCheckType<float>(const mxArray* array) {
    return mxGetClassID(array) == mxSINGLE_CLASS && !mxIsComplex(array);
}

// check the type of an array (int)
template <> 
inline bool mexCheckType<int>(const mxArray* array) {
    return mxGetClassID(array) == mxINT32_CLASS && !mxIsComplex(array);
}

// check the type of an array (int)
template <> 
inline bool mexCheckType<bool>(const mxArray* array) {
    return mxGetClassID(array) == mxLOGICAL_CLASS && !mxIsComplex(array);
}

// get a scalar from a struct
template <typename T> 
inline T getScalarStruct(const mxArray* pr_struct, const char* name) {
    mxArray *pr_field = mxGetField(pr_struct,0,name);
    if (!pr_field) {
        mexPrintf("Missing field: ");
        mexErrMsgTxt(name);
    }
    return static_cast<T>(mxGetScalar(pr_field));
}

// get a string from a struct
inline void getStringStruct(const mxArray* pr_struct, const char* name,
        char* field, const mwSize length) {
    mxArray *pr_field = mxGetField(pr_struct,0,name);
    if (!pr_field) {
        mexPrintf("Missing field: ");
        mexErrMsgTxt(name);
    }
    mxGetString(pr_field,field,length);
}

// get a scalar from a struct
inline bool checkField(const mxArray* pr_struct,
      const char* name) {
    mxArray *pr_field = mxGetField(pr_struct,0,name);
    if (!pr_field) {
        mexPrintf("Missing field: ");
        mexPrintf(name);
        return false;
    }
    return true;
}

// get a scalar from a struct and provide a default value
template <typename T>
inline T getScalarStructDef(const mxArray* pr_struct, const char* name,
        const T def) {
    mxArray *pr_field = mxGetField(pr_struct,0,name);
    return pr_field ? (T)(mxGetScalar(pr_field)) : def;
}

// Create a m x n matrix
template <typename T>
inline mxArray* mexCreateMatrix(int m, int n);

// Create a m x n double matrix
template <> 
inline mxArray* mexCreateMatrix<double>(int m, int n) {
    return mxCreateNumericMatrix(static_cast<mwSize>(m),
            static_cast<mwSize>(n),mxDOUBLE_CLASS,mxREAL);
}

// Create a m x n float matrix
template <> 
inline mxArray* mexCreateMatrix<float>(int m, int n) {
    return mxCreateNumericMatrix(static_cast<mwSize>(m),
            static_cast<mwSize>(n),mxSINGLE_CLASS,mxREAL);
}

// Create a m x n int matrix
template <> 
inline mxArray* mexCreateMatrix<int>(int m, int n) {
    return mxCreateNumericMatrix(static_cast<mwSize>(m),
            static_cast<mwSize>(n),mxINT32_CLASS,mxREAL);
}


///////////////////////////////////////////////////////////////////////////////
////   INTERACTIVE MATLAB                                                  ////
///////////////////////////////////////////////////////////////////////////////

// opens new matlab figure      
inline void mexFigure() {
    mexCallMATLAB(0,NULL,0,NULL,"figure");
}

// drawnow      
inline void mexDrawnow() {
    mexCallMATLAB(0,NULL,0,NULL,"drawnow");
}

// plots an array      
void mexPlot(const int l, const double* y) {
    mxArray* in = mexArray2Mex(l,y);
    mexCallMATLAB(0,NULL,1,&in,"plot");
}
void mexPlot(const int l, const double* y, const char* color) {
    mxArray* in[] = { mexArray2Mex(l,y), mxCreateString(color) };
    mexCallMATLAB(0,NULL,2,in,"plot");
}
void mexPlot(const int l, const double* x, const double* y) {
    mxArray* in[] = { mexArray2Mex(l,y), mexArray2Mex(l,y) };
    mexCallMATLAB(0,NULL,2,in,"plot");
}
void mexPlot(const int l, const double* x, const double* y, const char* color) {
    mxArray* in[] = { mexArray2Mex(l,x), mexArray2Mex(l,y), mxCreateString(color) };
    mexCallMATLAB(0,NULL,3,in,"plot");
}
// plots histogram
void mexHist(const int l, const double* y) {
    mxArray* in = mexArray2Mex(l,y);
    mexCallMATLAB(0,NULL,1,&in,"hist");
}
void mexHist(const int l, const double* y, const int nbins) {
    mxArray* n = mexScalar2Mat(nbins);
    mxArray* in[] = { mexArray2Mex(l,y), mexScalar2Mat(nbins) };
    mexCallMATLAB(0,NULL,2,in,"hist");
}

// adds a title to a figure
void mexTitle(const char* msg) {
    mxArray* in = mxCreateString(msg);
    mexCallMATLAB(0,NULL,1,&in,"title");
}

// holds on existing figure 
void mexHoldOn() {
    mxArray* in = mxCreateString("on");
    mexCallMATLAB(0,NULL,1,&in,"hold");
}

// legend
void mexLegend(const int n, const char** s) {
    custom_assert(n<=10,"maximal 10 strings in legend");
    mxArray* in[10];
    for (int i=0; i<n; ++i) in[i]=mxCreateString(s[i]);
    mexCallMATLAB(0,NULL,n,in,"legend");
}

// decide if continue (with or without debug mode) 
void mexContinue(bool& debugging) {
    mxArray* out;
    mxArray* in[] = {mxCreateString("CONTINUE DEBUGGING?\n"
            "  <n>: continue without debugging\n"
            "  <q>: quit program\n"
            "  other: continue debugging\n"), 
        mxCreateString("s")};
    mexCallMATLAB(1,&out,2,in,"input");
    const int m = mxGetM(out);
    const int n = mxGetN(out);
    mxChar* panswer = mxGetChars(out);
    if (m==1 && n==1) {
        if (panswer[0] == 'q') mexErrMsgTxt("Aborting.");
        if (panswer[0] == 'n') {
            debugging=false;
            mexPrintf("Continuing without debug mode.\n");
            return;
        }
    }
}

// pause
void mexPause() {
    mexPrintf("Press any key to continue!\n");
    mexCallMATLAB(0,NULL,0,NULL,"pause");
}




#endif


