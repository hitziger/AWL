/*!
 *
 *                File mexEAWL.cpp
 *                by Sebastian Hitziger
 * 
 * function epoched adaptive waveform learning (EAWL) for decomposing a set of
 * epochs into latency adative waveforms
 *
 * Usage (in Matlab): D = mexEAWL(X,param);
 *
 * INPUT:
 * X: matrix containing epoched signals as columns
 * param: struct with options (see parsing below)
 *
 * OUTPUT: 
 * D: contains learned waveforms 
 *
 * The function uses implementations of linear algebra routines and least angle
 * regression (LARS) from the SPAMS software package by Julien Mairal under the 
 * GNU General Public License, see <http://www.gnu.org/licenses/>.
 * */


#include <mexutils.h>
#include <EAWL.h>

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
    const int nlhs,const int nrhs) {

    /*************************************************************************/
    /***** PARSE PARAMETERS **************************************************/
    /*************************************************************************/
    
    ///// check consistency of arguments //////////////////////////////////////
    if (!mexCheckType<T>(prhs[0])) mexErrMsgTxt("argument 1 should be scalar");
    if (!mxIsStruct(prhs[1]))      mexErrMsgTxt("argument 2 should be struct");


    ///// read signals ////////////////////////////////////////////////////////
    Matrix<T> *X;
    const mwSize* dimsX=mxGetDimensions(prhs[0]);   
    int n = static_cast<int>(dimsX[0]);             // sample points per signal
    int M = static_cast<int>(dimsX[1]);             // number of signals
    T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0])); 
    X = new Matrix<T>(prX,n,M);

    ///// define number of threads ////////////////////////////////////////////
    int NUM_THREADS = 1;

    ///// read initial dict ///////////////////////////////////////////////////
    mxArray* pr_D = mxGetField(prhs[1],0,"D");
    if (!pr_D) mexErrMsgTxt("initial dictionary has to be provided");
    T* prD = reinterpret_cast<T*>(mxGetPr(pr_D));
    const mwSize* dimsD=mxGetDimensions(pr_D);    
    int K = static_cast<int>(dimsD[1]);             // number of atoms
    int nD =  static_cast<int>(dimsD[0]);           // sample points per atom
    Matrix<T> D1(prD,nD,K);               
     
    ///// other parameters ////////////////////////////////////////////////////
    ParamLearn<T> param;
    param.iter                  // number of iterations of entire algorithm 
        = getScalarStruct<int>(prhs[1],"iter");
    param.lambda                // l_1 penalty
        = getScalarStructDef<T>(prhs[1],"lambda",1e-3);             
    param.lambda2               // l_2 penalty
        = getScalarStructDef<T>(prhs[1],"lambda2",10e-10);
    param.eps                   // change in dictionary for stopping
        = getScalarStructDef<T>(prhs[1],"eps",1e-16);
    param.L                     // max of active atoms, default all
        = getScalarStructDef<int>(prhs[1],"L",K);
    param.clean                 // cleans dictionary from unused atoms
        = getScalarStructDef<bool>(prhs[1],"clean",true);
    param.align                 // align atoms w.r.t. mean latency
        = getScalarStructDef<bool>(prhs[1],"align",true);
    param.posAlpha              // positivity constraint on coefficients
        = getScalarStructDef<bool>(prhs[1],"posAlpha",false);
    param.silent = getScalarStructDef<bool>(prhs[1],"silent",false);
    param.verbose = getScalarStructDef<bool>(prhs[1],"verbose",false);
    param.reorder = getScalarStructDef<bool>(prhs[1],"reorder",true);
    param.mcopies = getScalarStructDef<bool>(prhs[1],"mcopies",false);
    param.lars_lasso = getScalarStructDef<bool>(prhs[1],"lars_lasso",false);
    param.nThreads = NUM_THREADS;
    
    if (nlhs>2) param.all_conv_measures = true;

    /*************************************************************************/
    /***** TRAIN DICTIONARY **************************************************/
    /*************************************************************************/
      
    ///// initialize and run trainer: see dictsJA_circ.h //////////////////////
    Trainer<T> trainer(D1,*X,param);
    trainer.train();

    /*************************************************************************/
    /***** SET UP OUTPUT DATA ************************************************/
    /*************************************************************************/

    ///// copy D to output ////////////////////////////////////////////////////
    plhs[0] = createMatrix<T>(nD,K);
    T* prD2 = reinterpret_cast<T*>(mxGetPr(plhs[0]));
    trainer.getD(prD2,K*nD);

    ///// set up additional info (optional) ///////////////////////////////////
    if (nlhs>1) {
        const int n_fields = 2;
        const char* field_names[n_fields];
        int field_count = 0;
        mxArray* fields[n_fields];
        T* data=NULL;
        int* dataInt;
        data = createMatrixField<T>(field_names,field_count,"A",fields,K,M);
        trainer.getA(data,K*M);
        dataInt = createMatrixField<int>(field_names,field_count,"Delta",fields,K,M);
        trainer.getDelta(dataInt,K*M);
        plhs[1] = createStructMex(field_names,field_count,fields);
    }
    if (nlhs>1) {
        const int n_fields = 4;
        const char* field_names[n_fields];
        int field_count = 0;
        mxArray* fields[n_fields];
        T* data=NULL;
        const int err_length=trainer.getIter();
        data = createMatrixField<T>(field_names,field_count,"av_error",fields,
                1,err_length);
        trainer.getErrors(data,err_length);
        data = createMatrixField<T>(field_names,field_count,"av_error_reg",fields,
                1,err_length);
        trainer.getRegErr(data,err_length);
        data = createMatrixField<T>(field_names,field_count,"change_dict",fields,
                1,err_length);
        trainer.getChangeD(data,err_length);
        data = createMatrixField<T>(field_names,field_count,"time",fields,
                1,1);
        *data = trainer.getT();
        plhs[2] = createStructMex(field_names,field_count,fields);
    }


    /*************************************************************************/
    /***** CLEAN UP **********************************************************/
    /*************************************************************************/

    delete(X);
}



void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) mexErrMsgTxt("Bad number of input arguments");
    if ((nlhs != 1) && (nlhs != 2) && (nlhs != 3))
        mexErrMsgTxt("Bad number of output arguments");
    if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS)
        callFunction<double>(plhs,prhs,nlhs,nrhs);
    else
        mexErrMsgTxt("Currently only implementation for double");
}




