/*!
 *
 *                File EAWL.h
 *                by Sebastian Hitziger
 *
 * Contains class Trainer to learn waveform representations of a set of signal
 * epochs. Uses sparse coding techniques, implemented as a modified version of
 * least angle regression (LARS). 
 *
 * Called by mex-function in mexEAWL.cpp
 *
 * */

#ifndef EAWL_H
#define EAWL_H

#include <modifiedLARS.h>

#ifdef FFT_CONV
#include <fftw3.h>
#endif


/*****************************************************************************/
/*****************************************************************************/
/********* DECLARATIONS ******************************************************/
/*****************************************************************************/
/*****************************************************************************/

///////////////////////////////////////////////////////////////////////////////
////   STRUCT PARAM_LEARN                                                  ////
///////////////////////////////////////////////////////////////////////////////
////       -stores parameters for training                                 ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct ParamLearn {
    public:
        ParamLearn() : iter(0), nThreads(1), 
            lambda(T()), lambda2(T()), eps(1e-20), L(0), 
            clean(true), align(true), posAlpha(false), silent(false),
            verbose(false), spacing(1), all_conv_measures(false),
            nfD(1000), reorder(true), mcopies(false), lars_lasso(true) { };
        ~ParamLearn() { };
        int             iter;
        int             nThreads;
        T               lambda;
        T               lambda2;
        T               eps;
        int             L;
        bool            clean;
        bool            align;
        bool            posAlpha;
        bool            silent;
        bool            verbose;
        int             spacing;
        bool            all_conv_measures;
        int             nfD;
        bool            reorder;
        bool            mcopies;
        bool            lars_lasso;
};

///////////////////////////////////////////////////////////////////////////////
////   CLASS: TRAINER                                                      ////
///////////////////////////////////////////////////////////////////////////////
////       -called in mexEAWL                                              ////
////       -main method train: learns dictionary for given data            ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
class Trainer {
public:
    /// Constructor with initial dictionary
    Trainer(Matrix<T>& D, Matrix<T>& X, const ParamLearn<T>& param);

    void train();

    /// Accessors
    void getA(Matrix<T>& A) const            { A.copy(_A);                 };
    void getA(T* A,const int n) const        { _A.copyRaw(n,A);         };
    void getDelta(Matrix<int>& De) const     { De.copy(_Delta);            };
    void getDelta(int* De,const int n) const { _Delta.copyRaw(n,De);    };
    void getD(Matrix<T>& D) const            { D.copy(_D);                 };
    void getD(T* D,const int n) const        { _D.copyRaw(n,D);         };
    void getErrors(Vector<T>& e) const       { e.copy(_errors);            };
    void getErrors(T* e,const int n) const   { _errors.extractRaw(e,n);    };
    void getRegErr(Vector<T>& e) const       { e.copy(_regErr);            };
    void getRegErr(T* e,const int n) const   { _regErr.extractRaw(e,n);    };
    void getChangeD(Vector<T>& e) const      { e.copy(_changeD);           };
    void getChangeD(T* e,const int n) const  { _changeD.extractRaw(e,n);   };
    int getIter() const                      { return _i;          };
    T   getT() const                         { return _tnet;          };

private:
    ///// Forbid lazy copies //////////////////////////////////////////////////
    explicit Trainer<T>(const Trainer<T>& trainer);
    Trainer<T>& operator=(const Trainer<T>& trainer);

    ///// Routines used by train //////////////////////////////////////////////
    void _calc_res(Matrix<T>& R);
    void _clean_dict(Matrix<T>& G, const T maxCorrel = 0.99);
    void _align_dict();
    void _update_coeffs(const Vector<int>& ind, const Vector<T>& coeffs,
            const int m);
    void _subtr_contr_res(Matrix<T>& R, const int k);
    void _add_contr_res(Matrix<T>& R, const int k);
    void _update_atom(const int k, const Matrix<T>& R);
    void _reorder_dict();
    void _correct_signs();

    /// Members
    Matrix<T>       _A;         // coefficient matrix
    Matrix<int>     _Delta;     // latency matrix
    Matrix<T>       _D;         // dictionary
    Matrix<T>       _tempD;     // temporary dictionary
    ThreadMat<Matrix<T> > _fD;  // former dictionaries
    int             _nD;        // no. samples per atom
    int             _K;         // no. atoms
    Matrix<T>       _X;         // signals
    int             _n;         // no. samples per signal
    int             _M;         // no. signals 
    int             _S;         // no. latencies of each atom
    int             _KK;        // no. shifted versions of all atoms
    int             _L;         // maximally active atoms
    int             _maxIter;
    int             _i;
    int             _NUM_THREADS;
    T               _lambda;
    T               _lambda2;
    T               _eps;
    int             _spacing;
    bool            _clean;
    bool            _align;
    bool            _posAlpha; 
    bool            _silent;
    bool            _verbose;
    bool            _converged;
    bool            _finish;
    bool            _write_errors;
    bool            _reordering;
    bool            _fft;
    bool            _mcopies;
    bool            _lars_lasso;
    Vector<T>       _errors;
    Vector<T>       _regErr;
    Vector<T>       _changeD;
    int             _nfD;
    T               _tnet;
};


/*****************************************************************************/
/*****************************************************************************/
/********* IMPLEMENTATIONS ***************************************************/
/*****************************************************************************/
/*****************************************************************************/


///////////////////////////////////////////////////////////////////////////////
////   CONSTRUCTOR (class Trainer)                                         ////
///////////////////////////////////////////////////////////////////////////////
////       -only for initialization purpose                                ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
Trainer<T>::Trainer(Matrix<T>& D, Matrix<T>& X, 
        const ParamLearn<T>& param) : 
        _A(D.n(),X.n()), 
        _Delta(D.n(),X.n()), 
        _D(D.rawX(),D.m(),D.n()), 
        _tempD(D.m(),D.n()), 
        _nD(D.m()),
        _K(D.n()),
        _X(X.rawX(),X.m(),X.n()),
        _n(X.m()),
        _M(X.n()),
        _S(D.m()-X.m()+1),
        _KK(D.n()*(D.m()-X.m()+1)),
        _L(MIN(param.L,D.n())), 
        _maxIter(param.iter), 
        _i(-1),
        _NUM_THREADS(param.nThreads),
        _lambda(param.lambda),
        _lambda2(MAX(param.lambda2,1e-10)),
        _eps(param.eps),
        _clean(param.clean),
        _align(param.align && (D.m()-X.m()>1) ),
        _posAlpha(param.posAlpha),
        _silent(param.silent),
        _verbose(param.verbose && !param.silent),
        _reordering(param.reorder),
        _converged(false),
        _finish(false),
        _errors(),
        _regErr(),
        _changeD(),
        _spacing(param.spacing),
        _nfD(param.nfD),
        _fD(param.nfD,D.m(),D.n(),false),
        _fft(false),
        _tnet(T()),
        _mcopies(param.mcopies),
        _lars_lasso(param.lars_lasso),
        _write_errors(param.all_conv_measures)
{
    if (_NUM_THREADS == -1) {
#ifdef _OPENMP
        _NUM_THREADS = MIN(MAX_THREADS,omp_get_num_procs());
#else         
        _NUM_THREADS = 1;
#endif
    }
    _NUM_THREADS = init_omp(_NUM_THREADS);
    if (_write_errors) {
        _spacing=1;
        _errors.resize(param.iter);
        _regErr.resize(param.iter);
        _changeD.resize(param.iter);
    }
}


///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: CLEAN_DICT (class trainer)                          ////
///////////////////////////////////////////////////////////////////////////////
////       -replace zero-valued and strongly correlated atoms              ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::_clean_dict(Matrix<T>& G, const T maxCorrel) {
    T* const pr_G = G.rawX();

    for (int i = 0; i<_K; ++i) {
        // check if atom i has non-vanishing norm
        if (abs(pr_G[i*_K+i]) > 1e-4) {
            // check if an atom j is highly correlated with atom i
            int j;
            for (j = i+1; j<_K; ++j)
                if (abs(pr_G[i*_K+j])/sqrt(pr_G[i*_K+i]*pr_G[j*_K+j]) 
                        > maxCorrel) {
                    break;
                }
            if (j==_K) continue;
        }
            
        // something not right, replace atom i 
        Vector<T> di, g;
        _D.refCol(i,di);
        di.setAleat();
        di.normalize();
        G.refCol(i,g);
        _D.multTrans(di,g);
        if (_verbose) mexPrintf("Replace atom %d (it %d)\n", i, _i); 
   }
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: ALIGN DICT (class trainer)                          ////
///////////////////////////////////////////////////////////////////////////////
////       -algin atoms w.r.t. mean latency                                ////
///////////////////////////////////////////////////////////////////////////////
template <typename T> // TODO: could be written more efficiently
void Trainer<T>::_align_dict() {
    Matrix<T> temp;
    T meankT;
    int meank=0;
    int shift=0;
    T ak;
    T akm;
    T* p_A = _A.rawX();
    int* p_De = _Delta.rawX();
    Vector<T> dk;
    Vector<T> dTemp;
    bool first_found = true;

    for (int k=0; k<_K; ++k) {
        meankT = T();
        ak = T();
        for (int j=0; j<_M; ++j) {
            akm = abs(p_A[k + j*_K]);
            if (akm != T()) {
                meankT += akm * p_De[k + j*_K];
                ak += akm;
            }
        }
        if (ak==0) meankT = 0;
        else meankT /= ak;
        meank = floor(meankT + 0.5);
        // check if mean deviates from 0
        if (meank != 0) {
            if (_verbose) {
                if (first_found) {
                    mexPrintf("Aligning atoms (it %d): ", _i);
                    first_found=false;
                }
                shift = abs(meank);
                if (meank<0) shift*=-1;
                mexPrintf("%d (%d st) ", k, shift);
            }
            _D.refCol(k,dk);
            dTemp.copy(dk);
            dk.setZeros();
            // do alignment (TODO verify direction) 
            if (shift<0)    dk.copy(dTemp,_nD+shift,-shift,0);
            else            dk.copy(dTemp,_nD-shift,0,shift);
            dk.normalize();
        }
    }
    if (_verbose && !first_found) mexPrintf("\n");
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: UPDATE_COEFFS          (class trainer)              ////
///////////////////////////////////////////////////////////////////////////////
////       -updates coefficients and latencies                             ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::_update_coeffs(const Vector<int>& ind, 
        const Vector<T>& coeffs, const int m) {
    Vector<T> Am;
    _A.refCol(m,Am);
    Am.setZeros();
    Vector<int> Deltam;
    _Delta.refCol(m,Deltam);
    Deltam.setZeros();
    for (int l = 0; l<_L; ++l) {
        if (ind[l] == -1) {
            break;
        } else {
            Am[ind[l]/_S]     = coeffs[l];
            Deltam[ind[l]/_S] = (ind[l]%_S)-_S/2;
//            if (abs(Deltam[ind[l]/_S]) > _S/2)
//                mexErrMsgTxt("something went wrong with delta");
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: CALC_RES (class trainer)                            ////
///////////////////////////////////////////////////////////////////////////////
////       -calculates the residual                                        ////
///////////////////////////////////////////////////////////////////////////////
template <typename T> 
void Trainer<T>::_calc_res(Matrix<T>& R) {
    R.copy(_X);
    Vector<T> Rj;
    Vector<T> Dk;
    for (int j=0; j<_M; ++j) {
        R.refCol(j,Rj);
        for (int k=0; k<_K; ++k) {
            _D.refCol(k,Dk);
            Rj.add_subvector(Dk,_n,_S/2-_Delta(k,j),0,-_A(k,j));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: SUBTR_CONTR_RES (class trainer)                     ////
///////////////////////////////////////////////////////////////////////////////
////       -subtracts contribution of atom k from residual                 ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::_subtr_contr_res(Matrix<T>& R, int k) {
    Vector<T> Rj;
    Vector<T> Dk;
    _D.refCol(k,Dk);
    for (int j=0; j<_M; ++j) {
        R.refCol(j,Rj);
        Rj.add_subvector(Dk,_n,_S/2-_Delta(k,j),0,_A(k,j));
    }
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: ADD_CONTR_RES (class trainer)                       ////
///////////////////////////////////////////////////////////////////////////////
////       -adds contribution of atom k to residual                        ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::_add_contr_res(Matrix<T>& R, int k) {
    Vector<T> Rj;
    Vector<T> Dk;
    _D.refCol(k,Dk);
    for (int j=0; j<_M; ++j) {
        R.refCol(j,Rj);
        Rj.add_subvector(Dk,_n,_S/2-_Delta(k,j),0,-_A(k,j));
    }
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: UPDATE_ATOM (class trainer)                         ////
///////////////////////////////////////////////////////////////////////////////
////       -updates atom k                                                 ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::_update_atom(int k, const Matrix<T>& R) {
    const Vector<T>* Rj;
    Vector<T> Dk;
    _D.refCol(k,Dk);
    Dk.setZeros();
    for (int j=0; j<_M; ++j) {
        Rj=R.refCol(j);
        Dk.add_subvector(*Rj,_n,0,_S/2-_Delta(k,j),_A(k,j));
        delete(Rj);
    }
    Dk.normalize();
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: REORDER_DICT (class trainer)                        ////
///////////////////////////////////////////////////////////////////////////////
////       -reorder atoms, largest energy first                            ////
///////////////////////////////////////////////////////////////////////////////
template <typename T> 
void Trainer<T>::_reorder_dict() {
    Vector<T> norms;
    Vector<int> new_order(_K);
    for (int i=0; i<_K; ++i) new_order[i]=i;
    _A.norm_l1_rows(norms);
    norms.sort2(new_order,false);
    Matrix<T> newD, newA;
    Matrix<int> newDelta;
    newD.copy(_D);
    newA.copy(_A); 
    newDelta.copy(_Delta); 
    Vector<T> d, newd, newa;
    Vector<int> newdelta; 
    for (int i=0; i<_K; ++i) {
        _D.refCol(i,d);
        newD.refCol(new_order[i],newd);
        d.copy(newd);
        newA.copyRow(new_order[i],newa);  // TODO: make more efficient
        _A.setRow(i,newa);
        newDelta.copyRow(new_order[i],newdelta);
        _Delta.setRow(i,newdelta);
    }
}

///////////////////////////////////////////////////////////////////////////////
////   PRIVATE METHOD: CORRECT_SIGNS (class trainer)                       ////
///////////////////////////////////////////////////////////////////////////////
////       -reorder atoms, largest energy first                            ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::_correct_signs() {
    Vector<T> sums;
    _A.sum_rows(sums);
    Vector<T> d, a; 
    for (int i=0; i<_K; ++i) {
        if (sums[i]<0) {
            _D.refCol(i,d);
            d.neg();
            _A.copyRow(i,a);
            a.neg();
            _A.setRow(i,a);
        }
    }
    _A.sum_rows(sums);
}

///////////////////////////////////////////////////////////////////////////////
////   METHOD: TRAIN (class trainer)                                       ////
///////////////////////////////////////////////////////////////////////////////
////       -train dictionary on training data                              ////
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void Trainer<T>::train() {

    ///// start timer /////////////////////////////////////////////////////////
    Timer timeALL, timePREP, timeSC1, timeSC1a, timeSC1b, timeSC2, timeSC3,
          timeSC, timeDU, timeMAIN, timeSCMISC, timeMISC, timeCORR1, timeFFT,
          timeCORR2, timeCopy, timeShMat;
    if (_verbose) timePREP.start();
    if (_verbose) timeALL.start();

    /*************************************************************************/
    /***** SET UP ************************************************************/
    /*************************************************************************/
    
    ///// set seed for random number generator ////////////////////////////////
    srandom(0);
    seed=0;
    
    if (!_silent) {
        mexPrintf("*****Epoched Adaptive Waveform Learning*****\n");
        if (_verbose) {
            mexPrintf(" mode with parameters:\n");
            mexPrintf("   lambda: %g\n", _lambda);
            mexPrintf("   X: %d x %d\n", _X.m(), _X.n());
            mexPrintf("   D: %d x %d\n", _D.m(), _D.n());
            mexPrintf("   S: %d\n",_S);
            mexPrintf("   num threads: %d\n", _NUM_THREADS ); 
            if (_posAlpha) mexPrintf("Pos. constr. on coefficients\n"); 
            if (_clean) mexPrintf("Cleaning activated\n"); 
            if (_align) mexPrintf("Aligning activated\n"); 
            if (_mcopies) mexPrintf("Multiple copies allowed\n"); 
            if (_lars_lasso) mexPrintf("Lars lasso used\n"); 
            if (_L<_D.n()) mexPrintf("Maximal no. of active atoms is %d\n",_L);
        }
    }
    if (_mcopies) _L=_KK;
    if (_S>4*log2((T)_D.m())) {
        _fft=true;
    }
#ifdef FFT_CONV
    // optimize fft length
    const int fft_len=2*_nD;
    int fft_len2=1;
    int n=_nD;
    
    Vector<int> factors(4);
    factors[0]=2;
    factors[1]=3;
    factors[2]=5;
    factors[3]=7;

    //calculate optimal fft length
    while (n!=2) {
        for (int i=0; i<factors.n();) {
            if (n%factors[i]==0) {
                n/=factors[i];
                fft_len2*=factors[i];
            }
            else ++i;
        }
        ++n;
    }

    int poss2=1;
    n=_nD;
    while (n!=2) {
        while (n%2==0) {
            n/=2;
            poss2*=2;
        }
        ++n;
    } 
    if (poss2<fft_len2) {
        fft_len2=poss2;
    }


    // ATTENTION: this can cause segmentation fault if fftw3 library is
    // dynamically linked
    T* fft_buffer = (T*) fftw_malloc(sizeof(T)*fft_len);
    T* fft_buffer2 = (T*) fftw_malloc(sizeof(T)*fft_len);
    fftw_plan p = fftw_plan_r2r_1d(fft_len,fft_buffer,fft_buffer,FFTW_R2HC,0);
    fftw_plan p2 = fftw_plan_r2r_1d(fft_len2,fft_buffer,fft_buffer,FFTW_R2HC,0);
    fftw_plan ip = fftw_plan_r2r_1d(fft_len,fft_buffer,fft_buffer,FFTW_HC2R,0);
    fftw_plan ip2 = fftw_plan_r2r_1d(fft_len2,fft_buffer,fft_buffer,FFTW_HC2R,0);
    Matrix<T> Dpadded(fft_len,_K);           
    Matrix<T> Df(fft_len2,_K);           

#endif

    ///// initializations /////////////////////////////////////////////////////
    Matrix<T> DD(_n,_KK);       // unrolled dictionary // TODO not really needed
    CovShMat<T> DtD(_K,_S);
    DtD.setFft(&p,&ip,fft_len,fft_buffer,&Dpadded);
    Vector<int> perm(_M);       // random number sequence
    Matrix<T> R(_n,_M);         // residual matrix
    Vector<T> normsD(_K);       // norms of atoms
    Vector<T> normsRsq(_M);     // residual norms (sq)
    Vector<T> normsXsq(_M);     // norms (sq) of training examples
    _Delta.setZeros();
    _A.setZeros();
    T dist;

    ///// norms of training examples //////////////////////////////////////////
    _X.norm_2sq_cols(normsXsq);
    T sumNormsX = normsXsq.sum();

    ///// initializations for the different threads (sparse coding) ///////////
    ThreadVec<Vector<int> > indT    (_NUM_THREADS,_L,true);
    ThreadVec<Vector<T> >   coeffsT (_NUM_THREADS,_L,true);
    ThreadVec<Vector<T> >   DtRT    (_NUM_THREADS,_KK,true);
    ThreadVec<Vector<T> >   uT      (_NUM_THREADS,_L,true); // TODO: 0s needed?
    ThreadMat<Matrix<T> >   GGsT    (_NUM_THREADS,_L,_L,true);
    ThreadMat<Matrix<T> >   GGaT    (_NUM_THREADS,_KK,_L,true);
    ThreadMat<Matrix<T> >   invGGsT (_NUM_THREADS,_L,_L,true);
    ThreadVec<Vector<T> >   work1T   (_NUM_THREADS,_KK,true);
    ThreadVec<Vector<T> >   work2T   (_NUM_THREADS,_KK,true);
    ThreadVec<Vector<T> >   work3T   (_NUM_THREADS,_KK,true);
  
    if (_verbose) timePREP.stop();
    if (_verbose) timeMAIN.start();

    /*************************************************************************/
    /***** MAIN LOOP OVER SPARSE CODING AND DICT UPDATES *********************/
    /*************************************************************************/
    
    bool iteration_complete;
    ///// loop breaks after sparse coding if converged or max. ////////////////
    ///// number of iterations is reached /////////////////////////////////////
    while (true) {
        if (_verbose) timeSCMISC.start();

        ///// increment iteration and display /////////////////////////////////
        ++_i;
        if (_verbose && _i%(_maxIter/10+1)==0) mexPrintf("Iteration: %d\n", _i);

        ///// copy dictionary to measure convergence //////////////////////////
        int index = (_i)%_nfD;
        _fD[index].copy(_D); // TODO: could be made more efficient

        ///// align atoms /////////////////////////////////////////////////////
        if (_align && (_i<_maxIter)) _align_dict();

        ///// replace bad atoms by Gaussian noise /////////////////////////////
        if (_clean) {
            Matrix<T> G;
            _D.XtX(G);
            this->_clean_dict(G);
        }

        ///// unroll dictionary ///////////////////////////////////////////////
        if (!_fft) {
            if (_verbose) timeSC1.start();
            Vector<T> dd,d;
            int index = 0;
            for (int k = 0; k<_K; ++k) {
                _D.refCol(k,d);
                for (int l=0;l<_S;++l) {
                    DD.refCol(index,dd);
                    dd.copy(d,_n,_S-l-1,0);
                    ++index;
                }
            }
            if (_verbose) timeSC1.stop();
        }

        ///// set up covariance matrix GG of unrolled dictionary //////////////
        if (_verbose) timeShMat.start();
        DtD.set(_D);
        if (_verbose) timeShMat.stop();
        DtD.addDiag(_lambda2);

#ifdef FFT_CONV
        Vector<T> Dj,Dfk;
        for (int k = 0; k<_K; ++k) {
            _D.refCol(k,Dj);
            memset(fft_buffer,0,sizeof(T)*fft_len2);
            Dj.extractRaw(fft_buffer,Dj.n());
            fftw_execute(p2);
            Df.refCol(k,Dfk);
            Dfk.copy(fft_buffer,fft_len2);
        }
#endif

         ///// generate random sequence to cycle through data //////////////////
        perm.randperm(_M);
        iteration_complete = false;
        if (_verbose) timeSCMISC.stop();
        while (!iteration_complete) {
            

            /*****************************************************************/
            /***** PARALLEL LOOP OVER BATCH FOR SPARSE CODING ****************/
            /*****************************************************************/

            if (_verbose) timeSC.start();
            int j;
            for (j = 0; j<_M; ++j) {
                int numT=0;

                const int m=perm[j];
                Vector<int>& indj= indT[numT];    
                Vector<T>&   coeffsj= coeffsT[numT]; 
                Vector<T>&   DtRj= DtRT[numT];    
                const Vector<T>* Xm;      
                Vector<T>&   uj= uT[numT];      
                Matrix<T>&   GGsj= GGsT[numT];     
                Matrix<T>&   GGaj= GGaT[numT];     
                Matrix<T>&   invGGsj= invGGsT[numT];  
                Vector<T>&   work1j= work1T[numT];   
                Vector<T>&   work2j= work2T[numT];   
                Vector<T>&   work3j= work3T[numT];   
                Xm=_X.refCol(m);
                if (!_fft) {
                    if (_verbose) timeSC1.start();
                    DD.multTrans(*Xm,DtRj);
                    if (_verbose) timeSC1.stop();
                }
                else {
                    Vector<T> Dk;
                    Xm->extractRaw(fft_buffer,Xm->n());
                    std::memset(fft_buffer+Xm->n(),0,(fft_len2-Xm->n())*sizeof(T));
                    fftw_execute(p2);
                    cblas_copy<T>(fft_len2,fft_buffer,1,fft_buffer2,1);
                    if (_verbose) timeSC1a.start();
                    for (int k = 0; k<_K; ++k) {
                        if (k>0) cblas_copy<T>(fft_len2,fft_buffer2,1,fft_buffer,1);
                        Df.refCol(k,Dk);
                        for (int i=0;i<fft_len2/2+1;++i)
                            fft_buffer[i]*=Dk[i];
                        for (int i=fft_len2/2+1;i<fft_len2;++i)
                            fft_buffer[i]*=-Dk[fft_len2-i]; //'-' for cross-corr. 
                        for (int i=1;i<(fft_len2+1)/2;++i)
                            fft_buffer[i]+=Dk[fft_len2-i]*fft_buffer2[fft_len2-i]; //'+' for cross-corr. 
                        for (int i=fft_len2/2+1;i<fft_len2;++i)
                            fft_buffer[i]+=Dk[i]*fft_buffer2[fft_len2-i];
                    if (_verbose) timeSC1b.start();
                        fftw_execute(ip2);
                    if (_verbose) timeSC1b.stop();
                        for (int l=0;l<_S;++l) { 
                            DtRj[k*_S+l]=fft_buffer[_S-l-1]/fft_len2;
                        }
                    }
                    if (_verbose) timeSC1a.stop();
                }

                T normX=normsXsq[m];
                coeffsj.setZeros();
                indj.setZeros();
             
                ///// sparse coding ///////////////////////////////////////////
                if (_verbose) timeSC2.start();
                modLARS(DtRj,DtD,GGsj,GGaj,invGGsj,uj,coeffsj,indj,
                        work1j,work2j,work3j,normX,_lambda,_posAlpha);
                if (_verbose) timeSC2.stop();
               
                ///// update coefficient and latencies for current example ////
                if (_verbose) timeSC3.start();
                _update_coeffs(indj,coeffsj,m);
                if (_verbose) timeSC3.stop();
            }
            iteration_complete = true;
            if (_verbose) timeSC.stop();
            /*****************************************************************/
            ///// continue sparse coding in first and last iterations /////////
            if (!iteration_complete && (_i==0 || _i>= _maxIter || _converged )) 
                continue;

            if (_verbose) timeDU.start();

            ///// stop if converged or max no. of iteration reached ///////////
            if (_converged || _i==_maxIter) {
                _finish = true;
                if (_verbose) timeDU.stop();
                break;
            }

            ///// calculate current residual //////////////////////////////////
            _calc_res(R);


            /*****************************************************************/
            /***** DICTIONARY UPDATE *****************************************/
            /*****************************************************************/

            ///// perform iterations over block coordinate descent ////////////
            for (int k=0; k<_K; ++k) {
                // subtract contribution of atom k from residual 
                _subtr_contr_res(R,k);
                // subtract residual from current atom 
                _update_atom(k,R);
                // add contribution of updated atom k to residual 
                _add_contr_res(R,k);
            }
                 
            if (_verbose) timeDU.stop();

            /***** END (Dictionary Update) ***********************************/
        }
        if (_finish) break;

        if (_verbose) timeMISC.start();
        
        // compare to previous dictionary, verify if converged
        dist = 1.;
        for (int k=0; k<MIN(_i,_nfD); ++k) {
            _tempD.copy(_fD[k]);
            _tempD.sub(_D);
            dist=MIN(_tempD.asum()/(_nD*_K),dist); // zero (machine prec)
        }
        if (dist<_eps) { 
            if (!_silent) mexPrintf("Converged after %d iterations\n", _i+1);
            _converged = true;
        }
        if (_write_errors) {
            _changeD[_i] = dist;
            T res = R.normFsq();
            _errors[_i] = 0.5* res / sumNormsX;
            _regErr[_i] = (0.5* res + _lambda*_A.asum()
                + 0.5*_lambda2*_A.normFsq())/ sumNormsX;
        }
        if (_verbose) timeMISC.stop();
    }
    if (_reordering) {
        _reorder_dict();
        _correct_signs();
    }
    if (_write_errors) {
        _errors.setn(_i);
        _regErr.setn(_i);
        _changeD.setn(_i);
    }
    if (_verbose) timeMAIN.stop();
    if (_verbose) timeALL.stop();
    if (_verbose) {
        timeALL.printElapsed("Time elapsed       ");
        timePREP.printElapsed("  -set up          ");
        timeMAIN.printElapsed("  -main part       ");
        timeMISC.printElapsed("    -error calc    ");
        timeDU.printElapsed("    -dict. update  ");
        timeSC.printElapsed("    -sparse coding ");
        timeSCMISC.printElapsed("      +misc        ");
        timeShMat.printElapsed("      +DtD         ");
        timeSC1.printElapsed("      +XtD (direct)");
        timeSC1a.printElapsed("      +XtD (fft)   ");
        timeSC1b.printElapsed("        -FFT       ");
        mexPrintf("        (%d)\n", fft_len2); 
        timeSC2.printElapsed("      +LARS        ");
        _tnet=timeALL.getElapsed()-timeMISC.getElapsed();
    }
};
            

template <typename T>
void writeLog(const Matrix<T>& D, const T time, int iter, 
      char* name) {
   std::ofstream f;
   f.precision(12);
   f.flags(std::ios_base::scientific);
   f.open(name, ofstream::trunc);
   f << time << " " << iter << std::endl;
   for (int i = 0; i<D.n(); ++i) {
      for (int j = 0; j<D.m(); ++j) {
         f << D[i*D.m()+j] << " ";
      }
      f << std::endl;
   }
   f << std::endl;
   f.close();
};


#endif

