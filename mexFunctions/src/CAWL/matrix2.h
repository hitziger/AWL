/*!
 *
 *                File matrix2.h
 *
 * Efficient implementations for calculating cross-correlations 
 *
 * CovShMat: 
 * - Covariance matrix of shift-invariant dictionary
 *
 * CorShMat:
 * - Correlations between signal and shift-invariant dictionary
 *
 * Two implementations:
 * - fft-based cross-correlations using fftw3 library (most efficient), requires
 *   compilation with -DFFT_CONV
 * - naive calculation of all correlations, generally slower
 *
 * */

#ifndef MATRIX2_H
#define MATRIX2_H

#include <fftw3.h>
#include <matrix.h>
#include <my_utils_sh.h>
#include <my_mex_sh.h>

#ifdef FFT_CONV
///////////////////////////////////////////////////////////////////////////////
////   CLASS FFT_BUFFER                                                    ////
///////////////////////////////////////////////////////////////////////////////
////       -sets up a buffer for fftw, optimizes length                    ////
///////////////////////////////////////////////////////////////////////////////
class FftBuffer {
public:
    FftBuffer(): _p(NULL), _ip(NULL), _len(0), _buffer(NULL) {}
    FftBuffer(const unsigned length, const bool opt=true ) { set_up(length, opt); }
    ~FftBuffer() {
        if (_buffer!=NULL) fftw_free(_buffer); 
    }
    inline unsigned optimize_fft_len(const unsigned len) {
        unsigned n=len;
        unsigned len2=1;
    
        Vector<int> factors(4);
        factors[0]=2;
        factors[1]=3;
        factors[2]=5;
        factors[3]=7;

        // increase to length that has only small prime factors
        while (n!=2) {
            for (unsigned i=0; i<factors.l();) {
                if (n%factors[i]==0) {
                    n/=factors[i];
                    len2*=factors[i];
                }
                else ++i;
            }
            ++n;
        }
        n=len;
        unsigned len3=1;
        while (n!=2) {
            while (n%2==0) {
                n/=2;
                len3*=2;
            }
            ++n;
        } 
        if (len3<len2) { return len3; }
        return len2;
    }
    // allocate memory and set up fftw plan 
    inline void set_up(const unsigned length, const bool opt=true) {
        _len=length;
        if (opt) _len = optimize_fft_len(_len);
        _buffer = (double*) fftw_malloc(sizeof(double)*_len);
        if (_buffer==NULL) mexErrMsgTxt("fft: unable to allocate buffer");
        _p = fftw_plan_r2r_1d(_len,_buffer,_buffer,FFTW_R2HC,FFTW_ESTIMATE);
        _ip = fftw_plan_r2r_1d(_len,_buffer,_buffer,FFTW_HC2R,FFTW_ESTIMATE);
    }
    inline void set_up2(const unsigned length, const bool opt=true) {
        _len=length;
        if (opt) _len = optimize_fft_len(_len);
        _buffer = (double*) fftw_malloc(sizeof(double)*_len);
        if (_buffer==NULL) mexErrMsgTxt("fft: unable to allocate buffer");
        _p = fftw_plan_r2r_1d(_len,_buffer,_buffer,FFTW_R2HC,FFTW_ESTIMATE);
        _ip = fftw_plan_r2r_1d(_len,_buffer,_buffer,FFTW_HC2R,FFTW_ESTIMATE);
    }

    
    inline void fft() { 
        assert(_len>0 && "fft: length not positive"); 
        fftw_execute(_p); 
    }
    inline void ifft() { 
        assert(_len>0 && "fft: length not positive"); 
        fftw_execute(_ip); 
    }
    inline double* buffer() { 
        assert(_len>0 && "buffer: length not positive"); 
        return _buffer; 
    }
    inline unsigned len() { return _len; }
    // operation corresponding to cross-correlation in Fourier domain,
    // half-complex format
    inline void operator*=(const double* array) {
        unsigned l;
        double temp;
        _buffer[0] *=array[0];
        if (_len%2==0) _buffer[_len/2] *=array[_len/2];
        for (l=1;           l<(_len+1)/2; ++l) {
            temp=_buffer[l];
            _buffer[l]*=array[l];
            _buffer[l]+=_buffer[_len-l]*array[_len-l];
            _buffer[_len-l]*=array[l];
            _buffer[_len-l]-=temp*array[_len-l];
        }
    }

private:
    fftw_plan    _p;
    fftw_plan    _ip;
    unsigned     _len;
    double*      _buffer;
};
#endif

/// Class representing Covariance matrix of a shiftable dictionary
template<typename T> 
class CovShMat: public DataObject<T> {
    typedef DataObject<T> base;
public:
   ///// Constructors, destructor /////////////////////////////////////////////
   CovShMat();
   CovShMat(const unsigned K, const unsigned n, const bool symmetric=true);

//   ~CovShMat();

   ///// Allocate /////////////////////////////////////////////////////////////
   inline void allocate(const unsigned K, const unsigned n, const bool symmetric=true);

   ///// Set data /////////////////////////////////////////////////////////////
   inline void set(const Matrix<T>& D);
   inline void set(const Matrix<T>& D1, const Matrix<T>& D2);

   ///// Set up FFT ///////////////////////////////////////////////////////////
#ifdef FFT_CONV
   inline void setDwork(Matrix<T>& Dwork) { _Dwork=Dwork; } 
#endif

   ///// Getting dimensions ///////////////////////////////////////////////////
   inline unsigned K() const { return _K;}
   inline unsigned n() const { return _n;}
    
   // add to diagonal
   inline void addDiag(const T diag);

   ///// Elementwise access (const) ///////////////////////////////////////////
   inline T operator()(const unsigned k1, const unsigned k2, const int diff12) const;
   inline T* ptr(const unsigned k1, const unsigned k2, const int diff12=0) const;

private:
   inline unsigned _getOffset(const unsigned k1, const unsigned k2) const;

    unsigned   _K;
    unsigned   _Kmax;
    unsigned   _n;
    bool       _symmetric;
#ifdef FFT_CONV
    FftBuffer     _fft;
    Matrix<T>     _Dwork;
    Vector<T>     _work;
#endif
};

/// Class representing Correlation matrix of a shiftable dictionary with a signal
template<typename T> 
class CorShMat: public Matrix<T> {
    typedef Matrix<T> base; 
public:
    ///// Constructors, destructor /////////////////////////////////////////////
    CorShMat(const unsigned N, const unsigned mD, const unsigned maxK,
            const bool do_fft);
//    ~CorShMat();

    ///// Set data /////////////////////////////////////////////////////////////
    inline void setX(const Vector<T>& X);
    inline void update(const Matrix<T>& D);
    inline void update(const Vector<T>& d);

private:

    unsigned   _mD;
    Vector<T> _X;
#ifdef FFT_CONV
    FftBuffer _fft;
#endif
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const CovShMat<T>& A) {
    out << "CovShMat (" << A.K() << "," << A.n() << ")";
    if (A.K()==0 || A.n()==0) {
        out << std::endl;
        return out;
    }
    out << ":" << std::endl;
    for (unsigned k2=0; k2<A.K(); ++k2) {
        for (unsigned k1=0; k1<=k2; ++k1) {
            out << "[";
            out.width(ENTRY_WIDTH-1);
            out << A(k1,k2,-A.n()+1);
            unsigned curr_pos=ENTRY_WIDTH;
            int start = - (int) A.n()+2;
            int end = A.n();
            for (int j=start; j<end; ++j) {
                curr_pos+=ENTRY_WIDTH;
                out.width(ENTRY_WIDTH);
                out << A(k1,k2,j);
                if (curr_pos>LINE_WIDTH-ENTRY_WIDTH) {
                    out << std::endl;
                    curr_pos = 0;
                }
            }
            out << "]" << std::endl;
        }
    }
    return out;
}


/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS COVMATRIXSHIFT ****************************/
/*****************************************************************************/

template <typename T>
CovShMat<T>::CovShMat(): base(), _Kmax(0), _K(0), _n(0) { }

template <typename T>
CovShMat<T>::CovShMat(const unsigned K, const unsigned n, const bool symmetric): _K(0) {
    this->allocate(K,n,symmetric);
}

 
template <typename T>
inline void CovShMat<T>::allocate(const unsigned K, const unsigned n, const bool symmetric) {
    _symmetric=symmetric;
    int nel;
    if (symmetric)  nel=K*(K+1)*(2*n-1)/2;
    else            nel=K*K*(2*n-1);
    if (base::_lmax<nel) {
        base::allocate(nel);
    }
    _Kmax=K;
    _n=n;
#ifdef FFT_CONV
    _fft.set_up(2*_n);
    _Dwork.resize(_fft.len(),K);
    if (!symmetric) _work.resize(_fft.len());
#endif
}


template <typename T>
inline void CovShMat<T>::set(const Matrix<T>& D) {
    assert(_symmetric && "CovShMat::set: two inputs needed for non-symmetric version!");
    _K=D.n();
    if (_n!=D.m() || _K>_Kmax ) {
        std::cout << "WARNING! Setting up covariance matrix..." << std::endl;
        allocate(D.n(),D.m()); 
    }
    int l,count=-1;
    unsigned k1, k2;
#ifdef FFT_CONV
    //mexPrintf("CovShMat::set: using fft\n");
    //mexEvalString("drawnow");
    for (k2=0; k2<_K; ++k2) {
        D.extract(_fft.buffer(),k2);
        std::memset(_fft.buffer()+_n,0,(_fft.len()-_n)*sizeof(T));
        _fft.fft();
        _Dwork.copy(_fft.buffer(),k2);
        for (k1=0; k1<=k2; ++k1) {
            _Dwork.extract(_fft.buffer(),k2);
            _fft*=_Dwork.col(k1);
            _fft.ifft();
            for (l=-(int)(_n-1);l<(int)_n;++l) { 
                ++count;
                base::_data[count]=_fft.buffer()[(_fft.len()+l)%_fft.len()]/(T) _fft.len();
            }
        }
    } 
#else
    //mexPrintf("CovShMat::set: not using fft\n");
    //mexEvalString("drawnow");
    T* pr_D = const_cast<T*>(D.ptr());
    unsigned off1, off2;
    for (k2=0; k2<_K; ++k2) {
        off2 = k2*D.m();
        for (k1=0; k1<=k2; ++k1) {
            off1 = k1*D.m();
            for (l=-(int) (_n-1); l<(int) _n; ++l) {
                base::_data[++count]=blas_dot<T>(D.m()-abs(l),pr_D+off1+max(0,-l),1,
                        pr_D+off2+max(0,l),1);
            }
        }
    }
#endif
}

template <typename T>
inline void CovShMat<T>::set(const Matrix<T>& D1, const Matrix<T>& D2) {
    _K=D1.n();
    assert(!_symmetric && "CovShMat::set: Only one input for symmetric version!");
    assert(D1.m()==D2.m() && D1.n()==D2.n() && "CovShMat::set: matrices must be of same dimensions!");
    if (_n!=D1.m() || _K>_Kmax ) {
        std::cout << "WARNING! CovShMat::set, need to reallocate..." << std::endl;
        allocate(D1.n(),D1.m()); 
    }
    int l,count=-1;
    unsigned k1, k2;
#ifdef FFT_CONV
    for (k1=0; k1<_K; ++k1) {
        D1.extract(_fft.buffer(),k1);
        std::memset(_fft.buffer()+_n,0,(_fft.len()-_n)*sizeof(T));
        _fft.fft();
        _Dwork.copy(_fft.buffer(),k1);
    }
    for (k2=0; k2<_K; ++k2) {
        D2.extract(_fft.buffer(),k2);
        std::memset(_fft.buffer()+_n,0,(_fft.len()-_n)*sizeof(T));
        _fft.fft();
        _work.copy(_fft.buffer());
        for (k1=0; k1<_K; ++k1) {
            _work.extract(_fft.buffer());
            _fft*=_Dwork.col(k1);
            _fft.ifft();

            for (l=-(int)(_n-1);l<(int)_n;++l) { 
                base::_data[++count]=_fft.buffer()[(_fft.len()+l)%_fft.len()]/(T) _fft.len();
            }
        }
    } 
     
#else
    mexErrMsgTxt("CovShMat<T>::set: non-symmetric version currently only implemented with fft based convolution");
#if 0
    T* pr_D = const_cast<T*>(D.ptr());
    unsigned off1, off2;
    for (k2=0; k2<_K; ++k2) {
        off2 = k2*D.m();
        for (k1=0; k1<=k2; ++k1) {
            off1 = k1*D.m();
                std::cout << "outside";
                l=-(int) (_n-1);
                std::cout << "l=" << l << std::endl;
                std::cout << "n=" << _n << std::endl;
                std::cout << "l<_n? " << (l<(int)_n) << std::endl;
                
            for (l=-(int) (_n-1); l<(int) _n; ++l) {
                std::cout << "inside";
                base::_data[++count]=blas_dot<T>(D.m()-abs(l),pr_D+off1+max(0,l),1,
                        pr_D+off2+max(0,-l),1);
                std::cout << blas_dot<T>(D.m()-abs(l),pr_D+off1+max(0,l),1,
                        pr_D+off2+max(0,-l),1);
            }
        }
    }
#endif

#endif
}


template <typename T>
inline void CovShMat<T>::addDiag(const T diag) {
    int offset = -1;
    unsigned i,k1,n2=2*_n-1;
    for (k1=0; k1<_K; ++k1) {
        offset+=k1*n2;
        for (i=0; i<n2; ++i) base::_data[++offset]+=diag; //TODO: blas
    }
}

template <typename T>
inline T CovShMat<T>::operator()(const unsigned k1, const unsigned k2, 
        const int diff12) const {
    assert(k1 < _K && k2 < _K && "CovShMat::operator(): indices out of range");
    if (abs(diff12)>_n-1) return 0.;
    return *ptr(k1,k2,diff12);
}

template <typename T>
inline T* CovShMat<T>::ptr(const unsigned k1, const unsigned k2, 
        const int diff12) const {
    assert(k1 < _K && k2 < _K && "CovShMat::ptr(): indices out of range");
    assert(abs(diff12)<=_n-1 && "CovShMat::ptr(): indices out of range");
    if (k1<=k2) return base::_data+_n-1+diff12+_getOffset(k1,k2);
    return base::_data+_n-1-diff12+_getOffset(k1,k2);
    }

template <typename T>
inline unsigned CovShMat<T>::_getOffset(const unsigned k1, const unsigned k2) const {
    assert(k1 < _K && k2 < _K && "CovShMat::getOffset(): indices out of range");
    if (!_symmetric)    return (k2*_K+k1)*(2*_n-1);
    else {
        if (k1>k2) return _getOffset(k2,k1);
        else       return ((k2*(k2+1))/2+k1)*(2*_n-1);
    }
}


/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS CORMATRIXSHIFT ****************************/
/*****************************************************************************/

template <typename T>
CorShMat<T>::CorShMat(const unsigned N, const unsigned mD, 
        const unsigned maxK, const bool do_fft): base(N-mD+1,maxK), _mD(mD) {
#ifdef FFT_CONV
    if (do_fft) { 
        _fft.set_up(N); 
        _X.resize(_fft.len());
        return;
    }
#endif
    _X.resize(N);
}

template <typename T>
inline void CorShMat<T>::setX(const Vector<T>& X) {
#ifdef FFT_CONV
    if (_fft.len()>0) {
        //mexPrintf("setX: Using fft\n");
        //mexEvalString("drawnow");
        assert(_X.l()>=X.l() && "CorShMat::setX(): argument has too many elements");

        //std::fill(_fft.buffer(),_fft.buffer()+_X.l(),T());
        std::memset(_fft.buffer(),T(),_X.l()*sizeof(T));
        X.extract(_fft.buffer());
#ifdef DEBUG_MATRIX2_CORSHMAT_SETX
        my_mex_figure();
        my_mex_plot(_X.l(),_fft.buffer());
        my_mex_drawnow();
        bool dummy;
        my_mex_continue(dummy);
        for (int i=0; i<_X.l(); ++i) mexPrintf ("%g ",_fft.buffer()[_X.l()-1-i]);
        my_mex_continue(dummy);
#endif
        _fft.fft();
#ifdef DEBUG_MATRIX2_CORSHMAT_SETX
        for (int i=0; i<_X.l(); ++i) mexPrintf ("%g ",_fft.buffer()[i]);
        //std::cout << X;
        //std::cout << X.sum();
        my_mex_figure();
        my_mex_plot(_X.l(),_fft.buffer(),"r");
        my_mex_title("CorShMat, fft of X");
#endif

        _X.copy(_fft.buffer());
        return;
    } 
#endif
    //mexPrintf("setX: not using fft\n");
    //mexEvalString("drawnow");
    assert(_X.l()==X.l() && "CorShMat::setX(): argument has false number of elements");
    _X.copy(X.ptr());
}

template <typename T>
inline void CorShMat<T>::update(const Matrix<T>& D) {
    assert(base::_n==D.n() && "CorShMat::update(): dictionary dimensions not consistent");
    assert(_mD==D.m() && "CorShMat::update(): dictionary dimensions not consistent");
    unsigned k, S=base::_m, K=base::_n;
#ifdef FFT_CONV
    if (_fft.len()>0) {
        //mexPrintf("CorShMat::update: Using fft\n");
        //mexEvalString("drawnow");
        for (k=0; k<K; ++k) {
            Vector<T> Dk(D,k);
            Dk.extract(_fft.buffer());
            std::memset(_fft.buffer()+_mD,0,(_fft.len()-_mD)*sizeof(T));
            _fft.fft();
            _fft*=_X.ptr();
            _fft.ifft();
            base::_data[k*S] = _fft.buffer()[0];
            blas_copy<T>(S-1,_fft.buffer()+_fft.len()-(S-1),-1,base::_data+k*S+1,1);
            blas_scal<T>(S,1./(T)_fft.len(),base::_data+k*S,1);
        }
        return;
    }
#endif
    //mexPrintf("CorShMat::update: not fft\n");
    //mexEvalString("drawnow");
    T* pr_D = const_cast<T*>(D.ptr());
    unsigned s,offset1;
    int count=-1;
    for (k=0; k<K; ++k) {
        offset1=k*D.m();
        for (s=0; s<S; ++s) {
            base::_data[++count]=blas_dot<T>(D.m(),pr_D+offset1,1,_X.ptr()+s,1);
        }
    }
}

template <typename T>
inline void CorShMat<T>::update(const Vector<T>& d) {
    const Matrix<T> D(const_cast<T*>(d.ptr()),d.l(),1);
    this->update(D);
}

#endif
