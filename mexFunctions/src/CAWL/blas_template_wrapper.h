#ifndef BLAS_TEMPLATE_WRAPPER_H
#define BLAS_TEMPLATE_WRAPPER_H



extern "C" {
#include <blas.h>
#include <lapack.h>

#ifdef _WIN32
#define BLAS_EXPANSION(x) x
#define LAPACK_EXPANSION(x) x    
#else 
#define BLAS_EXPANSION(x) x##_
#define LAPACK_EXPANSION(x) x##_    
#endif

}

namespace blas {
    static char NoTrans = 'N';
    static char Trans = 'T';
    static char ConjTrans = 'C';
    static char Upper = 'U';
    static char Lower = 'L';
    static char NonUnit = 'N';
    static char Unit = 'U';
    static char Left = 'L';
    static char Right = 'R';    
}

/// a few static variables for lapack
namespace lapack {
    static char low = 'l';
    static char nonUnit = 'n';
    static char upper = 'u';
    static ptrdiff_t info = 0;
    static char incr= 'I';
    static char decr= 'D';
}

// Interfaces to a few BLAS function, Level 1
template <typename T> 
inline T blas_nrm2(ptrdiff_t n, T* X, ptrdiff_t incX);
template <typename T> 
inline void blas_copy(ptrdiff_t n, T* X, ptrdiff_t incX, T* Y,
        ptrdiff_t incY);
template <typename T> 
inline void blas_axpy(ptrdiff_t n, T a, T* X, ptrdiff_t incX,
        T* Y, ptrdiff_t incY);
template <typename T>
inline void blas_scal(ptrdiff_t n, T a, T* X, ptrdiff_t incX);
template <typename T> 
inline T blas_asum(ptrdiff_t n, T* X, ptrdiff_t incX);
template <typename T> 
inline T blas_dot(ptrdiff_t n, T* X, ptrdiff_t incX, T* Y,
        ptrdiff_t incY);
template <typename T>
inline int blas_iamax(ptrdiff_t n, T* X, ptrdiff_t incX);

// Interfaces to a few BLAS function, Level 2
template <typename T>
inline void blas_gemv(char TransA, ptrdiff_t M, ptrdiff_t N,
        T alpha, T *A, ptrdiff_t lda, T *X, ptrdiff_t incX,
        T beta,T *Y,  ptrdiff_t incY);
template <typename T> 
inline void blas_trmv(char Uplo, char TransA, char Diag,
        ptrdiff_t N, T *A, ptrdiff_t lda, T *X, ptrdiff_t incX);
template <typename T> 
inline void blas_syr( char Uplo, ptrdiff_t N, T alpha, T *X,
        ptrdiff_t incX, T *A,  ptrdiff_t lda);
template <typename T> 
inline void blas_symv(char Uplo, ptrdiff_t N, T alpha, T *A,
        ptrdiff_t lda, T *X, ptrdiff_t incX, T beta,T *Y,
        ptrdiff_t incY);


// Interfaces to a few BLAS function, Level 3
template <typename T> 
void blas_gemm(char TransA, char TransB, ptrdiff_t M, ptrdiff_t N,
        ptrdiff_t K, T alpha, T *A, ptrdiff_t lda, T *B,
        ptrdiff_t ldb, T beta, T *C, ptrdiff_t ldc);
template <typename T> 
void blas_syrk(char Uplo, char Trans, ptrdiff_t N, ptrdiff_t K,
        T alpha, T *A, ptrdiff_t lda, T beta, T*C,
        ptrdiff_t ldc);
template <typename T> 
void blas_ger(ptrdiff_t M, ptrdiff_t N, T alpha, T *X,
        ptrdiff_t incX, T* Y, ptrdiff_t incY, T*A, ptrdiff_t lda);
template <typename T>
void blas_trmm(char Side, char Uplo, char TransA,
        char Diag, ptrdiff_t M, ptrdiff_t N, T alpha, T*A,
        ptrdiff_t lda,T *B, ptrdiff_t ldb);

// Interfaces to a few LAPACK functions
template <typename T>
void lapack_gesv(ptrdiff_t N, ptrdiff_t NRHS, T *A, ptrdiff_t lda, ptrdiff_t *ipiv,
        T *B, ptrdiff_t ldb);



/* ******************
 * Implementations
 * *****************/


// Implementations of the interfaces, BLAS Level 1
template <>
inline double blas_nrm2<double>(ptrdiff_t n, double* X, ptrdiff_t incX) {
    return BLAS_EXPANSION(dnrm2)(&n,X,&incX);
}
template <> 
inline float blas_nrm2<float>(ptrdiff_t n, float* X, ptrdiff_t incX) {
    return BLAS_EXPANSION(snrm2)(&n,X,&incX);
}

template <> 
inline void blas_copy<double>(ptrdiff_t n, double* X, ptrdiff_t incX,
        double* Y, ptrdiff_t incY) {
    BLAS_EXPANSION(dcopy)(&n,X,&incX,Y,&incY);
}
template <> 
inline void blas_copy<float>(ptrdiff_t n, float* X, ptrdiff_t incX, 
        float* Y, ptrdiff_t incY) {
   BLAS_EXPANSION(scopy)(&n,X,&incX,Y,&incY);
}
template <> 
inline void blas_copy<int>(ptrdiff_t n, int* X, ptrdiff_t incX, 
        int* Y, ptrdiff_t incY) {
    for (int i = 0; i<n; ++i) Y[incY*i]=X[incX*i];
}
template <> 
inline void blas_copy<bool>(ptrdiff_t n, bool* X, ptrdiff_t incX, 
        bool* Y, ptrdiff_t incY) {
    for (int i = 0; i<n; ++i) Y[incY*i]=X[incX*i];
}

template <>
inline void blas_axpy<double>(ptrdiff_t n, double a, double* X, 
        ptrdiff_t incX, double* Y, ptrdiff_t incY) {
    BLAS_EXPANSION(daxpy)(&n,&a,X,&incX,Y,&incY);
}
template <> 
inline void blas_axpy<float>(ptrdiff_t n, float a, float* X,
        ptrdiff_t incX, float* Y, ptrdiff_t incY) {
    BLAS_EXPANSION(saxpy)(&n,&a,X,&incX,Y,&incY);
}
template <> 
inline void blas_axpy<int>(ptrdiff_t n, int a, int* X,
        ptrdiff_t incX, int* Y, ptrdiff_t incY) {
    for (int i = 0; i<n; ++i) Y[i] += a*X[i];
}
template <>
inline void blas_axpy<bool>(ptrdiff_t n, bool a, bool* X,
        ptrdiff_t incX, bool* Y, ptrdiff_t incY) {
    for (int i = 0; i<n; ++i) Y[i] = a*X[i];
}

template <>
inline void blas_scal<double>(ptrdiff_t n, double a, double* X,
        ptrdiff_t incX) {
    BLAS_EXPANSION(dscal)(&n,&a,X,&incX);
}
template <> 
inline void blas_scal<float>(ptrdiff_t n, float a, float* X, 
        ptrdiff_t incX) {
    BLAS_EXPANSION(sscal)(&n,&a,X,&incX);
}
template <> 
inline void blas_scal<int>(ptrdiff_t n, int a, int* X, 
        ptrdiff_t incX) {
    for (int i = 0; i<n; ++i) X[i*incX]*=a;
}

template <>
inline double blas_asum<double>(ptrdiff_t n, double* X, ptrdiff_t incX){
    return BLAS_EXPANSION(dasum)(&n,X,&incX);
}
template <>
inline float blas_asum<float>(ptrdiff_t n, float* X, ptrdiff_t incX) {
    return BLAS_EXPANSION(sasum)(&n,X,&incX);
}
template <> 
inline int blas_asum<int>(ptrdiff_t n, int* X, ptrdiff_t incX) {
    int sum = 0;
    for (int i=0; i<n*incX; i+=incX) sum += abs(X[i]);
    return sum;
}

template <> 
inline double blas_dot<double>(ptrdiff_t n, double* X, ptrdiff_t incX,
        double* Y,ptrdiff_t incY) {
    return BLAS_EXPANSION(ddot)(&n,X,&incX,Y,&incY);
}
template <>
inline float blas_dot<float>(ptrdiff_t n, float* X, ptrdiff_t incX,
        float* Y,ptrdiff_t incY) {
    return BLAS_EXPANSION(sdot)(&n,X,&incX,Y,&incY);
}
template <>
inline int blas_dot<int>(ptrdiff_t n, int* X, ptrdiff_t incX, 
        int* Y,ptrdiff_t incY) {
    int j = 0;
    int total = 0;
    for (int i = 0; i<n; ++i) {
        total+=X[i*incX]*Y[j];
        j+=incY;
    }
    return total;
}

// Implementations of the interfaces, BLAS Level 2
template <> 
inline void blas_gemv<double>(char TransA, ptrdiff_t M, ptrdiff_t N,
        double alpha, double *A, ptrdiff_t lda, double *X,
        ptrdiff_t incX, double beta, double *Y, ptrdiff_t incY) {
    BLAS_EXPANSION(dgemv)(&TransA,&M,&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
}
template <>
inline void blas_gemv<float>(char TransA, ptrdiff_t M, ptrdiff_t N,
        float alpha, float *A, ptrdiff_t lda, float *X, 
        ptrdiff_t incX, float beta, float *Y, ptrdiff_t incY) {
    BLAS_EXPANSION(sgemv)(&TransA,&M,&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
}

template <>
inline void blas_ger<double>(ptrdiff_t M, ptrdiff_t N, double alpha,
        double *X, ptrdiff_t incX, double* Y, ptrdiff_t incY, 
        double *A, ptrdiff_t lda) {
    BLAS_EXPANSION(dger)(&M,&N,&alpha,X,&incX,Y,&incY,A,&lda);
}
template <>
inline void blas_ger<float>(ptrdiff_t M, ptrdiff_t N, float alpha,
        float *X, ptrdiff_t incX, float* Y, ptrdiff_t incY,
        float *A, ptrdiff_t lda) {
    BLAS_EXPANSION(sger)(&M,&N,&alpha,X,&incX,Y,&incY,A,&lda);
}

template <>
inline void blas_trmv<double>(char Uplo, char TransA,
        char Diag, ptrdiff_t N, double *A, ptrdiff_t lda,
        double *X, ptrdiff_t incX) {
    BLAS_EXPANSION(dtrmv)(&Uplo,&TransA,&Diag,&N,A,&lda,X,&incX);
}
template <>
inline void blas_trmv<float>(char Uplo,
      char TransA, char Diag, ptrdiff_t N,
      float *A, ptrdiff_t lda, float *X, ptrdiff_t incX) {
    BLAS_EXPANSION(strmv)(&Uplo,&TransA,&Diag,&N,A,&lda,X,&incX);
}

template <>
inline void blas_syr( char Uplo, ptrdiff_t N, double alpha,
        double*X, ptrdiff_t incX, double *A, ptrdiff_t lda) {
    BLAS_EXPANSION(dsyr)(&Uplo,&N,&alpha,X,&incX,A,&lda);
}
template <>
inline void blas_syr( char Uplo, ptrdiff_t N, float alpha,
        float*X, ptrdiff_t incX, float *A, ptrdiff_t lda) {
    BLAS_EXPANSION(ssyr)(&Uplo,&N,&alpha,X,&incX,A,&lda);
}

template <>
inline void blas_symv(char Uplo, ptrdiff_t N, float alpha,
        float *A, ptrdiff_t lda, float *X, ptrdiff_t incX,
        float beta,float *Y,  ptrdiff_t incY) {
    BLAS_EXPANSION(ssymv)(&Uplo,&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
}
template <>
inline void blas_symv(char Uplo, ptrdiff_t N, double alpha,
        double *A, ptrdiff_t lda, double *X, ptrdiff_t incX,
        double beta,double *Y,  ptrdiff_t incY) {
    BLAS_EXPANSION(dsymv)(&Uplo,&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
}

// Implementations of the interfaces, BLAS Level 3
template <>
inline void blas_gemm<double>(char TransA, char TransB,
        ptrdiff_t M, ptrdiff_t N, ptrdiff_t K, double alpha,
        double *A, ptrdiff_t lda, double *B, ptrdiff_t ldb,
        double beta, double *C, ptrdiff_t ldc) {
    BLAS_EXPANSION(dgemm)(&TransA,&TransB,&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,
            C,&ldc);
}
template <> 
inline void blas_gemm<float>(char TransA, char TransB,
        ptrdiff_t M, ptrdiff_t N, ptrdiff_t K, float alpha, 
        float *A, ptrdiff_t lda, float *B, ptrdiff_t ldb,
        float beta, float *C, ptrdiff_t ldc) {
    BLAS_EXPANSION(sgemm)(&TransA,&TransB,&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,
            C,&ldc);
}

template <>
inline void blas_syrk<double>(char Uplo, char Trans, ptrdiff_t N, 
        ptrdiff_t K, double alpha, double *A, ptrdiff_t lda,
        double beta, double *C, ptrdiff_t ldc) {
    BLAS_EXPANSION(dsyrk)(&Uplo,&Trans,&N,&K,&alpha,A,&lda,&beta,C,&ldc);
}
template <>
inline void blas_syrk<float>(char Uplo, char Trans, ptrdiff_t N, 
        ptrdiff_t K, float alpha, float *A, ptrdiff_t lda,
        float beta, float *C, ptrdiff_t ldc) { 
    BLAS_EXPANSION(ssyrk)(&Uplo,&Trans,&N,&K,&alpha,A,&lda,&beta,C,&ldc);
}

template <>
inline void blas_trmm<double>(char Side, char Uplo, 
        char TransA, char Diag, ptrdiff_t M, ptrdiff_t N, 
        double alpha, double *A, ptrdiff_t lda,double *B, 
        ptrdiff_t ldb) {
    BLAS_EXPANSION(dtrmm)(&Side,&Uplo,&TransA,&Diag,&M,&N,&alpha,A,&lda,B,
            &ldb);
}
template <>
inline void blas_trmm<float>(char Side, char Uplo,
        char TransA, char Diag, ptrdiff_t M, ptrdiff_t N, 
        float alpha, float *A, ptrdiff_t lda,float *B, 
        ptrdiff_t ldb) {
    BLAS_EXPANSION(strmm)(&Side,&Uplo,&TransA,&Diag,&M,&N,&alpha,A,&lda,B,
            &ldb);
}

template <> 
inline int blas_iamax<double>(ptrdiff_t n, double* X, ptrdiff_t incX) {
    return (BLAS_EXPANSION(idamax)(&n,X,&incX)-1);
}
template <>
inline int blas_iamax<float>(ptrdiff_t n, float* X, ptrdiff_t incX) {
    return (BLAS_EXPANSION(isamax)(&n,X,&incX)-1);
}
template <>
inline int blas_iamax<int>(ptrdiff_t n, int* X, ptrdiff_t incX) {
    int max=0;
    int ind=0;
    for (int i=0; i<n*incX; i++) if (X[i*incX]>max) {
        max=X[i*incX];
        ind=i;
    }
    return ind;
}


// LAPACK routines
template <> 
inline void lapack_gesv<double>(ptrdiff_t N, ptrdiff_t NRHS, double *A,
        ptrdiff_t lda, ptrdiff_t *ipiv, double *B, ptrdiff_t ldb) {
    LAPACK_EXPANSION(dgesv)(&N,&NRHS,A,&lda,ipiv,B,&ldb,&lapack::info);
}
template <> 
inline void lapack_gesv<float>( ptrdiff_t N, ptrdiff_t NRHS, float *A,
        ptrdiff_t lda, ptrdiff_t *ipiv, float *B, ptrdiff_t ldb) {
    LAPACK_EXPANSION(sgesv)(&N,&NRHS,A,&lda,ipiv,B,&ldb,&lapack::info);
}



#endif 
