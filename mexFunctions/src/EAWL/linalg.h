/* \file
 *
 *                File linalg.h
 * Contains Matrix, Vector classes and linear algebra routines
 *
 * Modified from the SPAMS software package by Julien Mairal under the 
 * GNU General Public License, see <http://www.gnu.org/licenses/>.
 *
 * */

#ifndef LINALG_H
#define LINALG_H

#include "misc.h"
#include "cblas_template.h"
#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif
#include <list>
#include <vector>
#include <algorithm>

#ifdef FFT_CONV
#include <fftw3.h>
#endif

#ifdef NEW_MATLAB
   typedef ptrdiff_t INTT;
#else
   typedef int INTT;
#endif

#include <utils.h>

#undef max
#undef min

/// Matrix class
template<typename T>
class Matrix;
/// Vector class
template<typename T> 
class Vector;

typedef std::list< int > group;
typedef std::list< group > list_groups;
typedef std::vector< group > vector_groups;

template <typename T> 
static inline bool isZero(const T lambda) {
   return static_cast<double>(abs<T>(lambda)) < 1e-99;
}

template <typename T> 
static inline bool isEqual(const T lambda1, const T lambda2) {
   return static_cast<double>(abs<T>(lambda1-lambda2)) < 1e-99;
}


template <typename T>
static inline T softThrs(const T x, const T lambda) {
   if (x > lambda) {
      return x-lambda;
   } else if (x < -lambda) {
      return x+lambda;
   } else {
      return 0;
   }
};

template <typename T>
static inline T hardThrs(const T x, const T lambda) {
   return (x > lambda || x < -lambda) ? x : 0;
};


template <typename T>
static inline T xlogx(const T x) {
   if (x < -1e-20) {
      return INFINITY;
   } else if (x < 1e-20) {
      return 0;
   } else {
      return x*log(x);
   }
}

template <typename T>
static inline T logexp(const T x) {
   if (x < -30) {
      return 0;
   } else if (x < 30) {
      return log( T(1.0) + exp_alt<T>( x ) );
   } else {
      return x;
   }
}

/// Data class, abstract class, useful in the class image.
template <typename T> 
class Data {
   public:
      virtual void getData(Vector<T>& data, const int i) const = 0;
      virtual void getGroup(Matrix<T>& data, const vector_groups& groups,
            const int i) const = 0;
      virtual inline T operator[](const int index) const = 0;
      virtual int n() const = 0;
      virtual int m() const = 0;
      virtual int V() const = 0;
      virtual void norm_2sq_cols(Vector<T>& norms) const { };
      virtual ~Data() { };
};

/// Abstract matrix class
template <typename T> 
class AbstractMatrixB {
public:
    virtual int n() const = 0;
    virtual int m() const = 0;

    ///// Display functions ////////////////////////////////////////////////////
    virtual void print(const string& name) const = 0;
    
    ///// Access-routines (const) /////////////////////////////////////////////
    virtual void copyRow(const int i, Vector<T>& x) const = 0;
    virtual void copyTo(Matrix<T>& mat) const = 0;

    ///// Basic linear algebra /////////////////////////////////////////////////
    /// b <- alpha A'x + beta b
    virtual void multTrans(const Vector<T>& x, Vector<T>& b,
            const T alpha = 1.0, const T beta = 0.0) const = 0;
    virtual void mult(const Vector<T>& x, Vector<T>& b, 
        const T alpha = 1.0, const T beta = 0.0) const = 0;
    /// perform C = a*A*B + b*C, possibly transposing A or B.
    virtual void mult(const Matrix<T>& B, Matrix<T>& C, 
        const bool transA = false, const bool transB = false,
        const T a = 1.0, const T b = 0.0) const = 0;
    /// perform C = a*B*A + b*C, possibly transposing A or B.
    virtual void multSwitch(const Matrix<T>& B, Matrix<T>& C, 
        const bool transA = false, const bool transB = false,
        const T a = 1.0, const T b = 0.0) const = 0;
    virtual void XtX(Matrix<T>& XtX) const = 0;
    virtual T dot(const Matrix<T>& mat) const = 0;
    virtual ~AbstractMatrixB() { };
};

/// Abstract matrix class
template <typename T> 
class AbstractMatrix {
public:
    ///// Getting dimensions //////////////////////////////////////////////////
    virtual int n() const = 0;
    virtual int m() const = 0;

    ///// Access-routines (const) /////////////////////////////////////////////
    virtual void copyCol(const int j, T* raw) const = 0;
    virtual void copyCol(const int j, Vector<T>& col) const = 0;
    virtual void addCol(const int j, T* raw, const T a) const = 0;
    virtual void copyDiag(T* raw) const = 0;
    virtual void copyDiag(Vector<T>& dv) const = 0;
    virtual inline T operator()(const int index1, const int index2) const = 0;
    virtual ~AbstractMatrix() { };
};


/// Class Matrix
template<typename T> 
class Matrix : public Data<T>, public AbstractMatrix<T>, 
    public AbstractMatrixB<T> {
    friend class Vector<T>;
public:

   ///// Constructors, destructor /////////////////////////////////////////////
   Matrix();
   Matrix(int m, int n);
   Matrix(T* X, int m, int n);
   virtual ~Matrix() { clear(); };

   ///// Display functions ////////////////////////////////////////////////////
   inline void print(const string& name) const;

   ///// Getting dimensions ///////////////////////////////////////////////////
   inline int m() const { return _m; };
   inline int n() const { return _n; };

   ///// Setting values, sizes ////////////////////////////////////////////////
   inline void clear();
   inline void resize(int m, int n);
   /// Set specific values
   inline void setZeros();
   inline void setAleat();
   inline void set(const T a);
   inline void eye();
   /// Use existing data by reference (no copy)  
   inline void setData(const T* X, int m, int n);
   inline void copyRef(const Matrix<T>& mat);
   /// Copy existing data  
   inline void copy(const Matrix<T>& mat);
   inline void setDiag(const Vector<T>& d);
   inline void setDiag(const T val);
   inline void fillRow(const Vector<T>& row); //TODO
   inline void setRow(const int i, const Vector<T>& row); //TODO
   /// Add existing data  
   inline void addDiag(const Vector<T>& diag);
   inline void addDiag(const T diag);
   inline void addValsToCols(const Vector<T>& row); // TODO
   inline void addValsToRows(const Vector<T>& col, const T a = 1.0);
   inline void addRow(const int i, const Vector<T>& row, const T a=1.0);
   inline void add(const Matrix<T>& mat, const T alpha = 1.0);
   inline void add(const T alpha);
  
   ///// Elementwise access ///////////////////////////////////////////////////
   // const
   inline T operator()(const int i, const int j) const { return _X[j*_m+i]; }
   inline T operator[](const int index) const          { return _X[index];  }
   inline const T* rawX() const                        { return _X;         }
   /// modifiable
   inline T& operator()(const int i, const int j) { return _X[j*_m+i];  }
   inline T& operator[](const int index)          { return _X[index];   }
   inline T* rawX()                               { return _X;          }

   ///// Columnwise access ////////////////////////////////////////////////////
   inline void refCol(const int j, Vector<T>& x);
   inline const Vector<T>* refCol(const int j) const;
   inline void refSubMat(const int j1, const int j2, Matrix<T>& mat);

   ///// "Copy to"-routines (const) ///////////////////////////////////////////
   inline void copyRow(const int i, T* raw        ) const;
   inline void copyRow(const int i, Vector<T>& row) const;
   inline void copyCol(const int j, T* raw        ) const;
   inline void copyCol(const int j, Vector<T>& col) const;
   inline void copyDiag(T* raw      ) const;
   inline void copyDiag(Vector<T>& d) const;
   inline void copyTo(Matrix<T>& mat) const { mat.copy(*this); };
   inline void copyRaw(const int n, T* raw) const;
  
   ///// "Add to"-routines (const) ////////////////////////////////////////////
   inline void addCol(const int i, T* x, const T a=T(1.0)          ) const;
   inline void addCol(const int i, Vector<T>& col, const T a=T(1.0)) const;

   ///// Convenience modifiers ////////////////////////////////////////////////
   inline void scal(const T a);
   inline void multDiagLeft(const Vector<T>& diag);
   inline void multDiagRight(const Vector<T>& diag);
   inline void transpose(Matrix<T>& trans);
   inline void neg();
   inline void incrDiag();
 
   ///// Analysis functions (scalar) //////////////////////////////////////////
   inline bool isNormalized() const;
   inline int max() const;
   inline T maxval() const;
   inline int fmax() const;
   inline T fmaxval() const;
   inline int fmin() const;
   inline T trace() const;
   inline T asum() const;
   inline T normF() const;
   inline T mean() const;
   inline T normFsq() const;
   inline T nrm2sq() const { return this->normFsq(); };
   inline T norm_inf_2_col() const;
   inline T norm_1_2_col() const;
   inline T dot(const Matrix<T>& mat) const;

   ///// Analysis functions (vector) //////////////////////////////////////////
   inline void sum_cols(Vector<T>& sum) const;
   inline void sum_rows(Vector<T>& sum) const;
   inline void meanCol(Vector<T>& mean) const;
   inline void meanRow(Vector<T>& mean) const;
   inline void norm_2_cols(Vector<T>& norms) const;
   inline void norm_2_rows(Vector<T>& norms) const;
   inline void norm_inf_cols(Vector<T>& norms) const;
   inline void norm_inf_rows(Vector<T>& norms) const;
   inline void norm_l1_rows(Vector<T>& norms) const;
   inline void norm_l1_cols(Vector<T>& norms) const;
   inline void norm_2sq_cols(Vector<T>& norms) const;
   inline void norm_2sq_rows(Vector<T>& norms) const;

   ///// Elementwise modifications ////////////////////////////////////////////
   inline void mult_elementWise(const Matrix<T>& B, Matrix<T>& C) const;
   inline void div_elementWise(const Matrix<T>& B, Matrix<T>& C) const;
   inline void inv_elem();
   inline void inv() { this->inv_elem(); }; // TODO: redundant
   inline void exp();
   inline void Sqrt();
   inline void sqr();
   inline void Invsqrt();

   ///// Dictionary functions /////////////////////////////////////////////////
   inline void clean();
   inline void normalize();
   inline void normalize2();
   inline void center();
   inline void center_rows();
   inline void center(Vector<T>& centers);
   inline void whiten(const int V); // TODO
   inline void whiten(Vector<T>& mean, const bool pattern = false);
   inline void whiten(Vector<T>& mean, const Vector<T>& mask);
   inline void unwhiten(Vector<T>& mean, const bool pattern = false);
   inline void merge(const Matrix<T>& B, Matrix<T>& C) const;

   ///// Manual manipulations /////////////////////////////////////////////////
   inline void setm(const int m) { _m = m; }; //DANGEROUS
   inline void setn(const int n) { _n = n; }; //DANGEROUS
   inline void fakeSize(const int m, const int n) { _n = n; _m=m;};

   ///// Symmetric matrices ///////////////////////////////////////////////////
   /// Make symmetric: copy upper-right into lower-left
   inline void fillSymmetric();
   inline void fillSymmetric2();
   /// Extract sub-matrix
   inline void subMatrixSym(const Vector<int>& indices, 
           Matrix<T>& subMatrix) const;
   /// Find eigenvector to largest eigenvalue
   inline void eigLargestSymApprox(const Vector<T>& u0, Vector<T>& u) const;
   inline T eigLargestMagnSym(const Vector<T>& u0, Vector<T>& u) const;
   inline T eigLargestMagnSym() const;
   // Invert
   inline void invSym();

   ///// Basic linear algebra /////////////////////////////////////////////////
   /// perform b = alpha*A'x + beta*b
   inline void multTrans(const Vector<T>& x, Vector<T>& b, const T alpha = 1.0,
           const T beta = 0.0) const;
   inline void multTrans(const Vector<T>& x, Vector<T>& b,
           const Vector<bool>& active) const; // TODO
   /// perform b = alpha*A*x+beta*b
   inline void mult(const Vector<T>& x, Vector<T>& b, const T alpha = 1.0,
           const T beta = 0.0) const;
   /// perform C = a*A*B + b*C, possibly transposing A or B.
   inline void mult(const Matrix<T>& B, Matrix<T>& C, 
         const bool transA = false, const bool transB = false,
         const T a = 1.0, const T b = 0.0) const;
   /// perform C = a*B*A + b*C, possibly transposing A or B.
   inline void multSwitch(const Matrix<T>& B, Matrix<T>& C, 
         const bool transA = false, const bool transB = false,
         const T a = 1.0, const T b = 0.0) const;
   /// Covariance matrix
   inline void XtX(Matrix<T>& XtX) const;
   inline void XXt(Matrix<T>& XXt) const;
   inline void upperTriXXt(Matrix<T>& XXt, const int L) const;

   ///// High-level linear algebra //////////////////////////////////////////// 
   inline void svdRankOne(const Vector<T>& u0, Vector<T>& u,
           Vector<T>& v) const;
   inline void singularValues(Vector<T>& u) const;
   inline void svd(Matrix<T>& U, Vector<T>& S, Matrix<T>&V) const;
   inline void softThrshold(const T nu);
   inline void hardThrshold(const T nu);
   inline void thrshold(const T nu);
   inline void thrsmax(const T nu);
   inline void thrsmin(const T nu);
   inline void thrsabsmin(const T nu);
   inline void thrsPos();
   inline void rank1Update(const Vector<T>& vec1, const Vector<T>& vec2,
           const T alpha = 1.0);
   inline void conjugateGradient(const Vector<T>& b, Vector<T>& x,
           const T tol = 1e-4, const int = 4) const;
   inline void drop(char* fileName) const;
   inline void NadarayaWatson(const Vector<int>& ind, const T sigma);
   inline void blockThrshold(const T nu, const int sizeGroup);
   inline void sparseProject(Matrix<T>& out, const T thrs, const int mode = 1,
           const T lambda1 = 0, const T lambda2 = 0, const T lambda3 = 0,
           const bool pos = false, const int numThreads=-1);
   inline void transformFilter();


   ///// Redundant stuff //////////////////////////////////////////////////////
   /// Copy the column i into x
   inline void getData(Vector<T>& data, const int i) const;
   inline void sub(const Matrix<T>& mat); // TODO: redundant
   /// make a reference of the matrix to a vector vec 
   inline void toVect(Vector<T>& vec) const;

   ///// Misc /////////////////////////////////////////////////////////////////
   virtual void getGroup(Matrix<T>& data, const vector_groups& groups,
         const int i) const;
   inline int V() const { return 1;}; // third dimension 
   inline void copyMask(Matrix<T>& out, Vector<bool>& mask) const;

protected:
   ///// Forbidding lazy copies ///////////////////////////////////////////////
   explicit Matrix<T>(const Matrix<T>& matrix);
   Matrix<T>& operator=(const Matrix<T>& matrix);

   ///// Members //////////////////////////////////////////////////////////////
   bool _externAlloc;   // if false, data is allocated automatically
   T*   _X;             // ptr to data
   int _m;              // no. rows
   int _n;              // no. columns
};

/// Class Vector
template<typename T> 
class Vector {
   friend class Matrix<T>;
   public:
   ///// Constructors, destructor /////////////////////////////////////////////
   Vector();
   Vector(T* X, int n);
   Vector(int n);
   explicit Vector<T>(const Vector<T>& vec);
   explicit Vector<T>(const Vector<T>* vec); //TODO
   Vector<T>(const Matrix<T>& M, int j);
   ~Vector() { clear(); };
   ///// Display functions ////////////////////////////////////////////////////
   inline void print(const char* name) const;

   ///// Getting dimension ////////////////////////////////////////////////////
   inline int n() const;

   ///// Setting values, sizes ////////////////////////////////////////////////
   inline void clear();
   inline void resize(const int n);
   /// Set specific values
   inline void setZeros();
   inline void set(const T val);
   inline void randperm(int n);
   inline void setAleat();
   inline void logspace(const int n, const T a, const T b);
   /// Use existing data (no copy)  
   inline void setPointer(T* X, const int n);
   inline void setData(T* X, const int n) { this->setPointer(X,n); };
   /// Copy existing data  
   inline void copy(const Vector<T>& x);
   inline void copy(const Vector<T>& x, const int n, const int off_x, 
           const int off_this);
   inline void copy(const T* x, int n);

   ///// Elementwise access ///////////////////////////////////////////////////
   // Const
   inline T operator[](const int index) const;
   // Modifiable
   inline T& operator[](const int index);
   inline T* rawX();
   inline const T* rawX() const;

   ///// Extract data /////////////////////////////////////////////////////////
   inline void extractRaw(T* x, const int n) const;

   ///// Thresholding /////////////////////////////////////////////////////////
   inline void softThrshold(const T nu);
   inline void hardThrshold(const T nu);
   inline void thrsmax(const T nu);
   inline void thrsmin(const T nu);
   inline void thrsabsmin(const T nu);
   inline void thrsabsmax(const T nu);
   inline void thrshold(const T nu);
   inline void thrsPos();

   ///// Analysis functions (scalar) //////////////////////////////////////////
   inline int max() const;
   inline int min() const;
   inline int fmax() const;
   inline int fmin() const;
   inline T maxval() const;
   inline T minval() const;
   inline T fmaxval() const;
   inline T fminval() const;
   inline T nrm2() const;
   inline T nrm2sq() const;
   inline T mean() const;
   inline T std() const;
   inline T asum() const;
   inline T lzero() const;
   inline T afused() const;
   inline T sum() const;
   inline int nnz() const;
   inline bool alltrue() const;
   inline bool allfalse() const;

   ///// Analysis functions (vector) //////////////////////////////////////////
   inline void sort(Vector<T>& out, const bool mode) const;
   inline void sort2(Vector<T>& out, Vector<int>& key, const bool mode) const;

   ///// Convenience modifiers ////////////////////////////////////////////////
   inline void sort(const bool mode);
   inline void sort2(Vector<int>& key, const bool mode);

   ///// Elementwise modifications ////////////////////////////////////////////
   inline void exp();
   inline void logexp();
   inline void Sqrt();
   inline void sqr();
   inline void Invsqrt();
   inline void inv();
   inline void scal(const T a);
   inline void neg();
   inline void add(const Vector<T>& x, const T a = 1.0);
   inline void add_subvector(const Vector<T>& x, const int n, const int off_x,
           const int off_this, const T a = 1.0); 
   inline void add(const T a, const int n=0);
   inline void sub(const Vector<T>& x);
   inline void div(const Vector<T>& x);
   inline void mult(const Vector<T>& x, const Vector<T>& y);
   inline void div(const Vector<T>& x, const Vector<T>& y);
   inline void sqr(const Vector<T>& x);
   inline void Sqrt(const Vector<T>& x);
   inline void Invsqrt(const Vector<T>& x);
   inline void inv(const Vector<T>& x);

//   inline void mult_elementWise(const Vector<T>& B, Vector<T>& C) const { C.mult(*this,B); };

   inline T softmax(const int y); //TODO

   /// Algebric operations
   inline T dot(const Vector<T>& x) const;
   inline T KL(const Vector<T>& X) const;
   inline void normalize();
   inline void normalize2();
   inline void whiten(Vector<T>& mean, const bool pattern = false);
   inline void whiten(Vector<T>& mean, const Vector<T>& mask); //TODO
   inline void whiten(const int V);
   inline void unwhiten(Vector<T>& mean, const bool pattern = false);
   inline void sign(Vector<T>& signs) const;
   /// projects the vector onto the l1 ball of radius thrs,
   inline void l1project(Vector<T>& out, const T thrs, 
           const bool simplex = false) const;
   inline void l1project_weighted(Vector<T>& out, const Vector<T>& weights, 
           const T thrs, const bool residual = false) const;
   inline void l1l2projectb(Vector<T>& out, const T thrs, const T gamma, 
           const bool pos = false, const int mode = 1);
   inline void sparseProject(Vector<T>& out, const T thrs, const int mode = 1,
           const T lambda1 = 0, const T lambda2 = 0, const T lambda3 = 0,
           const bool pos = false);
   inline void project_sft(const Vector<int>& labels, const int clas);
   inline void project_sft_binary(const Vector<T>& labels);
   inline void l1l2project(Vector<T>& out, const T thrs, const T gamma, 
           const bool pos = false) const;
   inline void fusedProject(Vector<T>& out, const T lambda1, const T lambda2, 
           const int itermax);
   inline void fusedProjectHomotopy(Vector<T>& out, const T lambda1,
           const T lambda2, const T lambda3 = 0,
         const bool penalty = true);
   inline void applyBayerPattern(const int offset);


   ///// Misc /////////////////////////////////////////////////////////////////
   inline void fakeSize(const int n) { _n = n; }; //DANGEROUS
   inline void setn(const int n) { _n = n; }; //DANGEROUS
   inline void copyMask(Vector<T>& out, Vector<bool>& mask) const;
private:
   inline void setConstPointer(Vector<T> t, T* X) { throw(1); }
   inline void setConstPointer(const Vector<T> t, const T* X) const { t._X = X; }

   ///// Forbidding lazy copies ///////////////////////////////////////////////
   Vector<T>& operator=(const Vector<T>& vec);

   ///// Memebers /////////////////////////////////////////////////////////////
   mutable bool _externAlloc;    // if false, data is allocated automatically
   mutable T* _X;        // ptr to data
   mutable int _n;               // no. elements
};

/// Class representing Covariance matrix of a shiftable dictionary
template<typename T> 
class CovShMat {
public:
   ///// Constructors, destructor /////////////////////////////////////////////
   CovShMat();
   CovShMat(const int K, const int n);
   CovShMat(const Matrix<T>& D, const int n=0);
   ~CovShMat();

   ///// Set data /////////////////////////////////////////////////////////////
   inline void setDims(const int K, const int n);
   inline void set(const Matrix<T>& D);
#ifdef FFT_CONV
   inline void setFft(fftw_plan* p, fftw_plan* ip, const int fft_len, 
           double* _buffer, Matrix<T>* _Dwork);
#endif

   ///// Getting dimensions ///////////////////////////////////////////////////
   inline int K() const { return _K;}
   inline int n1() const { return _n1;}
   inline int n2() const { return _n2;}
    
   // add to diagonal
   inline void addDiag(const T diag);

   ///// Elementwise access (const) ///////////////////////////////////////////
   inline T operator()(const int k1, const int k2, const int diff12) const;
//  inline T* operator[](const int i) const { return _X; };
   inline T* rawX(const int k1, const int k2) const;
   inline void generateG(T* x) const;
   inline void generateGk(const int k2, const int s2, T* x) const;

private:
   inline void _allocate();
   inline void _deallocate();
   inline int _getOffset(const int k1, const int k2) const;


  T*    _X;
  int   _K;
  int   _n1;
  int   _n2;
  bool  _allocated;
#ifdef FFT_CONV
  fftw_plan*    _p;
  fftw_plan*    _ip;
  int           _fft_len;
  double*       _buffer;
  Matrix<T>*    _Dwork;
#endif
};

/// Special block sparse matrix
template<typename T> 
class BlSpMat {
public:
   ///// Constructors, destructor /////////////////////////////////////////////
   BlSpMat();
   BlSpMat(const int n2, const int K, const int L);
   ~BlSpMat();

    inline void setOffset(const int j, const int first, const int nel);
    inline int getFirst(const int j) const { return _first[j]; };
    inline int getNumEl(const int j) const { return _numel[j]; };
    inline T* rawX() const { return _X; }

private:
  T*    _X;
  int*  _first;
  int*  _numel;
  int   _n2;
  int   _K;
  int   _L;
  bool  _allocated;

};

#if 1



/// Class representing the product of two matrices
template<typename T> 
class ProdMatrix : public AbstractMatrix<T> {
public:
   ///// Constructors, destructor /////////////////////////////////////////////
   ProdMatrix();
   ProdMatrix(const Matrix<T>& D, const bool high_memory = true);
   ProdMatrix(const Matrix<T>& D, const Matrix<T>& X,
           const bool high_memory = true);
   ~ProdMatrix() { delete(_DtX);} ;

   ///// Getting dimensions ///////////////////////////////////////////////////
   inline int n() const { return _n;};
   inline int m() const { return _m;};
    
   ///// Set matrices /////////////////////////////////////////////////////////
   inline void setMatrices(const Matrix<T>& D, const bool high_memory=true);
   inline void setMatrices(const Matrix<T>& D, const Matrix<T>& X,
          const bool high_memory=true);
   // add to diagonal
   void inline addDiag(const T diag);

   ///// Elementwise access (const) ///////////////////////////////////////////
  inline T operator()(const int index1, const int index2) const;
  inline T operator[](const int index) const;

   ///// "Copy to"-routines (const) ///////////////////////////////////////////
   inline void copyCol(const int i, Vector<T>& DtXi) const;
   inline void extract_rawCol(const int i,T* DtXi) const;
  void inline diag(Vector<T>& diag) const;

   ///// "Add to"-routines (const) ////////////////////////////////////////////
   virtual void add_rawCol(const int i, T* DtXi, const T a) const;

private:
  /// Depending on the mode, DtX is a matrix, or two matrices
  Matrix<T>* _DtX;
  const Matrix<T>* _X;
  const Matrix<T>* _D;
  bool _high_memory;
  int _n;
  int _m;
  T _addDiag;
};
#endif

#if 1

/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS MATRIX ************************************/
/*****************************************************************************/


///// Constructors, Destructors ///////////////////////////////////////////////
// Constructor
template <typename T>
Matrix<T>::Matrix(): _externAlloc(true), _X(NULL), _m(0), _n(0) { };
// Constructor with allocation 
template <typename T>
Matrix<T>::Matrix(int m, int n) : _externAlloc(false), _m(m), _n(n)  {
#pragma omp critical 
    {
        _X= new T[_n*_m];
    }
}
// Constructor with data
template <typename T>
Matrix<T>::Matrix(T* X, int m, int n): _externAlloc(true), _X(X), _m(m),
    _n(n) {
}

///// Display functions ///////////////////////////////////////////////////////
// Print the matrix to std::cerr
template <typename T>
inline void Matrix<T>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _m << " x " << _n << std::endl;
   for (int i = 0; i<_m; ++i) {
      for (int j = 0; j<_n; ++j) std::cerr << _X[j*_m+i] << " ";
      std::cerr << std::endl;
   }
   std::cerr << std::endl;
}

///// Setting values, sizes ///////////////////////////////////////////////////
/// Clear the matrix
template <typename T> 
inline void Matrix<T>::clear() {
   if (!_externAlloc) delete[](_X);
   _n=0;
   _m=0;
   _X=NULL;
   _externAlloc=true;
}
/// Resize the matrix
template <typename T> 
inline void Matrix<T>::resize(int m, int n) {
   if (_n==n && _m==m) return;
   if (!_externAlloc) delete[](_X);
   _n=n;
   _m=m;
   _externAlloc=false;
#pragma omp critical
   {
      _X=new T[_n*_m];
   }
}
/// Set all the values to zero
template <typename T> 
inline void Matrix<T>::setZeros() {
   memset(_X,0,_n*_m*sizeof(T));
}
/// Put white Gaussian noise in the matrix 
template <typename T> 
inline void Matrix<T>::setAleat() {
   for (int i = 0; i<_n*_m; ++i) _X[i]=normalDistrib<T>();
}
/// Set all the values to a scalar
template <typename T> 
inline void Matrix<T>::set(const T a) {
   for (int i = 0; i<_n*_m; ++i) _X[i]=a; //TODO: efficiently!
}
/// set the matrix to the identity
template <typename T> 
inline void Matrix<T>::eye() {
   this->setZeros();
   this->setDiag(T(1.0));
}
/// Change sizes and use external data 
template <typename T> 
inline void Matrix<T>::setData(const T* X, int m, int n) {
   if (!_externAlloc) delete[](_X);
   _X=X;
   _m=m;
   _n=n;
   _externAlloc=true;
}
/// create a new view on the matrix mat in the current matrix
template <typename T> 
inline void Matrix<T>::copyRef(const Matrix<T>& mat) {
   this->setData(mat.rawX(),mat.m(),mat.n());
}
/// make a copy of the matrix mat in the current matrix
template <typename T>
inline void Matrix<T>::copy(const Matrix<T>& mat) {
   this->resize(mat._m,mat._n);
   cblas_copy<T>(_m*_n,mat._X,1,_X,1);
}
/// set the diagonal from vector
template <typename T> 
inline void Matrix<T>::setDiag(const Vector<T>& dv) {
   int size_diag=MIN(_n,_m);
   const T* const d = dv.rawX();
   for (int i = 0; i<size_diag; ++i) _X[i*_m+i]=d[i];
}
/// set the diagonal to value
template <typename T> 
inline void Matrix<T>::setDiag(const T val) {
   int size_diag=MIN(_n,_m);
   for (int i = 0; i<size_diag; ++i) _X[i*_m+i]=val;
}
/// fill the matrix with the row given
template <typename T> 
inline void Matrix<T>::fillRow(const Vector<T>& row) {
   for (int i = 0; i<_n; ++i) {
      T val = row[i];
      std::fill_n(_X+i*_m,_m,val);
   }
}
/// set row i of the matrix
template <typename T> 
inline void Matrix<T>::setRow(const int i, const Vector<T>& row) {
    cblas_copy(_n,row._X,1,_X+i,_m);
}




/// add to the diagonal from vector
template <typename T> 
inline void Matrix<T>::addDiag( const Vector<T>& diag) {
   T* d= diag.rawX();
   for (int i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] += d[i];
}
/// add value diagonal
template <typename T> 
inline void Matrix<T>::addDiag(const T diag) {
   for (int i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] += diag;
}
template <typename T> 
inline void Matrix<T>::addValsToCols(const Vector<T>& row) {
   Vector<T> col;
   for (int i = 0; i<_n; ++i) {
      this->refCol(i,col);      
      col.add(row[i]);
   }
}
template <typename T> 
inline void Matrix<T>::addValsToRows(const Vector<T>& col, const T a) {
   Vector<T> coli;
   for (int i = 0; i<_n; ++i) {
      this->refCol(i,coli);      
      coli.add(col,a);
   }
}
/// fill the matrix with the row given
template <typename T> 
inline void Matrix<T>::addRow(const int i, const Vector<T>& row, const T a) {
    cblas_axpy<T>(_n,a,row,1,_X+i,_m);
}
/// add alpha*mat to the current matrix
template <typename T> 
inline void Matrix<T>::add(const Matrix<T>& mat, const T alpha) {
   assert(mat._m == _m && mat._n == _n);
   cblas_axpy<T>(_n*_m,alpha,mat._X,1,_X,1);
}
/// add alpha to the current matrix
template <typename T> 
inline void Matrix<T>::add(const T alpha) {
   for (int i = 0; i<_n*_m; ++i) _X[i]+=alpha; //TODO: more efficiently (?)
}

///// Columnwise access ///////////////////////////////////////////////////////

/// Reference the column i into the vector x
template <typename T>
inline void Matrix<T>::refCol(const int i, Vector<T>& x) {
   assert(i >= 0 && i<_n);
   if (!x._externAlloc) delete[] x._X;
   x._X=_X+i*_m;
   x._n=_m;
   x._externAlloc=true; 
}
/// Reference the column i into a const vector and return it
template <typename T>
inline const Vector<T>* Matrix<T>::refCol(const int j) const {
   assert(j >= 0 && j<_n);
   return new const Vector<T>(_X+j*_m,_m);
}

/// Reference the column i to i+n into the Matrix mat
template <typename T> 
inline void Matrix<T>::refSubMat(const int i, const int n, Matrix<T>& mat) {
   mat.setData(_X+i*_m,_m,n);
}

///// "Copy to"-routines //////////////////////////////////////////////////////
/// Copy row i to raw 
template <typename T> 
inline void Matrix<T>::copyRow(const int i, T* raw) const {
   assert(i >= 0 && i<_m);
   cblas_copy<T>(_n,_X+i,_m,raw,1);
}
/// Copy row i into vector
template <typename T> 
inline void Matrix<T>::copyRow(const int i, Vector<T>& row) const {
   row.resize(_n);
   this->copyRow(i,row._X);
}
/// Copy column j to raw 
template <typename T> 
inline void Matrix<T>::copyCol(const int j, T* raw) const {
   assert(j >= 0 && j<_n);
   cblas_copy<T>(_m,_X+j*_m,1,raw,1);
}
/// Copy column j into vector x
template <typename T>
inline void Matrix<T>::copyCol(const int j, Vector<T>& col) const {
   col.resize(_m);
   this->copyCol(j,col._X);
}
/// Copy diagonal to raw 
template <typename T> 
inline void Matrix<T>::copyDiag(T* raw) const {
   for (int i = 0; i<MIN(_n,_m); ++i) raw[i]=_X[i*_m+i];
}
/// Copy diagonal to vector 
template <typename T> 
inline void Matrix<T>::copyDiag(Vector<T>& dv) const {
   dv.resize(MIN(_n,_m));
   this->copyDiag(dv.rawX());
}
/// Copy n first elements to x
template <typename T> 
inline void Matrix<T>::copyRaw(const int n, T* x) const {
   assert(n >= 0 && n<=_m*_n);
   cblas_copy<T>(n,_X,1,x,1);
}


///// "Add to"-routines ///////////////////////////////////////////////////////
/// Add column j to raw
template <typename T>
inline void Matrix<T>::addCol(const int j, T* raw, const T a) const {
   assert(j >= 0 && j<_n);
   cblas_axpy<T>(_m,a,_X+j*_m,1,raw,1);
}
/// Add column j to vector 
template <typename T>
inline void Matrix<T>::addCol(const int j, Vector<T>& col, const T a) const {
    col.resize(_m);
    this->addCol(j,col.rawX(),a);
}




//////// Convenience modifiers ////////////////////////////////////////////////
/// scale the matrix by a
template <typename T> 
inline void Matrix<T>::scal(const T a) {
   cblas_scal<T>(_n*_m,a,_X,1);
}
/// mult by a diagonal matrix on the left
template <typename T>
inline void Matrix<T>::multDiagLeft(const Vector<T>& diag) {
    assert(diag.n() == _m);
    const T* d = diag.rawX();
    for (int i=0; i<_m; ++i) cblas_scal<T>(_n,d[i],_X+i,_m);
}
/// mult by a diagonal matrix on the right
template <typename T> 
inline void Matrix<T>::multDiagRight( const Vector<T>& diag) {
    assert(diag.n() == _n);
    const T* d = diag.rawX();
    for (int j=0; j<_n; ++j) cblas_scal<T>(_m,d[j],_X+j*_m,1);
}
/// transpose matrix and write result to trans
template <typename T> 
inline void Matrix<T>::transpose(Matrix<T>& trans) {
    trans.resize(_n,_m);
    T* out = trans._X;
    for (int j=0; j<_n; ++j) cblas_copy<T>(_m,_X+j*_m,1,out+j,_n);
}
/// A <- -A
template <typename T> 
inline void Matrix<T>::neg() { this->scal(T(-1.0)); }
/// increment the diagonal
template <typename T> 
inline void Matrix<T>::incrDiag() {
   for (int i = 0; i<MIN(_n,_m); ++i) ++_X[i*_m+i];
}


///// Analysis functions (scalar) /////////////////////////////////////////////
/// Check wether the columns of the matrix are normalized or not
template <typename T> 
inline bool Matrix<T>::isNormalized() const {
   for (int j = 0; j<_n; ++j) {
      T norm=cblas_nrm2<T>(_m,_X+_m*j,1);
      if (fabs(norm - 1.0) > 1e-6) return false;
   }
   return true;
}

///// Const accessors /////////////////////////////////////////////////////////

///// Redundant stuff /////////////////////////////////////////////////////////
/// Copy the column i into x
template <typename T> 
inline void Matrix<T>::getData(Vector<T>& x, const int i) const {
   this->copyCol(i,x);
}

template <typename T> 
inline void Matrix<T>::getGroup(Matrix<T>& data, 
      const vector_groups& groups, const int i) const {
   const group& gr = groups[i];
   const int N = gr.size();
   data.resize(_m,N);
   int count=0;
   for (group::const_iterator it = gr.begin(); it != gr.end(); ++it) {
      cblas_copy<T>(_m,_X+(*it)*_m,1,data._X+count*_m,1);
      ++count;
   }
};



/// clean a dictionary matrix
template <typename T>
inline void Matrix<T>::clean() {
   this->normalize();
   Matrix<T> G;
   this->XtX(G);
   T* prG = G._X;
   /// remove the diagonal
   for (int i = 0; i<_n; ++i) {
      for (int j = i+1; j<_n; ++j) {
         if (prG[i*_n+j] > 0.99) {
            // remove nasty column j and put random values inside
            Vector<T> col;
            this->refCol(j,col);
            col.setAleat();
            col.normalize();
         }
      }
   }
};

/// return the 1D-index of the greatest value
template <typename T> 
inline int Matrix<T>::max() const {
    int imax=0;
    T max=_X[0];
    for (int j = 1; j<_n; ++j) {
        T cur = _X[j];
        if (cur > max) {
            imax=j;
            max = cur;
        }
    }
   return imax;
};

/// return the greatest value
template <typename T> 
inline T Matrix<T>::maxval() const {
    int imax=0;
    T max=_X[0];
    for (int j = 1; j<_n; ++j) {
        T cur = _X[j];
        if (cur > max) {
            imax=j;
            max = cur;
        }
    }
    return _X[imax];
};


/// return the 1D-index of the value of greatest magnitude
template <typename T> 
inline int Matrix<T>::fmax() const {
   return cblas_iamax<T>(_n*_m,_X,1);
};

/// return the value of greatest magnitude
template <typename T> 
inline T Matrix<T>::fmaxval() const {
   return _X[cblas_iamax<T>(_n*_m,_X,1)];
};


/// return the 1D-index of the value of lowest magnitude
template <typename T> 
inline int Matrix<T>::fmin() const {
   return cblas_iamin<T>(_n*_m,_X,1);
};

/// extract a sub-matrix of a symmetric matrix
template <typename T> 
inline void Matrix<T>::subMatrixSym(
      const Vector<int>& indices, Matrix<T>& subMatrix) const {
   int L = indices.n();
   subMatrix.resize(L,L);
   T* out = subMatrix._X;
   int* rawInd = indices.rawX();
   for (int i = 0; i<L; ++i)
      for (int j = 0; j<=i; ++j)
         out[i*L+j]=_X[rawInd[i]*_n+rawInd[j]];
   subMatrix.fillSymmetric();
};

/// Normalize all columns to unit l2 norm
template <typename T> 
inline void Matrix<T>::normalize() {
   //T constant = 1.0/sqrt(_m);
   for (int i = 0; i<_n; ++i) {
      T norm=cblas_nrm2<T>(_m,_X+_m*i,1);
      if (norm > 1e-10) {
         T invNorm=1.0/norm;
         cblas_scal<T>(_m,invNorm,_X+_m*i,1);
      }  else {
         // for (int j = 0; j<_m; ++j) _X[_m*i+j]=constant;
         Vector<T> d;
         this->refCol(i,d);
         d.setAleat();
         d.normalize();
      } 
   }
};

/// Normalize all columns which l2 norm is greater than one.
template <typename T> 
inline void Matrix<T>::normalize2() {
   for (int i = 0; i<_n; ++i) {
      T norm=cblas_nrm2<T>(_m,_X+_m*i,1);
      if (norm > 1.0) {
         T invNorm=1.0/norm;
         cblas_scal<T>(_m,invNorm,_X+_m*i,1);
      } 
   }
};

/// center the matrix
template <typename T> 
inline void Matrix<T>::center() {
   for (int i = 0; i<_n; ++i) {
      Vector<T> col;
      this->refCol(i,col);
      T sum = col.sum();
      col.add(-sum/static_cast<T>(_m));
   }
};

/// center the matrix
template <typename T> 
inline void Matrix<T>::center_rows() {
   Vector<T> mean_rows(_m);
   mean_rows.setZeros();
   for (int i = 0; i<_n; ++i) 
      for (int j = 0; j<_m; ++j) 
         mean_rows[j] += _X[i*_m+j];
   mean_rows.scal(T(1.0)/_n);
   for (int i = 0; i<_n; ++i) 
      for (int j = 0; j<_m; ++j) 
         _X[i*_m+j] -= mean_rows[j];
};

/// center the matrix and keep the center values
template <typename T> 
inline void Matrix<T>::center(Vector<T>& centers) {
   centers.resize(_n);
   for (int i = 0; i<_n; ++i) {
      Vector<T> col;
      this->refCol(i,col);
      T sum = col.sum()/static_cast<T>(_m);
      centers[i]=sum;
      col.add(-sum);
   }
};



/// make the matrix symmetric by copying the upper-right part
/// into the lower-left part
//TODO: could be stored more efficiently
template <typename T> 
inline void Matrix<T>::fillSymmetric() {
   for (int i = 0; i<_n; ++i) {
      for (int j =0; j<i; ++j) {
         _X[j*_m+i]=_X[i*_m+j];
      }
   }
};
template <typename T> 
inline void Matrix<T>::fillSymmetric2() {
   for (int i = 0; i<_n; ++i) {
      for (int j =0; j<i; ++j) {
         _X[i*_m+j]=_X[j*_m+i];
      }
   }
};


template <typename T> 
inline void Matrix<T>::whiten(const int V) {
   const int sizePatch=_m/V;
   for (int i = 0; i<_n; ++i) {
      for (int j = 0; j<V; ++j) {
         T mean = 0;
         for (int k = 0; k<sizePatch; ++k) {
            mean+=_X[i*_m+sizePatch*j+k];
         }
         mean /= sizePatch;
         for (int k = 0; k<sizePatch; ++k) {
            _X[i*_m+sizePatch*j+k]-=mean;
         }
      }
   }
};

template <typename T> 
inline void Matrix<T>::whiten(Vector<T>& mean, const bool pattern) {
   mean.setZeros();
   if (pattern) {
      const int n =static_cast<int>(sqrt(static_cast<T>(_m)));
      int count[4];
      for (int i = 0; i<4; ++i) count[i]=0;
      for (int i = 0; i<_n; ++i) {
         int offsetx=0;
         for (int j = 0; j<n; ++j) {
            offsetx= (offsetx+1) % 2;
            int offsety=0;
            for (int k = 0; k<n; ++k) {
               offsety= (offsety+1) % 2;
               mean[2*offsetx+offsety]+=_X[i*_m+j*n+k];
               count[2*offsetx+offsety]++;
            }
         }
      }
      for (int i = 0; i<4; ++i)
         mean[i] /= count[i];
      for (int i = 0; i<_n; ++i) {
         int offsetx=0;
         for (int j = 0; j<n; ++j) {
            offsetx= (offsetx+1) % 2;
            int offsety=0;
            for (int k = 0; k<n; ++k) {
               offsety= (offsety+1) % 2;
               _X[i*_m+j*n+k]-=mean[2*offsetx+offsety];
            }
         }
      }
   } else  {
      const int V = mean.n();
      const int sizePatch=_m/V;
      for (int i = 0; i<_n; ++i) {
         for (int j = 0; j<V; ++j) {
            for (int k = 0; k<sizePatch; ++k) {
               mean[j]+=_X[i*_m+sizePatch*j+k];
            }
         }
      }
      mean.scal(T(1.0)/(_n*sizePatch));
      for (int i = 0; i<_n; ++i) {
         for (int j = 0; j<V; ++j) {
            for (int k = 0; k<sizePatch; ++k) {
               _X[i*_m+sizePatch*j+k]-=mean[j];
            }
         }
      }
   }
};

template <typename T> 
inline void Matrix<T>::whiten(Vector<T>& mean, const
      Vector<T>& mask) {
   const int V = mean.n();
   const int sizePatch=_m/V;
   mean.setZeros();
   for (int i = 0; i<_n; ++i) {
      for (int j = 0; j<V; ++j) {
         for (int k = 0; k<sizePatch; ++k) {
            mean[j]+=_X[i*_m+sizePatch*j+k];
         }
      }
   }
   for (int i = 0; i<V; ++i)
      mean[i] /= _n*cblas_asum(sizePatch,mask._X+i*sizePatch,1);
   for (int i = 0; i<_n; ++i) {
      for (int j = 0; j<V; ++j) {
         for (int k = 0; k<sizePatch; ++k) {
            if (mask[sizePatch*j+k])
               _X[i*_m+sizePatch*j+k]-=mean[j];
         }
      }
   }
};


template <typename T> 
inline void Matrix<T>::unwhiten(Vector<T>& mean, const bool pattern) {
   if (pattern) {
      const int n =static_cast<int>(sqrt(static_cast<T>(_m)));
      for (int i = 0; i<_n; ++i) {
         int offsetx=0;
         for (int j = 0; j<n; ++j) {
            offsetx= (offsetx+1) % 2;
            int offsety=0;
            for (int k = 0; k<n; ++k) {
               offsety= (offsety+1) % 2;
               _X[i*_m+j*n+k]+=mean[2*offsetx+offsety];
            }
         }
      }
   } else {
      const int V = mean.n();
      const int sizePatch=_m/V;
      for (int i = 0; i<_n; ++i) {
         for (int j = 0; j<V; ++j) {
            for (int k = 0; k<sizePatch; ++k) {
               _X[i*_m+sizePatch*j+k]+=mean[j];
            }
         }
      }
   }
};


/// perform a rank one approximation uv' using the power method
/// u0 is an initial guess for u (can be empty).
template <typename T> 
inline void Matrix<T>::svdRankOne(const Vector<T>& u0,
      Vector<T>& u, Vector<T>& v) const {
   int i;
   const int max_iter=MAX(_m,MAX(_n,200));
   const T eps=1e-10;
   u.resize(_m);
   v.resize(_n);
   T norm=u0.nrm2();
   Vector<T> up(u0);
   if (norm < EPSILON) up.setAleat();
   up.normalize();
   multTrans(up,v);
   for (i = 0; i<max_iter; ++i) {
      mult(v,u);
      norm=u.nrm2();
      u.scal(1.0/norm);
      multTrans(u,v);
      T theta=u.dot(up);
      if (i > 10 && (1 - fabs(theta)) < eps) break;
      up.copy(u);
   }
};

template <typename T> 
inline void Matrix<T>::singularValues(Vector<T>& u) const {
   u.resize(MIN(_m,_n));
   if (_m > 10*_n) {
      Matrix<T> XtX;
      this->XtX(XtX);
      syev<T>(no,lower,_n,XtX.rawX(),_n,u.rawX());
      u.thrsPos();
      u.Sqrt();
   } else if (_n > 10*_m) { 
      Matrix<T> XXt;
      this->XXt(XXt);
      syev<T>(no,lower,_m,XXt.rawX(),_m,u.rawX());
      u.thrsPos();
      u.Sqrt();
   } else {
      T* vu, *vv;
      Matrix<T> copyX;
      copyX.copy(*this);
      gesvd<T>(no,no,_m,_n,copyX._X,_m,u.rawX(),vu,1,vv,1);
   }
};

template <typename T> 
inline void Matrix<T>::svd(Matrix<T>& U, Vector<T>& S, Matrix<T>&V) const {
   const int num_eig=MIN(_m,_n);
   S.resize(num_eig);
   U.resize(_m,num_eig);
   V.resize(num_eig,_n);
   if (_m > 10*_n) {
      Matrix<T> Vt(_n,_n);
      this->XtX(Vt);
      syev<T>(allV,lower,_n,Vt.rawX(),_n,S.rawX());
      S.thrsPos();
      S.Sqrt();
      this->mult(Vt,U);
      Vt.transpose(V);
      Vector<T> inveigs;
      inveigs.copy(S);
      for (int i = 0; i<num_eig; ++i) 
         if (S[i] > 1e-10) {
            inveigs[i]=T(1.0)/S[i];
         } else {
            inveigs[i]=T(1.0);
         }
      U.multDiagRight(inveigs);
   } else if (_n > 10*_m) {
      this->XXt(U);
      syev<T>(allV,lower,_m,U.rawX(),_m,S.rawX());
      S.thrsPos();
      S.Sqrt();
      U.mult(*this,V,true,false);
      Vector<T> inveigs;
      inveigs.copy(S);
      for (int i = 0; i<num_eig; ++i) 
         if (S[i] > 1e-10) {
            inveigs[i]=T(1.0)/S[i];
         } else {
            inveigs[i]=T(1.0);
         }
      V.multDiagLeft(inveigs);
   } else {
      Matrix<T> copyX;
      copyX.copy(*this);
      gesvd<T>(reduced,reduced,_m,_n,copyX._X,_m,S.rawX(),U.rawX(),_m,V.rawX(),num_eig);
   }
};

/// find the eigenvector corresponding to the largest eigenvalue
/// when the current matrix is symmetric. u0 is the initial guess.
/// using two iterations of the power method
template <typename T> 
inline void Matrix<T>::eigLargestSymApprox(
      const Vector<T>& u0, Vector<T>& u) const {
   int i,j;
   const int max_iter=100;
   const T eps=10e-6;
   u.copy(u0);
   T norm = u.nrm2();
   T theta;
   u.scal(1.0/norm);
   Vector<T> up(u);
   Vector<T> uor(u);
   T lambda=T();

   for (j = 0; j<2;++j) {
      up.copy(u);
      for (i = 0; i<max_iter; ++i) {
         mult(up,u);
         norm = u.nrm2();
         u.scal(1.0/norm);
         theta=u.dot(up);
         if ((1 - fabs(theta)) < eps) break;
         up.copy(u);
      }
      lambda+=theta*norm;
      if isnan(lambda) {
         std::cerr << "eigLargestSymApprox failed" << std::endl;
         exit(1);
      }
      if (j == 1 && lambda < eps) {
         u.copy(uor);
         break;
      }
      if (theta >= 0) break;
      u.copy(uor);
      for (i = 0; i<_m; ++i) _X[i*_m+i]-=lambda;
   }
};

/// find the eigenvector corresponding to the eivenvalue with the 
/// largest magnitude when the current matrix is symmetric,
/// using the power method. It 
/// returns the eigenvalue. u0 is an initial guess for the 
/// eigenvector.
template <typename T> 
inline T Matrix<T>::eigLargestMagnSym(
      const Vector<T>& u0, Vector<T>& u) const {
   const int max_iter=1000;
   const T eps=10e-6;
   u.copy(u0);
   T norm = u.nrm2();
   u.scal(1.0/norm);
   Vector<T> up(u);
   T lambda=T();

   for (int i = 0; i<max_iter; ++i) {
      mult(u,up);
      u.copy(up);
      norm=u.nrm2();
      if (norm > 0) u.scal(1.0/norm);
      if (norm == 0 || fabs(norm-lambda)/norm < eps) break;
      lambda=norm;
   }
   return norm;
};

/// returns the value of the eigenvalue with the largest magnitude
/// using the power iteration.
template <typename T> 
inline T Matrix<T>::eigLargestMagnSym() const {
   const int max_iter=1000;
   const T eps=10e-6;
   Vector<T> u(_m);
   u.setAleat();
   T norm = u.nrm2();
   u.scal(1.0/norm);
   Vector<T> up(u);
   T lambda=T();
   for (int i = 0; i<max_iter; ++i) {
      mult(u,up);
      u.copy(up);
      norm=u.nrm2();
      if (fabs(norm-lambda) < eps) break;
      lambda=norm;
      u.scal(1.0/norm);
   }
   return norm;
};

/// inverse the matrix when it is symmetric
template <typename T> 
inline void Matrix<T>::invSym() {
 //  int lwork=2*_n;
 //  T* work;
//#ifdef USE_BLAS_LIB
//   INTT* ipiv;
//#else
//   int* ipiv;
//#endif
//#pragma omp critical
//   {
//      work= new T[lwork];
//#ifdef USE_BLAS_LIB
///      ipiv= new INTT[lwork];
//#else
//      ipiv= new int[lwork];
//#endif
//   }
//   sytrf<T>(upper,_n,_X,_n,ipiv,work,lwork);
//   sytri<T>(upper,_n,_X,_n,ipiv,work);
//   sytrf<T>(upper,_n,_X,_n);
   sytri<T>(upper,_n,_X,_n);
   this->fillSymmetric();
//   delete[](work);
//   delete[](ipiv);
};

/// perform b = alpha*A'x + beta*b
template <typename T> 
inline void Matrix<T>::multTrans(const Vector<T>& x, 
      Vector<T>& b, const T a, const T c) const {
   b.resize(_n);
   //   assert(x._n == _m && b._n == _n);
   cblas_gemv<T>(CblasColMajor,CblasTrans,_m,_n,a,_X,_m,x._X,1,c,b._X,1);
};


template <typename T> 
inline void Matrix<T>::multTrans(
      const Vector<T>& x, Vector<T>& b, const Vector<bool>& active) const {
   b.setZeros();
   Vector<T> col;
   bool* pr_active=active.rawX();
   for (int i = 0; i<_n; ++i) {
      if (pr_active[i]) {
         this->refCol(i,col);
         b._X[i]=col.dot(x);
      }
   }
};

/// perform b = alpha*A*x+beta*b
template <typename T> 
inline void Matrix<T>::mult(const Vector<T>& x, 
      Vector<T>& b, const T a, const T c) const {
   //  assert(x._n == _n && b._n == _m);
   b.resize(_m);
   cblas_gemv<T>(CblasColMajor,CblasNoTrans,_m,_n,a,_X,_m,x._X,1,c,b._X,1);
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename T> 
inline void Matrix<T>::mult(const Matrix<T>& B, 
      Matrix<T>& C, const bool transA, const bool transB,
      const T a, const T b) const {
   CBLAS_TRANSPOSE trA,trB;
   int m,k,n;
   if (transA) {
      trA = CblasTrans;
      m = _n;
      k = _m;
   } else {
      trA= CblasNoTrans;
      m = _m;
      k = _n;
   }
   if (transB) {
      trB = CblasTrans;
      n = B._m; 
      //  assert(B._n == k);
   } else {
      trB = CblasNoTrans;
      n = B._n; 
      // assert(B._m == k);
   }
   C.resize(m,n);
   cblas_gemm<T>(CblasColMajor,trA,trB,m,n,k,a,_X,_m,B._X,B._m,
         b,C._X,C._m);
};

/// perform C = a*B*A + b*C, possibly transposing A or B.
template <typename T>
inline void Matrix<T>::multSwitch(const Matrix<T>& B, Matrix<T>& C, 
      const bool transA, const bool transB,
      const T a, const T b) const {
   B.mult(*this,C,transB,transA,a,b);
};


/// C = A .* B, elementwise multiplication
template <typename T> 
inline void Matrix<T>::mult_elementWise(
      const Matrix<T>& B, Matrix<T>& C) const {
   assert(_n == B._n && _m == B._m);
   C.resize(_m,_n);
   vMul<T>(_n*_m,_X,B._X,C._X);
};

/// C = A .* B, elementwise multiplication
template <typename T> 
inline void Matrix<T>::div_elementWise(
      const Matrix<T>& B, Matrix<T>& C) const {
   assert(_n == B._n && _m == B._m);
   C.resize(_m,_n);
   vDiv<T>(_n*_m,_X,B._X,C._X);
};


/// XtX = A'*A
template <typename T> 
inline void Matrix<T>::XtX(Matrix<T>& xtx) const {
   xtx.resize(_n,_n);
   cblas_syrk<T>(CblasColMajor,CblasUpper,CblasTrans,_n,_m,T(1.0),
         _X,_m,T(),xtx._X,_n);
   xtx.fillSymmetric();
};

/// XXt = A*At
template <typename T> 
inline void Matrix<T>::XXt(Matrix<T>& xxt) const {
   xxt.resize(_m,_m);
   cblas_syrk<T>(CblasColMajor,CblasUpper,CblasNoTrans,_m,_n,T(1.0),
         _X,_m,T(),xxt._X,_m);
   xxt.fillSymmetric();
};

/// XXt = A*A' where A is an upper triangular matrix
template <typename T> 
inline void Matrix<T>::upperTriXXt(Matrix<T>& XXt, const int L) const {
   XXt.resize(L,L);
   for (int i = 0; i<L; ++i) {
      cblas_syr<T>(CblasColMajor,CblasUpper,i+1,T(1.0),_X+i*_m,1,XXt._X,L);
   }
   XXt.fillSymmetric();
}





/// each element of the matrix is replaced by its exponential
template <typename T> 
inline void Matrix<T>::exp() {
   vExp<T>(_n*_m,_X,_X);
};
template <typename T> 
inline void Matrix<T>::Sqrt() {
   vSqrt<T>(_n*_m,_X,_X);
};
template <typename T> 
inline void Matrix<T>::sqr() {
   vSqr<T>(_n*_m,_X,_X);
};

template <typename T> 
inline void Matrix<T>::Invsqrt() {
   vInvSqrt<T>(_n*_m,_X,_X);
};


/// add alpha*mat to the current matrix
template <typename T> 
inline T Matrix<T>::dot(const Matrix<T>& mat) const {
   assert(mat._m == _m && mat._n == _n);
   return cblas_dot<T>(_n*_m,mat._X,1,_X,1);
};



/// substract the matrix mat to the current matrix
template <typename T> 
inline void Matrix<T>::sub(const Matrix<T>& mat) {
   vSub<T>(_n*_m,_X,mat._X,_X);
};

/// compute the sum of the magnitude of the matrix values
template <typename T> 
inline T Matrix<T>::asum() const {
   return cblas_asum<T>(_n*_m,_X,1);
};

/// returns the trace of the matrix
template <typename T> 
inline T Matrix<T>::trace() const {
   T sum=T();
   int m = MIN(_n,_m);
   for (int i = 0; i<m; ++i) 
      sum += _X[i*_m+i];
   return sum;
};

/// return ||A||_F
template <typename T> 
inline T Matrix<T>::normF() const {
   return cblas_nrm2<T>(_n*_m,_X,1);
};

template <typename T> 
inline T Matrix<T>::mean() const {
   Vector<T> vec;
   this->toVect(vec);
   return vec.mean();
};

/// return ||A||_F^2
template <typename T> 
inline T Matrix<T>::normFsq() const {
   return cblas_dot<T>(_n*_m,_X,1,_X,1);
};

/// return ||At||_{inf,2}
template <typename T> 
inline T Matrix<T>::norm_inf_2_col() const {
   Vector<T> col;
   T max = -1.0;
   for (int i = 0; i<_n; ++i) {
      refCol(i,col);
      T norm_col = col.nrm2();
      if (norm_col > max) 
         max = norm_col;
   }
   return max;
};

/// return ||At||_{1,2}
template <typename T> 
inline T Matrix<T>::norm_1_2_col() const {
   Vector<T> col;
   T sum = 0.0;
   for (int i = 0; i<_n; ++i) {
      refCol(i,col);
      sum += col.nrm2();
   }
   return sum;
};

/// returns the l2 norms of the columns
template <typename T> 
inline void Matrix<T>::norm_2_rows(
      Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (int i = 0; i<_n; ++i) 
      for (int j = 0; j<_m; ++j) 
         norms[j] += _X[i*_m+j]*_X[i*_m+j];
   for (int j = 0; j<_m; ++j) 
      norms[j]=sqrt(norms[j]);
};

/// returns the l2 norms of the columns
template <typename T> 
inline void Matrix<T>::norm_2sq_rows(
      Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (int i = 0; i<_n; ++i) 
      for (int j = 0; j<_m; ++j) 
         norms[j] += _X[i*_m+j]*_X[i*_m+j];
};


/// returns the l2 norms of the columns
template <typename T> 
inline void Matrix<T>::norm_2_cols(
      Vector<T>& norms) const {
   norms.resize(_n);
   const Vector<T>* col;
   for (int i = 0; i<_n; ++i) {
      col=this->refCol(i);
      norms[i] = col->nrm2();
      delete col;
   }
};


/// returns the linf norms of the columns
template <typename T> 
inline void Matrix<T>::norm_inf_cols(Vector<T>& norms) const {
   norms.resize(_n);
   Vector<T> col;
   for (int i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.fmaxval();
   }
};

/// returns the linf norms of the columns
template <typename T> 
inline void Matrix<T>::norm_inf_rows(Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (int i = 0; i<_n; ++i) 
      for (int j = 0; j<_m; ++j) 
         norms[j] = MAX(abs<T>(_X[i*_m+j]),norms[j]);
};

/// returns the linf norms of the columns
template <typename T>
inline void Matrix<T>::norm_l1_rows(Vector<T>& norms) const {
    norms.resize(_m);
    for (int j = 0; j<_m; ++j) 
        norms[j] = cblas_asum<T>(_n,_X+j,_m);
};
/// returns the linf norms of the columns
template <typename T>
inline void Matrix<T>::norm_l1_cols(Vector<T>& norms) const {
   norms.resize(_n);
   const Vector<T> col(_m);
   for (int i = 0; i<_n; ++i) {
       refCol(i,col);
       norms[i] = col.asum();
   }
};

/// returns the l2 norms of the columns
template <typename T>
inline void Matrix<T>::norm_2sq_cols(Vector<T>& norms) const {
    norms.resize(_n);
    for (int j=0; j<_n; ++j) {
        const Vector<T> col(_X+j*_m,_m);
        norms[j] = col.nrm2sq();
   }
}

template <typename T> 
inline void Matrix<T>::sum_rows(Vector<T>& sum) const {
    sum.resize(_m);
    sum.setZeros();
    for (int j=0; j<_n; ++j) {
        const Vector<T> col(_X+j*_m,_m);
        sum.add(col);
    }
}

template <typename T> 
inline void Matrix<T>::sum_cols(Vector<T>& sum) const {
   sum.resize(_n);
   const Vector<T> col;
   for (int i = 0; i<_n; ++i) {
      refCol(i,col);
      sum[i] = col.sum();
   }
}
/// Compute the mean of the columns
template <typename T> 
inline void Matrix<T>::meanCol(Vector<T>& mean) const {
   Vector<T> ones(_n);
   ones.set(T(1.0/_n));
   this->mult(ones,mean,1.0,0.0);
}
/// Compute the mean of the rows
template <typename T> 
inline void Matrix<T>::meanRow(Vector<T>& mean) const {
   Vector<T> ones(_m);
   ones.set(T(1.0/_m));
   this->multTrans(ones,mean,1.0,0.0);
}
/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::softThrshold(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.softThrshold(nu);
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::hardThrshold(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.hardThrshold(nu);
};


/// perform thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::thrsmax(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.thrsmax(nu);
};

/// perform thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::thrsmin(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.thrsmin(nu);
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::inv_elem() {
   Vector<T> vec;
   toVect(vec);
   vec.inv();
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::blockThrshold(const T nu,
      const int sizeGroup) {
   for (int i = 0; i<_n; ++i) {
      int j;
      for (j = 0; j<_m-sizeGroup+1; j+=sizeGroup) {
         T nrm=0;
         for (int k = 0; k<sizeGroup; ++k)
            nrm += _X[i*_m +j+k]*_X[i*_m +j+k];
         nrm=sqrt(nrm);
         if (nrm < nu) {
            for (int k = 0; k<sizeGroup; ++k)
               _X[i*_m +j+k]=0;
         } else {
            T scal = (nrm-nu)/nrm;
            for (int k = 0; k<sizeGroup; ++k)
               _X[i*_m +j+k]*=scal;
         }
      }
      j -= sizeGroup;
      for ( ; j<_m; ++j)
         _X[j]=softThrs<T>(_X[j],nu);
   }
}

template <typename T> 
inline void Matrix<T>::sparseProject(Matrix<T>& Y, 
      const T thrs,   const int mode, const T lambda1,
      const T lambda2, const T lambda3, const bool pos,
      const int numThreads) {

   int NUM_THREADS=init_omp(numThreads);
   Vector<T>* XXT= new Vector<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      XXT[i].resize(_m);
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< _n; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      this->refCol(i,Xi);
      Vector<T> Yi;
      Y.refCol(i,Yi);
      Vector<T>& XX = XXT[numT];
      XX.copy(Xi);
      XX.sparseProject(Yi,thrs,mode,lambda1,lambda2,lambda3,pos);
   }
   delete[](XXT);
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::thrsPos() {
   Vector<T> vec;
   toVect(vec);
   vec.thrsPos();
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> 
inline void Matrix<T>::thrshold(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.thrshold(nu);
};



/// perform A <- A + alpha*vec1*vec2'
template <typename T> 
inline void Matrix<T>::rank1Update(
      const Vector<T>& vec1, const Vector<T>& vec2, const T alpha) {
   cblas_ger<T>(CblasColMajor,_m,_n,alpha,vec1._X,1,vec2._X,1,_X,_m);
};


/// compute x, such that b = Ax, 
template <typename T> 
inline void Matrix<T>::conjugateGradient(
      const Vector<T>& b, Vector<T>& x, const T tol, const int itermax) const {
   Vector<T> R,P,AP;
   R.copy(b);
   this->mult(x,R,T(-1.0),T(1.0));
   P.copy(R);
   int k = 0;
   T normR = R.nrm2sq();
   T alpha;
   while (normR > tol && k < itermax) {
      this->mult(P,AP);
      alpha = normR/P.dot(AP);
      x.add(P,alpha);
      R.add(AP,-alpha);
      T tmp = R.nrm2sq();
      P.scal(tmp/normR);
      normR = tmp;
      P.add(R,T(1.0));
      ++k;
   };
};

template <typename T> 
inline void Matrix<T>::drop(char* fileName) const {
   std::ofstream f;
   f.precision(12);
   f.flags(std::ios_base::scientific);
   f.open(fileName, ofstream::trunc);
   std::cout << "Matrix written in " << fileName << std::endl;
   for (int i = 0; i<_n; ++i) {
      for (int j = 0; j<_m; ++j) 
         f << _X[i*_m+j] << " ";
      f << std::endl;
   }
   f.close();
};

/// compute a Nadaraya Watson estimator
template <typename T> 
inline void Matrix<T>::NadarayaWatson(
      const Vector<int>& ind, const T sigma) {
   if (ind.n() != _n) return;

   init_omp(MAX_THREADS);

   const int Ngroups=ind.maxval();
   int i;
#pragma omp parallel for private(i)
   for (i = 1; i<=Ngroups; ++i) {
      Vector<int> indicesGroup(_n);
      int count = 0;
      for (int j = 0; j<_n; ++j)
         if (ind[j] == i) indicesGroup[count++]=j;
      Matrix<T> Xm(_m,count);
      Vector<T> col, col2;
      for (int j= 0; j<count; ++j) {
         this->refCol(indicesGroup[j],col);
         Xm.refCol(j,col2);
         col2.copy(col);
      }
      Vector<T> norms;
      Xm.norm_2sq_cols(norms);
      Matrix<T> weights;
      Xm.XtX(weights);
      weights.scal(T(-2.0));
      Vector<T> ones(Xm.n());
      ones.set(T(1.0));
      weights.rank1Update(ones,norms);
      weights.rank1Update(norms,ones);
      weights.scal(-sigma);
      weights.exp();
      Vector<T> den;
      weights.mult(ones,den);
      den.inv();
      weights.multDiagRight(den);
      Matrix<T> num;
      Xm.mult(weights,num);
      for (int j= 0; j<count; ++j) {
         this->refCol(indicesGroup[j],col);
         num.refCol(j,col2);
         col.copy(col2);
      }
   }
};


/// make a reference of the matrix to a vector vec 
template <typename T> 
inline void Matrix<T>::toVect(
      Vector<T>& vec) const {
   vec.clear();
   vec._externAlloc=true;
   vec._n=_n*_m;
   vec._X=_X;
};

/// merge two dictionaries
template <typename T> 
inline void Matrix<T>::merge(const Matrix<T>& B,
      Matrix<T>& C) const {
   const int K =_n; 
   Matrix<T> G;
   this->mult(B,G,true,false);
   std::list<int> list;
   for (int i = 0; i<G.n(); ++i) {
      Vector<T> g;
      G.refCol(i,g);
      T fmax=g.fmaxval();
      if (fmax < 0.995) list.push_back(i);
   }
   C.resize(_m,K+list.size());

   for (int i = 0; i<K; ++i) {
      Vector<T> d, d2;
      C.refCol(i,d);
      this->refCol(i,d2);
      d.copy(d2);
   }
   int count=0;
   for (std::list<int>::const_iterator it = list.begin();
         it != list.end(); ++it) {
      Vector<T> d, d2;
      C.refCol(K+count,d);
      B.refCol(*it,d2);
      d.copy(d2);
      ++count;
   }
};

/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS VECTOR ************************************/
/*****************************************************************************/

///////////////////////////////////////////////////////////////////////////////
////   CONSTRUCTORS, DESTRUCTOR                                            ////
///////////////////////////////////////////////////////////////////////////////

///// Empty constructor /////////////////////////////////////////////
template <typename T>
Vector<T>::Vector(): _externAlloc(true), _X(NULL), _n(0) {
}
///// Empty constructor, allocates memory ///////////////////////////
template <typename T>
Vector<T>::Vector(int n): _externAlloc(false), _n(n) {
#pragma omp critical
    {
        _X=new T[_n];
    }
}
///// Constructor with data /////////////////////////////////////////
template <typename T>
Vector<T>::Vector(T* X, int n): _externAlloc(true), _X(X),  _n(n) {  
}
///// Copy constructor, allocates memory ////////////////////////////
template <typename T>
Vector<T>::Vector(const Vector<T>& vec): _externAlloc(false), _n(vec._n) {
#pragma omp critical
    {
        _X=new T[_n];
    }
    cblas_copy<T>(_n,vec._X,1,_X,1);
}
///// Ref  constructor //////////////////////////////////////////////
template <typename T>
Vector<T>::Vector(const Vector<T>* vec): 
        _externAlloc(true), _n(vec->_n) {
    _X = vec->_X;
}

///////////////////////////////////////////////////////////////////////////////
////   DISPLAY FUNCTIONS                                                   ////
///////////////////////////////////////////////////////////////////////////////

///// Prints out vector /////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::print(const char* name) const {
    cout << name << ", " << _n << endl;
    for (int i = 0; i<_n; ++i) cout << _X[i] << " ";
    cout << endl;
};

///////////////////////////////////////////////////////////////////////////////
////   GETTING DIMENSIONS                                                  ////
///////////////////////////////////////////////////////////////////////////////

///// Returns number of elements ////////////////////////////////////
template <typename T> 
inline int Vector<T>::n() const { return _n; };

///////////////////////////////////////////////////////////////////////////////
////   SETTING VALUES, SIZES                                               ////
///////////////////////////////////////////////////////////////////////////////

///// Clear the vector //////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::clear() {
   if (!_externAlloc) delete[](_X);
   _n=0;
   _X=NULL;
   _externAlloc=true;
}
///// Resize the vector /////////////////////////////////////////////
template <typename T>
inline void Vector<T>::resize(const int n) {
    if (_n == n) return;
    if (!_externAlloc) delete[](_X); 
#pragma omp critical
    {
        _X=new T[n];
    }
    _n=n;
    _externAlloc=false;
}
///// Set all values to zero ////////////////////////////////////////
template <typename T> 
inline void Vector<T>::setZeros() {
   memset(_X,0,_n*sizeof(T));
}
///// Set each value of the vector to val ///////////////////////////
template <typename T> 
inline void Vector<T>::set(const T val) {
   for (int i = 0; i<_n; ++i) _X[i]=val;
}
///// Put a random permutation of size n (for integral vectors) /////
template <typename T>
inline void Vector<T>::randperm(int n) { cerr
    << "WARNING: no permutation generated!" << endl; }
template <>
inline void Vector<int>::randperm(int n) {
   resize(n);
   Vector<int> table(n);
   for (int i = 0; i<n; ++i)
      table[i]=i;
   int size=n;
   for (int i = 0; i<n; ++i) {
      const int ind=random() % size;
      _X[i]=table[ind];
      table[ind]=table[size-1];
      --size;
   }
};
///// Put random values in the vector (white Gaussian Noise) ////////
template <typename T> 
inline void Vector<T>::setAleat() {
   for (int i = 0; i<_n; ++i) _X[i]=normalDistrib<T>();
};

///// Generates logarithmically spaced values ///////////////////////
template <typename T>
inline void Vector<T>::logspace(const int n, const T a, const T b) {
    T first=log10(a);
    T last=log10(b);
    T step = (last-first)/(n-1);
    this->resize(n);
    _X[0]=first;
    for (int i = 1; i<_n; ++i)
        _X[i]=_X[i-1]+step;
    for (int i = 0; i<_n; ++i)
        _X[i]=pow(T(10.0),_X[i]);
}
///// Generates logarithmically spaced values ///////////////////////
template <>
inline void Vector<int>::logspace(const int n, const int a, const int b) {
    Vector<double> tmp(n);
    tmp.logspace(n,double(a),double(b));
    this->resize(n);
    _X[0]=a;
    _X[n-1]=b;
    for (int i = 1; i<_n-1; ++i) {
        int candidate=static_cast<int>(floor(tmp[i]));
        _X[i]= candidate > _X[i-1] ? candidate : _X[i-1]+1;
    }
}
///// Change the data of the vector /////////////////////////////////
template <typename T>
inline void Vector<T>::setPointer(T* X, const int n) {
   if (!_externAlloc) delete[](_X);
   _externAlloc=true;
   _X=X;
   _n=n;
};
///// Make a copy of vector x into current vector ///////////////////
template <typename T>
inline void Vector<T>::copy(const Vector<T>& x) {
   this->resize(x.n());
   cblas_copy<T>(_n,x._X,1,_X,1);
};
///// Make a copy of vector x into current vector (partially) ///////
template <typename T>
inline void Vector<T>::copy(const Vector<T>& x, const int n,
        const int off_x, const int off_this) {
    assert(n+off_this <= _n);
    assert(n+off_x <= x._n);
    cblas_copy<T>(n,x._X+off_x,1,_X+off_this,1);
};

///// Make a copy of data x into current vector /////////////////////
template <typename T>
inline void Vector<T>::copy(const T* x, int n) {
   assert(n <= _n);
   cblas_copy<T>(_n,const_cast<T*>(x),1,_X,1);
};

///////////////////////////////////////////////////////////////////////////////
////   ELEMENTWISE ACCESS                                                  ////
///////////////////////////////////////////////////////////////////////////////

///// Returns value at index i //////////////////////////////////////
template <typename T>
inline T Vector<T>::operator[](const int i) const { return _X[i]; };
///// Returns modifiable value at index /////////////////////////////
template <typename T>
inline T& Vector<T>::operator[](const int i) { return _X[i]; };
///// Returns modifiable reference to value at index ////////////////
template <typename T>
inline T* Vector<T>::rawX() { return _X; };
///// Returns non-modifiable reference to value at index ////////////
template <typename T>
inline const T* Vector<T>::rawX() const { return _X; };

///////////////////////////////////////////////////////////////////////////////
////   THRESHOLDING FUNCTIONS                                              ////
///////////////////////////////////////////////////////////////////////////////

///// Copy n first elements to x //////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::extractRaw(T* x, const int n) const {
   assert(n >= 0 && n<=_n);
   cblas_copy<T>(n,_X,1,x,1);
};

///////////////////////////////////////////////////////////////////////////////
////   THRESHOLDING FUNCTIONS                                              ////
///////////////////////////////////////////////////////////////////////////////

///// Soft low-down-thresholding ////////////////////////////////////
template <typename T>
inline void Vector<T>::softThrshold(const T nu) {
   for (int i = 0; i<_n; ++i) {
      if (_X[i] > nu) {
         _X[i] -= nu;
      } else if (_X[i] < -nu) {
         _X[i] += nu;
      } else {
         _X[i] = T();
      }
   }
};
///// Hard low-down-thresholding (threshold included) ////////////////
template <typename T>
inline void Vector<T>::hardThrshold(const T nu) {
   for (int i = 0; i<_n; ++i) {
      if (!(_X[i] > nu || _X[i] < -nu)) {
         _X[i] = 0;
      }
   }
};
///// Hard low-down-thresholding (threshold excluded) ///////////////
template <typename T>
inline void Vector<T>::thrshold(const T nu) {
   for (int i = 0; i<_n; ++i) 
      if (abs<T>(_X[i]) < nu) 
         _X[i]=0;
}

///// Hard low-down-thresholding (positive) /////////////////////////
template <typename T>
inline void Vector<T>::thrsPos() {
   for (int i = 0; i<_n; ++i) {
      if (_X[i] < 0) _X[i]=0;
   }
};

///// Hard low-up-thresholding (positive) ///////////////////////////
template <typename T>
inline void Vector<T>::thrsmax(const T nu) {
   for (int i = 0; i<_n; ++i) 
      _X[i]=MAX(_X[i],nu);
}
///// Hard low-up-thresholding (absolute) ///////////////////////////
template <typename T>
inline void Vector<T>::thrsabsmax(const T nu) {
   for (int i = 0; i<_n; ++i) 
       if (abs<T>(_X[i])<nu) {
           if (_X[i]>= 0) _X[i]=nu;
           else _X[i]=-nu;
       }
}
///// Hard high-down-thresholding (positive) ////////////////////////
template <typename T>
inline void Vector<T>::thrsmin(const T nu) {
   for (int i = 0; i<_n; ++i) 
      _X[i]=MIN(_X[i],nu);
}
///// Hard high-down-thresholding (absolute) ////////////////////////
template <typename T>
inline void Vector<T>::thrsabsmin(const T nu) {
   for (int i = 0; i<_n; ++i) 
      _X[i]=MAX(MIN(_X[i],nu),-nu);
}




///////////////////////////////////////////////////////////////////////////////
////   ANALYSIS FUNCTIONS (SCALAR)                                         ////
///////////////////////////////////////////////////////////////////////////////

///// Returns the index of the maximal value ////////////////////////
template <typename T>
inline int Vector<T>::max() const {
    int imax=0;
    T max=_X[0];
    for (int j = 1; j<_n; ++j) {
        T cur = _X[j];
        if (cur > max) {
            imax=j;
            max = cur;
        }
    }
    return imax;
};
///// Returns the index of the minimal value ////////////////////////
template <typename T>
inline int Vector<T>::min() const {
    int imin=0;
    T min=_X[0];
    for (int j = 1; j<_n; ++j) {
        T cur = _X[j];
        if (cur < min) {
            imin=j;
            min = cur;
        }
    }
    return imin;
};
///// Returns the index of the value with largest magnitude /////////
template <typename T> 
inline int Vector<T>::fmax() const {
   return cblas_iamax<T>(_n,_X,1);
};
///// Returns the index of the value with smallest magnitude ////////
template <typename T> 
inline int Vector<T>::fmin() const {
   return cblas_iamin<T>(_n,_X,1);
};
///// Returns the maximal value /////////////////////////////////////
template <typename T>
inline T Vector<T>::maxval() const {
    return _X[this->max()];
};
///// Returns the minimal value /////////////////////////////////////
template <typename T> 
inline T Vector<T>::minval() const {
    return _X[this->min()];
};
///// Returns the largest magnitude /////////////////////////////////
template <typename T>
inline T Vector<T>::fmaxval() const {
    return fabs(_X[this->fmax()]);
};
///// Returns the smallest magnitude ////////////////////////////////
template <typename T> 
inline T Vector<T>::fminval() const {
    return fabs(_X[this->fmin()]);
};
///// Returns ||A||_2 ///////////////////////////////////////////////
template <typename T> 
inline T Vector<T>::nrm2() const {
   return cblas_nrm2<T>(_n,_X,1);
};

///// Returns ||A||_2^2 /////////////////////////////////////////////
template <typename T>
inline T Vector<T>::nrm2sq() const {
   return cblas_dot<T>(_n,_X,1,_X,1);
};
///// Returns mean //////////////////////////////////////////////////
template <typename T>
inline T Vector<T>::mean() const { return this->sum()/_n; }
///// Returns std ///////////////////////////////////////////////////
template <typename T>
inline T Vector<T>::std() const {
   T E = this->mean();
   T std=0;
   for (int i = 0; i<_n; ++i) {
      T tmp=_X[i]-E;
      std += tmp*tmp;
   }
   std /= _n;
   return sqr_alt<T>(std);
}

///// Returns the number of non-zero entries ////////////////////////
template <typename T>
inline int Vector<T>::nnz() const {
    int sum=0;
    for (int i = 0; i<_n; ++i) 
        if (_X[i] != T()) ++sum;
    return sum;
};
/// computes the sum of the magnitudes of the vector
template <typename T>
inline T Vector<T>::asum() const {
   return cblas_asum<T>(_n,_X,1);
};
template <typename T>
inline T Vector<T>::lzero() const {
   int count=0;
   for (int i = 0; i<_n; ++i) 
      if (_X[i] != 0) ++count;
   return count;
};
template <typename T>
inline T Vector<T>::afused() const {
   T sum = 0;
   for (int i = 1; i<_n; ++i) {
      sum += abs<T>(_X[i]-_X[i-1]);
   }
   return sum;
}
/// returns the sum of the vector
template <typename T> 
inline T Vector<T>::sum() const {
   T sum=T();
   for (int i = 0; i<_n; ++i) sum +=_X[i]; 
   return sum;
};
///// Returns true if all entries are true (boolean) ////////////////
template <>
inline bool Vector<bool>::alltrue() const {
   for (int i = 0; i<_n; ++i) {
      if (!_X[i]) return false;
   }
   return true;
};
///// Returns true if all entries are false (boolean) ///////////////
template <>
inline bool Vector<bool>::allfalse() const {
   for (int i = 0; i<_n; ++i) {
      if (_X[i]) return false;
   }
   return true;
};

///////////////////////////////////////////////////////////////////////////////
////   ELEMENTWISE MODIFICATIONS                                           ////
///////////////////////////////////////////////////////////////////////////////

///// A <- -A ///////////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::neg() {
   for (int i = 0; i<_n; ++i) _X[i]=-_X[i];
};

/// Replace each value by its exponential ///////////////////////////
template <typename T> 
inline void Vector<T>::exp() {
   vExp<T>(_n,_X,_X);
};

/// ??? 
template <typename T> 
inline void Vector<T>::logexp() {
   for (int i = 0; i<_n; ++i) {
      if (_X[i] < -30) {
         _X[i]=0;
      } else if (_X[i] < 30) {
         _X[i]= log( T(1.0) + exp_alt<T>( _X[i] ) );
      }
   }
};
///// Scale the vector by a /////////////////////////////////////////
template <typename T> 
inline void Vector<T>::scal(const T a) {
   return cblas_scal<T>(_n,a,_X,1);
};
///// A <- sqr(A) ///////////////////////////////////////////////
template <typename T>
inline void Vector<T>::sqr() {
   vSqr<T>(_n,_X,_X);
}
///// A <- invsqrt(A) ///////////////////////////////////////////////
template <typename T>
inline void Vector<T>::Invsqrt() {
   vInvSqrt<T>(_n,_X,_X);
}
///// A <- sqrt(A) //////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::Sqrt() {
   vSqrt<T>(_n,_X,_X);
}
///// A <- 1./A /////////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::inv() {
   vInv<T>(_n,_X,_X);
};
///// A <- A + a*x //////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::add(const Vector<T>& x, const T a) {
    assert(_n == x._n);
    cblas_axpy<T>(_n,a,x._X,1,_X,1);
};
///// Adds subvector of x to subvector of current vector ////////////
template <typename T>
inline void Vector<T>::add_subvector(const Vector<T>& x, const int n, 
        const int off_x, const int off_this, const T a) {
    assert(n+off_this <= _n);
    assert(n+off_x <= x._n);
    cblas_axpy<T>(n,a,x._X+off_x,1,_X+off_this,1);
};

///// Adds a to each value in the vector ////////////////////////////
template <typename T> 
inline void Vector<T>::add(const T a, const int n) {
    int end = MIN(_n,_n-n);
    for (int i = MAX(0,n); i<end; ++i) _X[i]+=a; //TODO use blas
};
///// A <- A - x ////////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::sub(const Vector<T>& x) {
    assert(_n == x._n);
    vSub<T>(_n,_X,x._X,_X);
};
///// A <- A ./ x ///////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::div(const Vector<T>& x) {
    assert(_n == x._n);
    vDiv<T>(_n,_X,x._X,_X);
};
///// A <- x ./ y ///////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::div(const Vector<T>& x, const Vector<T>& y) {
    assert(_n == x._n);
    vDiv<T>(_n,x._X,y._X,_X);
};
///// A <- x .^ 2 ///////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::sqr(const Vector<T>& x) {
   this->resize(x._n);
   vSqr<T>(_n,x._X,_X);
}
///// A <- invsqrt(x) ///////////////////////////////////////////////
template <typename T>
inline void Vector<T>::Invsqrt(const Vector<T>& x) {
   this->resize(x._n);
   vInvSqrt<T>(_n,x._X,_X);
}
///// A <- sqrt(x) //////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::Sqrt(const Vector<T>& x) {
   this->resize(x._n);
   vSqrt<T>(_n,x._X,_X);
}
///// A <- 1./x /////////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::inv(const Vector<T>& x) {
   this->resize(x.n());
   vInv<T>(_n,x._X,_X);
};
///// A <- x .* y ///////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::mult(const Vector<T>& x, const Vector<T>& y) {
   this->resize(x.n());
   vMul<T>(_n,x._X,y._X,_X);
};


///////////////////////////////////////////////////////////////////////////////
////   ALGEBRAIC OPERATIONS                                                ////
///////////////////////////////////////////////////////////////////////////////

///// Returns  A'x ////////////////////////////////////////////////// 
template <typename T> 
inline T Vector<T>::dot(const Vector<T>& x) const {
   assert(_n == x._n);
   return cblas_dot<T>(_n,_X,1,x._X,1);
};
///// Normalize the vector //////////////////////////////////////////
template <typename T> 
inline void Vector<T>::normalize() {
   T norm=nrm2();
   if (norm > EPSILON) scal(1.0/norm);
};
///// Project vector onto l2 unit ball //////////////////////////////
template <typename T> 
inline void Vector<T>::normalize2() {
   T norm=nrm2();
   if (norm > T(1.0)) scal(1.0/norm);
};
///// Whiten ////////////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::whiten(
      Vector<T>& meanv, const bool pattern) {
   if (pattern) {
      const int n =static_cast<int>(sqrt(static_cast<T>(_n)));
      int count[4];
      for (int i = 0; i<4; ++i) count[i]=0;
      int offsetx=0;
      for (int j = 0; j<n; ++j) {
         offsetx= (offsetx+1) % 2;
         int offsety=0;
         for (int k = 0; k<n; ++k) {
            offsety= (offsety+1) % 2;
            meanv[2*offsetx+offsety]+=_X[j*n+k];
            count[2*offsetx+offsety]++;
         }
      }
      for (int i = 0; i<4; ++i)
         meanv[i] /= count[i];
      offsetx=0;
      for (int j = 0; j<n; ++j) {
         offsetx= (offsetx+1) % 2;
         int offsety=0;
         for (int k = 0; k<n; ++k) {
            offsety= (offsety+1) % 2;
            _X[j*n+k]-=meanv[2*offsetx+offsety];
         }
      }
   } else {
      const int V = meanv.n();
      const int sizePatch=_n/V;
      for (int j = 0; j<V; ++j) {
         T mean = 0;
         for (int k = 0; k<sizePatch; ++k) {
            mean+=_X[sizePatch*j+k];
         }
         mean /= sizePatch;
         for (int k = 0; k<sizePatch; ++k) {
            _X[sizePatch*j+k]-=mean;
         }
         meanv[j]=mean;
      }
   }
};
///// Whiten ////////////////////////////////////////////////////////
template <typename T>
inline void Vector<T>::whiten(
      Vector<T>& meanv, const Vector<T>& mask) {
   const int V = meanv.n();
   const int sizePatch=_n/V;
   for (int j = 0; j<V; ++j) {
      T mean = 0;
      for (int k = 0; k<sizePatch; ++k) {
         mean+=_X[sizePatch*j+k];
      }
      mean /= cblas_asum(sizePatch,mask._X+j*sizePatch,1);
      for (int k = 0; k<sizePatch; ++k) {
         if (mask[sizePatch*j+k])
            _X[sizePatch*j+k]-=mean;
      }
      meanv[j]=mean;
   }
};
///// Whiten ////////////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::whiten(const int V) {
   const int sizePatch=_n/V;
   for (int j = 0; j<V; ++j) {
      T mean = 0;
      for (int k = 0; k<sizePatch; ++k) {
         mean+=_X[sizePatch*j+k];
      }
      mean /= sizePatch;
      for (int k = 0; k<sizePatch; ++k) {
         _X[sizePatch*j+k]-=mean;
      }
   }
};
///// Whiten ////////////////////////////////////////////////////////
template <typename T>
inline T Vector<T>::KL(const Vector<T>& Y) const {
   T sum = 0;
   T* prY = Y.rawX();
   // Y.print("Y");
   // this->print("X");
   // stop();
   for (int i = 0; i<_n; ++i) {
      if (_X[i] > 1e-20) {
         if (prY[i] < 1e-60) {
            sum += 1e200;
         } else {
            sum += _X[i]*log_alt<T>(_X[i]/prY[i]);
         }
         //sum += _X[i]*log_alt<T>(_X[i]/(prY[i]+1e-100));
      }
   }
   sum += T(-1.0) + Y.sum();
   return sum;
};
///// Unwhiten //////////////////////////////////////////////////////
template <typename T> 
inline void Vector<T>::unwhiten(
      Vector<T>& meanv, const bool pattern) {
   if (pattern) {
      const int n =static_cast<int>(sqrt(static_cast<T>(_n)));
      int offsetx=0;
      for (int j = 0; j<n; ++j) {
         offsetx= (offsetx+1) % 2;
         int offsety=0;
         for (int k = 0; k<n; ++k) {
            offsety= (offsety+1) % 2;
            _X[j*n+k]+=meanv[2*offsetx+offsety];
         }
      }
   } else  {
      const int V = meanv.n();
      const int sizePatch=_n/V;
      for (int j = 0; j<V; ++j) {
         T mean = meanv[j];
         for (int k = 0; k<sizePatch; ++k) {
            _X[sizePatch*j+k]+=mean;
         }
      }
   }
};
///// Puts in signs, the sign of each point in the vector ///////////
template <typename T>
inline void Vector<T>::sign(Vector<T>& signs) const {
   T* prSign=signs.rawX();
   for (int i = 0; i<_n; ++i) {
      if (_X[i] == 0) {
         prSign[i]=0.0; 
      } else {
         prSign[i] = _X[i] > 0 ? 1.0 : -1.0;
      }
   }
};
///// ??? 
template <typename T> 
inline T Vector<T>::softmax(const int y) {
   this->add(-_X[y]);
   _X[y]=-INFINITY;
   T max=this->maxval();
   if (max > 30) {
      return max;
   } else if (max < -30) {
      return 0;
   } else {
      _X[y]=T(0.0);
      this->exp();
      return log(this->sum());
   }
};
/// projects the vector onto the l1 ball of radius thrs,
/// returns true if the returned vector is null
template <typename T> 
inline void Vector<T>::l1project(Vector<T>& out,
      const T thrs, const bool simplex) const {
   out.copy(*this);
   if (simplex) {
      out.thrsPos();
   } else {
      vAbs<T>(_n,out._X,out._X);
   }
   T norm1 = out.sum();
   if (norm1 <= thrs) {
      if (!simplex) out.copy(*this);
      return;
   }
   T* prU = out._X;
   int sizeU = _n;

   T sum = T();
   int sum_card = 0;

   while (sizeU > 0) {
      // put the pivot in prU[0]
      swap(prU[0],prU[sizeU/2]);
      T pivot = prU[0];
      int sizeG=1;
      T sumG=pivot;

      for (int i = 1; i<sizeU; ++i) {
         if (prU[i] >= pivot) {
            sumG += prU[i];
            swap(prU[sizeG++],prU[i]);
         }
      }

      if (sum + sumG - pivot*(sum_card + sizeG) <= thrs) {
         sum_card += sizeG;
         sum += sumG;
         prU +=sizeG;
         sizeU -= sizeG;
      } else {
         ++prU;
         sizeU = sizeG-1;
      }
   }
   T lambda = (sum-thrs)/sum_card;
   out.copy(*this);
   if (simplex) {
      out.thrsPos();
   }
   out.softThrshold(lambda);
};
/// projects the vector onto the l1 ball of radius thrs,
/// returns true if the returned vector is null
template <typename T> 
inline void Vector<T>::l1project_weighted(Vector<T>& out,
        const Vector<T>& weights, const T thrs,
        const bool residual) const {
   out.copy(*this);
   if (thrs==0) {
      out.setZeros();
      return;
   }
   vAbs<T>(_n,out._X,out._X);
   out.div(weights);
   Vector<int> keys(_n);
   for (int i = 0; i<_n; ++i) keys[i]=i;
   out.sort2(keys,false);
   T sum1=0;
   T sum2=0;
   T lambda=0;
   for (int i = 0; i<_n; ++i) {
      const T lambda_old=lambda;
      const T fact=weights[keys[i]]*weights[keys[i]];
      lambda=out[i];
      sum2 += fact;
      sum1 += fact*lambda;
      if (sum1 - lambda*sum2 >= thrs) {
         sum2-=fact;
         sum1-=fact*lambda;
         lambda=lambda_old;
         break;
      }
   }
   lambda=MAX(0,(sum1-thrs)/sum2);

   if (residual) {
      for (int i = 0; i<_n; ++i) {
         out._X[i]=_X[i] > 0 ? MIN(_X[i],lambda*weights[i]) :
             MAX(_X[i],-lambda*weights[i]);
      }
   } else {
      for (int i = 0; i<_n; ++i) {
         out._X[i]=_X[i] > 0 ? MAX(0,_X[i]-lambda*weights[i]) : 
             MIN(0,_X[i]+lambda*weights[i]);
      }
   }
};
template <typename T>
inline void Vector<T>::project_sft_binary(const Vector<T>& y) {
   T mean = this->mean();
   T thrs=mean;
   while (abs(mean) > EPSILON) {
      int n_seuils=0;
      for (int i = 0; i< _n; ++i) {
         _X[i] = _X[i]-thrs;
         const T val = y[i]*_X[i];
         if (val > 0) {
            ++n_seuils;
            _X[i]=0;
         } else if (val < -1.0) {
            ++n_seuils;
            _X[i] = -y[i];
         }
      }
      mean = this->mean();
      thrs= mean * _n/(_n-n_seuils);
   }
};
template <typename T>
inline void Vector<T>::project_sft(const Vector<int>& labels, const int clas) {
   T mean = this->mean();
   T thrs=mean;

   while (abs(mean) > EPSILON) {
      int n_seuils=0;
      for (int i = 0; i< _n; ++i) {
         _X[i] = _X[i]-thrs;
         if (labels[i]==clas) {
            if (_X[i] < -1.0) {
               _X[i]=-1.0;
               ++n_seuils;
            }
         } else {
            if (_X[i] < 0) {
               ++n_seuils;
               _X[i]=0;
            }
         }
      }
      mean = this->mean();
      thrs= mean * _n/(_n-n_seuils);
   }
};
template <typename T>
inline void Vector<T>::sparseProject(Vector<T>& out, const T thrs, 
        const int mode, const T lambda1,
      const T lambda2, const T lambda3, const bool pos) {
   if (mode == 1) {
      /// min_u ||b-u||_2^2 / ||u||_1 <= thrs
      this->l1project(out,thrs,pos);
   } else if (mode == 2) {
      /// min_u ||b-u||_2^2 / ||u||_2^2 + lambda1||u||_1 <= thrs
      if (lambda1 > 1e-10) {
         this->scal(lambda1);
         this->l1l2project(out,thrs,2.0/(lambda1*lambda1),pos);
         this->scal(T(1.0/lambda1));
         out.scal(T(1.0/lambda1));
      } else {
         out.copy(*this);
         out.normalize2();
         out.scal(sqrt(thrs));
      }
   } else if (mode == 3) {
      /// min_u ||b-u||_2^2 / ||u||_1 + (lambda1/2) ||u||_2^2 <= thrs
      this->l1l2project(out,thrs,lambda1,pos);
   } else if (mode == 4) {
      /// min_u 0.5||b-u||_2^2  + lambda1||u||_1 / ||u||_2^2 <= thrs
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.softThrshold(lambda1);
      T nrm=out.nrm2sq();
      if (nrm > thrs)
         out.scal(sqr_alt<T>(thrs/nrm));
   } else if (mode == 5) {
      /// min_u 0.5||b-u||_2^2  + lambda1||u||_1 +
      //lambda2 Fused(u) / ||u||_2^2 <= thrs
      //      this->fusedProject(out,lambda1,lambda2,100);
      //      T nrm=out.nrm2sq();
      //      if (nrm > thrs)
      //         out.scal(sqr_alt<T>(thrs/nrm));
      //  } else if (mode == 6) {
      /// min_u 0.5||b-u||_2^2  + lambda1||u||_1
      //+lambda2 Fused(u) +0.5lambda_3 ||u||_2^2 
      this->fusedProjectHomotopy(out,lambda1,lambda2,lambda3,true);
} else if (mode==6) {
   /// min_u ||b-u||_2^2  /  lambda1||u||_1 +lambda2 Fused(u) 
   //+ 0.5lambda3||u||_2^2 <= thrs
   this->fusedProjectHomotopy(out,lambda1/thrs,lambda2/thrs,lambda3/thrs,
           false);
} else {
   /// min_u ||b-u||_2^2 / (1-lambda1)*||u||_2^2 + lambda1||u||_1 <= thrs
   if (lambda1 < 1e-10) {
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.normalize2();
      out.scal(sqrt(thrs));
   } else if (lambda1 > 0.999999) {
      this->l1project(out,thrs,pos);
   } else {
      this->sparseProject(out,thrs/(1.0-lambda1),2,lambda1/(1-lambda1),0,0,
              pos);
   }
}
};

/// returns true if the returned vector is null
template <typename T>
inline void Vector<T>::l1l2projectb(Vector<T>& out, const T thrs, const T gamma, const bool pos,
      const int mode) {
   if (mode == 1) {
      /// min_u ||b-u||_2^2 / ||u||_2^2 + gamma ||u||_1 <= thrs
      this->scal(gamma);
      this->l1l2project(out,thrs,2.0/(gamma*gamma),pos);
      this->scal(T(1.0/gamma));
      out.scal(T(1.0/gamma));
   } else if (mode == 2) {
      /// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
      this->l1l2project(out,thrs,gamma,pos);
   } else if (mode == 3) {
      /// min_u 0.5||b-u||_2^2  + gamma||u||_1 / ||u||_2^2 <= thrs
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.softThrshold(gamma);
      T nrm=out.nrm2();
      if (nrm > thrs)
         out.scal(thrs/nrm);
   }
}

/// returns true if the returned vector is null
/// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
template <typename T>
inline void Vector<T>::l1l2project(Vector<T>& out, const T thrs, 
        const T gamma, const bool pos) const {
      if (gamma == 0) 
         return this->l1project(out,thrs,pos);
      out.copy(*this);
      if (pos) {
         out.thrsPos();
      } else {
         vAbs<T>(_n,out._X,out._X);
      }
      T norm = out.sum() + gamma*out.nrm2sq();
      if (norm <= thrs) {
         if (!pos) out.copy(*this);
         return;
      }

      /// BEGIN
      T* prU = out._X;
      int sizeU = _n;

      T sum = 0;
      int sum_card = 0;

      while (sizeU > 0) {
         // put the pivot in prU[0]
         swap(prU[0],prU[sizeU/2]);
         T pivot = prU[0];
         int sizeG=1;
         T sumG=pivot+0.5*gamma*pivot*pivot;

         for (int i = 1; i<sizeU; ++i) {
            if (prU[i] >= pivot) {
               sumG += prU[i]+0.5*gamma*prU[i]*prU[i];
               swap(prU[sizeG++],prU[i]);
            }
         }
         if (sum + sumG - pivot*(1+0.5*gamma*pivot)*(sum_card + sizeG) <
               thrs*(1+gamma*pivot)*(1+gamma*pivot)) {
            sum_card += sizeG;
            sum += sumG;
            prU +=sizeG;
            sizeU -= sizeG;
         } else {
            ++prU;
            sizeU = sizeG-1;
         }
      }
      T a = gamma*gamma*thrs+0.5*gamma*sum_card;
      T b = 2*gamma*thrs+sum_card;
      T c=thrs-sum;
      T delta = b*b-4*a*c;
      T lambda = (-b+sqrt(delta))/(2*a);

      out.copy(*this);
      if (pos) {
         out.thrsPos();
      }
      out.softThrshold(lambda);
      out.scal(T(1.0/(1+lambda*gamma)));
};

template <typename T>
static inline T fusedHomotopyAux(const bool& sign1,
      const bool& sign2,
      const bool& sign3,
      const T& c1,
      const T& c2) {
   if (sign1) {
      if (sign2) {
         return sign3 ? 0 : c2;
      } else {
         return sign3 ? -c2-c1 : -c1;
      }
   } else {
      if (sign2) {
         return sign3 ? c1 : c1+c2;
      } else {
         return sign3 ? -c2 : 0;
      }
   }
};

template <typename T>
inline void Vector<T>::fusedProjectHomotopy(Vector<T>& alpha, 
      const T lambda1,const T lambda2,const T lambda3,
      const bool penalty) {
   T* pr_DtR=_X;
   const int K = _n;
   alpha.setZeros();
   Vector<T> u(K); // regularization path for gamma
   Vector<T> Du(K); // regularization path for alpha
   Vector<T> DDu(K); // regularization path for alpha
   Vector<T> gamma(K); // auxiliary variable
   Vector<T> c(K); // auxiliary variables
   Vector<T> scores(K); // auxiliary variables
   gamma.setZeros();
   T* pr_gamma = gamma.rawX();
   T* pr_u = u.rawX();
   T* pr_Du = Du.rawX();
   T* pr_DDu = DDu.rawX();
   T* pr_c = c.rawX();
   T* pr_scores = scores.rawX();
   Vector<int> ind(K+1);
   Vector<bool> signs(K);
   ind.set(K);
   int* pr_ind = ind.rawX();
   bool* pr_signs = signs.rawX();

   /// Computation of DtR
   T sumBeta = this->sum();

   /// first element is selected, gamma and alpha are updated
   pr_gamma[0]=sumBeta/K;
   /// update alpha
   alpha.set(pr_gamma[0]);
   /// update DtR
   this->sub(alpha);
   for (int j = K-2; j>=0; --j) 
      pr_DtR[j] += pr_DtR[j+1];

   pr_DtR[0]=0;
   pr_ind[0]=0;
   pr_signs[0] = pr_DtR[0] > 0;
   pr_c[0]=T(1.0)/K;
   int currentInd=this->fmax();
   T currentLambda=abs<T>(pr_DtR[currentInd]);
   bool newAtom = true;

   /// Solve the Lasso using simplified LARS
   for (int i = 1; i<K; ++i) {
      /// exit if constraints are satisfied
      /// min_u ||b-u||_2^2  +  lambda1||u||_1 +lambda2 Fused(u) 
      //+ 0.5lambda3||u||_2^2 
      if (penalty && currentLambda <= lambda2) break;
      if (!penalty) {
         /// min_u ||b-u||_2^2  /  lambda1||u||_1 +lambda2 Fused(u)
         //+ 0.5lambda3||u||_2^2 <= 1.0
         scores.copy(alpha);
         scores.softThrshold(lambda1*currentLambda/lambda2);
         scores.scal(T(1.0/(1.0+lambda3*currentLambda/lambda2)));
         if (lambda1*scores.asum()+lambda2*scores.afused()+0.5*
               lambda3*scores.nrm2sq() >= T(1.0)) break;
      }

      /// Update pr_ind and pr_c
      if (newAtom) {
         int j;
         for (j = 1; j<i; ++j) 
            if (pr_ind[j] > currentInd) break;
         for (int k = i; k>j; --k) {
            pr_ind[k]=pr_ind[k-1];
            pr_c[k]=pr_c[k-1];
            pr_signs[k]=pr_signs[k-1];
         }
         pr_ind[j]=currentInd;
         pr_signs[j]=pr_DtR[currentInd] > 0;
         pr_c[j-1]=T(1.0)/(pr_ind[j]-pr_ind[j-1]);
         pr_c[j]=T(1.0)/(pr_ind[j+1]-pr_ind[j]);
      }

      // Compute u
      pr_u[0]= pr_signs[1] ? -pr_c[0] : pr_c[0];
      if (i == 1) {
         pr_u[1]=pr_signs[1] ? pr_c[0]+pr_c[1] : -pr_c[0]-pr_c[1];
      } else {
         pr_u[1]=pr_signs[1] ? pr_c[0]+pr_c[1] : -pr_c[0]-pr_c[1];
         pr_u[1]+=pr_signs[2] ? -pr_c[1] : pr_c[1];
         for (int j = 2; j<i; ++j) {
            pr_u[j]=2*fusedHomotopyAux<T>(pr_signs[j-1],
                  pr_signs[j],pr_signs[j+1], pr_c[j-1],pr_c[j]);
         }
         pr_u[i] = pr_signs[i-1] ? -pr_c[i-1] : pr_c[i-1];
         pr_u[i] += pr_signs[i] ? pr_c[i-1]+pr_c[i] : -pr_c[i-1]-pr_c[i];
      } 

      // Compute Du 
      pr_Du[0]=pr_u[0];
      for (int k = 1; k<pr_ind[1]; ++k)
         pr_Du[k]=pr_Du[0];
      for (int j = 1; j<=i; ++j) {
         pr_Du[pr_ind[j]]=pr_Du[pr_ind[j]-1]+pr_u[j];
         for (int k = pr_ind[j]+1; k<pr_ind[j+1]; ++k)
            pr_Du[k]=pr_Du[pr_ind[j]];
      }

      /// Compute DDu 
      DDu.copy(Du);
      for (int j = K-2; j>=0; --j) 
         pr_DDu[j] += pr_DDu[j+1];

      /// Check constraints
      T max_step1 = INFINITY;
      if (penalty) {
         max_step1 = currentLambda-lambda2;
      } 

      /// Check changes of sign
      T max_step2 = INFINITY;
      int step_out = -1;
      for (int j = 1; j<=i; ++j) {
         T ratio = -pr_gamma[pr_ind[j]]/pr_u[j];
         if (ratio > 0 && ratio <= max_step2) {
            max_step2=ratio;
            step_out=j;
         }
      }
      T max_step3 = INFINITY;
      /// Check new variables entering the active set
      for (int j = 1; j<K; ++j) {
         T sc1 = (currentLambda-pr_DtR[j])/(T(1.0)-pr_DDu[j]);
         T sc2 = (currentLambda+pr_DtR[j])/(T(1.0)+pr_DDu[j]);
         if (sc1 <= 1e-10) sc1=INFINITY;
         if (sc2 <= 1e-10) sc2=INFINITY;
         pr_scores[j]= MIN(sc1,sc2);
      }
      for (int j = 0; j<=i; ++j) {
         pr_scores[pr_ind[j]]=INFINITY;
      }
      currentInd = scores.fmin();
      max_step3 = pr_scores[currentInd];
      T step = MIN(max_step1,MIN(max_step3,max_step2));
      if (step == 0 || step == INFINITY) break; 

      /// Update gamma, alpha, DtR, currentLambda
      for (int j = 0; j<=i; ++j) {
         pr_gamma[pr_ind[j]]+=step*pr_u[j];
      }
      alpha.add(Du,step);
      this->add(DDu,-step);
      currentLambda -= step;
      if (step == max_step2) {
         /// Update signs,pr_ind, pr_c
         for (int k = step_out; k<=i; ++k) 
            pr_ind[k]=pr_ind[k+1];
         pr_ind[i]=K;
         for (int k = step_out; k<=i; ++k) 
            pr_signs[k]=pr_signs[k+1];
         pr_c[step_out-1]=T(1.0)/(pr_ind[step_out]-pr_ind[step_out-1]);
         pr_c[step_out]=T(1.0)/(pr_ind[step_out+1]-pr_ind[step_out]);
         i-=2;
         newAtom=false;
      } else {
         newAtom=true;
      }
   }

   if (penalty) {
      alpha.softThrshold(lambda1);
      alpha.scal(T(1.0/(1.0+lambda3)));
   } else {
      alpha.softThrshold(lambda1*currentLambda/lambda2);
      alpha.scal(T(1.0/(1.0+lambda3*currentLambda/lambda2)));
   }
};

template <typename T>
inline void Vector<T>::fusedProject(Vector<T>& alpha, const T lambda1, const T lambda2,
      const int itermax) {
   T* pr_alpha= alpha.rawX();
   T* pr_beta=_X;
   const int K = alpha.n();

   T total_alpha =alpha.sum();
   /// Modification of beta
   for (int i = K-2; i>=0; --i) 
      pr_beta[i]+=pr_beta[i+1];

   for (int i = 0; i<itermax; ++i) {
      T sum_alpha=0;
      T sum_diff = 0;
      /// Update first coordinate
      T gamma_old=pr_alpha[0];
      pr_alpha[0]=(K*gamma_old+pr_beta[0]-
            total_alpha)/K;
      T diff = pr_alpha[0]-gamma_old;
      sum_diff += diff;
      sum_alpha += pr_alpha[0];
      total_alpha +=K*diff;

      /// Update alpha_j
      for (int j = 1; j<K; ++j) {
         pr_alpha[j]+=sum_diff;
         T gamma_old=pr_alpha[j]-pr_alpha[j-1];
         T gamma_new=softThrs((K-j)*gamma_old+pr_beta[j]-
               (total_alpha-sum_alpha),lambda2)/(K-j);
         pr_alpha[j]=pr_alpha[j-1]+gamma_new;
         T diff = gamma_new-gamma_old;
         sum_diff += diff;
         sum_alpha+=pr_alpha[j];
         total_alpha +=(K-j)*diff;
      }
   }
   alpha.softThrshold(lambda1);

};

/// sort the vector
template <typename T>
inline void Vector<T>::sort(const bool mode) {
   if (mode) {
      lasrt<T>(incr,_n,_X);
   } else {
      lasrt<T>(decr,_n,_X);
   }
};


/// sort the vector
template <typename T>
inline void Vector<T>::sort(Vector<T>& out, const bool mode) const {
   out.copy(*this);
   out.sort(mode);
};

template <typename T>
inline void Vector<T>::sort2(Vector<int>& key, const bool mode) {
   quick_sort(key.rawX(),_X,0,_n-1,mode);
};


template <typename T>
inline void Vector<T>::sort2(Vector<T>& out, Vector<int>& key, const bool mode) const {
   out.copy(*this);
   out.sort2(key,mode);
}

template <typename T>
inline void Vector<T>::applyBayerPattern(const int offset) {
   int sizePatch=_n/3;
   int n = static_cast<int>(sqrt(static_cast<T>(sizePatch)));
   if (offset == 0) {
      // R
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 1 : 2;
         const int off = (i % 2) ? 0 : 1;
         for (int j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (int i = 0; i<n; ++i) {
         const int step = 2;
         const int off = (i % 2) ? 1 : 0;
         for (int j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 2 : 1;
         const int off = 0;
         for (int j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   } else if (offset == 1) {
      // R
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 2 : 1;
         const int off = (i % 2) ? 1 : 0;
         for (int j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (int i = 0; i<n; ++i) {
         const int step = 2;
         const int off = (i % 2) ? 0 : 1;
         for (int j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 1 : 2;
         const int off = 0;
         for (int j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   } else if (offset == 2) {
      // R
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 1 : 2;
         const int off = 0;
         for (int j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (int i = 0; i<n; ++i) {
         const int step = 2;
         const int off = (i % 2) ? 0 : 1;
         for (int j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 2 : 1;
         const int off = (i % 2) ? 1 : 0;
         for (int j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   } else if (offset == 3) {
      // R
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 2 : 1;
         const int off = 0;
         for (int j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (int i = 0; i<n; ++i) {
         const int step = 2;
         const int off = (i % 2) ? 1 : 0;
         for (int j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (int i = 0; i<n; ++i) {
         const int step = (i % 2) ? 1 : 2;
         const int off = (i % 2) ? 0 : 1;
         for (int j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   }
};

template <typename T>
inline void Vector<T>::copyMask(Vector<T>& out, Vector<bool>& mask) const {
   out.resize(_n);
   int pointer=0;
   for (int i = 0; i<_n; ++i) {
      if (mask[i])
         out[pointer++]=_X[i];
   }
   out.setn(pointer);
};

template <typename T>
inline void Matrix<T>::copyMask(Matrix<T>& out, Vector<bool>& mask) const {
   out.resize(_m,_n);
   int count=0;
   for (int i = 0; i<mask.n(); ++i)
      if (mask[i])
         ++count;
   out.setm(count);
   for (int i = 0; i<_n; ++i) {
      int pointer=0;
      for (int j = 0; j<_m; ++j) {
         if (mask[j]) {
            out[i*count+pointer]=_X[i*_m+j];
            ++pointer;
         }
      }
   }
};



/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS COVMATRIXSHIFT ****************************/
/*****************************************************************************/

template <typename T>
CovShMat<T>::CovShMat(): _X(NULL), _K(0), _n1(0), _n2(0), _allocated(false)
#ifdef FFT_CONV
    , _p(NULL), _ip(NULL), _fft_len(0), _buffer(NULL), _Dwork(NULL)
#endif
{ 
}

template <typename T>
CovShMat<T>::CovShMat(const int K, const int n): _K(K), _n1(n-1), _n2(2*n-1)  
#ifdef FFT_CONV
    , _p(NULL), _ip(NULL), _fft_len(0), _buffer(NULL), _Dwork(NULL)
#endif
{
    _allocate();
}

template <typename T>
CovShMat<T>::CovShMat(const Matrix<T>& D, const int n): _K(D.n())  
#ifdef FFT_CONV
    , _p(NULL), _ip(NULL), _fft_len(0), _buffer(NULL), _Dwork(NULL)
#endif
{
    _n1 = (n>0) ? n-1 : D.m()-1;
    _n2 = 2*_n1+1;
    _allocate();
    this->set(D);
}

template <typename T>
CovShMat<T>::~CovShMat() { if (_allocated) _deallocate(); }

// TODO: use fft
 
template <typename T>
inline void CovShMat<T>::setDims(const int K, const int n) {
    if (!_allocated){
        _K=K;
        _n1=n-1; 
        _n2=2*n-1;
        _allocate();
    }
}



template <typename T>
inline void CovShMat<T>::setFft(fftw_plan* p, fftw_plan* ip, const int fft_len, 
           double* buffer, Matrix<T>* Dwork) {
    _p=p;
    _ip=ip;
    _fft_len=fft_len;
    _buffer=buffer;
    _Dwork=Dwork;
}

template <typename T>
inline void CovShMat<T>::set(const Matrix<T>& D) {
    if (_K!=D.n() || _n1>D.m()-1)
        mexErrMsgTxt("CovShMat::set: Dimensions of dict inconsistent");
#ifdef FFT_CONV1
    if (_p==NULL || _ip==NULL || _fft_len==0 || _buffer==NULL || _Dwork==NULL) 
        mexErrMsgTxt("Fft plan has to be set to set up CovShMat");
    Vector<T> Dk1,Dk2;
    const Vector<T>* Dj;
    int count=-1;
    int k1, k2, l;
    for (k2=0; k2<_K; ++k2) {
        Dj=D.refCol(k2);
        Dj->extractRaw(_buffer,Dj->n());
        std::memset(_buffer+Dj->n(),0,(_fft_len-Dj->n())*sizeof(T));
        fftw_execute(*_p);
        _Dwork->refCol(k2,Dk2);
        Dk2.copy(_buffer,_fft_len);
        for (k1=0; k1<=k2; ++k1) {
            _Dwork->refCol(k1,Dk1);
            for (l=0;           l<_fft_len/2+1;   ++l) _buffer[l] =Dk1[l]*          Dk2[l];
            for (l=1;           l<(_fft_len+1)/2; ++l) _buffer[l]+=Dk1[_fft_len-l]* Dk2[_fft_len-l]; //'+' for cross-corr. 
            for (l=_fft_len/2+1;l<_fft_len;       ++l) _buffer[l] =Dk1[l]*          Dk2[_fft_len-l];
            for (l=_fft_len/2+1;l<_fft_len;       ++l) _buffer[l]-=Dk1[_fft_len-l]* Dk2[l]; //'-' for cross-corr. 
            fftw_execute(*_ip);
            for (l=-_n1;l<=_n1;++l) _X[++count]=_buffer[(_fft_len+l)%_fft_len]/_fft_len;
        }
    } 
#else
    T* pr_D = const_cast<T*>(D.rawX());
    int count=-1;
    int off1, off2, k1, k2, l;
    for (k2=0; k2<_K; ++k2) {
        off2 = k2*D.m();
        for (k1=0; k1<=k2; ++k1) {
            off1 = k1*D.m();
            for (l=-_n1; l<=_n1; ++l) {
                _X[++count]=cblas_dot<T>(D.m()-abs(l),pr_D+off1+MAX(0,l),1,
                        pr_D+off2+MAX(0,-l),1);
            }
        }
    }
#endif
}


template <typename T>
inline void CovShMat<T>::addDiag(const T diag) {
    int i,k1,offset = -1;
    for (k1=0; k1<_K; ++k1) {
        offset+=k1*_n2;
        for (i=0; i<_n2; ++i) _X[++offset]+=diag; //TODO: blas
    }
}

template <typename T>
inline T CovShMat<T>::operator()(const int k1, const int k2, 
        const int diff12) const {
    if (abs(diff12)>_n1) return 0.;
    return _X[_n1+abs(_getOffset(k1,k2)+diff12)];
}

template <typename T>
inline void CovShMat<T>::generateG(T* x) const {
    for (int k2=0; k2<_K; ++k2)
        for (int s2=0; s2<=_n1; ++s2) {
            this->generateGk(k2,s2,x);
            x+=(_n1+1)*_K;
        }
}

template <typename T>
inline void CovShMat<T>::generateGk(const int k2, const int s2, T* x) const {
    for (int k1=0; k1<_K; ++k1) {
        if (k1<=k2)
            cblas_copy<T>(_n1+1,this->rawX(k1,k2)+s2,-1,x+k1*(_n1+1),1);
        else
            cblas_copy<T>(_n1+1,this->rawX(k1,k2)+_n1-s2,1,x+k1*(_n1+1),1);
    }
}

#endif

template <typename T>
inline T* CovShMat<T>::rawX(const int k1, const int k2) const {
    return _X+abs(_getOffset(k1,k2));
}

template <typename T>
inline int CovShMat<T>::_getOffset(const int k1, const int k2) const {
    if (k1>k2) return -_getOffset(k2,k1);
    return ((k2*(k2+1))/2+k1)*_n2;
}

template <typename T>
inline void CovShMat<T>::_allocate() { 
    _X = new T[_K*(_K+1)/2*_n2];
    _allocated = true;
}

template <typename T>
inline void CovShMat<T>::_deallocate() { 
    delete[](_X);
    _allocated = false;
}

/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS BLSPMAT ***********************************/
/*****************************************************************************/
template <typename T>
BlSpMat<T>::BlSpMat(): _X(NULL),_first(NULL),_numel(NULL),_n2(0),_K(0),_L(0),
    _allocated(false) { }
template <typename T>
BlSpMat<T>::BlSpMat(const int n2,const int K,const int L):_n2(n2),_K(K),_L(L) {
    _X = new T[_n2*K*L];
    _first = new int[L];
    _numel = new int[L];
    _allocated = true;
}
template <typename T>
BlSpMat<T>::~BlSpMat() { 
    if (_allocated) {
        delete[] _X; 
        delete[] _first; 
        delete[] _numel; 
    }
}
template <typename T>
inline void BlSpMat<T>::setOffset(const int j, const int first, const int nel){
    _first[j]=first;
    _numel[j]=nel;
}

/// Constructor. Matrix D'*X is represented
/*****************************************************************************/
/********* IMPLEMENTATION OF CLASS PRODMATRIX ********************************/
/*****************************************************************************/

template <typename T>
ProdMatrix<T>::ProdMatrix()  {
   _DtX= NULL; 
   _X=NULL; 
   _D=NULL; 
   _high_memory=true;
   _n=0;
   _m=0;
   _addDiag=0;
};

/// Constructor. Matrix D'*X is represented
template <typename T>
ProdMatrix<T>::ProdMatrix(const Matrix<T>& D, const bool high_memory) {
   if (high_memory) _DtX = new Matrix<T>();
   this->setMatrices(D,high_memory);
};

/// Constructor. Matrix D'*X is represented
template <typename T>
ProdMatrix<T>::ProdMatrix(const Matrix<T>& D, const Matrix<T>& X,
        const bool high_memory) {
   if (high_memory) _DtX = new Matrix<T>();
   this->setMatrices(D,X,high_memory);
};

template <typename T> 
inline void ProdMatrix<T>::setMatrices(const Matrix<T>& D, const Matrix<T>& X,
      const bool high_memory)  {
   _high_memory=high_memory;
   _m = D.n(); 
   _n = X.n();
   if (high_memory) {
      D.mult(X,*_DtX,true,false);
   } else {
      _X=&X;
      _D=&D;
      _DtX=NULL;
   }
   _addDiag=0;
};

template <typename T> 
inline void ProdMatrix<T>::setMatrices( const Matrix<T>& D,
        const bool high_memory) {
   _high_memory=high_memory;
   _m = D.n(); 
   _n = D.n();
   if (high_memory) {
      D.XtX(*_DtX);
   } else {
      _X=&D;
      _D=&D;
      _DtX=NULL;
   } 
   _addDiag=0;
};

/// compute DtX(:,i)
template <typename T>
inline void ProdMatrix<T>::copyCol(const int i, Vector<T>& DtXi) const {
   if (_high_memory) {
      _DtX->copyCol(i,DtXi);
   } else {
      Vector<T> Xi;
      _X->refCol(i,Xi);
      _D->multTrans(Xi,DtXi);
      if (_addDiag && _m == _n) DtXi[i] += _addDiag;
   } 
};

/// compute DtX(:,i)
template <typename T> 
inline void ProdMatrix<T>::extract_rawCol(const int i,T* DtXi) const {
   if (_high_memory) {
      _DtX->extract_rawCol(i,DtXi);
   } else {
      Vector<T> Xi;
      Vector<T> vDtXi(DtXi,_m);
      _X->refCol(i,Xi);
      _D->multTrans(Xi,vDtXi);
      if (_addDiag && _m == _n) DtXi[i] += _addDiag;
   } 
};

template <typename T> 
inline void ProdMatrix<T>::add_rawCol(const int i,T* DtXi, const T a) const {
   if (_high_memory) {
      _DtX->add_rawCol(i,DtXi,a);
   } else {
      Vector<T> Xi;
      Vector<T> vDtXi(DtXi,_m);
      _X->refCol(i,Xi);
      _D->multTrans(Xi,vDtXi,a,T(1.0));
      if (_addDiag && _m == _n) DtXi[i] += a*_addDiag;
   } 
};

template <typename T>
void inline ProdMatrix<T>::addDiag(const T diag) {
   if (_m == _n) {
      if (_high_memory) {
         _DtX->addDiag(diag);
      } else {
         _addDiag=diag;
      }
   }
};

template <typename T>
inline T ProdMatrix<T>::operator[](const int index) const {
   if (_high_memory) {
      return (*_DtX)[index];
   } else {
      const int index2=index/this->_m;
      const int index1=index-this->_m*index2;
      Vector<T> col1, col2;
      _D->refCol(index1,col1);
      _X->refCol(index2,col2);
      return col1.dot(col2);
   }
};


template <typename T>
inline T ProdMatrix<T>::operator()(const int index1, const int index2) const {
   if (_high_memory) {
      return (*_DtX)(index1,index2);
   } else {
      Vector<T> col1, col2;
      _D->refCol(index1,col1);
      _X->refCol(index2,col2);
      return col1.dot(col2);
   }
};

template <typename T>
void inline ProdMatrix<T>::diag(Vector<T>& diag) const {
   if (_m == _n) {
      if (_high_memory) {
         _DtX->diag(diag);
      } else {
         Vector<T> col1, col2;
         for (int i = 0; i <_m; ++i) {
            _D->refCol(i,col1);
            _X->refCol(i,col2);
            diag[i] = col1.dot(col2);
         }
      }
   }
};


/*****************************************************************************/
/********* CLASS SUBMATRIX, DEF AND IMPLEMENTATION ***************************/
/*****************************************************************************/

template <typename T>
class SubMatrix : public AbstractMatrix<T> {

   public:
      SubMatrix(AbstractMatrix<T>& G, Vector<int>& indI, Vector<int>& indJ);

      void inline convertIndicesI(Vector<int>& ind) const;
      void inline convertIndicesJ(Vector<int>& ind) const;
      int inline n() const { return _indicesJ.n(); };
      int inline m() const { return _indicesI.n(); };
      void inline extract_rawCol(const int i, T* pr) const;
      /// compute DtX(:,i)
      inline void copyCol(const int i, Vector<T>& DtXi) const;
      /// compute DtX(:,i)
      inline void add_rawCol(const int i, T* DtXi, const T a) const;
      /// compute DtX(:,i)
      inline void diag(Vector<T>& diag) const;
      inline T operator()(const int index1, const int index2) const;

   private:
      Vector<int> _indicesI;
      Vector<int> _indicesJ;
      AbstractMatrix<T>* _matrix;
};

template <typename T> 
SubMatrix<T>::SubMatrix(AbstractMatrix<T>& G, Vector<int>& indI, Vector<int>& indJ) {
   _matrix = &G;
   _indicesI.copy(indI);
   _indicesJ.copy(indJ);
};

template <typename T>
void inline SubMatrix<T>::convertIndicesI(
      Vector<int>& ind) const {
   int* pr_ind = ind.rawX();
   for (int i = 0; i<ind.n(); ++i) {
      if (pr_ind[i] == -1) break;
      pr_ind[i]=_indicesI[pr_ind[i]];
   }
};

template <typename T> 
void inline SubMatrix<T>::convertIndicesJ(
      Vector<int>& ind) const {
   int* pr_ind = ind.rawX();
   for (int i = 0; i<ind.n(); ++i) {
      if (pr_ind[i] == -1) break;
      pr_ind[i]=_indicesJ[pr_ind[i]];
   }
};

template <typename T> 
void inline SubMatrix<T>::extract_rawCol(const int i, T* pr) const {
   int* pr_ind=_indicesI.rawX();
   int* pr_ind2=_indicesJ.rawX();
   for (int j = 0; j<_indicesI.n(); ++j) {
      pr[j]=(*_matrix)(pr_ind[j],pr_ind2[i]);
   }
};

template <typename T> 
inline void SubMatrix<T>::copyCol(const int i, 
      Vector<T>& DtXi) const {
   this->extract_rawCol(i,DtXi.rawX());
};

template <typename T> 
void inline SubMatrix<T>::add_rawCol(const int i, T* pr,
      const T a) const {
   int* pr_ind=_indicesI.rawX();
   int* pr_ind2=_indicesJ.rawX();
   for (int j = 0; j<_indicesI.n(); ++j) {
      pr[j]+=a*(*_matrix)(pr_ind[j],pr_ind2[i]);
   }
};

template <typename T> 
void inline SubMatrix<T>::diag(Vector<T>& diag) const {
   T* pr = diag.rawX();
   int* pr_ind=_indicesI.rawX();
   for (int j = 0; j<_indicesI.n(); ++j) {
      pr[j]=(*_matrix)(pr_ind[j],pr_ind[j]);
   }
};

template <typename T> 
inline T SubMatrix<T>::operator()(const int index1, 
      const int index2) const {
   return (*_matrix)(_indicesI[index1],_indicesJ[index2]);
}


/*****************************************************************************/
/********* CLASSES THREAD_VEC AND THREAD_MAT, FOR AUTOMATIC MEMORY MAN. ******/
/*****************************************************************************/


template<typename V>
class ThreadVec{
public:
    ThreadVec(int nThreads, int n, bool init) {
        data = new V[nThreads];
        for (int i = 0; i<nThreads; ++i) {
            data[i].resize(n);
            if (init) data[i].setZeros();
        }
    }
    ~ThreadVec() { delete[] data; }

    V& operator[](int i) { return data[i]; }
private:
    V* data;
};

template<typename M>
class ThreadMat{
public:
    ThreadMat(int nThreads, int m, int n, bool init) {
        data = new M[nThreads];
        for (int i = 0; i<nThreads; ++i) {
            data[i].resize(m,n);
            if (init) data[i].setZeros();
        }
    }
    ~ThreadMat() { delete[] data; }

    M& operator[](int i) { return data[i]; }
private:
    M*  data;
};


#endif
