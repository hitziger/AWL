/*!
 *
 *                File matrix.h
 * Simple matrix vector and linear algebra routines
 *
 * Efficient implementations based on BLAS/LAPACK
 *
 * Main classes:
 * - Vector
 * - Matrix
 *
 * */

#ifndef MATRIX_H
#define MATRIX_H


#include <cstdlib>
#include <algorithm>
#include <functional>
#include <numeric>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include <blas_template_wrapper.h>

#ifdef LINE_WIDTH
#undef LINE_WIDTH
#endif
#define LINE_WIDTH 80
#ifdef ENTRY_WIDTH
#undef ENTRY_WIDT
#endif
#define ENTRY_WIDTH 12 

#ifndef assert
#include <cassert>
#endif


// forward definitions
template<typename T>
class DataObject;
template<typename T>
class LinearAlgebraObject;
template<typename T>
class Vector;
template<typename T>
class Matrix;


///////////////////////////////////////////////////////////////////////////////
////   CLASS DATA_OBJECT                                                   ////
///////////////////////////////////////////////////////////////////////////////
////       -abstract class for various containers                          ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
class DataObject {
public:
    // Constructors, Destructor
    DataObject();
    explicit DataObject(const unsigned lmax);
    DataObject(T* data, const unsigned lmax);
    explicit DataObject<T>(DataObject<T>& d): _data(d._data), _lmax(d._lmax), 
        _alloc_intern(false) { }
    virtual ~DataObject();

    // Access to pointer
          T* pos(const unsigned i);
    const T* pos(const unsigned i) const; 
          T* ptr();
    const T* ptr() const; 

    // Maximal length 
    unsigned maxLength() const;
    
    // Low-level access
    virtual T  operator[](const unsigned i) const;
    virtual T& operator[](const unsigned i);

    // Allocation
    void allocate(const unsigned lmax);
    void freeData();

protected:

    T*          _data;
    unsigned    _lmax;
    bool        _alloc_intern;
};


///////////////////////////////////////////////////////////////////////////////
////   CLASS LINEAR_ALGEBRA_OBJECT                                         ////
///////////////////////////////////////////////////////////////////////////////
////       -abstract class from which vector and matrix are derived        ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
class LinearAlgebraObject: public DataObject<T> {
    typedef DataObject<T> base;
public:
    // Modification operators
    virtual void operator+=(const T alpha) = 0;
    virtual void operator-=(const T alpha) = 0;
    virtual void operator*=(const T alpha) = 0;
    virtual void operator/=(const T alpha) = 0;
    
    // Initializers
    virtual void zeros() = 0;
    virtual void set(const T alpha) = 0;
    virtual void set(T* data) = 0;
    virtual void copy(const T* data) = 0;

    // Data export
    virtual void extract(T* data) const = 0;

    // Analytical functions, regarding object as array
    virtual T sum() const = 0;
    virtual T asum() const = 0;
    virtual T mean() const = 0;
    virtual T var() const = 0;
    virtual T std() const = 0;
    virtual T min() const = 0;
    virtual T max() const = 0;
    virtual T amin() const = 0;
    virtual T amax() const = 0;
    virtual unsigned argmin() const = 0;
    virtual unsigned argmax() const = 0;
    virtual unsigned argamin() const = 0;
    virtual unsigned argamax() const = 0;

protected:
    // Constructors, Destructor
    LinearAlgebraObject();
    LinearAlgebraObject(const unsigned lmax);
    LinearAlgebraObject(T* data, const unsigned lmax);
    
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const DataObject<T>& d) {
    out << "DataObject (" << d.maxLength() << ")";
    if (d.maxLength()==0) {
        out << std::endl;
        return out;
    }
    out << ":" << std::endl;
    out << "[";
    out.width(ENTRY_WIDTH-1);
    out << d[0];
    unsigned curr_pos=ENTRY_WIDTH;
    for (unsigned i=1; i<d.maxLength(); ++i) {
        curr_pos+=ENTRY_WIDTH;
        out.width(ENTRY_WIDTH);
        out << d[i];
        if (curr_pos>LINE_WIDTH-ENTRY_WIDTH) {
            out << std::endl;
            curr_pos = 0;
        }
    }
    out << "]" << std::endl;
    return out;
}


///////////////////////////////////////////////////////////////////////////////
////   CLASS VECTOR                                                        ////
///////////////////////////////////////////////////////////////////////////////
////       -vector with common linear algebra routines                     ////
////       -derived from abstract class LinearAlgebraObject                ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
class Vector : public LinearAlgebraObject<T> {
    typedef LinearAlgebraObject<T> base;
public:
    // Constructors, Destructor
    Vector();
    explicit Vector(const unsigned l);
    Vector(T* X, const unsigned l);
    explicit Vector<T>(const Vector<T>& v);
    explicit Vector<T>(Vector<T>* v);
    explicit Vector(const Matrix<T>& A);
    explicit Vector(Matrix<T>& A);
    Vector(const Matrix<T>& A, const unsigned j);
    Vector(Matrix<T>& A, const unsigned j);
    
    // New memory allocation, data may be destroyed
    void resize(const unsigned l);
   
    // Dimension 
    unsigned l() const;
    void     l(const unsigned l);
    unsigned lmax() const {return base::_lmax;}
 
    // Initializers
    void zeros();
    void set(const T alpha);
    void set(T* data, const unsigned l);
    void set(T* data);
    void copy(const T* data);

    // Copy l elements from pointer                
    void copy(const T* data, const unsigned l, const unsigned offset);
    void copy(const T* data, const unsigned l);
    
    // Data export
    void extract(T* data) const;
    void extract(T* data, const unsigned l) const;
    void extract(T* data, const unsigned l, const unsigned offset) const;

    // Low-level access
    virtual T  operator()(const unsigned i) const;
    virtual T& operator()(const unsigned i);
    virtual T  operator[](const unsigned i) const;
    virtual T& operator[](const unsigned i);
    T  last() const;
    T& last();
    void push(const T val);
    void pop();

    // Access to pointer
    const T* pos(const unsigned i) const; 
          T* pos(const unsigned i);
    const T* ptr() const;
          T* ptr();

    // Modification operators (elementwise)
    void operator+=(const T alpha);
    void operator-=(const T alpha);
    void operator*=(const T alpha);
    void operator/=(const T alpha);
    void operator+=(const Vector<T>& v);
    void operator-=(const Vector<T>& v);
    void operator*=(const Vector<T>& v);
    void operator/=(const Vector<T>& v);
    
    // other elementwise operations
    void add(const T alpha, const Vector<T>& v); 
    void add(const T alpha, const T* x); 
    void addTo(const T alpha, T*) const; 
    
    // Analytical functions, regarding object as array
    T sum() const;
    T asum() const;
    T mean() const;
    T var() const;
    T std() const;
    T min() const;
    T max() const;
    T amin() const;
    T amax() const;
    unsigned argmin() const;
    unsigned argmax() const;
    unsigned argamin() const;
    unsigned argamax() const;

    // Analytical functions, vector specific
    T dot(const Vector<T>& v) const;
    T norm2() const;
    T norm2sq() const;


protected:
    unsigned _l;
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& v) {
    out << "Vector (" << v.l() << ")";
    if (v.l()==0) {
        out << std::endl;
        return out;
    }
    out << ":" << std::endl;
    out << "[";
    out.width(ENTRY_WIDTH-1);
    out << v[0];
    unsigned curr_pos=ENTRY_WIDTH;
    for (unsigned i=1; i<v.l(); ++i) {
        curr_pos+=ENTRY_WIDTH;
        out.width(ENTRY_WIDTH);
        out << v[i];
        if (curr_pos>LINE_WIDTH-ENTRY_WIDTH) {
            out << std::endl;
            curr_pos = 0;
        }
    }
    out << "]" << std::endl;
    return out;
}



///////////////////////////////////////////////////////////////////////////////
////   CLASS MATRIX                                                        ////
///////////////////////////////////////////////////////////////////////////////
////       -matrix with common linear algebra routines                     ////
////       -derived from abstract class LinearAlgebraObject                ////
////       -elementwise routines taken from class Vector                   ////
///////////////////////////////////////////////////////////////////////////////
template<typename T>
class Matrix : public LinearAlgebraObject<T> {
    typedef LinearAlgebraObject<T> base;
public:
    // Constructors, Destructor
    Matrix();
    Matrix(const unsigned m, const unsigned n);
    Matrix(T* X, const unsigned m, const unsigned n);
    explicit Matrix(const Matrix<T>& A);
    Matrix(Matrix<T>* A);

    // Assignment operator (shallow)
    void operator=(Matrix<T>& A) {
        base::_lmax=A._lmax;
        base::_data=A._data;
        base::_alloc_intern=false;
        _n=A._n;
        _m=A._m;
    }

    // New memory allocation, data may be destroyed
    void resize(const unsigned m, const unsigned n);

    // Initializers
    void zeros();
    void set(const T alpha);
    void set(T* data, const unsigned m, const unsigned n);
    void set(T* data);
    void copy(const T* data);
    void copy(const T* data, const unsigned j);

    // Dimension
    unsigned m() const;
    unsigned n() const;
    void n(const unsigned n);
    
    // Low-level access
    virtual T  operator[](const unsigned i) const;
    virtual T& operator[](const unsigned i);

    // Access
    T& operator()(const unsigned i, const unsigned j);
    T  operator()(const unsigned i, const unsigned j) const;

    // Access to pointer
          T* pos(const unsigned i, const unsigned j);
    const T* pos(const unsigned i, const unsigned j) const; 
          T* col(const unsigned j);
    const T* col(const unsigned j) const; 

    // Data export
    void extract(T* data) const;
    void extract(T* data, const unsigned j) const;

    // Modification operators (elementwise)
    void operator+=(const T alpha);
    void operator-=(const T alpha);
    void operator*=(const T alpha);
    void operator/=(const T alpha);
    void operator+=(const Matrix<T>& M);
    void operator-=(const Matrix<T>& M);
    void operator*=(const Matrix<T>& M);
    void operator/=(const Matrix<T>& M);

    // other elementwise operations
    void add(const T alpha, const Matrix<T> v); 

    // Analytical functions, regarding object as array
    T sum() const;
    T asum() const;
    T mean() const;
    T var() const;
    T std() const;
    T min() const;
    T max() const;
    T amin() const;
    T amax() const;
    unsigned argmin() const;
    unsigned argmax() const;
    unsigned argamin() const;
    unsigned argamax() const;

    // Linear algebra    
    void copyUpperToLower();
    void copyLowerToUpper();
    void XtX(Matrix<T>& xtx) const;
    void XXt(Matrix<T>& xxt) const;
    void mult(const Vector<T>& x, Vector<T>& b = Vector<T>(),
            const bool trans=false, const T alpha = 1.0, 
            const T beta = 0.0) const;
    void mult(const Matrix<T>& B, Matrix<T>& C, const bool transA = false,
            const bool transB = false, const T a = 1.0, const T b = 0.0) const;

protected:
    unsigned _m;
    unsigned _n;
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& A) {
    out << "Matrix (" << A.m() << "," << A.n() << ")";
    if (A.m()==0 || A.n()==0) {
        out << std::endl;
        return out;
    }
    out << ":" << std::endl;
    for (unsigned i=0; i<A.m(); ++i) {
        out << "[";
        out.width(ENTRY_WIDTH-1);
        out << A(i,0);
        unsigned curr_pos=ENTRY_WIDTH;
        for (unsigned j=1; j<A.n(); ++j) {
            curr_pos+=ENTRY_WIDTH;
            out.width(ENTRY_WIDTH);
            out << A(i,j);
            if (curr_pos>LINE_WIDTH-ENTRY_WIDTH) {
                out << std::endl;
                curr_pos = 0;
            }
        }
        out << "]" << std::endl;
    }
    return out;
}


///////////////////////////////////////////////////////////////////////////////
////   IMPLEMENTATION CLASS LINEAR_ALGEBRA_OBJECT                          ////
///////////////////////////////////////////////////////////////////////////////
////       -abstract class from which vector and matrix are derived        ////
///////////////////////////////////////////////////////////////////////////////

// Protected constructors, destructor                  
template<typename T>
DataObject<T>::DataObject(): _data(NULL), _lmax(0), _alloc_intern(false) { }
template<typename T>
DataObject<T>::DataObject(const unsigned l): _data(NULL), _lmax(0), _alloc_intern(true) {
    this->allocate(l);
} 
template<typename T>
DataObject<T>::DataObject(T* data, const unsigned l): _data(data), _lmax(l), _alloc_intern(false) { }
template<typename T>
DataObject<T>::~DataObject() { 
    this->freeData(); 
}

// Allocation routines
template<typename T>
inline void DataObject<T>::allocate(const unsigned lmax) { 
    if (_lmax>=lmax) return;
    this->freeData();
    _data=(T*) calloc(lmax,sizeof(T));
    assert(_data!=NULL && "DataObject<T>::allocate:Allocation failed!");
    _lmax=lmax;
    _alloc_intern=true;
}

template<typename T>
inline void DataObject<T>::freeData() {
    if (_data!=NULL) {
        if (_alloc_intern) {
            free(_data);
            _alloc_intern=false;
        }
        _lmax=0;
        _data=NULL;
    }
}

// Access to pointer
template<typename T>
inline const T* DataObject<T>::pos(const unsigned i) const { 
    assert(i<_lmax && "DataObject::pos: selected index exceeds maximal length");
    return _data+i;
}
template<typename T>
inline T* DataObject<T>::pos(const unsigned i) { 
    assert(i<_lmax && "DataObject::pos: selected index exceeds maximal length");
    return _data+i;
}
template<typename T>
inline const T* DataObject<T>::ptr() const { 
    return this->pos(0);
}
template<typename T>
inline T* DataObject<T>::ptr() { 
    return this->pos(0);
}

// Low-level Access
template<typename T>
inline T DataObject<T>::operator[](const unsigned i) const { 
    assert(i<_lmax && "DataObject::operator[]: index out of range");
    return _data[i];
}
template<typename T>
inline T& DataObject<T>::operator[](const unsigned i) { 
    assert(i<_lmax && "DataObject::operator[]: index out of range");
    return _data[i];
}

// Maximal length
template<typename T>
inline unsigned DataObject<T>::maxLength() const {
    return _lmax;
} 


///////////////////////////////////////////////////////////////////////////////
////   IMPLEMENTATION CLASS LINEAR_ALGEBRA_OBJECT                          ////
///////////////////////////////////////////////////////////////////////////////
////       -abstract class from which vector and matrix are derived        ////
///////////////////////////////////////////////////////////////////////////////

// Protected constructors, destructor                  
template<typename T>
LinearAlgebraObject<T>::LinearAlgebraObject(): base() { }
template<typename T>
LinearAlgebraObject<T>::LinearAlgebraObject(const unsigned l): base(l) { } 
template<typename T>
LinearAlgebraObject<T>::LinearAlgebraObject(T* data, const unsigned l)
    : base(data,l) { }



///////////////////////////////////////////////////////////////////////////////
////   IMPLEMENTATION CLASS VECTOR                                         ////
///////////////////////////////////////////////////////////////////////////////
////       -vector with common linear algebra routines                     ////
////       -derived from abstract class LinearAlgebraObject                ////
///////////////////////////////////////////////////////////////////////////////

// Constructors, destructor
template<typename T>
Vector<T>::Vector(): base(), _l(0) { }
template<typename T>
Vector<T>::Vector(const unsigned l): base(l), _l(l) { } 
template<typename T>
Vector<T>::Vector(T* data, const unsigned l): base(data,l), _l(l) { }
template<typename T>
Vector<T>::Vector(const Vector<T>& v): base(v._lmax), _l(v._l) { 
    this->copy(v._data); 
}
template<typename T>
Vector<T>::Vector(Vector<T>* v): base(v->_data,v->_lmax), _l(v->_l) { }
template<typename T>
Vector<T>::Vector(const Matrix<T>& A): base(const_cast<T*>(A.ptr()),A.maxLength()), _l(A.m()*A.n()) { }
template<typename T>
Vector<T>::Vector(Matrix<T>& A): base(A.ptr(),A.maxLength()), _l(A.m()*A.n()) { }
template<typename T>
Vector<T>::Vector(const Matrix<T>& A, const unsigned j): base(const_cast<T*>(A.pos(0,j)),A.m()), _l(A.m()) { }
template<typename T>
Vector<T>::Vector(Matrix<T>& A, const unsigned j): base(A.pos(0,j),A.m()), _l(A.m()) { }


// Low-level Access
template<typename T>
inline T Vector<T>::operator()(const unsigned i) const { 
    assert(i<_l && "Vector::operator(): index out of range");
    return base::operator[](i);
}
template<typename T>
inline T& Vector<T>::operator()(const unsigned i) { 
    assert(i<_l && "Vector::operator(): index out of range");
    return base::operator[](i);
}template<typename T>
inline T Vector<T>::operator[](const unsigned i) const { 
    assert(i<base::_lmax && "Vector::operator[]: index out of range");
    return base::operator[](i);
}
template<typename T>
inline T& Vector<T>::operator[](const unsigned i) { 
    assert(i<base::_lmax && "Vector::operator[]: index out of range");
    return base::operator[](i);
}
template<typename T>
inline T Vector<T>::last() const { 
    assert(_l>0 && "Vector::last: length must be positive");
    return base::operator[](_l-1);
}
template<typename T>
inline T& Vector<T>::last() { 
    assert(_l>0 && "Vector::last: length must be positive");
    return base::operator[](_l-1);
}
template<typename T>
inline void Vector<T>::push(const T val) { 
    assert(_l<this->_lmax && "Vector::push: not enough capacity");
    this->_data[_l] = val;
    _l += 1;
}
template<typename T>
inline void Vector<T>::pop() { 
    assert(_l>0 && "Vector::pop: length must be positive");
    _l -= 1;
}

// Access to pointer
template<typename T>
inline const T* Vector<T>::pos(const unsigned i) const { 
    assert(i<_l && "Vector::pos: selected index exceeds vector length");
    return base::pos(i);
}
template<typename T>
inline T* Vector<T>::pos(const unsigned i) { 
    assert(i<_l && "Vector::pos: selected index exceeds vector length");
    return base::pos(i);
}
template<typename T>
inline const T* Vector<T>::ptr() const { 
    return this->pos(0);
}
template<typename T>
inline T* Vector<T>::ptr() { 
    return this->pos(0);
}

// New memory allocation   
template<typename T>
inline void Vector<T>::resize(const unsigned l) {
    _l=l;
    this->allocate(l);
}

// Initializations
template<typename T>
inline void Vector<T>::zeros() {
    memset(base::_data,T(),_l*sizeof(T));
} 
template<typename T>
inline void Vector<T>::set(const T alpha) {
    std::fill_n(base::_data,_l,alpha);
} 
template<typename T>
inline void Vector<T>::set(T* data) {
    this->freeData();
    base::_lmax=_l;
    base::_data=data;
}
template<typename T>
inline void Vector<T>::set(T* data, const unsigned l) {
    _l=l;
    this->set(data);
}

// Copy l elements from pointer                
template<typename T>
inline void Vector<T>::copy(const T* data, const unsigned l, const unsigned offset) {
    assert(offset+l<=_l && "Vector::copy: index out of range");
    blas_copy<T>(*const_cast<unsigned*>(&l),const_cast<T*>(data),1,base::_data+offset,1);
}    
template<typename T>
inline void Vector<T>::copy(const T* data, const unsigned l) {
    this->copy(data,l,0);
}    
template<typename T>
inline void Vector<T>::copy(const T* data) {
    this->copy(data,_l);
} 

// Data export                
template<typename T>
inline void Vector<T>::extract(T* data, const unsigned l, const unsigned offset) const {
    assert(offset+l<=_l && "Vector::extract: index out of range");
    blas_copy<T>(l,base::_data+offset,1,data,1);
} 
template<typename T>
inline void Vector<T>::extract(T* data, const unsigned l) const {
    this->extract(data,l,0);
}    
template<typename T>
inline void Vector<T>::extract(T* data) const {
    this->extract(data,_l);
} 
// Dimension 
template<typename T>
inline unsigned Vector<T>::l() const {
    return _l;
}
template<typename T>
inline void Vector<T>::l(const unsigned l) {
    assert(l<=base::_lmax && "Vector::l: capacity too small");
    _l=l;
}

// Modification operators
template<typename T>
inline void Vector<T>::operator+=(const T alpha) {
    for (unsigned i=0; i<_l; ++i) base::_data[i]+=alpha;
}
template<typename T>
inline void Vector<T>::operator-=(const T alpha) {
    for (unsigned i=0; i<_l; ++i) base::_data[i]-=alpha;
}
template<typename T>
inline void Vector<T>::operator*=(const T alpha) {
    blas_scal<T>(_l,alpha,base::_data,1);
}  
template<typename T>
inline void Vector<T>::operator/=(const T alpha) {
    assert(alpha>1e-20 && "Vector::operator/=: argument must be non vanishing");
    blas_scal<T>(_l,1./alpha,base::_data,1);
} 
template<typename T>
inline void Vector<T>::operator+=(const Vector<T>& v) {
    assert(v._l==_l  &&  "Vector::operator+=: length of argument not consistent");
    blas_axpy<T>(_l,1.,v._data,1,base::_data,1);
}
template<typename T>
inline void Vector<T>::operator-=(const Vector<T>& v) {
    assert(v._l==_l && "Vector::operator-=: length of argument not consistent");
    blas_axpy<T>(_l,-1.,v._data,1,base::_data,1);
}
template<typename T>
inline void Vector<T>::operator*=(const Vector<T>& v) {
    assert(v._l==_l && "Vector::operator*=: length of argument not consistent");
    std::transform(base::_data,base::_data+_l,v._data,base::_data,std::multiplies<T>());
}
template<typename T>
inline void Vector<T>::operator/=(const Vector<T>& v) {
    assert(v._l==_l && "Vector::operator/=: length of argument not consistent");
    std::transform(base::_data,base::_data+_l,v._data,base::_data,std::divides<T>());
}

// Other elementwise operations                   
template<typename T>
inline void Vector<T>::add(const T alpha, const Vector<T>& v) {
    assert(v._l==_l && "Vector::add: length of argument not consistent");
    add(alpha,v._data);
}
template<typename T>
inline void Vector<T>::add(const T alpha, const T* x) {
    blas_axpy<T>(_l,alpha,const_cast<T*>(x),1,base::_data,1);
}
template<typename T>
inline void Vector<T>::addTo(const T alpha, T* x) const {
    blas_axpy<T>(_l,alpha,base::_data,1,x,1);
}

// Analytical functions, regarding object as array
template<typename T>
inline T Vector<T>::sum() const {
    return std::accumulate(base::_data,base::_data+_l,T());
}
template<typename T>
inline T Vector<T>::asum() const {
   return blas_asum<T>(_l,base::_data,1);
}
template<typename T>
inline T Vector<T>::mean() const {
    return this->sum()/(T) _l;
}
template<typename T>
inline T Vector<T>::var() const {
    const T m=this->mean();
    T res =T();
    for (unsigned i=0; i<_l; ++i)
        res += pow(base::_data[i]-m,2);
    return res/(T) _l;

}
template<typename T>
inline T Vector<T>::std() const {
    return sqrt(this->var());
}
template<typename T>
inline T Vector<T>::min() const {
    return *std::min_element(base::_data,base::_data+_l);
}
template<typename T>
inline T Vector<T>::max() const {
    return *std::max_element(base::_data,base::_data+_l);
}
template<typename T>
inline T Vector<T>::amin() const {
    T min=abs(base::_data[0]);
    for (unsigned i=1; i<_l; ++i) {
        if (abs(base::_data[i])<min) min=abs(base::_data[i]);
    }
    return min;
}
template<typename T>
inline T Vector<T>::amax() const {
    return base::_data[this->argamax()];
}
template<typename T>
inline unsigned Vector<T>::argmin() const {
    return (unsigned) (std::min_element(base::_data,base::_data+_l)-base::_data);
}
template<typename T>
inline unsigned Vector<T>::argmax() const {
    return (unsigned) (std::max_element(base::_data,base::_data+_l)-base::_data);
}
template<typename T>
inline unsigned Vector<T>::argamin() const {
    T min=abs(base::_data[0]);
    unsigned k=0;
    for (unsigned i=1; i<_l; ++i) {
        if (abs(base::_data[i])<min) {
            min=abs(base::_data[i]);
            k=i;
        }
    }
    return k;
}

template<typename T>
inline unsigned Vector<T>::argamax() const {
    return blas_iamax<T>(_l,base::_data,1);
}

// Analytical functions, vector specific
template<typename T>
inline T Vector<T>::dot(const Vector<T>& v) const {
    assert(v.l()==_l && "Vector::dot: vectors must be of same length!");
    return blas_dot<T>(*const_cast<unsigned*>(&_l),base::_data,1,const_cast<T*>(v.ptr()),1);
}
template<typename T>
inline T Vector<T>::norm2sq() const {
    return blas_dot<T>(_l,base::_data,1,base::_data,1);
}
template<typename T>
inline T Vector<T>::norm2() const {
    return blas_nrm2<T>(_l,base::_data,1);
}

    
///////////////////////////////////////////////////////////////////////////////
////   IMPLEMENTATION CLASS MATRIX                                         ////
///////////////////////////////////////////////////////////////////////////////
////       -matrix with common linear algebra routines                     ////
////       -derived from abstract class LinearAlgebraObject                ////
////       -elementwise routines taken from class Vector                   ////
///////////////////////////////////////////////////////////////////////////////

// Constructors, Destructor
template<typename T>
Matrix<T>::Matrix(): base(), _m(0), _n(0) { }
template<typename T>
Matrix<T>::Matrix(const unsigned m, const unsigned n): base(m*n), _m(m), _n(n) { } 
template<typename T>
Matrix<T>::Matrix(T* data, const unsigned m, const unsigned n): base(data,m*n), _m(m), _n(n) { }
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& A): base(A._m*A._n), _m(A._m), _n(A._n) { this->copy(A._data); }
template<typename T>
Matrix<T>::Matrix(Matrix<T>* A): base(A._data,A._m*A._n), _m(A._m), _n(A._n) { }

// New memory allocation, data may be destroyed
template<typename T>
inline void Matrix<T>::resize(const unsigned m, const unsigned n) {
    _m=m;
    _n=n;
    this->allocate(m*n);
}

// Initializers
template<typename T>
inline void Matrix<T>::zeros() {
    Vector<T> v(*this);
    v.zeros();
} 
template<typename T>
inline void Matrix<T>::set(const T alpha) {
    Vector<T> v(*this);
    v.set(alpha);
} 
template<typename T>
inline void Matrix<T>::set(T* data) {
    this->freeData();
    base::_lmax=_m*_n;
    base::_data=data;
} 
template<typename T>
inline void Matrix<T>::set(T* data, const unsigned m, const unsigned n) {
    _m=m;
    _n=n;
    this->set(data);
} 
template<typename T>
inline void Matrix<T>::copy(const T* data) {
    Vector<T> v(*this);
    v.copy(data);
}
template<typename T>
inline void Matrix<T>::copy(const T* data, const unsigned j) {
    Vector<T> v(*this,j);
    v.copy(data);
} 
 

// Dimension
template<typename T>
inline unsigned Matrix<T>::m() const {
    return _m;
}
template<typename T>
inline unsigned Matrix<T>::n() const {
    return _n;
}
template<typename T>
inline void Matrix<T>::n(const unsigned n) {
    assert(_m*n<=base::_lmax && "Matrix::n: capacity too small");
    _n=n;
}

// Low-lewel access
template<typename T>
inline T& Matrix<T>::operator[](const unsigned i) {
    Vector<T> v(*this);
    return v[i];
}
template<typename T>
inline T Matrix<T>::operator[](const unsigned i) const {
    const Vector<T> v(*this);
    return v[i];
}

// Access
template<typename T>
inline T& Matrix<T>::operator()(const unsigned i, const unsigned j) {
    return *(this->pos(i,j));
}
template<typename T>
inline T Matrix<T>::operator()(const unsigned i, const unsigned j) const {
    return *(this->pos(i,j));
}
// Access to pointer
template<typename T>
inline T* Matrix<T>::pos(const unsigned i, const unsigned j) {
    assert(i<_m && j<_n && "Matrix::pos: index out of range");
    return base::_data+i+j*_m;
}
template<typename T>
inline const T* Matrix<T>::pos(const unsigned i, const unsigned j) const {
    assert(i<_m && j<_n && "Matrix::pos: index out of range");
    return base::_data+i+j*_m;
}
template<typename T>
inline T* Matrix<T>::col(const unsigned j) {
    return this->pos(0,j);
}
template<typename T>
inline const T* Matrix<T>::col(const unsigned j) const {
    return this->pos(0,j);
}

// Data export
template<typename T>
inline void Matrix<T>::extract(T* data) const { 
    const Vector<T> v(*this);
    v.extract(data);
}

// Data export
template<typename T>
inline void Matrix<T>::extract(T* data, const unsigned j) const { 
    const Vector<T> v(*this,j);
    v.extract(data);
}


// Modification operators
template<typename T>
inline void Matrix<T>::operator+=(const T alpha) {
    Vector<T> v(*this);
    v+=alpha;
}
template<typename T>
inline void Matrix<T>::operator-=(const T alpha) {
    Vector<T> v(*this);
    v-=alpha;
}
template<typename T>
inline void Matrix<T>::operator*=(const T alpha) {
    Vector<T> v(*this);
    v*=alpha;
}  
template<typename T>
inline void Matrix<T>::operator/=(const T alpha) {
    Vector<T> v(*this);
    v/=alpha;
} 
template<typename T>
inline void Matrix<T>::operator+=(const Matrix<T>& M) {
    assert(M._m=_m && M._n=_n && "Matrix::operator+=: dimensions of argument inconsistent");
    Vector<T> v(*this);
    const Vector<T> w(M);
    v+=w;
}
template<typename T>
inline void Matrix<T>::operator-=(const Matrix<T>& M) {
    assert(M._m=_m && M._n=_n && "Matrix::operator-=: dimensions of argument inconsistent");
    Vector<T> v(*this);
    const Vector<T> w(M);
    v-=w;
}
template<typename T>
inline void Matrix<T>::operator*=(const Matrix<T>& M) {
    assert(M._m=_m && M._n=_n && "Matrix::operator*=: dimensions of argument inconsistent");
    Vector<T> v(*this);
    const Vector<T> w(M);
    v*=w;
}
template<typename T>
inline void Matrix<T>::operator/=(const Matrix<T>& M) {
    assert(M._m=_m && M._n=_n && "Matrix::operator/=: dimensions of argument inconsistent");
    Vector<T> v(*this);
    const Vector<T> w(M);
    v/=w;
}

// Analytical functions, regarding object as array
template<typename T>
inline T Matrix<T>::sum() const {
    const Vector<T> v(*this);
    return v.sum();
}
template<typename T>
inline T Matrix<T>::asum() const {
    const Vector<T> v(*this);
    return v.asum();
}
template<typename T>
inline T Matrix<T>::mean() const {
    const Vector<T> v(*this);
    return v.mean();
}
template<typename T>
inline T Matrix<T>::var() const {
    const Vector<T> v(*this);
    return v.var();
}
template<typename T>
inline T Matrix<T>::std() const {
    const Vector<T> v(*this);
    return v.std();
}
template<typename T>
inline T Matrix<T>::min() const {
    const Vector<T> v(*this);
    return v.min();
}
template<typename T>
inline T Matrix<T>::max() const {
    const Vector<T> v(*this);
    return v.max();
}
template<typename T>
inline T Matrix<T>::amin() const {
    const Vector<T> v(*this);
    return v.amin();
}
template<typename T>
inline T Matrix<T>::amax() const {
    const Vector<T> v(*this);
    return v.amax();
}
template<typename T>
inline unsigned Matrix<T>::argmin() const {
    const Vector<T> v(*this);
    return v.argmin();
}
template<typename T>
inline unsigned Matrix<T>::argmax() const {
    const Vector<T> v(*this);
    return v.argmax();
}
template<typename T>
inline unsigned Matrix<T>::argamin() const {
    const Vector<T> v(*this);
    return v.argamin();
}
template<typename T>
inline unsigned Matrix<T>::argamax() const {
    const Vector<T> v(*this);
    return v.argamax();
}


// Linear algebra
// 
template <typename T> 
inline void Matrix<T>::copyUpperToLower() {
   for (int i = 0; i<_n; ++i) {
      for (int j =0; j<i; ++j) {
         this->_data[j*_m+i]=this->_data[i*_m+j];
      }
   }
}

template <typename T> 
inline void Matrix<T>::copyLowerToUpper() {
   for (int i = 0; i<_n; ++i) {
      for (int j =0; j<i; ++j) {
         this->_data[i*_m+j]=this->_data[j*_m+i];
      }
   }
}

// XtX = A'*A
template <typename T>
inline void Matrix<T>::XtX(Matrix<T>& AtA) const {
   AtA.resize(_n,_n);
   blas_syrk<T>(blas::Upper,blas::Trans,_n,_m,T(1.0), this->_data,_m,T(0.0),
           AtA._data,_n);
   AtA.copyUpperToLower();
}

// XXt = A*A'
template <typename T> inline void Matrix<T>::XXt(Matrix<T>& AAt) const {
   AAt.resize(_m,_m);
   blas_syrk<T>(blas::Upper,blas::NoTrans,_m,_n,T(1.0),
         this->_data,_m,T(0.0),AAt._X,_m);
   AAt.copyUpperToLower();
}

// b = a*Ax + c*b, (possibly transposing A)
template<typename T>
inline void Matrix<T>::mult(const Vector<T>& x, Vector<T>& b, const bool trans,
        const T a, const T c) const {
    if (trans) {
        if (b.l()==0) b.resize(_n);
        assert(b.l()==_n && x.l()==_m && "Matrix::mult: matrix and vector dimensions must agree!");
        blas_gemv<T>(blas::Trans,_m,_n,a,this->_data,_m,const_cast<T*>(x.ptr()),1,c,b.ptr(),1);
    }
    else {
        if (b.l()==0) b.resize(_m);
        assert(b.l()==_m && x.l()==_n && "Matrix::mult: matrix and vector dimensions must agree!");
        blas_gemv<T>(blas::NoTrans,_m,_n,a,this->_data,_m,const_cast<T*>(x.ptr()),1,c,b.ptr(),1);
    }
}

// C = a*A*B + b*C (possibly transposing A or B)
template<typename T>
inline void Matrix<T>::mult(const Matrix<T>& B, Matrix<T>& C, 
        const bool transA, const bool transB, const T a, const T b) const {
    char trA,trB;
    int m,kA,kB,n;
    if (transA) {
        trA = blas::Trans;
        m = _n;
        kA = _m;
    } else {
        trA= blas::NoTrans;
        m = _m;
        kA = _n;
    }
    if (transB) {
        trB = blas::Trans;
        kB = B.n();
        n = B.m(); 
    } else {
        trB = blas::Trans;
        kB = B.m(); 
        n = B.n(); 
    }
    assert(kA == kB &&  "Matrix::mult: matrix dimensions must be consistent" );
    if (!(m==C.m() && n==C.n())) {
        mexPrintf("Warning: Matrix::mult: resizing matrix");
        C.resize(m,n);
    }
    blas_gemm<T>(trA,trB,m,n,kA,a,this->_data,_m,B.ptr(),B.m(),
         b,C.ptr(),C.m());
}

template <typename T>
Vector<T> operator*(const Matrix<T> A, const Vector<T> v) {
    assert(A.n() == v.l() &&  
            "operator*: matrix and vector dimension inconsistent");
    Vector<T> res;
    A.mult(v,res);
    return res;
}

template <typename T>
Vector<T> operator*(const Vector<T> v, const Matrix<T> A) {
    assert(A.m() == v.l() &&  
            "operator*: matrix and vector dimension inconsistent");
    Vector<T> res;
    A.mult(v,res,true);
    return res;
}


#endif
