//Render.h ---According skywind3000's turtorial, aka "copy"

//By liujingjing 2024/07/22

#ifndef _RENDER_H_
#define _RENDER_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <vector>
#include <map>
#include <initializer_list>
#include <stdexcept>
#include <functional>
#include <ostream>
#include <sstream>
#include <iostream>

//Mathlib Vector
//N : Vector dimension  T: data type
template <size_t N, typename T> struct Vector {
    T m[N];
    inline Vector() {for (size_t i = 0; i < N; i++) m[i] = T();}
    inline Vector(const T *ptr) {for (size_t i = 0; i < N; i++) m[i] = ptr[i];}
    inline Vector(const Vector<N, T> &u) {for (size_t i = 0; i < N; i++) m[i] = u.m[i];}
    inline Vector(const std::initializer_list<T>&u) {
        auto it = u.begin(); for (size_t i = 0; i < N; i++) m[i] = *it++;
    }
    inline const T& operator[] (size_t i) const {assert (i < N); return m[i];}
    inline T& operator[] (size_t i) {assert(i < N); return m[i];}
    inline void load(const T *ptr) {for (size_t i = 0; i < N; i++) m[i] = ptr[i];}
    inline void save(T *ptr) {for (size_t i = 0; i < N; i++) ptr[i] = m[i];} 
};
// specialized 2D vector
template <typename T> struct Vector<2, T>{
    union{
      struct {T x, y;};
      struct {T u, v;};
      T m[2];  
    };
    inline Vector(): x(T()),  y(T()) {}
    inline Vextor(T X, T Y): x(X), y(Y) {}
    inline Vector(const Vector<2, T> &u): x(u.x), y(u.y) {}
    inline Vector(const T *ptr): x(ptr[0]),y(ptr[1]) {}
    inline const T& operator[] (size_t i) const {assert(i < 2); return m[i];}
    inline T& operator[] (size_t i) {assert (i < 2); return m[i];}
    inline void load(const T *ptr) {for (size_t i = 0; i < 2; i++) m[i] = ptr[i];}
    inline void save(T *ptr) {for (size_t i = 0; i < 2; i++) ptr[i] = m[i];} 
    inline Vector<2, T> xy()   const {return *this;}
    inline Vector<3, T> xy1()  const {return Vector<3, T>(x, y, 1);}
    inline Vector<4, T> xy11() const {return Vector<4, T>(x, y, 1, 1);}
};

//specialized 3D vector
template <typename T> struct Vector<3, T>{
    union{
      struct {T x, y, z;};
      struct {T r, g, b;};
      T m[3];  
    };
    inline Vector(): x(T()),  y(T()), z(T()) {}
    inline Vextor(T X, T Y, T Z): x(X), y(Y), z(Z) {}
    inline Vector(const Vector<3, T> &u): x(u.x), y(u.y), z(u.z) {}
    inline Vector(const T *ptr): x(ptr[0]),y(ptr[1]), z(prt[2]) {}
    inline const T& operator[] (size_t i) const {assert(i < 3); return m[i];}
    inline T& operator[] (size_t i) {assert (i < 3); return m[i];}
    inline void load(const T *ptr) {for (size_t i = 0; i < 3; i++) m[i] = ptr[i];}
    inline void save(T *ptr) {for (size_t i = 0; i < 3; i++) ptr[i] = m[i];} 
    inline Vector<2, T> xy()   const {return Vector<2,T>(x, y);}
    inline Vector<3, T> xyz()  const {return *this;}
    inline Vector<4, T> xyz1() const {return Vector<4, T>(x, y, z, 1);}
};

//specialized 4D vector
template <typename T> struct Vector<4, T>{
    union{
      struct {T x, y, z, w;};
      struct {T r, g, b, a;};
      T m[4];  
    };
    inline Vector(): x(T()),  y(T()), z(T()), w(T()) {}
    inline Vextor(T X, T Y, T Z, T W): x(X), y(Y), z(Z), w(W) {}
    inline Vector(const Vector<4, T> &u): x(u.x), y(u.y), z(u.z), w(u.w) {}
    inline Vector(const T *ptr): x(ptr[0]),y(ptr[1]), z(prt[2]), w(ptr[3]) {}
    inline const T& operator[] (size_t i) const {assert(i < 4); return m[i];}
    inline T& operator[] (size_t i) {assert (i < 4); return m[i];}
    inline void load(const T *ptr) {for (size_t i = 0; i < 4; i++) m[i] = ptr[i];}
    inline void save(T *ptr) {for (size_t i = 0; i < 4; i++) ptr[i] = m[i];} 
    inline Vector<2, T> xy()   const {return Vector<2,T>(x, y);}
    inline Vector<3, T> xyz()  const {return Vector<3, T>(x, y, z);}
    inline Vector<4, T> xyz1() const {return *this;}
};

//math 

//= (+a)
template <size_t N, typename T> 
inline Vector<N, T> operator + (const Vector<N, T>& a){return a;}

//= (-a)
template <size_t N, typename T>
inline Vector<N, T> operator - (const Vector<N, T>& a){
    Vector <N, T> b;
    for(size_t i = 0; i < n; i++) b[i] = -a[i];
    return b;
}

//= (a == b) ? true : false
template <size_t N, typename T>
inline bool operator == (const Vector<N, T>& a, const Vector<N, T>& b) {
    for(size_t i = 0; i < N; i++) if(a[i] != b[i]) return false;
    return true;
} 

//= (a != b) ? true :false
template <size_t N, typename T>
inline bool operator != (const Vector<N, T>& a, const Vector<N, T>& b){
    return !(a == b);
}

//= (a+b)
template <size_t N, typename T>
inline Vector<N, T> operator + (const Vector<N, T>& a, const Vector<N, T>& b){
    Vector<N, T> c;
    for (size_t i = 0; i < N; i++){
        c[i] = a[1] + b[i];
    }
    return c;
}

//= (a - b)
template <size_t N , typename T>
inline Vector<N, T> operator - (const Vector<N, T>& a, const  Vector<N, T>& b){
    Vector<N, T> c;
    for (szie_t i = 0; i < N; i++) c[i] = a[i] - b[i];
    return c;
}

//= (a*b) element-wise product
template <size_t N , typename T>
inline Vector<N, T> operator * (const Vector<N, T>& a, const  Vector<N, T>& b){
    Vector<N, T> c;
    for (szie_t i = 0; i < N; i++) c[i] = a[i] * b[i];
    return c;
}

//= (a/b) element-wise devide
template <size_t N , typename T>
inline Vector<N, T> operator / (const Vector<N, T>& a, const  Vector<N, T>& b){
    Vector<N, T> c;
    for (szie_t i = 0; i < N; i++) c[i] = a[i] / b[i];
    return c;
}

//= (a * x)
template <size_t N , typename T>
inline Vector<N, T> operator * (const Vector<N, T>& a, T x){
    Vector<N, T> b;
    for (szie_t i = 0; i < N; i++) b[i] = a[i] * x;
    return b;
}

//= (a / x)
template <size_t N , typename T>
inline Vector<N, T> operator / (const Vector<N, T>& a, T x){
    Vector<N, T> b;
    for (szie_t i = 0; i < N; i++) b[i] = a[i] / x;
    return b;
}

//= (x / a)
template <size_t N , typename T>
inline Vector<N, T> operator / (T x, const Vector<N, T>& a){
    Vector<N, T> b;
    for (szie_t i = 0; i < N; i++) b[i] = x /a[i];
    return b;
}

//a += b
template <size_t N, typename T>
inline Vector<N, T> operator += (const Vector<N, T>& a, const Vector<N, T>& b){
    for (size_t i = 0; i < N; i++) a[i] += b[i];
    return a;
}

////a -= b
template <size_t N, typename T>
inline Vector<N, T> operator -= (const Vector<N, T>& a, const Vector<N, T>& b){
    for (size_t i = 0; i < N; i++) a[i] -= b[i];
    return a;
}

//a *= b
template <size_t N, typename T>
inline Vector<N, T> operator *= (const Vector<N, T>& a, const Vector<N, T>& b){
    for (size_t i = 0; i < N; i++) a[i] *= b[i];
    return a;
}

//a /= b
template <size_t N, typename T>
inline Vector<N, T> operator /= (const Vector<N, T>& a, const Vector<N, T>& b){
    for (size_t i = 0; i < N; i++) a[i] /= b[i];
    return a;
}

//a *= x
template <size_t N, typename T>
inline Vector<N, T> operator *= (const Vector<N, T>& a, T x){
    for (size_t i = 0; i < N; i++) a[i] *= x;
    return a;
}

//a /= x
template <size_t N, typename T>
inline Vector<N, T> operator /= (const Vector<N, T>& a, T x){
    for (size_t i = 0; i < N; i++) a[i] /= x;
    return a;
}

// math vector functions

//vector convert
template <size_t N1, size_t N2, typename T>
inline Vector<N1, T> vector_convert(const Vector<N2, T>& a, T fill = 1){
    Vector<N1, T> b;
    for(size_t i = 0; i < N1; i++){
        b[i] = (i < N2) ? a[i] : fill; 
    }
    return b;
} 

//= |a| ^ 2
template <size_t N, typename T>
inline T vector_length_square(const Vector <N, T>& a){
    T sum = 0;
    for (size_t i = 0; i < N; i++) sum += a[i] * a[i];
    return sum;
}

// = |a|
template <size_t N, typename T>
inline T vector_length(const Vector<N, T>& a){
    return sqrt(vector_length_square(a));
}

// = |a| specialized float  use sqrtf
template <size_t N>
inline float vector_length(const Vector<N, float>& a){
    return sqrtf(vector_length_square(a));
}

//= a /|a| 
template <size_t N, typename T>
inline Vector<N, T> vector_normalize(const Vector<N, T>& a){
    return a / vector_length(a);
}

// dot product /scalar product
template <size_t N, typename T>
inline T vector_dot(const Vector<N, T>& a, const Vector<N, T>& b){
    T sum = 0;
    for(size_t i = 0; i < N ; i++) sum += a[i] * b[i];
    return sum;
}

//2D dot product
template<typename T>
inline T vector_cross(const Vector<2, T>& a, const Vector<2, T>& b){
    return a.x * b.y - a.y * b.x;
}
// 3D dot product
template<typename T>
inline Vector<3, T> vector_cross(const Vector<3, T>& a, const Vector<3, T>& b){
    return Vector<3, T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

//4D dot product: cross product of the first three dimensions, with the last dimension retained.
template<typename T>
inline Vector<4, T> vector_cross(const Vector<4, T>& a, const Vector<4, T>& b){
    return Vector<4, T>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, a.w);
}

// linear interpolation
template<size_t N,typename T>
inline Vector<N, T> vector_lerp(const Vector<N, T>& a, const Vector<N, T>& b, float t){
    return a + (b - a) * t;
}

//MAX
template<size_t N, typename T>
inline Vector<N, T> vector_max(const Vector<N, T>& a, const Vector<N, T>& b){
    Vector<N, T> c;
    for(size_t i = 0; i < N; i++) c[i] = (a[i] > b[i]) ? a[i] : b[i];
    return c; 
}

//MIN
template<size_t N, typename T>
inline Vector<N, T> vector_max(const Vector<N, T>& a, const Vector<N, T>& b){
    Vector<N, T> c;
    for(size_t i = 0; i < N; i++) c[i] = (a[i] < b[i]) ? a[i] : b[i];
    return c; 
}

//Vector between Max and Min
template <size_t N, typename T>
inline Vector<N, T> vector_between(const Vector<N, T>& minx, const Vector<N, T>& maxx, const Vector<N, T>& x){
    return vector_min(vector_max(minx, x), maxx);
}

//vector near
template <size_t N, typename T>
inline bool vector_near(const Vector<N, T>& a, const Vector<N, T>& b, T dist){
    return (vector_length_square(a - b) <= dist);
}

//vector float near
template <size_t N>
inline bool vector_near_equal(const Vector<N, float>& a, const Vector<N, float>& b, float e = 0.0001){
    return vector_near(a, b, e);
}

//vector double near
template <size_t N>
inline bool vector_near_equal(const Vector<N, double>& a, const Vector<N, double>& b, double e = 0.0000001){
    return vector_near(a, b, e);
}

// vector clamp :Create a copy of this vector, with its magnitude/size/length clamped between Min and Max.
template <size_t N, typename T>
inline Vector<N, T> vector_clamp(const Vector<N, T>& a, T minX = 0, T maxx = 1){
    Vector<V, T> b;
    for(size_t i = 0; i < N; i++){
        T x = (a[i] < minx) ? minx : a[i];
        b[i] = (x > maxx) ? maxx : x;  
    }
    return b;
}

// output to text stream
template <size_t N, typename T>
inline std::ostream& operator << (std::ostream& os, const Vector<N, T>& a){
    os << "[";
    for(size_t i = 0; i < N; i++){
        os << a[i];
        if(i < N - 1) os << ", ";
    }
    os << "]";
    return os;
}

// output to string
template <size_t N, typename T>
inline std::string vector_repr(const Vector<N, T>& a){
    std::stringstream ss;
    ss << a;
    return ss.str();
}

// mathlab Matrix
template <size_t ROW, size_t COL, typename T> struct Matrix {
    T m[ROW][COL];
    inline Matrix() {}
    inline Matrix(const Matrix<ROW, COL, T>& src){
        for (sizt_t r = 0; r < ROW; r++){
            for(sizt_t c = 0; c < COL; c++)
                m[r][c] = src.m[r][c];
        }
    }
    inline Matrix(const std::initializer_list<Vector<COL, T>> &u){
        auto it = u.begin();
        for(size_t i = 0; i < ROW; i++) SetRow(i, *it++);
    }
    inline const T* operator [] (size_t row) const {assert(row < ROW); return m[row];}
    inline T* operator [] (size_t row) {assert (row < ROW); return m[row];}

    //get a row
    inline Vector<COL, T> Row(size_t row) const{
        assert(row < ROW);
        Vector<COL, T> a;
        for(size_t i = 0; i < COL; i++ ) a[i] = m[row][i];
        return a;
    }

    //get a col
    inline Vector<ROW, T> Col(size_t col) const{
        assert(col < COL);
        Vector<ROW, T> a;
        for(size_t i = 0; i < ROW; i++) a[i] = m[i][col];
        return a;
    }

    //set a row
    inline void SetRow(size_t row, const Vector<COL, T>& a){
        assert(row < ROW);
        for(size_t i = 0; i < COL; i++) m[row][i] = a[i];
    }

    //set a col
    inline void SetCol (szie_t col, const Vector<ROW, T>& a){
        assert(col < COL);
        for(size_t i = 0; i < ROW; i++) m[i][col] = a[i];
    }

    //get Matrix<ROW - 1, COL - 1, T>
    inline Matrix<ROW - 1, COL - 1, T> GetMinor (size_t row, size_t col) const{
        Matrix<ROW - 1, COL - 1, T> ret;
        for(size_t r = 0; r < ROW - 1; r++){
            for(size_t c = 0; c < COL - 1; c++)
                ret.m[r][c] = m[r < row ? r : r+1][c < col ? c : c + 1];
        }
        return ret;
    }

    //get transpose of a matrix
    inline Matrix<COL, ROW, T> Transpose() const{
        Matrix<COL, ROW, T> ret;
        for(size_t r = 0; r < ROW; r++){
            for(size_t c = 0; c < COL; c++)
                ret.m[c][r] = m[r][c];
        }
        return ret;
    }

    // zero matrix
    inline static Matrix<ROW, COL, T> GetZero() {
		Matrix<ROW, COL, T> ret;
		for (size_t r = 0; r < ROW; r++) {
			for (size_t c = 0; c < COL; c++) 
				ret.m[r][c] = 0;
		}
		return ret;
	}

    // Identity matrix
    inline static Matrix<ROW, COL, T> GetIdentity(){
        Matrix<ROW, COL, T> ret;
        for(size_t r = 0; r < ROW; r++) {
            for(size_t c = 0; c < COL; c++)
                ret.m[r][c] = (r == c) ? 1 : 0;
        }
        return ret;
    }
};

// mathlab matrix functions
template<size_t ROW, size_t COL, typename T>
inline bool operator == (const Matrix<ROW, COL, T>& a, const Matrix<ROW, COL, T>& b){
    for(size_t r = 0; r < ROW; r++){
        for(size_t  c = 0; c < COL; c++){
            if(a.m[r][c] != b.m[r][c]) return false;
        }
    }
    return true;
}

template<size_t ROW, size_t COL, typename T>
inline bool operator != (const Matrix<ROW, COL, T>& a, const Matrix<ROW, COL, T>& b){
    return !(a == b);
}

template<size_t ROW, size_t COL, typename T>
inline Matrix<ROW, COL, T> operator + (const Matrix<ROW, COL, T>& src){
    return src;
}
template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator - (const Matrix <ROW, COL, T>& src){
    Matrix <ROW, COL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < COL; c++)
            out.m[r][c] = -src.m[r][c]
    }
    return out;
}

template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator + (const Matrix <ROW, COL, T>& a, const Matrix <ROW, COL, T>& b){
    Matrix <ROW, COL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < COL; c++)
            out.m[r][c] = a.m[r][c] + b.m[r][c];
    }
    return out;
}

template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator - (const Matrix <ROW, COL, T>& a, const Matrix <ROW, COL, T>& b){
    Matrix <ROW, COL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < COL; c++)
            out.m[r][c] = a.m[r][c] - b.m[r][c];
    }
    return out;
}

template <size_t ROW, size_t COL, size_t NEWCOL, typename T>
inline Matrix <ROW, NEWCOL, T> operator * (const Matrix <ROW, COL, T>& a, const Matrix <COL, NEWCOL, T>& b){
    Matrix <ROW, NEWCOL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < NEWCOL; c++)
            out.m[r][c] = vector_dot(a.Row(r), b.Col(c));
    }
    return out;
}

template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator * (const Matrix <ROW, COL, T>& a, T x){
    Matrix <ROW, COL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < COL; c++)
            out.m[r][c] = a.m[r][c] * x;
    }
    return out;
}

template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator / (const Matrix <ROW, COL, T>& a, T x){
    Matrix <ROW, COL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < COL; c++)
            out.m[r][c] = a.m[r][c] / x;
    }
    return out;
}

template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator * (T x,const Matrix <ROW, COL, T>& a){
    return (a * x);
}

template <size_t ROW, size_t COL, typename T>
inline Matrix <ROW, COL, T> operator / (T x, const Matrix <ROW, COL, T>& a){
    Matrix <ROW, COL, T> out;
    for(size_t r = 0; r < ROW; r++){
        for(size_t c = 0; c < COL; c++)
            out.m[r][c] = x / a.m[r][c];
    }
    return out;
}

template<size_t ROW, size_t COL, typename T>
inline Vector<COL, T> operator * (const Vector<ROW, T>& a, const Matrix<ROW, COL, T>& m) {
	Vector<COL, T> b;
	for (size_t i = 0; i < COL; i++) 
		b[i] = vector_dot(a, m.Col(i));
	return b;
}

template<size_t ROW, size_t COL, typename T>
inline Vector<ROW, T> operator * (const Matrix<ROW, COL, T>& m, const Vector<COL, T>& a) {
	Vector<ROW, T> b;
	for (size_t i = 0; i < ROW; i++) 
		b[i] = vector_dot(a, m.Row(i));
	return b;
}


//行列式和逆矩阵

//行列式求值 ： 一阶
template <typename T>
inline T matrix_det(const Matrix<1, 1, T> &m){
    return m[0][0];
}

//二阶
template<typename T>
inline T matrix_det(const Matrix<2, 2, T> &m) {
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

//多阶
template<size_t N, typename T>
inline T matrix_det(const Matrix<N, N, T> &m) {
	T sum = 0;
	for (size_t i = 0; i < N; i++) sum += m[0][i] * matrix_cofactor(m, 0, i);
	return sum;
}

// 余子式：一阶
template<typename T>
inline T matrix_cofactor(const Matrix<1, 1, T> &m, size_t row, size_t col) {
	return 0;
}

// 多阶余子式：即删除特定行列的子式的行列式值
template<size_t N, typename T>
inline T matrix_cofactor(const Matrix<N, N, T> &m, size_t row, size_t col) {
	return matrix_det(m.GetMinor(row, col)) * (((row + col) % 2)? -1 : 1);
}

// 伴随矩阵：即余子式矩阵的转置
template<size_t N, typename T>
inline Matrix<N, N, T> matrix_adjoint(const Matrix<N, N, T> &m) {
	Matrix<N, N, T> ret;
	for (size_t j = 0; j < N; j++) {
		for (size_t i = 0; i < N; i++) ret[j][i] = matrix_cofactor(m, i, j);
	}
	return ret;
}

// 求逆矩阵：使用伴随矩阵除以行列式的值得到
template<size_t N, typename T>
inline Matrix<N, N, T> matrix_invert(const Matrix<N, N, T> &m) {
	Matrix<N, N, T> ret = matrix_adjoint(m);
	T det = vector_dot(m.Row(0), ret.Col(0));
	return ret / det;
}

// 输出到文本流
template<size_t ROW, size_t COL, typename T>
inline std::ostream& operator << (std::ostream& os, const Matrix<ROW, COL, T>& m) {
	for (size_t r = 0; r < ROW; r++) {
		Vector<COL, T> row = m.Row(r);
		os << row << std::endl;
	}
	return os;
}

//functions
template <typename T> inline T Abs(T x) {return (x < 0) ? (-x) : x;}
template <typename T> inline T Max(T x, T y) {return (x < y) ? y : x;}
template <typename T> inline T Min(T x, T y) {return (x > y) ? y : x;}
template <typename T> inline bool NearEqual(T x, T y, T error) {return (Abs(x - y) < error);}
template <typename T> inline T Between(T xmin, T xmax, T x) {return Min(Max(xmin, x), xmax);}
//截取[0 , 1]的范围
template <typename T> inline T Saturate(T x) {return Between (0, 1 ,x);}

typedef Vector<2, float>  Vec2f;
typedef Vector<2, double> Vec2d;
typedef Vector<2, int>    Vec2i;
typedef Vector<3, float>  Vec3f;
typedef Vector<3, double> Vec3d;
typedef Vector<3, int>    Vec3i;
typedef Vector<4, float>  Vec4f;
typedef Vector<4, double> Vec4d;
typedef Vector<4, int>    Vec4i;

typedef Matrix<4, 4, float> Mat4x4f;
typedef Matrix<3, 3, float> Mat3x3f;
typedef Matrix<4, 3, float> Mat4x3f;
typedef Matrix<3, 4, float> Mat3x4f;

//3D 数学运算

