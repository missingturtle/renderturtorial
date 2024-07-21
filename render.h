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
