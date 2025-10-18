#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <cmath>      // for sin(), cos(), M_PI
#include <algorithm>  // for std::swap
#include <omp.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Complex = std::complex<double>;

template<typename T>
T clamp(T val, T minVal, T maxVal) {
    if (val < minVal) return minVal;
    if (val > maxVal) return maxVal;
    return val;
}

namespace Utils {
}