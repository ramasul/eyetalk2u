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

namespace Utils {
    void fft1D(std::vector<Complex>& a, bool invert);
    void fft2D(std::vector<std::vector<Complex>>& data, bool invert);
    std::vector<double> gaussianKernel1D(int ksize, double sigma);
    cv::Mat separableGaussian(const cv::Mat& gray, int ksize = 11, double sigma = 3.0);
    cv::Mat fftGaussianBlur(const cv::Mat& gray, double sigma);
}