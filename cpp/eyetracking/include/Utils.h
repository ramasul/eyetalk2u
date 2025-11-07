#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <cmath>      // for sin(), cos(), M_PI
#include <algorithm>  // for std::swap
#include <omp.h>

typedef int64_t Timestamp;
extern Timestamp maxTimestamp;

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

template<typename T> double ED(T p1, T p2) { return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)); }

#define CV_BLUE 	cv::Scalar(0xff,0xb0,0x00)
#define CV_GREEN 	cv::Scalar(0x03,0xff,0x76)
#define CV_RED 		cv::Scalar(0x00,0x3d,0xff)
#define CV_YELLOW	cv::Scalar(0x00,0xea,0xff)
#define CV_CYAN		cv::Scalar(0xff,0xff,0x18)
#define CV_MAGENT   cv::Scalar(0x81,0x40,0xff)
#define CV_WHITE	cv::Scalar(0xff,0xff,0xff)
#define CV_BLACK	cv::Scalar(0x00,0x00,0x00)
#define CV_ALMOST_BLACK	cv::Scalar(0x01,0x01,0x01)

namespace Utils {
    cv::Point3f estimateMarkerCenter(const std::vector<cv::Point2f> corners);
}