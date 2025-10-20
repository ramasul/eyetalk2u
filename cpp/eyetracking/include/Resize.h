#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vision {
    namespace resize {
        constexpr int INTER_NEAREST = 0;
        constexpr int INTER_LINEAR = 1;
        constexpr int INTER_AREA = 2;
		constexpr int INTER_CUBIC = 3;
        void resize(const cv::Mat& src, cv::Mat& dst, cv::Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);
        cv::Mat resize(const cv::Mat& src, cv::Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);
    }
}