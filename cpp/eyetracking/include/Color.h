#pragma once
#include <opencv2/opencv.hpp>

namespace vision {
    namespace color {
        void BGR2Gray(const cv::Mat& src, cv::Mat& dst);
		cv::Mat BGR2Gray(const cv::Mat& src);
    }
}