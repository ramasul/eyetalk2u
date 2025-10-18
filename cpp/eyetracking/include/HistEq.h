#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision {
	namespace histeq {
            /**
         * @brief Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
         *
         * @param src Input grayscale image (CV_8UC1)
         * @param dst Output image (CV_8UC1)
         * @param clipLimit Maximum contrast clip limit (>1 means contrast limited)
         * @param tileGridSize Size of grid, e.g., cv::Size(8,8)
         */
        void CLAHE(const cv::Mat& src, cv::Mat& dst, double clipLimit, cv::Size tileGridSize);
		cv::Mat CLAHE(const cv::Mat& src, double clipLimit, cv::Size tileGridSize);
	}
}