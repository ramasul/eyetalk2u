#pragma once
#include <opencv2/opencv.hpp>

namespace vision {
	namespace blur {
		void GaussianBlur(const cv::Mat& src, cv::Mat& dst, double sigma, int radius = 0, int borderType = cv::BORDER_REPLICATE);
		cv::Mat GaussianBlur(const cv::Mat& src, double sigma, int radius = 0, int borderType = cv::BORDER_REPLICATE);
		cv::Mat FFTGaussianBlur(const cv::Mat& src, double sigma);
	}
}