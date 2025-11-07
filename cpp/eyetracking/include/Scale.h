#pragma once
#include <opencv2/opencv.hpp>

namespace vision {
	namespace scale {
		// Default target height for processing/display
		static constexpr int kDefaultHeight = 324;

		inline double computeScaleForHeight(int srcHeight, int desiredHeight = kDefaultHeight) {
			if (srcHeight <= 0) return 1.0;
			return static_cast<double>(desiredHeight) / static_cast<double>(srcHeight);
		}

		inline cv::Mat resizeToHeight(const cv::Mat& src, int desiredHeight = kDefaultHeight, int interpolation = cv::INTER_AREA) {
			if (src.empty()) return src;
			double s = computeScaleForHeight(src.rows, desiredHeight);
			if (std::abs(s - 1.0) < 1e-6) return src;
			cv::Mat dst;
			cv::resize(src, dst, cv::Size(), s, s, interpolation);
			return dst;
		}

		inline void resizeToHeight(const cv::Mat& src, cv::Mat& dst, int desiredHeight = kDefaultHeight, int interpolation = cv::INTER_AREA) {
			if (src.empty()) { dst = src; return; }
			double s = computeScaleForHeight(src.rows, desiredHeight);
			if (std::abs(s - 1.0) < 1e-6) { dst = src; return; }
			cv::resize(src, dst, cv::Size(), s, s, interpolation);
		}
	}
}


