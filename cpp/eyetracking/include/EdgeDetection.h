#pragma once
#include <opencv2/opencv.hpp>
//#include <climits>

namespace vision {
	namespace canny {
		cv::Mat canny(const cv::Mat& in, bool blurImage = false, bool useL2 = true, int bins = 64, float nonEdgePixelsRatio = 0.7f, float lowHighThresholdRatio = 0.4f);
	}
}