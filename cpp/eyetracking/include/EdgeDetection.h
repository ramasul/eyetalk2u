#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
//#include <climits>

namespace vision {
	namespace canny {
		cv::Mat canny(const cv::Mat& in, bool blurImage, bool useL2, int bins, float nonEdgePixelsRatio, float lowHighThresholdRatio);
	}
}