#pragma once
#include <opencv2/opencv.hpp>

namespace vision {
	namespace edge {
		void filterEdges(cv::Mat& edges);
		cv::Mat filterEdges(const cv::Mat& edges, cv::Mat& dst);
	}
}