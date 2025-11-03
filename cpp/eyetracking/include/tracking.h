#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "callibrate.h"

namespace vision {
	namespace tracking {
		class Tracker {
		public:
			Tracker(const calibration::Poly2& model, int target_height, int target_width);
			int run(int camera_index = 0);
		private:
			calibration::Poly2 model;
			int height;
			int width;
			cv::Point2f map_to_screen(const cv::Point2f& p) const;
		};
	}
}


