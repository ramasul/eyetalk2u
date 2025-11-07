#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "callibrate.h"
#include "PupilDetector.h"

namespace vision {
	namespace tracking {
		class Tracker {
		public:
			Tracker(const calibration::Poly2& model, int target_height, int target_width, vision::detection::PupilDetector& detector);

			template<typename Src>
			int run(Src camera_index = 0, bool useHaar = false);
		private:
			calibration::Poly2 model;
			int height;
			int width;
			vision::detection::PupilDetector& detector;
			cv::Point2f map_to_screen(const cv::Point2f& p) const;
		};
	}
}


