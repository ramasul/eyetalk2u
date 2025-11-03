#pragma once

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

// Simple calibration routine:
// - Renders an n x n grid of points on a target image of size (width,height) with margins m
// - For each target point: prompts user, waits countdown, collects pupil positions for t seconds, and averages them
// - Returns pairwise mapping of target points (original) to measured points (captured)
namespace vision {
	namespace calibration {
		struct Poly2 {
			double a[6]; // U = sum a_i * [1, x, y, x^2, x*y, y^2]
			double b[6]; // V = sum b_i * [1, x, y, x^2, x*y, y^2]
		};
		class Calibrator {
		public:
			// Runs calibration and returns vector of ((original_x, original_y), (captured_x, captured_y))
			// height, width: target image size in pixels; m: margin in pixels; n: grid dimension; t: capture interval (seconds)
			std::vector<std::pair<cv::Point2f, cv::Point2f>> run(int height, int width, int m, int n, double t);

			// Fit 2nd-order polynomial mapping from measured (captured) to target (original)
			// pairs: ((target), (captured))
			static Poly2 fit_poly2(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& pairs);

		private:
			cv::Mat make_target(int height, int width, int m, int n, const cv::Point& highlight) const;
			double now_seconds() const;
		};
	}
}


