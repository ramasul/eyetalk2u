#pragma once

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace pure {
	struct Parameters{
		// Either set auto_pupil_diameter = true or specify min/max.
		// If auto, read min/max for automatic values.
		bool auto_pupil_diameter = true;
		double min_pupil_diameter = 0.0;
		double max_pupil_diameter = 0.0;
	};

	struct Confidence{
		double value = 0;
		double aspect_ratio = 0;
		double angular_speed = 0;
		double outline_contrast = 0;
	};

	struct Result {
		cv::Point2f center = { 0, 0 };
		cv::Size2f axes = { 0, 0 };// width = major axis, height = minor axis
		double angle = 0;
		Confidence confidence = { 0, 0, 0, 0 };
		bool operator<(const Result& other) const {
			return confidence.value < other.confidence.value;
		}
	};

	typedef std::vector<cv::Point> Segment;

}