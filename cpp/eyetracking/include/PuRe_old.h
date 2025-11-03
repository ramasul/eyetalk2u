#pragma once

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace pure_old {
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
		double angular_spread = 0;
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

	class Detector {
	public:
		Result detect(const cv::Mat& gray_img, cv::Mat* debug_color_img = nullptr);
		Parameters params;

	private: 
		// Preprocessing
		cv::Mat orig_img;
		cv::Mat debug_img;
		bool debug = false; // Set to true to enable debug image generation.
		double scaling_factor;
		bool preprocess(const cv::Mat& input_img);
		void postprocess(Result& final_result, const cv::Mat& input_img, cv::Mat* debug_color_img);

	private:
		//Edge detection and Morphological Manipulation
		cv::Mat edge_img;
		void detect_edges();

	private: 
		// Edge Segment Selection and Confidence Measure
		std::vector<Segment> segments;
		std::vector<Result> candidates;
		double min_pupil_diameter, max_pupil_diameter;
		void select_edge_segments();

		void evaluate_segment(const Segment& segment, Result& result) const;
		bool segment_large_enough(const Segment& segment) const;
		bool segment_diameter_valid(const Segment& segment) const;
		bool segment_curvature_valid(const Segment& segment) const;
		bool axes_ratio_is_invalid(double ratio) const;
		bool fit_ellipse(const Segment& segment, Result& result) const;
		bool segment_mean_in_ellipse(const Segment& segment, const Result& result) const;

		Confidence calculate_confidence(const Segment& segment, const Result& result) const;
		double angular_edge_spread(const Segment& segment, const Result& result) const;
		double ellipse_outline_contrast(const Result& result) const;

	private: 
		// Conditional Segment Combination
		void combine_segments();
		bool proper_intersection(const cv::Rect& r1, const cv::Rect& r2) const;
		Segment merge_segments(const Segment& seg1, const Segment& seg2) const;

	private:
		Result select_final_segment();

	};
}