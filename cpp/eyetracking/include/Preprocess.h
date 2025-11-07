#pragma once
#include <opencv2/opencv.hpp>

namespace vision {
	namespace pre {
		// Strong denoise while preserving edges (bilateral/NLMeans alternative)
		cv::Mat denoise(const cv::Mat& gray,
			int bilateralDiameter = 7,
			double sigmaColor = 50.0,
			double sigmaSpace = 7.0);

		// Local contrast enhancement with CLAHE
		cv::Mat clahe(const cv::Mat& gray,
			double clipLimit = 3.0,
			cv::Size tileGrid = cv::Size(8,8));

		// Unsharp masking to accentuate pupil-iris boundary
		cv::Mat unsharpMask(const cv::Mat& gray,
			double amount = 1.2,
			double sigma = 1.0);

		// Full pipeline to minimize noise and maximize boundary contrast
		cv::Mat enhanceForPupil(const cv::Mat& gray,
			double claheClip = 3.0,
			cv::Size tileGrid = cv::Size(8,8),
			int bilateralDiameter = 7,
			double sigmaColor = 50.0,
			double sigmaSpace = 7.0,
			double unsharpAmount = 1.2,
			double unsharpSigma = 1.0);
	}
}


