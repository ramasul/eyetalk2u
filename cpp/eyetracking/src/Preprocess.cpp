#include "Preprocess.h"

namespace vision {
	namespace pre {
		cv::Mat denoise(const cv::Mat& gray,
			int bilateralDiameter,
			double sigmaColor,
			double sigmaSpace)
		{
			cv::Mat g;
			if (gray.channels() == 3) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
			else g = gray;
			cv::Mat med;
			cv::medianBlur(g, med, 3);
			cv::Mat out;
			cv::bilateralFilter(med, out, bilateralDiameter, sigmaColor, sigmaSpace, cv::BORDER_REPLICATE);
			return out;
		}

		cv::Mat clahe(const cv::Mat& gray, double clipLimit, cv::Size tileGrid)
		{
			cv::Mat g;
			if (gray.channels() == 3) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
			else g = gray;
			cv::Ptr<cv::CLAHE> c = cv::createCLAHE(clipLimit, tileGrid);
			cv::Mat out;
			c->apply(g, out);
			return out;
		}

		cv::Mat unsharpMask(const cv::Mat& gray, double amount, double sigma)
		{
			cv::Mat g;
			if (gray.channels() == 3) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
			else g = gray;
			cv::Mat blur;
			cv::GaussianBlur(g, blur, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
			cv::Mat out;
			cv::addWeighted(g, 1.0 + amount, blur, -amount, 0, out);
			return out;
		}

		cv::Mat enhanceForPupil(const cv::Mat& gray,
			double claheClip,
			cv::Size tileGrid,
			int bilateralDiameter,
			double sigmaColor,
			double sigmaSpace,
			double unsharpAmount,
			double unsharpSigma)
		{
			cv::Mat g;
			if (gray.channels() == 3) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
			else g = gray;
			// Fast denoise: median + bilateral (no NLMeans)
			cv::Mat d = denoise(g, bilateralDiameter, sigmaColor, sigmaSpace);
			// Local contrast (small tiles for speed)
			cv::Mat c = clahe(d, claheClip, tileGrid);
			// Light unsharp to emphasize boundaries
			cv::Mat u = unsharpMask(c, unsharpAmount, unsharpSigma);
			return u;
		}
	}
}


