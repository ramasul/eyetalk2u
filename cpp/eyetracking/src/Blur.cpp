#include "Blur.h"
#include "Utils.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace vision {
	namespace blur {
        static void makeGaussianKernel(double sigma, int radius, std::vector<float>& kernel) {
            int len = 2 * radius + 1;
            kernel.resize(len);
            const double s2 = 2.0 * sigma * sigma;
            const double inv_sqrt = 1.0 / (std::sqrt(M_PI * s2));
            double sum = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                double v = std::exp(-(i * i) / s2) * inv_sqrt; // gaussian (unnormalized)
                kernel[i + radius] = static_cast<float>(v);
                sum += v;
            }
            // normalize
            for (int i = 0; i < len; ++i) kernel[i] = static_cast<float>(kernel[i] / sum);
        }

        void GaussianBlur(const cv::Mat& src, cv::Mat& dst, double sigma, int radius, int borderType) {
            if (src.empty()) {
                dst = src.clone();
                return;
            }
            if (sigma <= 0.0) { // no-op
                dst = src.clone();
                return;
            }

            CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
            CV_Assert(src.channels() == 1 || src.channels() == 3);

            // compute radius
            if (radius <= 0) {
                radius = static_cast<int>(std::ceil(3.0 * sigma));
                if (radius < 1) radius = 1;
            }
            const int klen = 2 * radius + 1;

            // make kernel
            std::vector<float> kernel;
            makeGaussianKernel(sigma, radius, kernel);

            // convert to float for processing
            cv::Mat srcF;
            src.convertTo(srcF, CV_32F);

            const int rows = src.rows;
            const int cols = src.cols;
            const int channels = src.channels();

            // temporary buffer after horizontal pass
            cv::Mat tmp = cv::Mat::zeros(rows, cols, CV_32FC(channels)); // CV_32F with channels

            // Horizontal pass (parallel over rows)
#pragma omp parallel for schedule(static)
            for (int y = 0; y < rows; ++y) {
                const float* srcRow = srcF.ptr<float>(y);
                float* tmpRow = tmp.ptr<float>(y);
                for (int x = 0; x < cols; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        double acc = 0.0;
                        // kernel convolution
                        for (int k = -radius; k <= radius; ++k) {
                            int ix = x + k;
                            // border handling - only replicate implemented (fast clamp)
                            if (borderType == cv::BORDER_REPLICATE) {
                                ix = (ix < 0) ? 0 : (ix >= cols ? cols - 1 : ix);
                            }
                            else if (borderType == cv::BORDER_REFLECT || borderType == cv::BORDER_REFLECT_101) {
                                if (ix < 0) ix = -ix - (borderType == cv::BORDER_REFLECT_101 ? 1 : 0);
                                if (ix >= cols) ix = (2 * cols - ix - 1) - (borderType == cv::BORDER_REFLECT_101 ? 1 : 0);
                                ix = clamp(ix, 0, cols - 1);
                            }
                            else {
                                ix = clamp(ix, 0, cols - 1);
                            }
                            acc += srcRow[ix * channels + c] * kernel[k + radius];
                        }
                        tmpRow[x * channels + c] = static_cast<float>(acc);
                    }
                }
            }

            // Vertical pass (parallel over rows)
            cv::Mat dstF = cv::Mat::zeros(rows, cols, CV_32FC(channels));
#pragma omp parallel for schedule(static)
            for (int y = 0; y < rows; ++y) {
                float* dstRow = dstF.ptr<float>(y);
                for (int x = 0; x < cols; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        double acc = 0.0;
                        for (int k = -radius; k <= radius; ++k) {
                            int iy = y + k;
                            if (borderType == cv::BORDER_REPLICATE) {
                                iy = (iy < 0) ? 0 : (iy >= rows ? rows - 1 : iy);
                            }
                            else if (borderType == cv::BORDER_REFLECT || borderType == cv::BORDER_REFLECT_101) {
                                if (iy < 0) iy = -iy - (borderType == cv::BORDER_REFLECT_101 ? 1 : 0);
                                if (iy >= rows) iy = (2 * rows - iy - 1) - (borderType == cv::BORDER_REFLECT_101 ? 1 : 0);
                                iy = clamp(iy, 0, rows - 1);
                            }
                            else {
                                iy = clamp(iy, 0, rows - 1);
                            }
                            const float* tmpRow = tmp.ptr<float>(iy);
                            acc += tmpRow[x * channels + c] * kernel[k + radius];
                        }
                        dstRow[x * channels + c] = static_cast<float>(acc);
                    }
                }
            }
            dstF.convertTo(dst, src.type());
        }

        cv::Mat GaussianBlur(const cv::Mat& src, double sigma, int radius, int borderType) {
            cv::Mat dst;
            GaussianBlur(src, dst, sigma, radius, borderType);
            return dst;
		}
	}
}