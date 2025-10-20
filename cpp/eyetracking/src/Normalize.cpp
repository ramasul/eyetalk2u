#include "Normalize.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace vision {
    namespace normalize {
        // Helper: set default alpha/beta
        static void setAlphaBeta(const cv::Mat& src, double& alpha, double& beta, int norm_type)
        {
            if (alpha >= 0 && beta >= 0) return;

            if (norm_type == NORM_MINMAX) {
                switch (src.depth()) {
                case CV_8U:  alpha = 0; beta = 255; break;
                case CV_16U: alpha = 0; beta = 65535; break;
                case CV_32F: alpha = 0.0; beta = 1.0; break;
                case CV_64F: alpha = 0.0; beta = 1.0; break;
                default:     alpha = 0.0; beta = 1.0; break;
                }
            }
            else {
                alpha = (alpha < 0) ? 1.0 : alpha;
                beta = 0.0;
            }
        }

        // Internal template function
        template<typename T>
        static void normalizeImpl(const cv::Mat& src,
            cv::Mat& dst,
            double alpha,
            double beta,
            int norm_type)
        {
            dst.create(src.size(), src.type());

            double minVal = 0, maxVal = 0, sum = 0, sumSq = 0;

            if (norm_type == NORM_MINMAX) {
                cv::minMaxLoc(src, &minVal, &maxVal);
                if (std::abs(maxVal - minVal) < std::numeric_limits<double>::epsilon()) {
                    dst.setTo(cv::Scalar(alpha));
                    return;
                }
            }
            else if (norm_type == NORM_INF) {
                maxVal = 0;
                for (int i = 0; i < src.rows; i++) {
                    const T* ptr = src.ptr<T>(i);
                    for (int j = 0; j < src.cols * src.channels(); j++)
                        maxVal = std::max(maxVal, std::abs(static_cast<double>(ptr[j])));
                }
                if (maxVal < std::numeric_limits<double>::epsilon()) {
                    dst.setTo(cv::Scalar(0));
                    return;
                }
            }
            else if (norm_type == NORM_L1 || norm_type == NORM_L2) {
                for (int i = 0; i < src.rows; i++) {
                    const T* ptr = src.ptr<T>(i);
                    for (int j = 0; j < src.cols * src.channels(); j++) {
                        double val = static_cast<double>(ptr[j]);
                        if (norm_type == NORM_L1) sum += std::abs(val);
                        else sumSq += val * val;
                    }
                }
                if ((norm_type == NORM_L1 && sum < std::numeric_limits<double>::epsilon()) ||
                    (norm_type == NORM_L2 && sumSq < std::numeric_limits<double>::epsilon())) {
                    dst.setTo(cv::Scalar(0));
                    return;
                }
            }

            for (int i = 0; i < src.rows; i++) {
                const T* srcPtr = src.ptr<T>(i);
                T* dstPtr = dst.ptr<T>(i);
                for (int j = 0; j < src.cols * src.channels(); j++) {
                    double val = static_cast<double>(srcPtr[j]);
                    double normVal = 0.0;
                    switch (norm_type) {
                    case NORM_MINMAX:
                        normVal = (val - minVal) / (maxVal - minVal) * (beta - alpha) + alpha;
                        break;
                    case NORM_INF:
                        normVal = val / maxVal * alpha;
                        break;
                    case NORM_L1:
                        normVal = val / sum * alpha;
                        break;
                    case NORM_L2:
                        normVal = val / std::sqrt(sumSq) * alpha;
                        break;
                    }
                    dstPtr[j] = static_cast<T>(normVal);
                }
            }
        }

        // Main type-dispatching normalize
        void normalize(const cv::Mat& src,
            cv::Mat& dst,
            double alpha,
            double beta,
            int norm_type)
        {
            CV_Assert(!src.empty());
            setAlphaBeta(src, alpha, beta, norm_type);

            switch (src.depth()) {
            case CV_8U:  normalizeImpl<uchar>(src, dst, alpha, beta, norm_type); break;
            case CV_16U: normalizeImpl<unsigned short>(src, dst, alpha, beta, norm_type); break;
            case CV_16S: normalizeImpl<short>(src, dst, alpha, beta, norm_type); break;
            case CV_32S: normalizeImpl<int>(src, dst, alpha, beta, norm_type); break;
            case CV_32F: normalizeImpl<float>(src, dst, alpha, beta, norm_type); break;
            case CV_64F: normalizeImpl<double>(src, dst, alpha, beta, norm_type); break;
            default: CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported Mat depth");
            }
        }

        cv::Mat normalize(const cv::Mat& src,
            double alpha,
            double beta,
            int norm_type)
        {
            cv::Mat dst;
            normalize(src, dst, alpha, beta, norm_type);
            return dst;
        }

    }
}
