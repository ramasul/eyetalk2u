#pragma once
#pragma once
#include <opencv2/core.hpp>

namespace vision {
    namespace normalize {
        constexpr int NORM_INF = 1;
        constexpr int NORM_L1 = 2;
        constexpr int NORM_L2 = 4;
        constexpr int NORM_MINMAX = 32;

        void normalize(const cv::Mat& src,
            cv::Mat& dst,
            double alpha = -1, // if -1, auto set based on type
            double beta = -1,
            int norm_type = NORM_L2);

        cv::Mat normalize(const cv::Mat& src,
            double alpha = -1,
            double beta = -1,
            int norm_type = NORM_L2);
    }
}
