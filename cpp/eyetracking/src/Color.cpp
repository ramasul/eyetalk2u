#include "Color.h"
#include "Utils.h"

namespace vision {
    namespace color {
        void BGR2Gray(const cv::Mat& src, cv::Mat& dst)
        {
            if (src.empty() || src.channels() != 3)
            {
                throw std::invalid_argument("Input image must be a non-empty 3-channel BGR image.");
            }

            dst.create(src.rows, src.cols, CV_8UC1);

            for (int y = 0; y < src.rows; ++y)
            {
                const uchar* srcRow = src.ptr<uchar>(y);
                uchar* dstRow = dst.ptr<uchar>(y);

                for (int x = 0; x < src.cols; ++x)
                {
                    uchar B = srcRow[x * 3 + 0];
                    uchar G = srcRow[x * 3 + 1];
                    uchar R = srcRow[x * 3 + 2];

                    dstRow[x] = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
                }
            }
        }
        cv::Mat BGR2Gray(const cv::Mat& src)
        {
            cv::Mat dst;
            BGR2Gray(src, dst);
            return dst;
		}
    }
}
