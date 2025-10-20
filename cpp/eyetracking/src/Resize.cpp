#include "Resize.h"
#include "Utils.h"
#include <cmath>

namespace vision {
    namespace resize {

        inline double cubicWeight(double x) {
            x = std::abs(x);
            if (x <= 1)
                return 1 - 2 * x * x + x * x * x;
            else if (x < 2)
                return 4 - 8 * x + 5 * x * x - x * x * x;
            return 0;
        }

        void resize(const cv::Mat& src,
            cv::Mat& dst,
            cv::Size dsize,
            double fx,
            double fy,
            int interpolation)
        {
            CV_Assert(!src.empty());

            // Compute scale factors
            if (dsize.width == 0 && dsize.height == 0) {
                dsize.width = cvRound(src.cols * fx);
                dsize.height = cvRound(src.rows * fy);
            }
            else if (fx == 0 && fy == 0) {
                fx = static_cast<double>(dsize.width) / src.cols;
                fy = static_cast<double>(dsize.height) / src.rows;
            }

            CV_Assert(fx > 0 && fy > 0);

            dst.create(dsize, src.type());
            int channels = src.channels();

            for (int y = 0; y < dsize.height; y++) {
                double sy = (y + 0.5) / fy - 0.5;  // Map dst -> src
                int y0 = std::floor(sy);
                int y1 = std::min(y0 + 1, src.rows - 1);
                sy -= y0;
                if (y0 < 0) { y0 = 0; sy = 0; }

                for (int x = 0; x < dsize.width; x++) {
                    double sx = (x + 0.5) / fx - 0.5;
                    int x0 = std::floor(sx);
                    int x1 = std::min(x0 + 1, src.cols - 1);
                    sx -= x0;
                    if (x0 < 0) { x0 = 0; sx = 0; }

                    if (interpolation == INTER_NEAREST) {
                        int nearestY = clamp(int(std::round((y + 0.5) / fy - 0.5)), 0, src.rows - 1);
                        int nearestX = clamp(int(std::round((x + 0.5) / fx - 0.5)), 0, src.cols - 1);

                        if (channels == 1) {
                            dst.at<uchar>(y, x) = src.at<uchar>(nearestY, nearestX);
                        }
                        else if (channels == 3) {
                            dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(nearestY, nearestX);
                        }
                    }
                    else if (interpolation == INTER_LINEAR) {
                        if (channels == 1) {
                            double v00 = src.at<uchar>(y0, x0);
                            double v01 = src.at<uchar>(y0, x1);
                            double v10 = src.at<uchar>(y1, x0);
                            double v11 = src.at<uchar>(y1, x1);

                            double val0 = v00 * (1 - sx) + v01 * sx;
                            double val1 = v10 * (1 - sx) + v11 * sx;
                            double val = val0 * (1 - sy) + val1 * sy;

                            dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
                        }
                        else if (channels == 3) {
                            for (int c = 0; c < 3; c++) {
                                double v00 = src.at<cv::Vec3b>(y0, x0)[c];
                                double v01 = src.at<cv::Vec3b>(y0, x1)[c];
                                double v10 = src.at<cv::Vec3b>(y1, x0)[c];
                                double v11 = src.at<cv::Vec3b>(y1, x1)[c];

                                double val0 = v00 * (1 - sx) + v01 * sx;
                                double val1 = v10 * (1 - sx) + v11 * sx;
                                double val = val0 * (1 - sy) + val1 * sy;

                                dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(val);
                            }
                        }
                    }
                    else if (interpolation == INTER_AREA) {
                        // Weighted average of the pixels in the source area
                        if (channels == 1) {
                            double sum = 0.0;
                            int count = 0;
                            for (int yy = y0; yy < y1; yy++) {
                                for (int xx = x0; xx < x1; xx++) {
                                    sum += src.at<uchar>(yy, xx);
                                    count++;
                                }
                            }
                            dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum / count);
                        }
                        else if (channels == 3) {
                            double sum[3] = { 0,0,0 };
                            int count = 0;
                            for (int yy = y0; yy < y1; yy++) {
                                for (int xx = x0; xx < x1; xx++) {
                                    for (int c = 0; c < 3; c++) sum[c] += src.at<cv::Vec3b>(yy, xx)[c];
                                    count++;
                                }
                            }
                            for (int c = 0; c < 3; c++) dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(sum[c] / count);
                        }
                    }
                    else if (interpolation == INTER_CUBIC) {
                        // Bicubic 4x4 neighborhood
                        for (int c = 0; c < channels; c++) {
                            double val = 0.0;
                            for (int m = -1; m <= 2; m++) {
                                int ym = clamp(y0 + m, 0, src.rows - 1);
                                double wy = cubicWeight(m - sy);
                                for (int n = -1; n <= 2; n++) {
                                    int xn = clamp(x0 + n, 0, src.cols - 1);
                                    double wx = cubicWeight(n - sx);
                                    if (channels == 1)
                                        val += src.at<uchar>(ym, xn) * wx * wy;
                                    else
                                        val += src.at<cv::Vec3b>(ym, xn)[c] * wx * wy;
                                }
                            }
                            if (channels == 1)
                                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
                            else
                                dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(val);
                        }
                    }
                }
            }
        }

        cv::Mat resize(const cv::Mat& src,
            cv::Size dsize,
            double fx,
            double fy,
            int interpolation)
        {
            cv::Mat dst;
            resize(src, dst, dsize, fx, fy, interpolation);
            return dst;
		}
	}
}
