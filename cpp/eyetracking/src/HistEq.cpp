#include "Histeq.h"
#include "Utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace vision {
	namespace histeq {

        //------------------------------------------------------
        // Helper: clip histogram to limit contrast enhancement
        //------------------------------------------------------
        static void clipHistogram(std::vector<int>& hist, int clipLimit)
        {
            // compute total excess pixels above clip limit
            int excess = 0;
            for (int& h : hist) {
                if (h > clipLimit) {
                    excess += (h - clipLimit);
                    h = clipLimit;
                }
            }

            // redistribute excess pixels equally
            int bins = static_cast<int>(hist.size());
            int distribute = excess / bins;
            int remainder = excess % bins;

            for (int& h : hist)
                h += distribute;

            // distribute the remaining pixels one by one
            for (int i = 0; remainder > 0; ++i, --remainder)
                hist[i % bins]++;
        }

        //------------------------------------------------------
        // Helper: compute cumulative distribution (CDF)
        //------------------------------------------------------
        static std::vector<uchar> makeLUT(const std::vector<int>& hist, int totalPixels)
        {
            std::vector<uchar> lut(hist.size(), 0);
            std::vector<int> cdf(hist.size(), 0);

            // build cumulative histogram
            std::partial_sum(hist.begin(), hist.end(), cdf.begin());

            // normalize to [0, 255]
            double scale = 255.0 / totalPixels;
            for (size_t i = 0; i < cdf.size(); ++i)
                lut[i] = static_cast<uchar>(clamp(int(cdf[i] * scale), 0, 255));

            return lut;
        }

        //------------------------------------------------------
        // Helper: compute histogram for a tile
        //------------------------------------------------------
        static std::vector<int> computeHistogram(const cv::Mat& tile)
        {
            std::vector<int> hist(256, 0);
            for (int y = 0; y < tile.rows; ++y) {
                const uchar* row = tile.ptr<uchar>(y);
                for (int x = 0; x < tile.cols; ++x)
                    hist[row[x]]++;
            }
            return hist;
        }

        //------------------------------------------------------
        // Main CLAHE implementation
        //------------------------------------------------------
        void CLAHE(const cv::Mat& src, cv::Mat& dst, double clipLimit, cv::Size tileGridSize)
        {
            CV_Assert(src.type() == CV_8UC1);

            dst.create(src.size(), src.type());

            const int nx = tileGridSize.width;
            const int ny = tileGridSize.height;
            const int tileWidth = std::ceil((float)src.cols / nx);
            const int tileHeight = std::ceil((float)src.rows / ny);

            // build LUT for each tile
            std::vector<std::vector<uchar>> luts(nx * ny);
            for (int ty = 0; ty < ny; ++ty) {
                for (int tx = 0; tx < nx; ++tx) {
                    // region boundaries
                    int x0 = tx * tileWidth;
                    int y0 = ty * tileHeight;
                    int x1 = std::min(x0 + tileWidth, src.cols);
                    int y1 = std::min(y0 + tileHeight, src.rows);

                    // extract tile
                    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
                    cv::Mat tile = src(roi);

                    // histogram and clipping
                    std::vector<int> hist = computeHistogram(tile);

                    // clip limit calculation: proportional to tile size
                    int limit = std::max(1, static_cast<int>(clipLimit * tile.total() / 256.0));
                    clipHistogram(hist, limit);

                    // make LUT from clipped histogram
                    luts[ty * nx + tx] = makeLUT(hist, tile.total());
                }
            }

            // interpolation between 4 nearest LUTs (bilinear)
            for (int y = 0; y < src.rows; ++y) {
                float gy = (float)y / tileHeight - 0.5f;
                int y1 = clamp(int(std::floor(gy)), 0, ny - 1);
                int y2 = clamp(y1 + 1, 0, ny - 1);
                float dy = gy - y1;

                for (int x = 0; x < src.cols; ++x) {
                    float gx = (float)x / tileWidth - 0.5f;
                    int x1 = clamp(int(std::floor(gx)), 0, nx - 1);
                    int x2 = clamp(x1 + 1, 0, nx - 1);
                    float dx = gx - x1;

                    // pixel intensity
                    int val = src.at<uchar>(y, x);

                    // fetch from 4 LUTs and interpolate
                    uchar lu = luts[y1 * nx + x1][val];
                    uchar ru = luts[y1 * nx + x2][val];
                    uchar lb = luts[y2 * nx + x1][val];
                    uchar rb = luts[y2 * nx + x2][val];

                    // bilinear interpolation
                    float top = lu + dx * (ru - lu);
                    float bottom = lb + dx * (rb - lb);
                    float result = top + dy * (bottom - top);

                    dst.at<uchar>(y, x) = static_cast<uchar>(clamp(int(result), 0, 255));
                }
            }
        }
        cv::Mat CLAHE(const cv::Mat& src, double clipLimit, cv::Size tileGridSize) {
            cv::Mat dst;
            CLAHE(src, dst, clipLimit, tileGridSize);
            return dst;
        }
	}
}