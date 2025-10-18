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

        using Complex = std::complex<double>;

        static inline int nextPow2(int n) {
            int p = 1;
            while (p < n) p <<= 1;
            return p;
        }

        static void bitReverse(std::vector<Complex>& a) {
            int n = static_cast<int>(a.size());
            int j = 0;
            for (int i = 1; i < n; ++i) {
                int bit = n >> 1;
                for (; j & bit; bit >>= 1)
                    j ^= bit;
                j ^= bit;
                if (i < j)
                    std::swap(a[i], a[j]);
            }
        }

        static void fft(std::vector<Complex>& a, bool invert) {
            int n = static_cast<int>(a.size());
            bitReverse(a);

            for (int len = 2; len <= n; len <<= 1) {
                double ang = 2 * M_PI / len * (invert ? -1 : 1);
                Complex wlen(std::cos(ang), std::sin(ang));

                for (int i = 0; i < n; i += len) {
                    Complex w(1);
                    for (int j = 0; j < len / 2; ++j) {
                        Complex u = a[i + j];
                        Complex v = a[i + j + len / 2] * w;
                        a[i + j] = u + v;
                        a[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }

            if (invert) {
                for (auto& x : a) x /= n;
            }
        }

        static std::vector<double> makeGaussian1D(int size, double sigma) {
            std::vector<double> kernel(size);
            double sigma2 = 2.0 * sigma * sigma;
            int half = size / 2;
            double sum = 0.0;
            for (int i = 0; i < size; ++i) {
                double x = i - half;
                kernel[i] = std::exp(-(x * x) / sigma2);
                sum += kernel[i];
            }
            for (double& k : kernel)
                k /= sum;
            return kernel;
        }

        static cv::Mat reflectPad(const cv::Mat& src, int pad) {
            cv::Mat dst;
            cv::copyMakeBorder(src, dst, pad, pad, pad, pad, cv::BORDER_REFLECT);
            return dst;
        }

        /*static void convolve1DFFT(cv::Mat& img, const std::vector<double>& kernel, bool horizontal) {
            int rows = img.rows;
            int cols = img.cols;
            int n = horizontal ? cols : rows;
            int k = static_cast<int>(kernel.size());
            int fftSize = nextPow2(n + k - 1);

            std::vector<Complex> kernelFFT(fftSize);
            for (int i = 0; i < k; ++i)
                kernelFFT[i] = kernel[i];
            for (int i = k; i < fftSize; ++i)
                kernelFFT[i] = 0.0;
            fft(kernelFFT, false);

            std::vector<Complex> A(fftSize);

            if (horizontal) {
                for (int y = 0; y < rows; ++y) {
                    for (int i = 0; i < n; ++i)
                        A[i] = static_cast<double>(img.at<uchar>(y, i));
                    for (int i = n; i < fftSize; ++i)
                        A[i] = 0.0;

                    fft(A, false);
                    for (int i = 0; i < fftSize; ++i)
                        A[i] *= kernelFFT[i];
                    fft(A, true);

                    for (int i = 0; i < n; ++i) {
                        double val = A[i].real();
                        val = clamp(val, 0.0, 255.0);
                        img.at<uchar>(y, i) = static_cast<uchar>(val);
                    }
                }
            }
            else {
                for (int x = 0; x < cols; ++x) {
                    for (int i = 0; i < n; ++i)
                        A[i] = static_cast<double>(img.at<uchar>(i, x));
                    for (int i = n; i < fftSize; ++i)
                        A[i] = 0.0;

                    fft(A, false);
                    for (int i = 0; i < fftSize; ++i)
                        A[i] *= kernelFFT[i];
                    fft(A, true);

                    for (int i = 0; i < n; ++i) {
                        double val = A[i].real();
                        val = clamp(val, 0.0, 255.0);
                        img.at<uchar>(i, x) = static_cast<uchar>(val);
                    }
                }
            }
        }*/

        static void convolve1DFFT(cv::Mat& img, const std::vector<double>& kernel, bool horizontal) {
            int rows = img.rows;
            int cols = img.cols;
            int n = horizontal ? cols : rows;
            int k = static_cast<int>(kernel.size());
            int fftSize = nextPow2(n + k - 1);

            // Precompute FFT of kernel (shared and read-only)
            std::vector<Complex> kernelFFT(fftSize);
            for (int i = 0; i < k; ++i)
                kernelFFT[i] = kernel[i];
            for (int i = k; i < fftSize; ++i)
                kernelFFT[i] = 0.0;
            fft(kernelFFT, false);

            // Parallelized loop
            if (horizontal) {
#pragma omp parallel for schedule(static)
                for (int y = 0; y < rows; ++y) {
                    std::vector<Complex> A(fftSize);  // thread-local buffer, correctly sized

                    // Copy one row
                    for (int i = 0; i < n; ++i)
                        A[i] = static_cast<double>(img.at<uchar>(y, i));
                    for (int i = n; i < fftSize; ++i)
                        A[i] = 0.0;

                    fft(A, false);
                    for (int i = 0; i < fftSize; ++i)
                        A[i] *= kernelFFT[i];
                    fft(A, true);

                    for (int i = 0; i < n; ++i) {
                        double val = A[i].real();
                        val = clamp(val, 0.0, 255.0);
                        img.at<uchar>(y, i) = static_cast<uchar>(val);
                    }
                }
            }
            else {
#pragma omp parallel for schedule(static)
                for (int x = 0; x < cols; ++x) {
                    std::vector<Complex> A(fftSize);  // thread-local buffer

                    // Copy one column
                    for (int i = 0; i < n; ++i)
                        A[i] = static_cast<double>(img.at<uchar>(i, x));
                    for (int i = n; i < fftSize; ++i)
                        A[i] = 0.0;

                    fft(A, false);
                    for (int i = 0; i < fftSize; ++i)
                        A[i] *= kernelFFT[i];
                    fft(A, true);

                    for (int i = 0; i < n; ++i) {
                        double val = A[i].real();
                        val = clamp(val, 0.0, 255.0);
                        img.at<uchar>(i, x) = static_cast<uchar>(val);
                    }
                }
            }
        }


        cv::Mat FFTGaussianBlur(const cv::Mat& src, double sigma) {
            CV_Assert(src.channels() == 1 || src.channels() == 3);

            int kernelRadius = std::max(1, static_cast<int>(std::ceil(3 * sigma)));
            int kernelSize = 2 * kernelRadius + 1;
            std::vector<double> kernel = makeGaussian1D(kernelSize, sigma);

            cv::Mat result(src.size(), src.type());

            auto processChannel = [&](const cv::Mat& channel) {
                cv::Mat padded = reflectPad(channel, kernelRadius);

                convolve1DFFT(padded, kernel, true);
                convolve1DFFT(padded, kernel, false);

                cv::Rect roi(kernelRadius, kernelRadius, channel.cols, channel.rows);
                return padded(roi).clone();
                };

            if (src.channels() == 1) {
                result = processChannel(src);
            }
            else {
                std::vector<cv::Mat> channels;
                cv::split(src, channels);
#pragma omp parallel for schedule(static)
                for (int i = 0; i < 3; ++i)
                    channels[i] = processChannel(channels[i]);
                cv::merge(channels, result);
            }

            return result;
        }
	}
}