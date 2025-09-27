#include "Utils.h"

template<typename T>
T clamp(T val, T minVal, T maxVal) {
    if (val < minVal) return minVal;
    if (val > maxVal) return maxVal;
    return val;
}


// ------------------- Iterative 1D FFT (radix-2) -------------------
void Utils::fft1D(std::vector<Complex>& a, bool invert) {
    int n = a.size();
    // Bit reversal
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Iterative FFT
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        Complex wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int k = 0; k < len / 2; ++k) {
                Complex u = a[i + k];
                Complex v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (auto& x : a) x /= n;
    }
}

// ------------------- 2D FFT using iterative 1D FFT -------------------
void Utils::fft2D(std::vector<std::vector<Complex>>& data, bool invert) {
    int rows = data.size();
    int cols = data[0].size();

    // FFT rows
    for (int i = 0; i < rows; i++) Utils::fft1D(data[i], invert);

    // FFT columns
    for (int j = 0; j < cols; j++) {
        std::vector<Complex> col(rows);
        for (int i = 0; i < rows; i++) col[i] = data[i][j];
        fft1D(col, invert);
        for (int i = 0; i < rows; i++) data[i][j] = col[i];
    }
}

// ------------------- Next Power of 2 -------------------
int nextPow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// ------------------- Separable Gaussian Kernel -------------------
std::vector<double> Utils::gaussianKernel1D(int ksize, double sigma) {
    std::vector<double> kernel(ksize);
    int half = ksize / 2;
    double sum = 0;
    for (int i = 0; i < ksize; i++) {
        int x = i - half;
        kernel[i] = std::exp(-x * x / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; i++) kernel[i] /= sum;
    return kernel;
}

// ------------------- Separable Convolution -------------------
cv::Mat Utils::separableGaussian(const cv::Mat& gray, int ksize, double sigma) {
    std::vector<double> kernel = Utils::gaussianKernel1D(ksize, sigma);
    int half = ksize / 2;

    // Temporary row convolution
    cv::Mat temp = cv::Mat::zeros(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            double sum = 0;
            for (int k = -half; k <= half; k++) {
                int xx = clamp(x + k, 0, gray.cols - 1);
                sum += gray.at<uchar>(y, xx) * kernel[k + half];
            }
            temp.at<float>(y, x) = sum;
        }
    }

    // Column convolution
    cv::Mat result = cv::Mat::zeros(gray.size(), CV_8U);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            double sum = 0;
            for (int k = -half; k <= half; k++) {
                int yy = clamp(y + k, 0, gray.rows - 1);
                sum += temp.at<float>(yy, x) * kernel[k + half];
            }
            result.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
        }
    }
    return result;
}