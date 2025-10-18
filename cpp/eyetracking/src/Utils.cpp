#include "Utils.h"

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

        // Precompute twiddles
        std::vector<Complex> wtable(len / 2);
        wtable[0] = 1;
        for (int k = 1; k < len / 2; ++k)
            wtable[k] = wtable[k - 1] * wlen;

        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; ++k) {
                Complex u = a[i + k];
                Complex v = a[i + k + len / 2] * wtable[k];
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }

    if (invert) {
        double inv_n = 1.0 / n;
        for (auto& x : a) x *= inv_n;
    }
}

void Utils::fft2D(std::vector<std::vector<Complex>>& data, bool invert) {
    int rows = data.size();
    int cols = data[0].size();

    // FFT rows
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
        fft1D(data[i], invert);

    // FFT cols (transpose trick would be faster, but this is simpler)
#pragma omp parallel for
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

// Check if n can be factorized only by 2, 3, 5
bool isOptimalDFTSize(int n) {
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}

// Equivalent of cv::getOptimalDFTSize
int getOptimalDFTSize(int n) {
    while (!isOptimalDFTSize(n)) {
        n++;
    }
    return n;
}

cv::Mat Utils::fftGaussianBlur(const cv::Mat& gray, double sigma) {
    // --- 1. Get optimal FFT size (custom)
    int rowsFFT = getOptimalDFTSize(gray.rows);
    int colsFFT = getOptimalDFTSize(gray.cols);

    // --- 2. Copy image into complex padded array
    std::vector<std::vector<Complex>> img(rowsFFT, std::vector<Complex>(colsFFT, 0));
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            img[y][x] = Complex((double)gray.at<uchar>(y, x), 0.0);
        }
    }

    // --- 3. Build Gaussian kernel (same size as padded image, centered)
    static std::vector<std::vector<Complex>> kernelFFT;
    static int cachedRows = -1, cachedCols = -1;
    static double cachedSigma = -1;

    if (kernelFFT.empty() || cachedRows != rowsFFT || cachedCols != colsFFT || cachedSigma != sigma) {
        std::vector<std::vector<Complex>> kernel(rowsFFT, std::vector<Complex>(colsFFT, 0));

        int cy = rowsFFT / 2;
        int cx = colsFFT / 2;
        double sum = 0.0;

        for (int y = 0; y < rowsFFT; y++) {
            for (int x = 0; x < colsFFT; x++) {
                int dy = (y <= rowsFFT / 2) ? y : y - rowsFFT;
                int dx = (x <= colsFFT / 2) ? x : x - colsFFT;
                double g = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[y][x] = Complex(g, 0.0);
                sum += g;
            }
        }


        // Normalize kernel
        for (int y = 0; y < rowsFFT; y++)
            for (int x = 0; x < colsFFT; x++)
                kernel[y][x] /= sum;

        // FFT of kernel
        kernelFFT = kernel;
        Utils::fft2D(kernelFFT, false);

        cachedRows = rowsFFT;
        cachedCols = colsFFT;
        cachedSigma = sigma;
    }

    // --- 4. FFT of image
    Utils::fft2D(img, true);

    // --- 5. Multiply in frequency domain
    for (int y = 0; y < rowsFFT; y++) {
        for (int x = 0; x < colsFFT; x++) {
            img[y][x] *= kernelFFT[y][x];
        }
    }

    // --- 6. Inverse FFT
    Utils::fft2D(img, false);

    // --- 7. Copy back to Mat (crop to original size)
    cv::Mat result(gray.size(), CV_8U);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            double val = img[y][x].real();
            result.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
        }
    }

    return result;
}