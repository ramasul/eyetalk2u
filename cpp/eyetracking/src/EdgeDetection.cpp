#include "EdgeDetection.h"
#include "Utils.h"

namespace vision {
	namespace canny {
		cv::Mat canny(const cv::Mat& in, bool blurImage, bool useL2, int bins, float nonEdgePixelsRatio, float lowHighThresholdRatio) {
			cv::Size workingSize;
			workingSize.width = in.cols;
			workingSize.height = in.rows;

			cv::Mat dx= cv::Mat::zeros(workingSize, CV_32F), dy = cv::Mat::zeros(workingSize, CV_32F), magnitude = cv::Mat::zeros(workingSize, CV_32F);
			cv::Mat edgeType = cv::Mat::zeros(workingSize, CV_8U), edge = cv::Mat::zeros(workingSize, CV_8U);

			(void)useL2; // Unused parameter, Sebenernya buat implement norm aja
			/* Step 1: Smoothing & Directional Gradients  */

			cv::Mat blurred;
			if (blurImage) {
				cv::Size blurSize(5, 5);
				cv::GaussianBlur(in, blurred, blurSize, 1.5, 1.5, cv::BORDER_REPLICATE);
			}
			else blurred = in;

			cv::Sobel(blurred, dx, dx.type(), 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
			cv::Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, 0, cv::BORDER_REPLICATE);

			/* Magnitude */
			double minMag = 0;
			double maxMag = 0;
			float* p_res;
			float* p_x, * p_y; // result, x, y

			cv::magnitude(dx, dy, magnitude);

			/* Normalization */
			cv::minMaxLoc(magnitude, &minMag, &maxMag);
			magnitude = magnitude / maxMag;

			/* Step 2: Threshold Selection (Based on Magnitude Historgram */
			float low_th = 0;
			float high_th = 0;

			int* histogram = new int[bins]();
			cv::Mat res_idx = (bins - 1) * magnitude; //value range [0,bins-1]
			res_idx.convertTo(res_idx, CV_16U);
			short* p_res_idx = 0;
			for (int i = 0; i < res_idx.rows; i++)
			{
				p_res_idx = res_idx.ptr<short>(i);
				for (int j = 0; j < res_idx.cols; j++)
					histogram[p_res_idx[j]]++;
			}

			int sum = 0;
			int nonEdgePixels = nonEdgePixelsRatio * in.rows * in.cols;
			for (int i = 0; i < bins; i++)
			{
				sum += histogram[i];
				if (sum > nonEdgePixels)
				{
					high_th = float(i + 1) / bins;
					break;
				}
			}
			low_th = lowHighThresholdRatio * high_th;

			/* Step 3: Non-Maximum Suppression, Thin out “fat” gradient edges into 1-pixel-wide edges */
			const float tg22_5 = 0.4142135623730950488016887242097f; // tan(22.5 derajat)
			const float tg67_5 = 2.4142135623730950488016887242097f; // tan(67.5 derajat)
			uchar* _edgeType;
			float* p_res_b, * p_res_t;
			edgeType.setTo(0);
			for (int i = 1; i < magnitude.rows - 1; i++)
			{
				_edgeType = edgeType.ptr<uchar>(i); // Gives _edgeType to point to the beginning of row i

				p_res = magnitude.ptr<float>(i);
				p_res_t = magnitude.ptr<float>(i - 1);
				p_res_b = magnitude.ptr<float>(i + 1);

				p_x = dx.ptr<float>(i);
				p_y = dy.ptr<float>(i);

				for (int j = 1; j < magnitude.cols - 1; j++)
				{
					float m = p_res[j];
					if (m < low_th)
						continue;

					float iy = p_y[j];
					float ix = p_x[j];
					float y = abs((double)iy);
					float x = abs((double)ix);

					uchar val = p_res[j] > high_th ? 255 : 128;

					float tg22_5x = tg22_5 * x;
					if (y < tg22_5x)
					{
						if (m > p_res[j - 1] && m >= p_res[j + 1])
							_edgeType[j] = val;
					}
					else
					{
						float tg67_5x = tg67_5 * x;
						if (y > tg67_5x)
						{
							if (m > p_res_b[j] && m >= p_res_t[j])
								_edgeType[j] = val;
						}
						else
						{
							if ((iy <= 0) == (ix <= 0))
							{
								if (m > p_res_t[j - 1] && m >= p_res_b[j + 1])
									_edgeType[j] = val;
							}
							else
							{
								if (m > p_res_b[j - 1] && m >= p_res_t[j + 1])
									_edgeType[j] = val;
							}
						}
					}
				}
			}

			/* Step 4: Hystheresis */
			int pic_x = edgeType.cols;
			int pic_y = edgeType.rows;
			int area = pic_x * pic_y;
			int lines_idx = 0;
			int idx = 0;

			std::vector<int> lines;
			edge.setTo(0);
			for (int i = 1; i < pic_y - 1; i++) {
				for (int j = 1; j < pic_x - 1; j++) {

					if (edgeType.data[idx + j] != 255 || edge.data[idx + j] != 0)
						continue;

					edge.data[idx + j] = 255;
					lines_idx = 1;
					lines.clear();
					lines.push_back(idx + j);
					int akt_idx = 0;

					while (akt_idx < lines_idx) {
						int akt_pos = lines[akt_idx];
						akt_idx++;

						if (akt_pos - pic_x - 1 < 0 || akt_pos + pic_x + 1 >= area)
							continue;

						for (int k1 = -1; k1 < 2; k1++)
							for (int k2 = -1; k2 < 2; k2++) {
								if (edge.data[(akt_pos + (k1 * pic_x)) + k2] != 0 || edgeType.data[(akt_pos + (k1 * pic_x)) + k2] == 0)
									continue;
								edge.data[(akt_pos + (k1 * pic_x)) + k2] = 255;
								lines.push_back((akt_pos + (k1 * pic_x)) + k2);
								lines_idx++;
							}
					}
				}
				idx += pic_x;
			}

			return edge;
		}
	}
}