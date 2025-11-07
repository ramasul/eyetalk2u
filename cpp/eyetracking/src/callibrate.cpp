#include "callibrate.h"
#include "PupilDetector.h"
#include "Scale.h"

#include <chrono>
#include <iostream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace vision {
	namespace calibration {
		using Pair = std::pair<cv::Point2f, cv::Point2f>;
		
		Calibrator::Calibrator(const std::string& faceCascadePath, const std::string& eyeCascadePath)
			: faceCascadePath(faceCascadePath), eyeCascadePath(eyeCascadePath)
		{
		}

		static inline std::vector<cv::Point2f> grid_points(int height, int width, int m, int n)
		{
			std::vector<cv::Point2f> pts;
			if (n <= 1) {
				pts.emplace_back(width / 2.0f, height / 2.0f);
				return pts;
			}
			const float w0 = static_cast<float>(m);
			const float h0 = static_cast<float>(m);
			const float w1 = static_cast<float>(width - m);
			const float h1 = static_cast<float>(height - m);
			const float dx = (w1 - w0) / (n - 1);
			const float dy = (h1 - h0) / (n - 1);
			for (int j = 0; j < n; ++j)
				for (int i = 0; i < n; ++i)
					pts.emplace_back(w0 + i * dx, h0 + j * dy);
			return pts;
		}

		double Calibrator::now_seconds() const
		{
			using clock = std::chrono::steady_clock;
			return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
		}

		cv::Mat Calibrator::make_target(int height, int width, int m, int n, const cv::Point& highlight) const
		{
			cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
			auto pts = grid_points(height, width, m, n);
			for (const auto& p : pts)
				cv::circle(img, p, 4, cv::Scalar(64, 64, 64), cv::FILLED, cv::LINE_AA);
			if (highlight.x >= 0)
				cv::circle(img, highlight, 8, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
			return img;
		}

		template<typename Src>
		std::vector<Pair> Calibrator::run(Src camera_index, int height, int width, int m, int n, double t, bool useHaar, vision::detection::PupilDetector& detector)
		{
			std::vector<Pair> result;
			auto targets = grid_points(height, width, m, n);
			if (targets.empty()) return result;

			cv::namedWindow("Calibration Target", cv::WINDOW_NORMAL);
			cv::resizeWindow("Calibration Target", width, height);

			// Draw one frame to set the window's client size, then center or fullscreen
			cv::Mat first = make_target(height, width, m, n, cv::Point(-1, -1));
			cv::imshow("Calibration Target", first);
			cv::waitKey(1);

#ifdef _WIN32
			int screenW = GetSystemMetrics(SM_CXSCREEN);
			int screenH = GetSystemMetrics(SM_CYSCREEN);
			if (width == screenW && height == screenH) {
				cv::setWindowProperty("Calibration Target", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
			} else {
				int x = (screenW - width) / 2;
				int y = (screenH - height) / 2;
				cv::moveWindow("Calibration Target", std::max(0, x), std::max(0, y));
			}
			cv::setWindowProperty("Calibration Target", cv::WND_PROP_TOPMOST, 1);
#endif

			std::cout << "Starting calibration with " << targets.size() << " points\n";

			// Camera
			cv::VideoCapture cap(camera_index);
			if (!cap.isOpened()) {
				std::cerr << "Cannot open camera\n";
				return result;
			}

			// Use shared PupilDetector workflow/state
			cv::Mat frame;

			std::cout << "Warming up camera for 2 seconds..." << std::endl;
			double start_warmup = now_seconds();
			while (now_seconds() - start_warmup < 2.0) // Your 2-second delay
			{
				cap >> frame; // Actively pull frames
				if (frame.empty()) {
					std::cerr << "Error: Camera failed during warm-up." << std::endl;
					cv::destroyWindow("Calibration Target"); // Clean up
					return result;
				}

				// This waitKey is tiny but important. It allows OpenCV to process
				// window events and prevents the loop from freezing.
				cv::waitKey(1);
			}
			std::cout << "Camera is ready. Starting calibration." << std::endl;

			for (size_t k = 0; k < targets.size(); ++k)
			{
				const auto& tp = targets[k];
				// Countdown prompt
				for (int c = 3; c >= 1; --c)
				{
					cv::Mat tgt = make_target(height, width, m, n, cv::Point(cv::saturate_cast<int>(tp.x), cv::saturate_cast<int>(tp.y)));
					cv::putText(tgt, "Look at point in " + std::to_string(c) + "...", { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 0.7, { 0,255,255 }, 2);
					cv::imshow("Calibration Target", tgt);
					cv::waitKey(250);
				}

				// Capture for t seconds
				double start = now_seconds();
				double sum_x = 0.0, sum_y = 0.0; int cnt = 0;
				while (now_seconds() - start < t)
				{
					cap >> frame; if (frame.empty()) break;
					// Use unified workflow with shared state (ROI/Haar locked preserved)
					Pupil p = detector.processFrame(frame, useHaar);
					if (p.size.width > 0)
					{
						sum_x += p.center.x;
						sum_y += p.center.y;
						cnt++;
					}
					// Keep showing target
					cv::Mat tgt = make_target(height, width, m, n, cv::Point(cv::saturate_cast<int>(tp.x), cv::saturate_cast<int>(tp.y)));
					cv::imshow("Calibration Target", tgt);
					cv::waitKey(1);
				}

				cv::Point2f captured = cnt > 0 ? cv::Point2f(static_cast<float>(sum_x / cnt), static_cast<float>(sum_y / cnt)) : cv::Point2f(-1,-1);
				result.emplace_back(tp, captured);
				std::cout << "Good, captured point " << (k + 1) << "/" << targets.size() << "\n";
			}

			std::cout << "Good, captured all points" << std::endl;
			cv::destroyWindow("Calibration Target");
			return result;
		}

		// Solve normal equations (A^T A) c = A^T y for c (6x1)
		static void solve_normal_6x6(double ATA[6][6], double ATy[6], double out[6])
		{
			// Gaussian elimination with partial pivoting
			double M[6][7];
			for (int i = 0; i < 6; ++i) {
				for (int j = 0; j < 6; ++j) M[i][j] = ATA[i][j];
				M[i][6] = ATy[i];
			}
			for (int col = 0; col < 6; ++col) {
				int piv = col;
				double best = std::abs(M[col][col]);
				for (int r = col + 1; r < 6; ++r) {
					double v = std::abs(M[r][col]);
					if (v > best) { best = v; piv = r; }
				}
				if (piv != col) for (int c = col; c < 7; ++c) std::swap(M[piv][c], M[col][c]);
				double diag = M[col][col];
				if (std::abs(diag) < 1e-12) continue;
				for (int c = col; c < 7; ++c) M[col][c] /= diag;
				for (int r = 0; r < 6; ++r) if (r != col) {
					double f = M[r][col];
					for (int c = col; c < 7; ++c) M[r][c] -= f * M[col][c];
				}
			}
			for (int i = 0; i < 6; ++i) out[i] = M[i][6];
		}

		Poly2 Calibrator::fit_poly2(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& pairs)
		{
			Poly2 model{};
			if (pairs.size() < 6) return model;
			double ATA[6][6] = {0}, ATU[6] = {0}, ATV[6] = {0};
			auto phi = [](double x, double y, double out[6]) {
				out[0] = 1.0;
				out[1] = x;
				out[2] = y;
				out[3] = x * x;
				out[4] = x * y;
				out[5] = y * y;
			};
			double f[6];
			for (const auto &pr : pairs) {
				const auto &target = pr.first;   // (U,V)
				const auto &meas   = pr.second;  // (x,y)
				phi(meas.x, meas.y, f);
				for (int i = 0; i < 6; ++i) {
					ATU[i] += f[i] * target.x;
					ATV[i] += f[i] * target.y;
					for (int j = 0; j < 6; ++j) ATA[i][j] += f[i] * f[j];
				}
			}
			solve_normal_6x6(ATA, ATU, model.a);
			solve_normal_6x6(ATA, ATV, model.b);
			return model;
		}
	}
}

template std::vector<vision::calibration::Pair>
vision::calibration::Calibrator::run<int>(int, int, int, int, int, double, bool, vision::detection::PupilDetector&);

template std::vector<vision::calibration::Pair>
vision::calibration::Calibrator::run<std::string>(std::string, int, int, int, int, double, bool, vision::detection::PupilDetector&);

