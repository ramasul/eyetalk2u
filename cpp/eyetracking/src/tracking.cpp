#include "tracking.h"
#include "PupilDetector.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace vision {
	namespace tracking {
		Tracker::Tracker(const calibration::Poly2& model, int target_height, int target_width, vision::detection::PupilDetector& detector)
			: model(model), height(target_height), width(target_width), detector(detector) {}

		cv::Point2f Tracker::map_to_screen(const cv::Point2f& p) const
		{
			double x = p.x, y = p.y;
			double phi[6] = { 1.0, x, y, x*x, x*y, y*y };
			double U = 0, V = 0;
			for (int i = 0; i < 6; ++i) { U += model.a[i] * phi[i]; V += model.b[i] * phi[i]; }
			return cv::Point2f(static_cast<float>(U), static_cast<float>(V));
		}

		template<typename Src>
		int Tracker::run(Src camera_index, bool useHaar)
		{
			cv::VideoCapture cap(camera_index);
			if (!cap.isOpened()) return -1;
			cv::Mat frame;
			cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
			cv::imshow("Tracking", cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0)));
			cv::waitKey(1);

#ifdef _WIN32
			int screenW = GetSystemMetrics(SM_CXSCREEN);
			int screenH = GetSystemMetrics(SM_CYSCREEN);
			if (width == screenW && height == screenH) {
				// Fix 1: Use the correct window name "Tracking"
				cv::setWindowProperty("Tracking", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
			}
			else {
				int x = (screenW - width) / 2;
				int y = (screenH - height) / 2;
				// Fix 1: Use the correct window name "Tracking"
				cv::moveWindow("Tracking", std::max(0, x), std::max(0, y));
			}
			// Fix 1: Use the correct window name "Tracking"
			cv::setWindowProperty("Tracking", cv::WND_PROP_TOPMOST, 1);
#endif

			while (true) {
				cap >> frame; if (frame.empty()) break;
				// Use unified workflow
				Pupil p = detector.processFrame(frame, useHaar);
				cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
				if (p.size.width > 0) {
					cv::Point2f s = map_to_screen(p.center);
					cv::circle(canvas, s, 15, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_AA);
				}
				cv::imshow("Tracking", canvas);
				int k = cv::waitKey(1);
				if (k == 'q' || k == 27) break;
			}
			cap.release();
			cv::destroyWindow("Tracking");
			return 0;
		}
	}
}

template int vision::tracking::Tracker::run<int>(int, bool);
template int vision::tracking::Tracker::run<std::string>(std::string, bool);
