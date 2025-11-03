#include "tracking.h"
#include "Pure.h"

namespace vision {
	namespace tracking {
		Tracker::Tracker(const calibration::Poly2& model, int target_height, int target_width)
			: model(model), height(target_height), width(target_width) {}

		cv::Point2f Tracker::map_to_screen(const cv::Point2f& p) const
		{
			double x = p.x, y = p.y;
			double phi[6] = { 1.0, x, y, x*x, x*y, y*y };
			double U = 0, V = 0;
			for (int i = 0; i < 6; ++i) { U += model.a[i] * phi[i]; V += model.b[i] * phi[i]; }
			return cv::Point2f(static_cast<float>(U), static_cast<float>(V));
		}

		int Tracker::run(int camera_index)
		{
			cv::VideoCapture cap(camera_index);
			if (!cap.isOpened()) return -1;
			PuRe detector;
			cv::Mat frame, gray;
			cv::namedWindow("Tracking", cv::WINDOW_AUTOSIZE);
			while (true) {
				cap >> frame; if (frame.empty()) break;
				cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
				Pupil p = detector.run(gray);
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


