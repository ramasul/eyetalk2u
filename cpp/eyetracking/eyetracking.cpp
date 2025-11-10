// eyetracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "callibrate.h"
#include "tracking.h"
#include "PupilDetector.h"

using namespace vision::detection;

// ------------------- Main -------------------
int main() {
    std::string faceCascadePath = "haarcascade_frontalface_default.xml";
    std::string eyeCascadePath = "haarcascade_eye.xml";

    PupilDetector detector(faceCascadePath, eyeCascadePath);

    //3 : OBS
	//0 : Kamera laptop
    int cam = 0;
	//std::string cam = "http://192.168.1.5:8080/video"; // for IP camera
	std::string url = "sample/ciel.mp4"; // for IP camera
    cv::VideoCapture cap(cam);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }

    cv::Mat frame;
    bool useHaar = true; // toggles Haar zoom acquisition
	int screenWidth = 1920;
	int screenHeight = 1080;

    // Store last calibration result for tracking demo
    static std::vector<std::pair<cv::Point2f, cv::Point2f>> last_pairs;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Process frame using unified workflow
        Pupil pupil = detector.processFrame(frame, useHaar);

        // Draw on working view (coordinates must match working frame)
        cv::Mat view = detector.getWorkingFrame();
        Pupil wp = detector.getWorkingPupil();
        if (wp.size.width > 0) {
            drawMarker(view, wp.center, cv::Scalar(0, 0, 255));
            ellipse(view, wp, cv::Scalar(0, 0, 255));
        }
        imshow("Results", view);

        //cv::ellipse(frame, cv::Point(result.center), cv::Size(result.axes), result.angle, 0, 360, cv::Scalar(0, 0, 255));

        //cv::imshow("Frame", frame);
        //cv::imshow("Debug", debug);
        // if (cv::waitKey(1) == 'q') break;
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'h' || key == 'H') {
            useHaar = !useHaar;
            std::cout << "Haar cascade: " << (useHaar ? "ON" : "OFF") << std::endl;
        }
        if (key == 'r' || key == 'R') {
            // Relock request: next detected eyes redefine the working frame
            detector.reset();
            std::cout << "Haar relock: next eyes will redefine the working frame.\n";
        }
        if (key == 'c') {
            // Run calibration without erasing existing processing
            cap.release();
            vision::calibration::Calibrator calib(faceCascadePath, eyeCascadePath);
            // Example params: full HD-like target, 60px margin, 5x5 grid, 1s per point
            auto pairs = calib.run(cam, screenHeight, screenWidth, 60, 3, 2.0, useHaar, detector);
            last_pairs = pairs;
            std::cout << "Calibration pairs (target -> measured):\n";
            for (const auto &pr : pairs) {
                std::cout << "(" << pr.first.x << "," << pr.first.y << ") -> ("
                          << pr.second.x << "," << pr.second.y << ")\n";
            }
            // reopen camera
            cap.open(cam);
            if (!cap.isOpened()) { std::cerr << "Cannot reopen camera after calibration\n"; return -1; }
        }
        if (key == 't') {
            if (last_pairs.size() >= 6) {
                cap.release();
                auto model = vision::calibration::Calibrator::fit_poly2(last_pairs);
                vision::tracking::Tracker tracker(model, screenHeight, screenWidth, detector);
                tracker.run(cam, useHaar);
                cap.open(cam);
                if (!cap.isOpened()) { std::cerr << "Cannot reopen camera after tracking demo\n"; return -1; }
            } else {
                std::cout << "Run calibration first (press 'c').\n";
            }
        }
    }

    while (false) {
        cap >> frame;
		if (frame.empty()) break;
		cv::Mat gray, blurred, lap, sharpened, debug;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // vision::histeq::CLAHE(gray, blurred, 2.0, cv::Size(8, 8));
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        cv::Laplacian(blurred, lap, CV_16S, 3);
        cv::convertScaleAbs(lap, lap);

        cv::addWeighted(blurred, 1.7, lap, -0.7, 0, sharpened);

        gray = sharpened;

		// pure_old::Detector detector;
        // auto result = detector.detect(gray, &debug);

        // cv::ellipse(frame, cv::Point(result.center), cv::Size(result.axes), result.angle, 0, 360, cv::Scalar(0, 0, 255));

        cv::imshow("Frame", frame);
		cv::imshow("Debug", debug);
        if (cv::waitKey(1) == 'q') break;
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        cv::Mat gray, smallGray, blurred, edge, clahe, filtered, norm;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Downscale frame for speed
        // smallGray = vision::resize::resize(gray, cv::Size(512, 512));

        // clahe = vision::histeq::CLAHE(smallGray, 2.0, cv::Size(8, 8));
        // blurred = vision::blur::GaussianBlur(clahe, 1.0);
        // edge = vision::canny::canny(blurred, false, true, 64, 0.8f, 0.5f);
        // filtered = vision::edge::filterEdges(edge, filtered);
        // norm = vision::normalize::normalize(gray, 100, 255, cv::NORM_MINMAX);
        // cv::imshow("Separable Gaussian Blur", filtered);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
