// eyetracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include "Utils.h"
#include "Resize.h"
#include "EdgeDetection.h"
#include "Blur.h"
#include "HistEq.h"
#include "Color.h"
#include "EdgeProcessing.h"
#include "PuRe_old.h"
#include "Pure.h"
#include "Normalize.h"
#include "callibrate.h"
#include "tracking.h"

// ------------------- Main -------------------
int main() {
    //3 : OBS
	//0 : Kamera laptop
    int cam = 0;
    cv::VideoCapture cap(cam);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }

    cv::Mat frame, gray, smallGray, blurred, edge, clahe, filtered, norm;
    cv::Mat debug;

    // Store last calibration result for tracking demo
    static std::vector<std::pair<cv::Point2f, cv::Point2f>> last_pairs;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
		//frame = cv::imread("sample/sample2.jpg");
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        vision::histeq::CLAHE(gray, clahe, 3.0, cv::Size(8, 8));

        //pure_old::Detector detector;
        PuRe detector;
        //auto result = detector.detect(gray, &debug);
		Pupil pupil = detector.run(clahe);

        drawMarker(frame, pupil.center, cv::Scalar(0, 0, 255));
        if (pupil.size.width > 0)
            ellipse(frame, pupil, cv::Scalar(0, 0, 255));

        imshow("Results", frame);

        //cv::ellipse(frame, cv::Point(result.center), cv::Size(result.axes), result.angle, 0, 360, cv::Scalar(0, 0, 255));

        //cv::imshow("Frame", frame);
        //cv::imshow("Debug", debug);
        // if (cv::waitKey(1) == 'q') break;
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'c') {
            // Run calibration without erasing existing processing
            cap.release();
            vision::calibration::Calibrator calib;
            // Example params: full HD-like target, 60px margin, 5x5 grid, 1s per point
            auto pairs = calib.run(1000, 1000, 60, 4, 1.0);
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
                vision::tracking::Tracker tracker(model, 720, 1280);
                tracker.run(cam);
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
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        vision::histeq::CLAHE(gray, blurred, 2.0, cv::Size(8, 8));
        cv::GaussianBlur(blurred, blurred, cv::Size(5, 5), 0);

        cv::Mat lap;
        cv::Laplacian(blurred, lap, CV_16S, 3);
        convertScaleAbs(lap, lap);

        cv::Mat sharpened;
        cv::addWeighted(blurred, 1.7, lap, -0.7, 0, sharpened);

        gray = sharpened;

		pure_old::Detector detector;
        auto result = detector.detect(gray, &debug);

        cv::ellipse(frame, cv::Point(result.center), cv::Size(result.axes), result.angle, 0, 360, cv::Scalar(0, 0, 255));

        cv::imshow("Frame", frame);
		cv::imshow("Debug", debug);
        if (cv::waitKey(1) == 'q') break;
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		//gray = vision::color::BGR2Gray(frame);
        

        // Downscale frame for speed
        smallGray = vision::resize::resize(gray, cv::Size(512, 512));

		clahe = vision::histeq::CLAHE(smallGray, 2.0, cv::Size(8, 8));

		blurred = vision::blur::GaussianBlur(clahe, 1.0);
		//blurred = vision::blur::FFTGaussianBlur(smallGray, 1.0);

        // Separable Gaussian blur
        //blurred = Utils::fftGaussianBlur(smallGray, 5);
		//edge = vision::canny::canny(smallGray, true, true, 128, 0.8f, 0.5f);
        //cv::Canny(smallGray, edge, 50, 150);

		edge = vision::canny::canny(blurred, false, true, 64, 0.8f, 0.5f);

        // Canny edge detection after blurring
        //cv::Mat edges;
        //cv::Canny(blurred, edges, 20, 60); // thresholds: adjust as needed

        //cv::imshow("Original Gray", smallGray);

		filtered = vision::edge::filterEdges(edge, filtered);
		norm = vision::normalize::normalize(gray, 100, 255, cv::NORM_MINMAX);
        cv::imshow("Separable Gaussian Blur", filtered);

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
