#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem> // C++17
#include "callibrate.h"
#include "tracking.h"
#include "PupilDetector.h"

using namespace vision::detection;
namespace fs = std::filesystem;

int testing() {
    std::string faceCascadePath = "haarcascade_frontalface_default.xml";
    std::string eyeCascadePath = "haarcascade_eye.xml";

    PupilDetector detector(faceCascadePath, eyeCascadePath);
    bool useHaar = false;

    std::string folderPath = "D:\\Capstone\\p1-left\\frames";
    int totalFrames = 939;
    int delay = 1000 / 24; // ~41 ms per frame for 24 FPS

    for (int i = 1; i <= totalFrames; ++i) {
        std::string imagePath = folderPath + "\\" + std::to_string(i) + "-eye.png";
        cv::Mat frame = cv::imread(imagePath);

        if (frame.empty()) {
            std::cerr << "Warning: Could not load " << imagePath << std::endl;
            continue;
        }

        Pupil pupil = detector.processFrame(frame, useHaar);
        cv::Mat view = detector.getWorkingFrame();
        Pupil wp = detector.getWorkingPupil();

        // Draw detection results
        if (wp.size.width > 0) {
            cv::drawMarker(view, wp.center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
            cv::ellipse(view, wp, cv::Scalar(0, 0, 255), 2);
        }

        // Print info once per frame
        if (wp.hasOutline()) {
            std::cout << i << " | "
                << wp.center.x << " "
                << wp.center.y << " "
                << wp.size.width << " "
                << wp.size.height << " "
                << wp.angle * CV_PI / 180.0
                << std::endl;
        }

        cv::imshow("Pupil Detection Result", view);

        // Wait ~41 ms; break if 'q' is pressed
        if (cv::waitKey(delay) == 'q')
            break;
    }

    cv::destroyAllWindows();
    return 0;
}
