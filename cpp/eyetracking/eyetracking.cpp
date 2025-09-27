// eyetracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include "Utils.h"

// ------------------- Main -------------------
int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }

    cv::Mat frame, gray, smallGray, blurred;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Downscale frame for speed
        cv::resize(gray, smallGray, cv::Size(256, 256));

        // Separable Gaussian blur
        blurred = Utils::fftGaussianBlur(smallGray, 3);

        // Canny edge detection after blurring
        cv::Mat edges;
        cv::Canny(blurred, edges, 20, 60); // thresholds: adjust as needed

        //cv::imshow("Original Gray", smallGray);
        cv::imshow("Separable Gaussian Blur", edges);

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
