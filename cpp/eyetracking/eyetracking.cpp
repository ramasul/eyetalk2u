// eyetracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>

// Function to brighten + enhance contrast manually
cv::Mat brightenManual(const cv::Mat& input, float alpha = 1.2f, int beta = 40) {
    cv::Mat output(input.rows, input.cols, input.type());

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            cv::Vec3b color = input.at<cv::Vec3b>(y, x);
            cv::Vec3b newColor;

            for (int c = 0; c < 3; c++) {
                int newValue = static_cast<int>(alpha * color[c] + beta);
                newColor[c] = cv::saturate_cast<uchar>(newValue); // clamp 0–255
            }

            output.at<cv::Vec3b>(y, x) = newColor;
        }
    }

    return output;
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Mat frame, brightFrame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Brighten dark frames
        brightFrame = brightenManual(frame, 1.3f, 20); // tweak alpha/beta

        cv::imshow("Brightened Webcam", brightFrame);

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
