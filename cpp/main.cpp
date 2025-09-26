#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "image.jpg";
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return -1;
    }
    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}
