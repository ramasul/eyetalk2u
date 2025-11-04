#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vision {
    namespace haar {

        struct EyeZoomResult {
            std::vector<cv::Mat> zoomedEyes;
            cv::Mat annotatedFrame;
            int eyeCount;
        };

        class EyeZoomer {
        public:
            EyeZoomer(const std::string& faceCascadePath,
                const std::string& eyeCascadePath,
                int zoomWidth = 200,
                int zoomHeight = 200);

            EyeZoomResult processFrame(const cv::Mat& frame);

        private:
            cv::CascadeClassifier faceCascade;
            cv::CascadeClassifier eyeCascade;
            int zoomW, zoomH;

            cv::Mat cropAndZoom(const cv::Mat& src, const cv::Rect& eyeRect);
        };

    } // namespace haar
} // namespace vision
#pragma once
