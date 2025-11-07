#include "haarcascade.h"
#include <iostream>

using namespace cv;
using namespace std;

namespace vision {
    namespace haar {

        EyeZoomer::EyeZoomer(const std::string& faceCascadePath,
            const std::string& eyeCascadePath,
            int zoomWidth,
            int zoomHeight)
            : zoomW(zoomWidth), zoomH(zoomHeight)
        {
            if (!faceCascade.load(faceCascadePath)) {
                cerr << "Error: cannot load face cascade from " << faceCascadePath << endl;
                throw std::runtime_error("Failed to load face cascade");
            }
            if (!eyeCascade.load(eyeCascadePath)) {
                cerr << "Error: cannot load eye cascade from " << eyeCascadePath << endl;
                throw std::runtime_error("Failed to load eye cascade");
            }
        }

        cv::Mat EyeZoomer::cropAndZoom(const cv::Mat& src, const cv::Rect& eyeRect) {
            int padX = max(2, (int)(eyeRect.width * 0.15));
            int padY = max(2, (int)(eyeRect.height * 0.15));

            Rect padded(eyeRect.x - padX, eyeRect.y - padY,
                eyeRect.width + 2 * padX, eyeRect.height + 2 * padY);

            padded &= Rect(0, 0, src.cols, src.rows);

            Mat cropped = src(padded).clone();
            Mat zoomed;
            resize(cropped, zoomed, Size(zoomW, zoomH), 0, 0, INTER_CUBIC);
            return zoomed;
        }

        EyeZoomResult EyeZoomer::processFrame(const cv::Mat& frame) {
            EyeZoomResult result;
            result.annotatedFrame = frame.clone();
            result.eyeCount = 0;

            Mat gray;
            if (frame.channels() == 3)
                cvtColor(frame, gray, COLOR_BGR2GRAY);
            else
                gray = frame.clone();
            equalizeHist(gray, gray);

            vector<Rect> faces;
            faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(80, 80));

            if (faces.empty()) {
                vector<Rect> eyes;
                eyeCascade.detectMultiScale(gray, eyes, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(20, 20));
                for (auto& e : eyes) {
                    result.zoomedEyes.push_back(cropAndZoom(frame, e));
                    result.eyeRects.push_back(e);  // ADD THIS
                    rectangle(result.annotatedFrame, e, Scalar(0, 255, 0), 2);
                    result.eyeCount++;
                }
                return result;
            }

            for (auto& f : faces) {
                Mat faceROIGray = gray(f);
                vector<Rect> eyes;
                eyeCascade.detectMultiScale(faceROIGray, eyes, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(20, 20));
                for (auto& e : eyes) {
                    Rect eyeInImg(e.x + f.x, e.y + f.y, e.width, e.height);
                    result.zoomedEyes.push_back(cropAndZoom(frame, eyeInImg));
                    result.eyeRects.push_back(eyeInImg);  // ADD THIS
                    rectangle(result.annotatedFrame, eyeInImg, Scalar(0, 255, 0), 2);
                    result.eyeCount++;
                }
            }

            return result;
        }

    } // namespace haar
} // namespace vision
