#if __has_include(<uwebsockets/App.h>)
#include <uwebsockets/App.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <algorithm>
#include "callibrate.h"
#include "tracking.h"
#include "PupilDetector.h"

using namespace vision::detection;
using namespace vision::calibration;

struct PerSocketData {
    std::string path;
};

// Global shared state
std::atomic<bool> backendActive{ false };
std::atomic<bool> calibrationActive{ false };
std::atomic<bool> useHaar{ true };
std::atomic<bool> calibrationRunning{ false }; // Prevent camera conflicts during calibration
std::mutex frameMutex;
std::mutex modelMutex;
std::mutex cameraMutex; // Protect camera access
cv::Mat latestFrame;

// --- Camera source selection ---
enum CameraType { NONE, CAM_INT, CAM_LINK };
struct CameraConfig {
    CameraType type = NONE;
    int camIndex = -1;
    std::string link;
};
CameraConfig currentCamera = { CAM_INT, 0, "" };

// Calibration model for mapping
Poly2 calibrationModel;
bool hasCalibrationModel{ false };
int screenWidth = 1920;
int screenHeight = 1080;

std::vector<uWS::WebSocket<false, true, PerSocketData>*> videoClients;
std::vector<uWS::WebSocket<false, true, PerSocketData>*> mappingClients;

void broadcastFrame(const cv::Mat& frame) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf);
    std::string encoded(buf.begin(), buf.end());
    for (auto* ws : videoClients) {
        ws->send(encoded, uWS::OpCode::BINARY);
    }
}

void broadcastMapping(const cv::Point2f& point) {
    char msg[64];
    snprintf(msg, sizeof(msg), "%.2f,%.2f", point.x, point.y);
    for (auto* ws : mappingClients) {
        ws->send(std::string(msg), uWS::OpCode::TEXT);
    }
}

// Map pupil center to screen coordinates using calibration model
cv::Point2f map_to_screen(const cv::Point2f& p) {
    std::lock_guard<std::mutex> lock(modelMutex);
    if (!hasCalibrationModel) {
        return cv::Point2f(-1, -1); // No calibration available
    }
    
    double x = p.x, y = p.y;
    double phi[6] = { 1.0, x, y, x*x, x*y, y*y };
    double U = 0, V = 0;
    for (int i = 0; i < 6; ++i) {
        U += calibrationModel.a[i] * phi[i];
        V += calibrationModel.b[i] * phi[i];
    }
    return cv::Point2f(static_cast<float>(U), static_cast<float>(V));
}

int web_services() { //Change to main()
    std::string faceCascadePath = "haarcascade_frontalface_default.xml";
    std::string eyeCascadePath = "haarcascade_eye.xml";

    PupilDetector detector(faceCascadePath, eyeCascadePath);
    using namespace std::chrono_literals;

    // Start background camera thread
    std::thread([&detector]() {
        cv::VideoCapture cap;
        bool cameraOpened = false;
        int consecutiveErrors = 0;
        const int maxErrors = 10;

        cv::Mat frame;
        while (true) {
            if (!backendActive.load()) {
                if (cameraOpened) {
                    cap.release();
                    cameraOpened = false;
                }
                std::this_thread::sleep_for(100ms);
                continue;
            }

            // Pause camera access during calibration to avoid conflicts
            if (calibrationRunning.load()) {
                if (cameraOpened) {
                    cap.release();
                    cameraOpened = false;
                }
                std::this_thread::sleep_for(100ms);
                continue;
            }

            // Open camera if not opened
            if (!cameraOpened) {
                std::lock_guard<std::mutex> lock(cameraMutex);
                if (currentCamera.type == CAM_LINK)
                    cap.open(currentCamera.link);
                else if (currentCamera.type == CAM_INT)
                    cap.open(currentCamera.camIndex);
                else {
                    std::cerr << "Please set the camera first using /camera/link or /camera/cam\n";
                    std::this_thread::sleep_for(1000ms);
                    continue;
                }
                if (cap.isOpened()) {
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                    cameraOpened = true;
                    consecutiveErrors = 0;
                    std::cout << "Camera opened\n";
                } else {
                    std::cerr << "Cannot open camera\n";
                    std::this_thread::sleep_for(1000ms);
                    continue;
                }
            }

            // Grab frame with error handling
            bool grabbed = false;
            {
                std::lock_guard<std::mutex> lock(cameraMutex);
                grabbed = cap.grab();
            }
            
            if (!grabbed) {
                consecutiveErrors++;
                if (consecutiveErrors >= maxErrors) {
                    std::cerr << "Too many camera errors, releasing camera\n";
                    cap.release();
                    cameraOpened = false;
                    consecutiveErrors = 0;
                    std::this_thread::sleep_for(500ms);
                }
                continue;
            }

            consecutiveErrors = 0;
            
            {
                std::lock_guard<std::mutex> lock(cameraMutex);
                cap.retrieve(frame);
            }
            
            if (frame.empty()) continue;

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = frame.clone();
            }

            // Process frame using unified workflow
            Pupil p = detector.processFrame(frame, useHaar.load());
            
            // Draw on working view (coordinates must match working frame)
            cv::Mat view = detector.getWorkingFrame();
            Pupil wp = detector.getWorkingPupil();
            if (wp.size.width > 0) {
                cv::drawMarker(view, wp.center, cv::Scalar(0, 0, 255));
                // Draw ellipse using RotatedRect properties
                cv::ellipse(view, wp.center, cv::Size(wp.size.width/2, wp.size.height/2), 
                           wp.angle, 0, 360, cv::Scalar(0, 0, 255));
            }

            broadcastFrame(view);

            // Map to screen coordinates if calibration is active and we have a valid pupil
            if (calibrationActive.load() && p.size.width > 0) {
                cv::Point2f mapped = map_to_screen(p.center);
                if (mapped.x >= 0 && mapped.y >= 0) {
                    broadcastMapping(mapped);
                }
            }

            std::this_thread::sleep_for(33ms);
        }
        }).detach();

    // ------------------- uWS App -------------------
    uWS::App app;

    auto setCORS = [](auto* res) {
        res->writeHeader("Access-Control-Allow-Origin", "*");
        res->writeHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res->writeHeader("Access-Control-Allow-Headers", "Content-Type");
        };

    // HTTP: start backend
    app.get("/start", [setCORS](auto* res, auto* req) {
        setCORS(res);
        backendActive = true;
        res->end("Backend activated - camera started");
        });

    app.get("/stop", [&detector, setCORS](auto* res, auto* req) {
        setCORS(res);
        backendActive = false;
        calibrationActive = false;
        detector.reset(); // Reset detector state when stopping
        res->end("Backend deactivated - camera stopped");
        });

    // HTTP: reset Haar (reset detector)
    app.get("/reset-haar", [&detector, setCORS](auto* res, auto* req) {
        setCORS(res);
        if (!backendActive.load()) {
            res->end("Backend not active - start backend first");
            return;
        }
        detector.reset();
        res->end("Haar reset - detector state cleared");
        });

    // HTTP: set camera link
    app.get("/camera/link", [setCORS](auto* res, auto* req) {
        setCORS(res);
        if (backendActive.load()) {
            res->end("Please turn off backend first");
            return;
        }

        std::string_view link = req->getQuery("link");
        if (link.empty()) {
            res->end("Missing ?link parameter");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(cameraMutex);
            currentCamera.type = CAM_LINK;
            currentCamera.link = std::string(link);
            currentCamera.camIndex = -1;
        }

        res->end("Camera set to link: " + std::string(link));
        });

    // HTTP: set camera index
    app.get("/camera/cam", [setCORS](auto* res, auto* req) {
        setCORS(res);
        if (backendActive.load()) {
            res->end("Please turn off backend first");
            return;
        }

        std::string_view camStr = req->getQuery("cam");
        if (camStr.empty()) {
            res->end("Missing ?cam parameter");
            return;
        }

        int camIndex = std::stoi(std::string(camStr));

        {
            std::lock_guard<std::mutex> lock(cameraMutex);
            currentCamera.type = CAM_INT;
            currentCamera.camIndex = camIndex;
            currentCamera.link.clear();
        }

        res->end("Camera set to index: " + std::to_string(camIndex));
        });


    // HTTP: start calibration
    app.get("/calibrate", [&detector, faceCascadePath, eyeCascadePath, setCORS](auto* res, auto* req) {
        setCORS(res);
        if(!backendActive.load()) {
            res->end("Backend not active - start backend first");
            return;
		}
        if (calibrationRunning.load()) {
            res->end("Calibration already running");
            return;
        }
        
        std::thread([&detector, faceCascadePath, eyeCascadePath]() {
            calibrationRunning = true;
            std::cout << "Starting calibration...\n";
            
            // Wait a bit for main camera thread to release camera
            std::this_thread::sleep_for(500ms);
            
            try {
                Calibrator calib(faceCascadePath, eyeCascadePath);
                // Example params: full HD-like target, 60px margin, 3x3 grid, 2.0s per point
                auto pairs = calib.run(0, screenHeight, screenWidth, 60, 3, 2.0, useHaar.load(), detector);
                
                if (pairs.size() >= 6) {
                    auto model = Calibrator::fit_poly2(pairs);
                    {
                        std::lock_guard<std::mutex> lock(modelMutex);
                        calibrationModel = model;
                        hasCalibrationModel = true;
                    }
                    std::cout << "Calibration completed with " << pairs.size() << " points\n";
                    
                    // Enable calibration mapping
                    calibrationActive = true;
                } else {
                    std::cerr << "❌ Calibration failed: insufficient points (" << pairs.size() << ")\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ Calibration error: " << e.what() << "\n";
                // Ensure calibration window is destroyed on error
                try {
                    cv::destroyWindow("Calibration Target");
                } catch (...) {}
            } catch (...) {
                std::cerr << "❌ Unknown calibration error\n";
                // Ensure calibration window is destroyed on error
                try {
                    cv::destroyWindow("Calibration Target");
                } catch (...) {}
            }
            
            calibrationRunning = false;
            std::cout << "Calibration thread finished\n";
        }).detach();
        res->end("✅ Calibration started in background");
        });

    // WebSocket handler
    app.ws<PerSocketData>("/*", {
        .upgrade = [](auto* res, auto* req, auto* context) {
            // Create per-socket data and capture the path
            std::string_view path = req->getUrl();
            res->template upgrade<PerSocketData>({std::string(path)}, req->getHeader("sec-websocket-key"),
                                                 req->getHeader("sec-websocket-protocol"),
                                                 req->getHeader("sec-websocket-extensions"),
                                                 context);
        },
        .open = [](auto* ws) {
            auto* data = ws->getUserData();
            if (data->path == "/video") {
                videoClients.push_back(ws);
                std::cout << "🎥 Video WS connected\n";
            }
            else if (data->path == "/mapping") {
                mappingClients.push_back(ws);
                std::cout << "🗺️ Mapping WS connected\n";
            }
            else {
                std::cout << "🌐 Unknown WS path: " << data->path << "\n";
            }
        },
        .message = [](auto* ws, std::string_view msg, uWS::OpCode) {
            std::cout << "📩 Received: " << msg << std::endl;
        },
        .close = [](auto* ws, int, std::string_view) {
            videoClients.erase(std::remove(videoClients.begin(), videoClients.end(), ws), videoClients.end());
            mappingClients.erase(std::remove(mappingClients.begin(), mappingClients.end(), ws), mappingClients.end());
            std::cout << "❌ Client disconnected\n";
        }
                });

    // Listen
    app.listen(9001, [](auto* token) {
        if (token)
            std::cout << "✅ Listening on port 9001\n";
        else
            std::cerr << "❌ Failed to bind port 9001\n";
        });

    app.run();
}

#else
#pragma message("uWebSocket not found — Please Install following the README on github")
#endif