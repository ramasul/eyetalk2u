#include "PupilDetector.h"
#include <iostream>

namespace vision {
	namespace detection {
		PupilDetector::PupilDetector(const std::string& faceCascadePath, const std::string& eyeCascadePath)
			: zoomer(faceCascadePath, eyeCascadePath, 200, 200)
			, haarLocked(false)
			, roiMargin(10)
			, hasPrevPupil(false)
			, hasSmooth(false)
			, roiScaleFactor(1.0)
		{
			try {
				detector.initHaar(faceCascadePath, eyeCascadePath);
			}
			catch (const std::exception& e) {
				std::cerr << "Failed to load Haar cascades: " << e.what() << std::endl;
			}
		}
		
		void PupilDetector::reset() {
			haarLocked = false;
			hasPrevPupil = false;
			hasSmooth = false;
		}
		
		Pupil PupilDetector::processFrame(const cv::Mat& frame, bool useHaar) {
			if (frame.empty()) {
				Pupil empty;
				return empty;
			}
			
			// Step 1: Resize frame to default height
			cv::Mat frameSmall = vision::scale::resizeToHeight(frame, 512);
			cv::flip(frameSmall, frameSmall, 1); // Mirror
			cv::Mat gray;
			cv::cvtColor(frameSmall, gray, cv::COLOR_BGR2GRAY);
			
			// Step 2: Haar detection and ROI locking
			if (useHaar && !haarLocked) {
				EyeZoomResult zr = zoomer.processFrame(gray);
				if (!zr.eyeRects.empty()) {
					cv::Rect acc;
					for (const auto& r : zr.eyeRects) acc |= r;
					if (acc.area() > 0) {
						acc.x = std::max(0, acc.x - roiMargin);
						acc.y = std::max(0, acc.y - roiMargin);
						acc.width = std::min(frameSmall.cols - acc.x, acc.width + 2 * roiMargin);
						acc.height = std::min(frameSmall.rows - acc.y, acc.height + 2 * roiMargin);
						lockedRoi = acc;
						haarLocked = true;
					}
				}
			}
			
			// Step 3: Extract working region
			cv::Mat working;
			if (haarLocked) {
				working = frameSmall(lockedRoi).clone();
				currentRoi = lockedRoi;
			} else {
				working = frameSmall;
				currentRoi = cv::Rect(0, 0, frameSmall.cols, frameSmall.rows);
			}
			
			// Step 4: Resize ROI if it's too small (KEY REQUIREMENT)
			// This ensures detector/purest run on properly sized images
			double originalHeight = working.rows;
			cv::Mat workingResized = vision::scale::resizeToHeight(working, vision::scale::kDefaultHeight);
			roiScaleFactor = (originalHeight > 0) ? (workingResized.rows / originalHeight) : 1.0;
			cv::cvtColor(workingResized, workingGray, cv::COLOR_BGR2GRAY);

			// Optional: Morphological closing to reduce noise
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
			cv::morphologyEx(workingGray, workingGray, cv::MORPH_CLOSE, kernel);
			
			// Store for visualization
			workingFrame = workingResized.clone();
			
			// Step 5: Preprocessing
			cv::Mat enhanced = vision::pre::enhanceForPupil(workingGray, 2.0, cv::Size(6, 6), 5, 40.0, 5.0, 1.0, 0.8);
			cv::Mat clahe = enhanced;
			
			// Step 6: Detection/tracking
			Pupil pupil = detectPupil(clahe);
			
			// Step 7: Update previous pupil for tracking
			if (pupil.size.width > 0) {
				prevPupil = pupil;
				hasPrevPupil = true;
			}
			
			// Step 8: Validation and smoothing
			bool valid = false;
			double contrastScore = 0.0;
			if (pupil.size.width > 0) {
				valid = validatePupil(pupil, workingGray);
				if (valid) {
					// Compute contrast score for smoothing
					float rIn = 0.25f * (float)std::min(pupil.size.width, pupil.size.height);
					float rOut = 0.5f * (float)std::max(pupil.size.width, pupil.size.height);
					
					auto insideMean = [&](const cv::Point2f& c, float r) -> double {
						int x0 = std::max(0, (int)(c.x - r)), x1 = std::min(workingGray.cols - 1, (int)(c.x + r));
						int y0 = std::max(0, (int)(c.y - r)), y1 = std::min(workingGray.rows - 1, (int)(c.y + r));
						double sum = 0; int cnt = 0;
						for (int y = y0; y <= y1; ++y) {
							const uchar* row = workingGray.ptr<uchar>(y);
							for (int x = x0; x <= x1; ++x) {
								float dx = x - c.x, dy = y - c.y;
								if (dx * dx + dy * dy <= r * r) { sum += row[x]; cnt++; }
							}
						}
						return cnt ? sum / cnt : 255;
					};
					
					auto ringMean = [&](const cv::Point2f& c, float r1, float r2) -> double {
						int x0 = std::max(0, (int)(c.x - r2)), x1 = std::min(workingGray.cols - 1, (int)(c.x + r2));
						int y0 = std::max(0, (int)(c.y - r2)), y1 = std::min(workingGray.rows - 1, (int)(c.y + r2));
						double sum = 0; int cnt = 0;
						for (int y = y0; y <= y1; ++y) {
							const uchar* row = workingGray.ptr<uchar>(y);
							for (int x = x0; x <= x1; ++x) {
								float dx = x - c.x, dy = y - c.y;
								float d2 = dx * dx + dy * dy;
								if (d2 <= r2 * r2 && d2 > r1 * r1) { sum += row[x]; cnt++; }
							}
						}
						return cnt ? sum / cnt : 0;
					};
					
					double mIn = insideMean(pupil.center, rIn);
					double mOut = ringMean(pupil.center, rIn, rOut);
					contrastScore = (mOut - mIn);
					updateSmooth(pupil, contrastScore);
				}
			}
			
			// Return smoothed pupil if available, otherwise raw (in WORKING space)
			Pupil result = hasSmooth ? smoothPupil : pupil;
			// Store working-space pupil for drawing with getWorkingFrame
			lastWorkingPupil = result;
			
			// Transform coordinates back to frame space for external consumers
			return transformToFrameSpace(result);
		}
		
		Pupil PupilDetector::transformToFrameSpace(const Pupil& p) const {
			if (p.size.width <= 0) return p;
			
			Pupil transformed = p;
			
			// If ROI was resized, scale coordinates back
			if (roiScaleFactor != 1.0 && roiScaleFactor > 0) {
				transformed.center.x /= roiScaleFactor;
				transformed.center.y /= roiScaleFactor;
				transformed.size.width /= roiScaleFactor;
				transformed.size.height /= roiScaleFactor;
			}
			
			// If Haar locked, shift coordinates to frame space
			if (haarLocked) {
				transformed.shift(cv::Point2f(currentRoi.x, currentRoi.y));
			}
			
			return transformed;
		}
		
		Pupil PupilDetector::detectPupil(const cv::Mat& clahe) {
			Pupil pupil;
			
			// Detection/tracking stack: PuRe for init, PuReST for tracking
			if (hasPrevPupil && haarLocked) {
				cv::Rect full(0, 0, clahe.cols, clahe.rows);
				cv::Rect roi = full;
				Pupil tracked;
				purest.run(clahe, roi, prevPupil, tracked);
				if (tracked.size.width > 0) {
					//std::cout << "Tracked pupil at: " << tracked.center << " size: " << tracked.size << std::endl;
					pupil = tracked;
				} else {
					// Fallback to full detection
					detector.run(clahe, pupil);
				}
			} else {
				detector.run(clahe, pupil);
			}
			
			return pupil;
		}
		
		bool PupilDetector::validatePupil(const Pupil& p, const cv::Mat& gray) const {
			if (p.size.width <= 0) return false;
			
			auto ellipseAspect = [](const Pupil& p) {
				double r = p.size.width / p.size.height;
				if (r < 1) r = 1.0 / r;
				return r;
			};
			
			auto insideMean = [&](const cv::Point2f& c, float r) -> double {
				int x0 = std::max(0, (int)(c.x - r)), x1 = std::min(gray.cols - 1, (int)(c.x + r));
				int y0 = std::max(0, (int)(c.y - r)), y1 = std::min(gray.rows - 1, (int)(c.y + r));
				double sum = 0; int cnt = 0;
				for (int y = y0; y <= y1; ++y) {
					const uchar* row = gray.ptr<uchar>(y);
					for (int x = x0; x <= x1; ++x) {
						float dx = x - c.x, dy = y - c.y;
						if (dx * dx + dy * dy <= r * r) { sum += row[x]; cnt++; }
					}
				}
				return cnt ? sum / cnt : 255;
			};
			
			auto ringMean = [&](const cv::Point2f& c, float r1, float r2) -> double {
				int x0 = std::max(0, (int)(c.x - r2)), x1 = std::min(gray.cols - 1, (int)(c.x + r2));
				int y0 = std::max(0, (int)(c.y - r2)), y1 = std::min(gray.rows - 1, (int)(c.y + r2));
				double sum = 0; int cnt = 0;
				for (int y = y0; y <= y1; ++y) {
					const uchar* row = gray.ptr<uchar>(y);
					for (int x = x0; x <= x1; ++x) {
						float dx = x - c.x, dy = y - c.y;
						float d2 = dx * dx + dy * dy;
						if (d2 <= r2 * r2 && d2 > r1 * r1) { sum += row[x]; cnt++; }
					}
				}
				return cnt ? sum / cnt : 0;
			};
			
			float rIn = 0.25f * (float)std::min(p.size.width, p.size.height);
			float rOut = 0.5f * (float)std::max(p.size.width, p.size.height);
			double mIn = insideMean(p.center, rIn);
			double mOut = ringMean(p.center, rIn, rOut);
			double contrastScore = (mOut - mIn);
			double asp = ellipseAspect(p);
			double area = CV_PI * 0.25 * p.size.width * p.size.height;
			double minArea = 0.0002 * gray.total();
			double maxArea = 0.15 * gray.total();
			
			return (contrastScore > 8.0) && (asp < 3.5) && (area > minArea) && (area < maxArea);
			//return (contrastScore > 8.0) && (asp < 6.0) && (area > minArea) && (area < maxArea);
		}
		
		void PupilDetector::updateSmooth(const Pupil& p, double contrastScore) {
			if (!hasSmooth) {
				smoothPupil = p;
				hasSmooth = true;
			} else {
				float eta = 0.25f;
				if (contrastScore < 15.0) eta = 0.15f;
				cv::Point2f c = smoothPupil.center * (1.0f - eta) + p.center * eta;
				cv::Size2f s = smoothPupil.size * (1.0f - eta) + p.size * eta;
				float a0 = smoothPupil.angle, a1 = p.angle, da = a1 - a0;
				if (da > 180) da -= 360;
				if (da < -180) da += 360;
				float a = a0 + eta * da;
				if (a < 0) a += 360;
				if (a >= 360) a -= 360;
				smoothPupil.center = c;
				smoothPupil.size = s;
				smoothPupil.angle = a;
			}
		}
	}
}

