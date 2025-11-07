#pragma once
#include <opencv2/opencv.hpp>
#include "Pure.h"
#include "PuReST.h"
#include "haarcascade.h"
#include "Preprocess.h"
#include "Scale.h"

using namespace vision::haar;

namespace vision {
	namespace detection {
		// Unified pupil detection/tracking workflow
		// Encapsulates: Haar locking, ROI resizing, preprocessing, detection, validation, smoothing
		class PupilDetector {
		public:
			PupilDetector(const std::string& faceCascadePath, const std::string& eyeCascadePath);
			
			// Process a frame and return detected pupil
			// Returns the smoothed/validated pupil if available, otherwise raw detection
			Pupil processFrame(const cv::Mat& frame, bool useHaar = false);
			
			// Reset state (e.g., when relocking Haar)
			void reset();
			
			// Get current working region (for visualization)
			cv::Mat getWorkingFrame() const { return workingFrame.clone(); }
			// Get last pupil in working-frame coordinates (matches getWorkingFrame)
			Pupil getWorkingPupil() const { return lastWorkingPupil; }
			
			// Check if Haar is currently locked
			bool isHaarLocked() const { return haarLocked; }
			
			// Transform pupil coordinates from working space to original frame space
			// Returns pupil with coordinates in the resized frame space (not original)
			Pupil transformToFrameSpace(const Pupil& p) const;
			
		private:
			PuRe detector;
			PuReST purest;
			vision::haar::EyeZoomer zoomer;
			
			bool haarLocked;
			cv::Rect lockedRoi;
			int roiMargin;
			
			Pupil prevPupil;
			bool hasPrevPupil;
			
			Pupil smoothPupil;
			bool hasSmooth;
			
			cv::Mat workingFrame;
			cv::Mat workingGray;
			Pupil lastWorkingPupil;
			
			// Coordinate transformation info
			cv::Rect currentRoi;  // ROI in resized frame space
			double roiScaleFactor; // Scale factor applied to ROI
			
			// Internal processing
			Pupil detectPupil(const cv::Mat& frame);
			bool validatePupil(const Pupil& p, const cv::Mat& gray) const;
			void updateSmooth(const Pupil& p, double contrastScore);
		};
	}
}

