#include "RANSAC.h"
#include <vector>
#include <random>
#include <numeric> // For std::iota
#include <algorithm> // For std::shuffle

/**
 * @brief Fits an ellipse to a set of 2D points using the RANSAC algorithm.
 */
cv::RotatedRect fitEllipseRANSAC(
    const std::vector<cv::Point>& points,
    int maxIterations,
    double distanceThreshold,
    int minInliers)
{
    // We need at least 5 points to fit an ellipse
    if (points.size() < 5) {
        return cv::RotatedRect(); // Return invalid rect
    }

    cv::RotatedRect bestEllipse;
    int bestInliersCount = 0;

    // Create a vector of indices to shuffle for random sampling
    std::vector<int> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    for (int i = 0; i < maxIterations; ++i)
    {
        // 1. Select a random subset of 5 points
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<cv::Point> subset;
        for (int j = 0; j < 5; ++j) {
            subset.push_back(points[indices[j]]);
        }

        // 2. Fit a candidate ellipse to the subset
        cv::RotatedRect candidateEllipse = cv::fitEllipse(subset);
        if (candidateEllipse.size.width == 0 || candidateEllipse.size.height == 0) {
            continue; // fitEllipse failed
        }

        // 3. Convert the candidate ellipse to a contour for distance testing
        std::vector<cv::Point> modelContour;
        cv::ellipse2Poly(candidateEllipse.center, candidateEllipse.size,
            candidateEllipse.angle, 0, 360, 5, modelContour);

        if (modelContour.empty()) {
            continue;
        }

        // 4. Count inliers
        int currentInliersCount = 0;
        std::vector<cv::Point> currentInlierSet;

        for (const auto& point : points)
        {
            // Use pointPolygonTest to find the signed distance to the contour.
            // A point "on" the contour is 0.
            double dist = cv::pointPolygonTest(modelContour, point, true);

            if (std::abs(dist) < distanceThreshold)
            {
                currentInliersCount++;
                currentInlierSet.push_back(point);
            }
        }

        // 5. Update the best model if this one is better
        if (currentInliersCount > bestInliersCount)
        {
            bestInliersCount = currentInliersCount;
            // Refit the ellipse using *all* the inliers from this set
            // to get a more accurate model.
            if (currentInlierSet.size() >= 5) {
                bestEllipse = cv::fitEllipse(currentInlierSet);
            }
            else {
                bestEllipse = candidateEllipse; // Fallback to the subset model
            }
        }
    }

    // 6. Check if the best model meets our minimum criteria
    if (bestInliersCount < minInliers) {
        return cv::RotatedRect(); // Return invalid rect
    }

    return bestEllipse;
}