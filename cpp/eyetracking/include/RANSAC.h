#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief Fits an ellipse to a set of 2D points using the RANSAC algorithm.
 *
 * This function is robust to outliers. It iteratively samples subsets of
 * points, fits a candidate ellipse, and finds the model that is
 * supported by the largest set of inliers.
 *
 * @param points The input vector of 2D points to fit.
 * @param maxIterations The maximum number of RANSAC iterations to perform.
 * @param distanceThreshold The maximum distance (in pixels) from a point to
 * the ellipse contour for it to be considered an inlier.
 * @param minInliers The minimum number of inliers required to consider a
 * model valid. If no model reaches this, an invalid
 * RotatedRect is returned.
 *
 * @return A cv::RotatedRect representing the best-fit ellipse. If a
 * sufficiently good model cannot be found (e.g., not enough inliers),
 * the returned RotatedRect will have size.width = 0 and size.height = 0.
 */
cv::RotatedRect fitEllipseRANSAC(
    const std::vector<cv::Point>& points,
    int maxIterations = 100,
    double distanceThreshold = 1.5,
    int minInliers = 10
);