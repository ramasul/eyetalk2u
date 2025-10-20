#include "PuRe.h"
#include "Resize.h"
#include "Utils.h"
#include "EdgeDetection.h"
#include "EdgeProcessing.h"
#include "HistEq.h"

#include <bitset>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <queue>
#include <tuple>
#include <array>

namespace pure {   
    Result Detector::detect(const cv::Mat& input_img, cv::Mat* debug_color_img)
    {
        // setup debugging
        debug = debug_color_img != nullptr;
        // NOTE: when debugging, the debug_color_img pointer will get filled in the end.
        // The intermediate debug image will potentially be downscaled in order to match
        // the working image. It will be upscaled to input image size in the end.

        // preprocessing
        if (!preprocess(input_img))
        {
            // preprocessing can fail for invalid parameters
            // TODO: add debug information?
            // still postprocess in order to return correct debug image
            Result dummy_result;
            postprocess(dummy_result, input_img, debug_color_img);
            return dummy_result;
        }

        detect_edges();

        if (debug)
        {
            // draw edges onto debug view
            cv::Mat edge_color;
            cvtColor(edge_img, edge_color, cv::COLOR_GRAY2BGR);
            debug_img = max(debug_img, 0.5 * edge_color);
        }

        select_edge_segments();
        combine_segments();

        if (debug)
        {
            // Draw all non-zero-confidence segments onto debug view with color coding
            // for confidence. Red: confidence == 0, green: confidence == 1 
            for (size_t i = 0; i < segments.size(); ++i)
            {
                const auto& segment = segments[i];
                const auto& result = candidates[i];
                const double c = result.confidence.value;
                if (c == 0) continue;
                const cv::Scalar color(0, 255 * std::min(1.0, 2.0 * c), 255 * std::min(1.0, 2.0 * (1 - c)));

                cv::Mat blend = debug_img.clone();
                ellipse(
                    blend,
                    cv::Point(result.center),
                    cv::Size(result.axes),
                    result.angle,
                    0, 360,
                    color,
                    cv::FILLED
                );
                debug_img = 0.9 * debug_img + 0.1 * blend;
                polylines(debug_img, segment, false, 0.8 * color);
            }

            // Draw ellipse min/max indicators onto debug view.
            cv::Point center(orig_img.cols / 2, orig_img.rows / 2);
            cv::Size size(orig_img.cols, orig_img.rows);
            const cv::Scalar white(255, 255, 255);
            const cv::Scalar black(0, 0, 0);
            const cv::Scalar blue(255, 150, 0);
            cv::Mat mask = cv::Mat::zeros(size, CV_8UC3);
            const int min_pupil_radius = static_cast<int>(round(min_pupil_diameter / 2));
            const int max_pupil_radius = static_cast<int>(round(max_pupil_diameter / 2));
            circle(mask, center, max_pupil_radius, white, cv::FILLED);
            circle(mask, center, min_pupil_radius, black, cv::FILLED);
            cv::Mat colored(size, CV_8UC3, blue);
            colored = min(mask, colored);
            debug_img = debug_img * 0.9 + colored * 0.1;
            circle(debug_img, center, max_pupil_radius, blue);
            circle(debug_img, center, min_pupil_radius, blue);
        }

        Result final_result = select_final_segment();

        // draw confidence and pupil diameter info
        if (debug)
        {
            // confidence indicator
            const double c = final_result.confidence.value;
            const cv::Scalar color(0, 255 * std::min(1.0, 2.0 * c), 255 * std::min(1.0, 2.0 * (1 - c)));
            int decimal = static_cast<int>(round(c * 10));
            std::string confidence_string = decimal >= 10 ? "1.0" : "0." + std::to_string(decimal);
            float font_scale = 0.4f;
            const cv::Scalar white(255, 255, 255);
            int pos = static_cast<int>(round(c * debug_img.cols));
            line(debug_img, cv::Point(pos, debug_img.rows), cv::Point(pos, debug_img.rows - 20), color, 2);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(confidence_string, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
            cv::putText(debug_img, confidence_string, cv::Point(c < 0.5 ? pos : pos - text_size.width, debug_img.rows - 20), cv::FONT_HERSHEY_SIMPLEX, font_scale, white);

            // if conf > 0, visualize pupil diameter
            if (c > 0)
            {
                cv::Point center(orig_img.cols / 2, orig_img.rows / 2);
                const cv::Scalar green(0, 255, 0);
                int diameter = static_cast<int>(round(std::max(final_result.axes.width, final_result.axes.height)));
                circle(debug_img, center, diameter, green);

                const float inverse_factor = scaling_factor != 0.0f ? static_cast<float>(1.0f / scaling_factor) : 1.0f;
                std::string diameter_text = std::to_string(diameter);
                text_size = cv::getTextSize(diameter_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
                cv::Point text_offset = cv::Point(text_size.width, -text_size.height) / 2;
                cv::putText(debug_img, diameter_text, center - text_offset, cv::FONT_HERSHEY_SIMPLEX, font_scale, white);
            }
        }

        postprocess(final_result, input_img, debug_color_img);

        return final_result;
    }

    bool Detector::preprocess(const cv::Mat& input_img) {
        constexpr int target_width = 192;
        constexpr int target_height = 192;
        constexpr int target_area = target_width * target_height;
        int input_area = input_img.cols * input_img.rows;

        // Cek apakah perlu dikecilin?
        if (input_area > target_area) {
            scaling_factor = sqrt(target_area / (double)input_area);
            orig_img = vision::resize::resize(input_img, cv::Size(0, 0), scaling_factor, scaling_factor, vision::resize::INTER_AREA);
        }
        else {
            scaling_factor = 0.0;
            orig_img = input_img.clone();
        }

        //vision::histeq::CLAHE(orig_img, orig_img, 2.0, cv::Size(8, 8));

        cv::normalize(orig_img, orig_img, 0, 255, cv::NORM_MINMAX);

        if (debug) {
            // Init debug view
			cv::cvtColor(orig_img, debug_img, cv::COLOR_GRAY2BGR);
			debug_img *= 0.4; // Dimmer
        }

        const double diameter_scaling_factor = scaling_factor == 0 ? 1.0 : scaling_factor;
        if (params.auto_pupil_diameter)
        {
            // compute automatic pupil radius bounds
            constexpr double min_pupil_diameter_ratio = 0.07 * 2 / 3;
            constexpr double max_pupil_diameter_ratio = 0.29;
            const double diagonal = sqrt(orig_img.cols * orig_img.cols + orig_img.rows * orig_img.rows);

            min_pupil_diameter = min_pupil_diameter_ratio * diagonal;
            max_pupil_diameter = max_pupil_diameter_ratio * diagonal;

            // report computed parameters back (scaled back)
            params.min_pupil_diameter = min_pupil_diameter / diameter_scaling_factor;
            params.max_pupil_diameter = max_pupil_diameter / diameter_scaling_factor;
        }
        else
        {
            // scale input parameters
            min_pupil_diameter = params.min_pupil_diameter * diameter_scaling_factor;
            max_pupil_diameter = params.max_pupil_diameter * diameter_scaling_factor;
        }

        bool success = (
            0 <= min_pupil_diameter &&
            0 <= max_pupil_diameter &&
            min_pupil_diameter <= max_pupil_diameter
        );

        if (!success && debug) {
            cv::putText(debug_img, "Invalid pupil size!", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        return success;
    }

    //Opsional aja
    void Detector::postprocess(Result& final_result, const cv::Mat& input_img, cv::Mat* debug_color_img) {
        //Gedein biar ga kekecilan
        if (scaling_factor != 0.0) {
            const float inverse_factor = static_cast<float>(1.0f / scaling_factor);
            final_result.axes *= inverse_factor;
            final_result.center *= inverse_factor;
        }

        if (debug) {
            if (scaling_factor != 0.0) {
				cv::Size input_size(input_img.cols, input_img.rows);
                vision::resize::resize(debug_img, *debug_color_img, input_size, 0.0, 0.0, vision::resize::INTER_CUBIC);
            }
            else {
                *debug_color_img = debug_img.clone();
            }
        }
    }

    void Detector::detect_edges() {
        edge_img = vision::canny::canny(orig_img, true);
        vision::edge::filterEdges(edge_img);
    }

    void Detector::select_edge_segments()
    {
        cv::findContours(edge_img, segments, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_KCOS);

        // NOTE: We are essentially re-using the result from previous runs. Need to make
        // sure that either all values will be overwritten or confidence will be set to
        // 0 for every result!
        candidates.resize(segments.size());

        for (size_t segment_i = 0; segment_i < segments.size(); ++segment_i)
        {
            evaluate_segment(segments[segment_i], candidates[segment_i]);
        }
    }

    void Detector::evaluate_segment(const Segment& segment, Result& result) const
    {
        // 3.3.1 Filter small segments
        if (!segment_large_enough(segment))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.2 Filter segments based on approximate diameter
        if (!segment_diameter_valid(segment))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.3 Filter segments based on curvature approximation
        if (!segment_curvature_valid(segment))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.4 Ellipse fitting
        if (!fit_ellipse(segment, result))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.5 Additional filter
        if (!segment_mean_in_ellipse(segment, result))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.4  Calculate confidence
        result.confidence = calculate_confidence(segment, result);
    }

    inline bool Detector::segment_large_enough(const Segment& segment) const
    {
        return segment.size() >= 5;
    }

    bool Detector::segment_diameter_valid(const Segment& segment) const
    {
        double approx_diameter = 0;
        const auto end = segment.end();
        for (auto p1 = segment.begin(); p1 != end; ++p1)
        {
            for (auto p2 = p1 + 1; p2 != end; ++p2)
            {
                approx_diameter = std::max(approx_diameter, cv::norm(*p1 - *p2));
                // we can early exit, because we will only get bigger
                if (approx_diameter > max_pupil_diameter)
                {
                    break;
                }
            }
            // we can early exit, because we will only get bigger
            if (approx_diameter > max_pupil_diameter)
            {
                break;
            }
        }
        return min_pupil_diameter < approx_diameter && approx_diameter < max_pupil_diameter;
    }

    bool Detector::segment_curvature_valid(const Segment& segment) const
    {
        auto rect = cv::minAreaRect(segment);
        double ratio = rect.size.width / rect.size.height;
        return !axes_ratio_is_invalid(ratio);
    }

    inline bool Detector::axes_ratio_is_invalid(double ratio) const
    {
        constexpr double axes_ratio_threshold = 0.2;
        constexpr double inverse_threshold = 1.0 / axes_ratio_threshold;

        return ratio < axes_ratio_threshold || ratio > inverse_threshold;
    }

    bool Detector::fit_ellipse(const Segment& segment, Result& result) const
    {
        // NOTE: This is a cv::RotatedRect, see https://stackoverflow.com/a/32798273 for
        // conversion to ellipse parameters. Also below.
        auto fit = cv::fitEllipse(segment);

        // 	(I) discard if center outside image boundaries
        if (fit.center.x < 0 || fit.center.y < 0 || fit.center.x > edge_img.cols || fit.center.y > edge_img.rows)
        {
            return false;
        }

        // 	(II) discard if ellipse is too skewed
        auto ratio = fit.size.width / fit.size.height;
        if (axes_ratio_is_invalid(ratio))
        {
            return false;
        }

        result.center = fit.center;
        result.angle = fit.angle;
        // NOTE: width always provides the first axis, which corresponds to the
        // angle. Height provides the second axis, which corresponds to angle +
        // 90deg. This is NOT related to major/minor axes! But we also don't
        // need the information of which is the major and which is the minor
        // axis.
        result.axes = {
            fit.size.width / 2.0f,
            fit.size.height / 2.0f
        };
        return true;
    }

    bool Detector::segment_mean_in_ellipse(const Segment& segment, const Result& result) const
    {
        cv::Point2f segment_mean(0, 0);
        for (const auto& p : segment)
        {
            segment_mean += cv::Point2f(p);
        }
        // NOTE: cv::Point operator /= does not work with size_t scalar
        segment_mean.x /= segment.size();
        segment_mean.y /= segment.size();

        // We need to test if the mean lies in the rhombus defined by the
        // rotated rect of the ellipse. Essentially each vertex of the
        // rhombus corresponds to a midpoint of the sides of the rect.
        // Testing is easiest if we don't rotate all points of the rect, but
        // rotate the segment_mean backwards, because then we can test
        // against the axis-aligned rhombus.

        // See the following rhombus for reference. Note that we only need
        // to test for Q1, since the we can center at (0,0) and the rest is
        // symmetry. (not in image coordinates, but y-up)
        //    /|\      |
        //   / | \  Q1 |
        //  /  |  \    |
        // ---------
        //  \  |  /
        //   \ | /
        //    \|/

        // Shift rotation to origin to center at (0,0).
        segment_mean -= result.center;
        // Rotate backwards with negative angle
        const auto angle_rad = -result.angle * M_PI / 180.0f;
        const float angle_cos = static_cast<float>(cos(angle_rad));
        const float angle_sin = static_cast<float>(sin(angle_rad));
        // We take the abs values to utilize symmetries. This way can do the
        // entire testing in Q1 of the rhombus.
        cv::Point2f unrotated(
            abs(segment_mean.x * angle_cos - segment_mean.y * angle_sin),
            abs(segment_mean.x * angle_sin + segment_mean.y * angle_cos)
        );

        // Discard based on testing first rhombus quadrant Q1. This tests
        // for containment in the axis-aligned triangle.
        return (
            (unrotated.x < result.axes.width) &&
            (unrotated.y < result.axes.height) &&
            ((unrotated.x / result.axes.width) + (unrotated.y / result.axes.height) < 1)
            );
    }

    Confidence Detector::calculate_confidence(const Segment& segment, const Result& result) const
    {
        Confidence conf;
        conf.aspect_ratio = result.axes.width / result.axes.height;
        if (conf.aspect_ratio > 1.0) conf.aspect_ratio = 1.0 / conf.aspect_ratio;

        conf.angular_spread = angular_edge_spread(segment, result);
        conf.outline_contrast = ellipse_outline_contrast(result);

        // compute value
        conf.value = (conf.aspect_ratio + conf.angular_spread + conf.outline_contrast) / 3.0;

        return conf;
    }

    double angular_edge_spread(const Segment& segment, const Result& result)
    {
        // Q2 | Q1
        // -------
        // Q3 | Q4
        // (not in image coordinates, but y-up)

        std::bitset<8> bins;

        for (const auto& p : segment)
        {
            const auto v = cv::Point2f(p.x - result.center.x, p.y - result.center.y);

            if (v.x > 0)
            {
                if (v.y > 0)
                {
                    if (v.x > v.y) bins[1] = true;
                    else bins[0] = true;
                }
                else
                {
                    if (v.x > v.y) bins[2] = true;
                    else bins[3] = true;
                }
            }
            else
            {
                if (v.y > 0)
                {
                    if (v.x > v.y) bins[7] = true;
                    else bins[6] = true;
                }
                else
                {
                    if (v.x > v.y) bins[4] = true;
                    else bins[5] = true;
                }
            }
            // early exit
            if (bins.count() == 8) break;
        }

        return bins.count() / 8.0;
    }

    double Detector::angular_edge_spread(const Segment& segment, const Result& result) const
    {
        // Divide the circle into 8 octants (45 degrees each)
        // Using image coordinates (y-down):
        //     7   0   1
        //      \ | /
        //   6 -- + -- 2
        //      / | \
        //     5   4   3

        std::bitset<8> bins;

        for (const auto& p : segment)
        {
            const float dx = p.x - result.center.x;
            const float dy = p.y - result.center.y;

            const float abs_x = std::abs(dx);
            const float abs_y = std::abs(dy);

            int octant;

            if (abs_x > abs_y) {
                // Dominated by horizontal component
                if (dx > 0) {
                    octant = (dy > 0) ? 3 : 1;  // Right-bottom or Right-top
                }
                else {
                    octant = (dy > 0) ? 5 : 7;  // Left-bottom or Left-top
                }
            }
            else {
                // Dominated by vertical component
                if (dy > 0) {
                    octant = (dx > 0) ? 4 : 6;  // Bottom-right or Bottom-left
                }
                else {
                    octant = (dx > 0) ? 0 : 2;  // Top-right or Top-left
                }
            }

            bins[octant] = true;

            // Early exit if all octants are filled
            if (bins.count() == 8) break;
        }

        return bins.count() / 8.0;
    }

    double Detector::ellipse_outline_contrast(const Result& result) const
    {
        double contrast = 0;
        constexpr double radian_per_degree = M_PI / 180.0;
        // Iterate circle with stride of 10 degrees (all in radians)
        constexpr double stride = 10 * radian_per_degree;
        double theta = 0;
        // NOTE: A for-loop: for(theta=0; theta < 2*PI; ...) will result
        // in 37 iterations because of rounding errors. This will result
        // in one line being counted twice.
        constexpr int n_iterations = 36;
        const double minor = std::min(result.axes.width, result.axes.height);
        const double cos_angle = cos(result.angle * radian_per_degree);
        const double sin_angle = sin(result.angle * radian_per_degree);
        const cv::Rect bounds = cv::Rect(0, 0, orig_img.cols, orig_img.rows);
        constexpr int bias = 5;
        // Mat tmp;
        // cvtColor(orig_img, tmp, COLOR_GRAY2BGR);
        for (int i = 0; i < n_iterations; ++i)
        {
            const double x = result.axes.width * cos(theta);
            const double y = result.axes.height * sin(theta);
            cv::Point2f offset(
                static_cast<float>(x * cos_angle - y * sin_angle),
                static_cast<float>(y * cos_angle + x * sin_angle)
            );
            cv::Point2f outline_point = result.center + offset;

            cv::Point2f offset_norm = offset / cv::norm(offset);
            cv::Point2f inner_pt = outline_point - (0.3 * minor) * offset_norm;
            cv::Point2f outer_pt = outline_point + (0.3 * minor) * offset_norm;

            if (!bounds.contains(inner_pt) || !bounds.contains(outer_pt))
            {
                theta += stride;
                continue;
            }

            double inner_avg = 0;
            cv::LineIterator inner_line(orig_img, inner_pt, outline_point);
            for (int j = 0; j < inner_line.count; j++, ++inner_line)
            {
                inner_avg += *(*inner_line);
            }
            inner_avg /= inner_line.count;

            double outer_avg = 0;
            cv::LineIterator outer_line(orig_img, outline_point, outer_pt);
            for (int j = 0; j < outer_line.count; j++, ++outer_line)
            {
                outer_avg += *(*outer_line);
            }
            outer_avg /= outer_line.count;

            if (inner_avg + bias < outer_avg) contrast += 1;

            // if (inner_avg + bias < outer_avg)
            //     line(tmp, inner_pt, outer_pt, Scalar(0, 255, 0));
            // else
            //     line(tmp, inner_pt, outer_pt, Scalar(0, 0, 255));

            theta += stride;
        }

        // imshow("pfa", tmp);
        // waitKey(-1);
        return contrast / n_iterations;
    }

    void Detector::combine_segments()
    {
        std::vector<Segment> combined_segments;
        std::vector<Result> combined_results;
        if (segments.size() == 0) return;
        size_t end1 = segments.size() - 1;
        size_t end2 = segments.size();
        for (size_t idx1 = 0; idx1 < end1; ++idx1)
        {
            auto& result1 = candidates[idx1];
            if (result1.confidence.value == 0) continue;
            auto& segment1 = segments[idx1];
            const auto rect1 = cv::boundingRect(segment1);
            for (size_t idx2 = idx1 + 1; idx2 < end2; ++idx2)
            {
                auto& result2 = candidates[idx2]; // Sic! Tadi ada bug disini idx1
                if (result2.confidence.value == 0) continue;
                auto& segment2 = segments[idx2];
                const auto rect2 = cv::boundingRect(segment2);

                if (proper_intersection(rect1, rect2))
                {
                    auto new_segment = merge_segments(segment1, segment2);
                    Result new_result;
                    evaluate_segment(new_segment, new_result);
                    if (new_result.confidence.value == 0) continue;
                    const auto previous_contrast = std::max(
                        result1.confidence.outline_contrast,
                        result2.confidence.outline_contrast
                    );
                    if (new_result.confidence.outline_contrast <= previous_contrast) continue;

                    combined_segments.push_back(new_segment);
                    combined_results.push_back(new_result);
                }
            }
        }
        segments.insert(segments.end(), combined_segments.begin(), combined_segments.end());
        candidates.insert(candidates.end(), combined_results.begin(), combined_results.end());
    }

    bool Detector::proper_intersection(const cv::Rect& r1, const cv::Rect& r2) const
    {
        const cv::Rect r = r1 & r2; // intersection
        return r.area() > 0 && r != r1 && r != r2;
    }

    Segment Detector::merge_segments(const Segment& s1, const Segment& s2) const
    {
        // Naive approach is to just take the convex hull of the union of both segments.
        // But there is no documentation.
        Segment combined;
        combined.insert(combined.end(), s1.begin(), s1.end());
        combined.insert(combined.end(), s2.begin(), s2.end());
        Segment hull;
        // NOTE: convexHull does not support in-place computation.
        cv::convexHull(combined, hull);
        return hull;
    }

    Result Detector::select_final_segment()
    {
        if (candidates.size() == 0)
        {
            return Result();
        }
        Result* initial_pupil = &*std::max_element(candidates.begin(), candidates.end());
        double semi_major = std::max(initial_pupil->axes.width, initial_pupil->axes.height);
        Result* candidate = nullptr;
        for (auto& result : candidates)
        {
            if (result.confidence.value == 0) continue;
            if (result.confidence.outline_contrast < 0.75) continue;
            if (&result == initial_pupil) continue;
            // NOTE: The initial paper mentions to discard candidates with a diameter
            // larger than the initial pupil's semi major, i.e. only candidates that are
            // half the size are considered. In dark environments this leads to bad
            // results though, as the pupil can be up to 80% of the iris size. Therefore
            // we use 0.8 as threshold.
            if (std::max(result.axes.width, result.axes.height) > 0.8 * semi_major) continue;
            if (norm(initial_pupil->center - result.center) > semi_major) continue;
            if (candidate && result.confidence.value <= candidate->confidence.value) continue;
            candidate = &result;
        }
        return (candidate) ? *candidate : *initial_pupil;
    }
}
