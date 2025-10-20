#include "EdgeProcessing.h"
#include "Utils.h"

namespace vision {
	namespace edge {
        // My Own Implementation (Rama)
        void filterEdges(cv::Mat& edges)
        {
            // Preconditions
            CV_Assert(!edges.empty());
            CV_Assert(edges.type() == CV_8UC1);

            const int margin = 5;
            const int start_x = margin;
            const int start_y = margin;
            const int end_x = edges.cols - margin;
            const int end_y = edges.rows - margin;

            // Ensure image is at least big enough for the largest neighborhood (±3 used below)
            CV_Assert(edges.cols > margin * 2 && edges.rows > margin * 2);

            // Create two buffers: src (initial snapshot) and tmp (destination reused)
            cv::Mat src = edges.clone();                 // read-only for the first pass
            cv::Mat tmp(edges.size(), edges.type());     // destination buffer (reused)
            cv::Mat dst = tmp;

            // Helper macro to swap src/dst
            auto swap_buffers = [&]() {
                std::swap(src, dst);
                };


            // ---- PASS 1: 3x3 cross pruning (original first stage) ----
            for (int j = start_y; j < end_y; ++j) {
                const uchar* row_m1 = src.ptr<uchar>(j - 1);
                const uchar* row = src.ptr<uchar>(j);
                const uchar* row_p1 = src.ptr<uchar>(j + 1);
                uchar* dstRow = dst.ptr<uchar>(j);

                // copy left margin and right margin untouched for this row (optional)
                // but we keep writing only central region
                for (int i = start_x; i < end_x; ++i) {
                    uchar center = row[i];
                    if (!center) {
                        dstRow[i] = 0;
                        continue;
                    }

                    uchar n1 = row_m1[i];    // up
                    uchar n3 = row[i - 1];   // left
                    uchar n5 = row[i + 1];   // right
                    uchar n7 = row_p1[i];    // down

                    if ((n5 && n7) || (n5 && n1) || (n3 && n7) || (n3 && n1))
                        dstRow[i] = 0;
                    else
                        dstRow[i] = center;
                }
            }
            swap_buffers();

            // ---- PASS 2: too many neighbours (>3) in 3x3 ----
            for (int j = start_y; j < end_y; ++j) {
                const uchar* row_m1 = src.ptr<uchar>(j - 1);
                const uchar* row = src.ptr<uchar>(j);
                const uchar* row_p1 = src.ptr<uchar>(j + 1);
                uchar* dstRow = dst.ptr<uchar>(j);

                for (int i = start_x; i < end_x; ++i) {
                    // count 3x3 neighbourhood (including center)
                    int neigh = 0;
                    neigh += (row_m1[i - 1] > 0);
                    neigh += (row_m1[i] > 0);
                    neigh += (row_m1[i + 1] > 0);
                    neigh += (row[i - 1] > 0);
                    neigh += (row[i] > 0);
                    neigh += (row[i + 1] > 0);
                    neigh += (row_p1[i - 1] > 0);
                    neigh += (row_p1[i] > 0);
                    neigh += (row_p1[i + 1] > 0);

                    if (neigh > 3)
                        dstRow[i] = 0;
                    else
                        dstRow[i] = row[i];
                }
            }
            swap_buffers();

            // ---- PASS 3: medium-range corrections (the box[17] logic) ----
            // we need rows j-1 .. j+2 and columns up to i+3 so margin=5 is safe
            for (int j = start_y; j < end_y; ++j) {
                const uchar* r_m1 = src.ptr<uchar>(j - 1);
                const uchar* r = src.ptr<uchar>(j);
                const uchar* r_p1 = src.ptr<uchar>(j + 1);
                const uchar* r_p2 = src.ptr<uchar>(j + 2);
                uchar* dstRow = dst.ptr<uchar>(j);

                for (int i = start_x; i < end_x; ++i) {
                    uchar center = r[i];
                    if (!center) { dstRow[i] = 0; continue; }

                    // 3x3 neighbors
                    uchar b0 = r_m1[i - 1];
                    uchar b1 = r_m1[i];
                    uchar b2 = r_m1[i + 1];
                    uchar b3 = r[i - 1];
                    uchar b5 = r[i + 1];
                    uchar b6 = r_p1[i - 1];
                    uchar b7 = r_p1[i];
                    uchar b8 = r_p1[i + 1];

                    // extended neighbours
                    uchar b9 = r[i + 2];         // src(j, i+2)
                    uchar b10 = r_p2[i];         // src(j+2, i)
                    uchar b11 = r[i + 3];        // src(j, i+3)
                    uchar b12 = r_m1[i + 2];     // src(j-1, i+2)
                    uchar b13 = r_p1[i + 2];     // src(j+1, i+2)
                    uchar b14 = src.ptr<uchar>(j + 3)[i]; // src(j+3, i)
                    uchar b15 = r_p2[i - 1];     // src(j+2, i-1)
                    uchar b16 = r_p2[i + 1];     // src(j+2, i+1)

                    // replicate original rules
                    if ((b10 && !b7) && (b8 || b6)) {
                        dst.ptr<uchar>(j + 1)[i - 1] = 0;
                        dst.ptr<uchar>(j + 1)[i + 1] = 0;
                        dstRow[i] = 255;
                        continue;
                    }

                    if ((b14 && !b7 && !b10) && ((b8 || b6) && (b16 || b15))) {
                        dst.ptr<uchar>(j + 1)[i + 1] = 0;
                        dst.ptr<uchar>(j + 1)[i - 1] = 0;
                        dst.ptr<uchar>(j + 2)[i + 1] = 0;
                        dst.ptr<uchar>(j + 2)[i - 1] = 0;
                        dst.ptr<uchar>(j + 1)[i] = 255;
                        dst.ptr<uchar>(j + 2)[i] = 255;
                        continue;
                    }

                    if ((b9 && !b5) && (b8 || b2)) {
                        dst.ptr<uchar>(j + 1)[i + 1] = 0;
                        dst.ptr<uchar>(j - 1)[i + 1] = 0;
                        dstRow[i + 1] = 255; // dst at this row, col i+1
                        dstRow[i] = r[i];    // keep center (no explicit center set in original; keep original)
                        continue;
                    }

                    if ((b11 && !b5 && !b9) && ((b8 || b2) && (b13 || b12))) {
                        dst.ptr<uchar>(j + 1)[i + 1] = 0;
                        dst.ptr<uchar>(j - 1)[i + 1] = 0;
                        dst.ptr<uchar>(j + 1)[i + 2] = 0;
                        dst.ptr<uchar>(j - 1)[i + 2] = 0;
                        dstRow[i + 1] = 255;
                        dst.ptr<uchar>(j)[i + 2] = 255;
                        dstRow[i] = r[i];
                        continue;
                    }

                    // if no rule matched, copy original center
                    dstRow[i] = r[i];
                }
            }
            swap_buffers();

            // ---- PASS 4: larger-range corrections (the big box[33] logic) ----
            // rows j-3 .. j+3 and cols i-3 .. i+3 (margin ensures safe access)
            for (int j = start_y; j < end_y; ++j) {
                // gather pointers for rows j-3 .. j+3 (helps performance)
                const uchar* rp3 = src.ptr<uchar>(j - 3);
                const uchar* rp2 = src.ptr<uchar>(j - 2);
                const uchar* rp1 = src.ptr<uchar>(j - 1);
                const uchar* r = src.ptr<uchar>(j);
                const uchar* rn1 = src.ptr<uchar>(j + 1);
                const uchar* rn2 = src.ptr<uchar>(j + 2);
                const uchar* rn3 = src.ptr<uchar>(j + 3);
                uchar* dstRow = dst.ptr<uchar>(j);

                for (int i = start_x; i < end_x; ++i) {
                    uchar center = r[i];
                    if (!center) { dstRow[i] = 0; continue; }

                    uchar b0 = rp1[i - 1];
                    uchar b1 = rp1[i];
                    uchar b2 = rp1[i + 1];
                    uchar b3 = r[i - 1];
                    uchar b5 = r[i + 1];
                    uchar b6 = rn1[i - 1];
                    uchar b7 = rn1[i];
                    uchar b8 = rn1[i + 1];

                    uchar b9 = rp1[i + 2];
                    uchar b10 = rp1[i - 2];
                    uchar b11 = rn1[i + 2];
                    uchar b12 = rn1[i - 2];

                    uchar b13 = rp2[i - 1];
                    uchar b14 = rp2[i + 1];
                    uchar b15 = rn2[i - 1];
                    uchar b16 = rn2[i + 1];

                    uchar b17 = rp3[i - 1];
                    uchar b18 = rp3[i + 1];
                    uchar b19 = rn3[i - 1];
                    uchar b20 = rn3[i + 1];

                    uchar b21 = rn1[i + 3];
                    uchar b22 = rn1[i - 3];
                    uchar b23 = rp1[i + 3];
                    uchar b24 = rp1[i - 3];

                    uchar b25 = rp2[i - 2];
                    uchar b26 = rn2[i + 2];
                    uchar b27 = rp2[i + 2];
                    uchar b28 = rn2[i - 2];

                    uchar b29 = rp3[i - 3];
                    uchar b30 = rn3[i + 3];
                    uchar b31 = rp3[i + 3];
                    uchar b32 = rn3[i - 3];

                    if (b7 && b2 && b9) { dstRow[i] = 0; continue; }
                    if (b7 && b0 && b10) { dstRow[i] = 0; continue; }
                    if (b1 && b8 && b11) { dstRow[i] = 0; continue; }
                    if (b1 && b6 && b12) { dstRow[i] = 0; continue; }

                    if (b0 && b13 && b17 && b8 && b11 && b21) { dstRow[i] = 0; continue; }
                    if (b2 && b14 && b18 && b6 && b12 && b22) { dstRow[i] = 0; continue; }
                    if (b6 && b15 && b19 && b2 && b9 && b23) { dstRow[i] = 0; continue; }
                    if (b8 && b16 && b20 && b0 && b10 && b24) { dstRow[i] = 0; continue; }

                    if (b0 && b25 && b2 && b27) { dstRow[i] = 0; continue; }
                    if (b0 && b25 && b6 && b28) { dstRow[i] = 0; continue; }
                    if (b8 && b26 && b2 && b27) { dstRow[i] = 0; continue; }
                    if (b8 && b26 && b6 && b28) { dstRow[i] = 0; continue; }

                    // box2 tests (reconstructed)
                    uchar box2_1 = r[i - 1];
                    uchar box2_2 = rp1[i - 2];
                    uchar box2_3 = rp2[i - 3];
                    uchar box2_4 = rp1[i + 1];
                    uchar box2_5 = rp2[i + 2];
                    uchar box2_6 = rn1[i - 2];
                    uchar box2_7 = rn2[i - 3];
                    uchar box2_8 = rn1[i + 1];
                    uchar box2_9 = rn2[i + 2];
                    uchar box2_10 = rn1[i];
                    uchar box2_11 = rn2[i + 1];
                    uchar box2_12 = rn3[i + 2];
                    uchar box2_13 = rn2[i - 1];
                    uchar box2_14 = rn3[i - 2];
                    uchar box2_15 = rp1[i - 1];
                    uchar box2_16 = rp2[i - 2];

                    if (box2_1 && box2_2 && box2_3 && box2_4 && box2_5) { dstRow[i] = 0; continue; }
                    if (box2_1 && box2_6 && box2_7 && box2_8 && box2_9) { dstRow[i] = 0; continue; }
                    if (box2_10 && box2_11 && box2_12 && box2_4 && box2_5) { dstRow[i] = 0; continue; }
                    if (box2_10 && box2_13 && box2_14 && box2_15 && box2_16) { dstRow[i] = 0; continue; }

                    // fallback: copy original center
                    dstRow[i] = r[i];
                }
            }
            swap_buffers();

            // After final swap, 'src' holds the latest processed image. Copy back to edges if necessary.
            if (src.data != edges.data) {
                src.copyTo(edges);
            }
        }

        cv::Mat filterEdges(const cv::Mat& edges, cv::Mat& dst)
        {
            dst.create(edges.size(), edges.type());
            edges.copyTo(dst);
            filterEdges(dst);
            return dst;
        }
	}
}