#include <iostream>
#include <cmath>
#include <thread>
#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <librealsense2/rs.hpp>  // rs.h is for C; rs.hpp is for C++...

#include <DarknetNet.h>

#include <darknet/images/image_opencv.h>
#include <darknet/images/http_stream.h>
#include <darknet/images/realsense-opencv-helpers.hpp>

#include <utility>

using namespace cv;
using namespace std;

#define POSE_MAX_PEOPLE 96
#define MODEL 0

template<typename T>
inline int intRound(const T a) {
    return lround(a);
}

template<typename T>
inline T fastMin(const T a, const T b) {
    return (a < b ? a : b);
}

class OpenposePostProcessor {
public:
    OpenposePostProcessor() {
        OpenposePostProcessor::initialize();
    }

    /**
     * // Result for BODY_25 (25 body parts consisting of COCO + foot)
    // const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
    //     {0,  "Nose"},
    //     {1,  "Neck"},
    //     {2,  "RShoulder"},
    //     {3,  "RElbow"},
    //     {4,  "RWrist"},
    //     {5,  "LShoulder"},
    //     {6,  "LElbow"},
    //     {7,  "LWrist"},
    //     {8,  "MidHip"},
    //     {9,  "RHip"},
    //     {10, "RKnee"},
    //     {11, "RAnkle"},
    //     {12, "LHip"},
    //     {13, "LKnee"},
    //     {14, "LAnkle"},
    //     {15, "REye"},
    //     {16, "LEye"},
    //     {17, "REar"},
    //     {18, "LEar"},
    //     {19, "LBigToe"},
    //     {20, "LSmallToe"},
    //     {21, "LHeel"},
    //     {22, "RBigToe"},
    //     {23, "RSmallToe"},
    //     {24, "RHeel"},
    //     {25, "Background"}
    // };
     *
     * @param frame
     * @param keyPoints
     * @param keyShape
     * @param threshold
     * @param scale
     */
    static void renderPoseKeyPoints(Mat &frame, const vector<float> &keyPoints, vector<int> keyShape,
                                    const float threshold, float scale) {
        initialize();

        const int nrKeyPoints = keyShape[1];
        const int pairs_size = OpenposePostProcessor::BODY_PART_PAIRS.size();
        const int number_colors = OpenposePostProcessor::COLORS.size();

        for (int person = 0; person < keyShape[0]; ++person) {
            // Draw lines
            for (int pair = 0u; pair < pairs_size; pair += 2) {
                const int index1 =
                        (int) (person * nrKeyPoints + OpenposePostProcessor::BODY_PART_PAIRS[pair]) * keyShape[2];
                const int index2 =
                        (int) (person * nrKeyPoints + OpenposePostProcessor::BODY_PART_PAIRS[pair + 1]) * keyShape[2];
                if (keyPoints[index1 + 2] > threshold && keyPoints[index2 + 2] > threshold) {
                    const int color_index = OpenposePostProcessor::BODY_PART_PAIRS[pair + 1] * 3;
                    Scalar color{OpenposePostProcessor::COLORS[(color_index + 2) % number_colors],
                                 OpenposePostProcessor::COLORS[(color_index + 1) % number_colors],
                                 OpenposePostProcessor::COLORS[(color_index + 0) % number_colors]};
                    Point keyPoint1{intRound(keyPoints[index1] * scale), intRound(keyPoints[index1 + 1] * scale)};
                    Point keyPoint2{intRound(keyPoints[index2] * scale), intRound(keyPoints[index2 + 1] * scale)};
                    line(frame, keyPoint1, keyPoint2, color, 2);
                }
            }
            // Draw circles
            for (int part = 0; part < nrKeyPoints; ++part) {
                const int index = (person * nrKeyPoints + part) * keyShape[2];
                if (keyPoints[index + 2] > threshold) {
                    const int color_index = part * 3;
                    Scalar color{OpenposePostProcessor::COLORS[(color_index + 2) % number_colors],
                                 OpenposePostProcessor::COLORS[(color_index + 1) % number_colors],
                                 OpenposePostProcessor::COLORS[(color_index + 0) % number_colors]};
                    Point center{intRound(keyPoints[index] * scale), intRound(keyPoints[index + 1] * scale)};
                    circle(frame, center, 3, color, -1);
                }
            }
        }
    }

    static void connectBodyparts(
            vector<float> &poseKeypoints, const float *const map, const float *const peaks, int mapW, int mapH,
            const int inter_min_above_th, const float inter_th, const int min_subset_cnt, const float min_subset_score,
            vector<int> &keypoint_shape) {
        initialize();

        keypoint_shape.resize(3);

        const int num_body_parts = MODEL_SIZE - 1;  // model part number
        const int num_body_part_pairs = (int) (OpenposePostProcessor::BODY_PART_PAIRS.size() / 2);
        std::vector<std::pair<std::vector<int>, double>> subset;
        const int subset_counter_index = num_body_parts;
        const int subset_size = num_body_parts + 1;
        const int peaks_offset = 3 * (POSE_MAX_PEOPLE + 1);
        const int map_offset = mapW * mapH;

        for (unsigned int pair_index = 0u; pair_index < num_body_part_pairs; ++pair_index) {
            const int body_partA = OpenposePostProcessor::BODY_PART_PAIRS[2 * pair_index];
            const int body_partB = OpenposePostProcessor::BODY_PART_PAIRS[2 * pair_index + 1];
            const float *candidateA = peaks + body_partA * peaks_offset;
            const float *candidateB = peaks + body_partB * peaks_offset;
            const int nA = (int) (candidateA[0]); // number of part A candidates
            const int nB = (int) (candidateB[0]); // number of part B candidates

            // add parts into the subset in special case
            if (nA == 0 || nB == 0) {
                // Change w.r.t. other
                // nB == 0 or not
                if (nA == 0) {
                    for (int i = 1; i <= nB; ++i) {
                        bool num = false;
                        for (auto &j : subset) {
                            const int off = body_partB * peaks_offset + i * 3 + 2;
                            if (j.first[body_partB] == off) {
                                num = true;
                                break;
                            }
                        }
                        if (!num) {
                            std::vector<int> row_vector(subset_size, 0);
                            // store the index
                            row_vector[body_partB] = body_partB * peaks_offset + i * 3 + 2;
                            // the parts number of that person
                            row_vector[subset_counter_index] = 1;
                            // total score
                            const float subsetScore = candidateB[i * 3 + 2];
                            subset.emplace_back(std::make_pair(row_vector, subsetScore));
                        }
                    }
                } else {
                    // if (nA != 0 && nB == 0)
                    for (int i = 1; i <= nA; i++) {
                        bool num = false;
                        for (auto &j : subset) {
                            const int off = body_partA * peaks_offset + i * 3 + 2;
                            if (j.first[body_partA] == off) {
                                num = true;
                                break;
                            }
                        }
                        if (!num) {
                            std::vector<int> row_vector(subset_size, 0);
                            // store the index
                            row_vector[body_partA] = body_partA * peaks_offset + i * 3 + 2;
                            // parts number of that person
                            row_vector[subset_counter_index] = 1;
                            // total score
                            const float subsetScore = candidateA[i * 3 + 2];
                            subset.emplace_back(std::make_pair(row_vector, subsetScore));
                        }
                    }
                }
            } else {
                std::vector<std::tuple<double, int, int>> temp;
                const int num_inter = 10;
                // limb PAF x-direction heatmap
                const float *const mapX = map + OpenposePostProcessor::POSE_MAP_INDEX[2 * pair_index] * map_offset;
                // limb PAF y-direction heatmap
                const float *const mapY = map + OpenposePostProcessor::POSE_MAP_INDEX[2 * pair_index + 1] * map_offset;
                // start greedy algorithm
                for (int i = 1; i <= nA; i++) {
                    for (int j = 1; j <= nB; j++) {
                        const int dX = (int) (candidateB[j * 3] - candidateA[i * 3]);
                        const int dY = (int) (candidateB[j * 3 + 1] - candidateA[i * 3 + 1]);
                        const auto norm_vec = float(std::sqrt(dX * dX + dY * dY));
                        // If the peaksPtr are coincident. Don't connect them.
                        if (norm_vec > 1e-6) {
                            const float sX = candidateA[i * 3];
                            const float sY = candidateA[i * 3 + 1];
                            const float vecX = ((float) dX) / norm_vec;
                            const float vecY = ((float) dY) / norm_vec;
                            float sum = 0.;
                            int count = 0;
                            for (int lm = 0; lm < num_inter; lm++) {
                                const int mX = fastMin(mapW - 1, intRound(sX + ((float) (lm * dX)) / num_inter));
                                const int mY = fastMin(mapH - 1, intRound(sY + ((float) (lm * dY)) / num_inter));
                                const int idx = mY * mapW + mX;
                                const float score = (vecX * mapX[idx] + vecY * mapY[idx]);
                                if (score > inter_th) {
                                    sum += score;
                                    ++count;
                                }
                            }

                            // parts score + connection score
                            if (count > inter_min_above_th) {
                                temp.emplace_back(std::make_tuple(sum / ((float) count), i, j));
                            }
                        }
                    }
                }
                // select the top minAB connection, assuming that each part occur only once
                // sort rows in descending order based on parts + connection score
                if (!temp.empty()) {
                    std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());
                }
                std::vector<std::tuple<int, int, double>> connectionK;

                const int minAB = fastMin(nA, nB);
                // assuming that each part occur only once, filter out same part1 to different part2
                std::vector<int> occurA(nA, 0);
                std::vector<int> occurB(nB, 0);
                int counter = 0;
                for (auto &row : temp) {
                    const float score = std::get<0>(row);
                    const int aidx = std::get<1>(row);
                    const int bidx = std::get<2>(row);
                    if (!occurA[aidx - 1] && !occurB[bidx - 1]) {
                        // save two part score "position" and limb mean PAF score
                        connectionK.emplace_back(std::make_tuple(body_partA * peaks_offset + aidx * 3 + 2,
                                                                 body_partB * peaks_offset + bidx * 3 + 2, score));
                        ++counter;
                        if (counter == minAB) {
                            break;
                        }
                        occurA[aidx - 1] = 1;
                        occurB[bidx - 1] = 1;
                    }
                }
                // Cluster all the body part candidates into subset based on the part connection
                // initialize first body part connection
                if (pair_index == 0) {
                    for (const auto connectionKI : connectionK) {
                        std::vector<int> row_vector(num_body_parts + 3, 0);
                        const int indexA = std::get<0>(connectionKI);
                        const int indexB = std::get<1>(connectionKI);
                        const double score = std::get<2>(connectionKI);
                        row_vector[OpenposePostProcessor::BODY_PART_PAIRS[0]] = indexA;
                        row_vector[OpenposePostProcessor::BODY_PART_PAIRS[1]] = indexB;
                        row_vector[subset_counter_index] = 2;
                        // add the score of parts and the connection
                        const double subset_score = peaks[indexA] + peaks[indexB] + score;
                        subset.emplace_back(std::make_pair(row_vector, subset_score));
                    }
                }
                    // Add ears connections (in case person is looking to opposite direction to camera)
                else if (pair_index == 17 || pair_index == 18) {
                    for (const auto &connectionKI : connectionK) {
                        const int indexA = std::get<0>(connectionKI);
                        const int indexB = std::get<1>(connectionKI);
                        for (auto &subsetJ : subset) {
                            auto &subsetJ_first = subsetJ.first[body_partA];
                            auto &subsetJ_first_plus1 = subsetJ.first[body_partB];
                            if (subsetJ_first == indexA && subsetJ_first_plus1 == 0) {
                                subsetJ_first_plus1 = indexB;
                            } else if (subsetJ_first_plus1 == indexB && subsetJ_first == 0) {
                                subsetJ_first = indexA;
                            }
                        }
                    }
                } else {
                    if (!connectionK.empty()) {
                        for (auto &i : connectionK) {
                            const int indexA = std::get<0>(i);
                            const int indexB = std::get<1>(i);
                            const double score = std::get<2>(i);
                            int num = 0;
                            // if A is already in the subset, add B
                            for (auto &j : subset) {
                                if (j.first[body_partA] == indexA) {
                                    j.first[body_partB] = indexB;
                                    ++num;
                                    j.first[subset_counter_index] = j.first[subset_counter_index] + 1;
                                    j.second = j.second + peaks[indexB] + score;
                                }
                            }
                            // if A is not found in the subset, create new one and add both
                            if (num == 0) {
                                std::vector<int> row_vector(subset_size, 0);
                                row_vector[body_partA] = indexA;
                                row_vector[body_partB] = indexB;
                                row_vector[subset_counter_index] = 2;
                                const auto subsetScore = (float) (peaks[indexA] + peaks[indexB] + score);
                                subset.emplace_back(std::make_pair(row_vector, subsetScore));
                            }
                        }
                    }
                }
            }
        }

        // Delete people below thresholds, and save to output
        int number_people = 0;
        std::vector<int> valid_subset_indexes;
        valid_subset_indexes.reserve(fastMin((size_t) POSE_MAX_PEOPLE, subset.size()));
        for (unsigned int index = 0; index < subset.size(); ++index) {
            const int subset_counter = subset[index].first[subset_counter_index];
            const double subset_score = subset[index].second;
            if (subset_counter >= min_subset_cnt && (subset_score / subset_counter) > min_subset_score) {
                ++number_people;
                valid_subset_indexes.emplace_back(index);
                if (number_people == POSE_MAX_PEOPLE) {
                    break;
                }
            }
        }

        // Fill and return pose_keypoints
        keypoint_shape = {number_people, (int) num_body_parts, 3};
        if (number_people > 0) {
            poseKeypoints.resize(number_people * (int) num_body_parts * 3);
        } else {
            poseKeypoints.clear();
        }
        for (unsigned int person = 0u; person < valid_subset_indexes.size(); ++person) {
            const auto &subsetI = subset[valid_subset_indexes[person]].first;
            for (int bodyPart = 0u; bodyPart < num_body_parts; bodyPart++) {
                const int base_offset = (int) (person * num_body_parts + bodyPart) * 3;
                const int body_part_index = subsetI[bodyPart];
                if (body_part_index > 0) {
                    poseKeypoints[base_offset] = peaks[body_part_index - 2];
                    poseKeypoints[base_offset + 1] = peaks[body_part_index - 1];
                    poseKeypoints[base_offset + 2] = peaks[body_part_index];
                } else {
                    poseKeypoints[base_offset] = 0.f;
                    poseKeypoints[base_offset + 1] = 0.f;
                    poseKeypoints[base_offset + 2] = 0.f;
                }
            }
        }
    }

    static void findHeatmapPeaks(const float *src, float *dst, const int SRCW, const int SRCH, const int SRC_CH,
                                 const float TH) {
        initialize();

        // find peaks (8-connected neighbor), weights with 7 by 7 area to get sub-pixel location and response
        const int SRC_PLANE_OFFSET = SRCW * SRCH;
        // add 1 for saving total people count, 3 for x, y, score
        const int DST_PLANE_OFFSET = (POSE_MAX_PEOPLE + 1) * 3;
        float *dstPtr = dst;
        int c = 0;
        int x = 0;
        int y = 0;
        int i = 0;
        int j = 0;
        // TODO: reduce multiplication by using pointer
        for (c = 0; c < SRC_CH - 1; ++c) {
            int num_people = 0;
            for (y = 1; y < SRCH - 1 && num_people != POSE_MAX_PEOPLE; ++y) {
                for (x = 1; x < SRCW - 1 && num_people != POSE_MAX_PEOPLE; ++x) {
                    int idx = y * SRCW + x;
                    float value = src[idx];
                    if (value > TH) {
                        const float TOPLEFT = src[idx - SRCW - 1];
                        const float TOP = src[idx - SRCW];
                        const float TOPRIGHT = src[idx - SRCW + 1];
                        const float LEFT = src[idx - 1];
                        const float RIGHT = src[idx + 1];
                        const float BUTTOMLEFT = src[idx + SRCW - 1];
                        const float BUTTOM = src[idx + SRCW];
                        const float BUTTOMRIGHT = src[idx + SRCW + 1];
                        if (value > TOPLEFT && value > TOP && value > TOPRIGHT && value > LEFT &&
                            value > RIGHT && value > BUTTOMLEFT && value > BUTTOM && value > BUTTOMRIGHT) {
                            float x_acc = 0;
                            float y_acc = 0;
                            float score_acc = 0;
                            for (i = -3; i <= 3; ++i) {
                                int ux = x + i;
                                if (ux >= 0 && ux < SRCW) {
                                    for (j = -3; j <= 3; ++j) {
                                        int uy = y + j;
                                        if (uy >= 0 && uy < SRCH) {
                                            float score = src[uy * SRCW + ux];
                                            x_acc += (float) ux * score;
                                            y_acc += (float) uy * score;
                                            score_acc += score;
                                        }
                                    }
                                }
                            }
                            x_acc /= score_acc;
                            y_acc /= score_acc;
                            score_acc = value;
                            dstPtr[(num_people + 1) * 3 + 0] = x_acc;
                            dstPtr[(num_people + 1) * 3 + 1] = y_acc;
                            dstPtr[(num_people + 1) * 3 + 2] = score_acc;
                            ++num_people;
                        }
                    }
                }
            }
            dstPtr[0] = (float) num_people;
            src += SRC_PLANE_OFFSET;
            dstPtr += DST_PLANE_OFFSET;
        }
    }

    static void postProcess(const DarknetNet &net, Mat &image, float *output, float scale) {
        initialize();

        // cout << "Resize net output back to input size..." << endl;
        // 7. resize net output back to input size to get heatMap
        auto *heatMap = new float[net.inW * net.inH * OpenposePostProcessor::NET_OUT_CHANNELS];
        for (int i = 0; i < OpenposePostProcessor::NET_OUT_CHANNELS; ++i) {
            Mat netOut(net.outH, net.outW, CV_32F, (output + net.outH * net.outW * i));
            Mat nmsin(net.inH, net.inW, CV_32F, heatMap + net.inH * net.inW * i);
            resize(netOut, nmsin, Size(net.inW, net.inH), 0, 0, INTER_CUBIC);
        }

        // cout << "Finding heatMap peaks..." << endl;
        // 8. get heatMap peaks
        auto heatMapSize = 3 * (POSE_MAX_PEOPLE + 1) * (OpenposePostProcessor::NET_OUT_CHANNELS - 1);
        // cout << "heatMapSize = " << heatMapSize << endl;
        auto *heatMapPeaks = new float[heatMapSize];
        OpenposePostProcessor::findHeatmapPeaks(heatMap, heatMapPeaks, net.inW, net.inH,
                                                OpenposePostProcessor::NET_OUT_CHANNELS, 0.05);

        // cout << "Linking parts..." << endl;
        // 9. link parts
        vector<float> keyPoints;
        vector<int> shape;
        OpenposePostProcessor::connectBodyparts(keyPoints, heatMap, heatMapPeaks, net.inW, net.inH, 9, 0.05, 6, 0.4,
                                                shape);

        // cout << "Key Points:" << endl;
        /*
        for (float keyPoint : keyPoints) {
            cout << keyPoint << ", ";
        }
        cout << endl;
        for (int shapePoint : shape) {
            cout << shapePoint << ", ";
        }
        cout << endl;
        //*/
        // cout << "Drawing result..." << endl;
        // 10. draw result
        OpenposePostProcessor::renderPoseKeyPoints(image, keyPoints, shape, 0.05, scale);

        // cout << "Showing result..." << endl;
        // 11. show result
        // cout << "people: " << shape[0] << endl;

        delete[] heatMapPeaks;
        delete[] heatMap;
    }

private:
    static int INITIALIZED;
    static int MODEL_SIZE;
    static int NET_OUT_CHANNELS;
    static vector<int> POSE_MAP_INDEX;
    static vector<int> BODY_PART_PAIRS;
    static vector<float> COLORS;

    static void initialize() {
        if (OpenposePostProcessor::INITIALIZED) {
            return;
        }
#if MODEL == 0
        OpenposePostProcessor::MODEL_SIZE = 26;

        // size = 2 * model_size
        OpenposePostProcessor::POSE_MAP_INDEX = {
                26, 27, 40, 41, 48, 49, 42, 43, 44, 45, 50, 51, 52, 53, 32, 33, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39,
                56, 57, 58, 59, 62, 63, 60, 61, 64, 65, 46, 47, 54, 55, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        };

        // size = variable
        OpenposePostProcessor::BODY_PART_PAIRS = {
                1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0,
                0, 15, 15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24,
        };

        // size = 3 * (model_size - 1)
        OpenposePostProcessor::COLORS = {
                255, 0, 85, 255, 0, 0, 255, 85, 0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 0,
                255, 0, 0, 0, 255, 85, 0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 255, 0, 170,
                170, 0, 255, 255, 0, 255, 85, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 255,
                0, 255, 255,
        };
#elif MODEL == 1
        OpenposePostProcessor::MODEL_SIZE = 19;

        // size = 2 * model_size
        OpenposePostProcessor::POSE_MAP_INDEX = {
                31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46,
        };

        // size = variable
        OpenposePostProcessor::BODY_PART_PAIRS = {
                1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11,
                11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17,
        };

        // size = 3 * (model_size - 1)
        OpenposePostProcessor::COLORS = {
                255, 0, 85, 255, 0, 0, 255, 85, 0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 0, 0, 255, 85,
                0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 255, 0, 170, 170, 0, 255, 255, 0, 255, 85, 0, 255,
        };
#endif

        OpenposePostProcessor::NET_OUT_CHANNELS = 3 * OpenposePostProcessor::MODEL_SIZE;
        OpenposePostProcessor::INITIALIZED = 1;
    }
};

int OpenposePostProcessor::INITIALIZED = 0;
int OpenposePostProcessor::MODEL_SIZE = 0;
int OpenposePostProcessor::NET_OUT_CHANNELS = 0;
vector<int> OpenposePostProcessor::POSE_MAP_INDEX;
vector<int> OpenposePostProcessor::BODY_PART_PAIRS;
vector<float> OpenposePostProcessor::COLORS;

class YoloPostProcessor {
public:
    YoloPostProcessor() {
        YoloPostProcessor::initialize();
    }

    static void postProcess(const DarknetNet &net, Mat &imageInput, float *output, float scale, float thresh = 0.25,
                            float hierarchyThresh = 0.5, int letterBox = 0, int printDetections = 0) {
        initialize();

        int nrBoxes = 0;
        detection *detections = get_network_boxes(net.net, imageInput.cols, imageInput.rows, thresh, hierarchyThresh,
                                                  nullptr, 1, &nrBoxes, letterBox);
        if (YoloPostProcessor::NMS != 0) {
            if (net.lastDetectionLayer.nms_kind == DEFAULT_NMS) {
                do_nms_sort(detections, nrBoxes, net.lastDetectionLayer.classes, YoloPostProcessor::NMS);
            } else {
                diounms_sort(detections, nrBoxes, net.lastDetectionLayer.classes, YoloPostProcessor::NMS,
                             net.lastDetectionLayer.nms_kind, net.lastDetectionLayer.beta_nms);
            }
        }

        // scale detections
        for (int i = 0; i < nrBoxes; i++) {
            float xScale = (scale * (float) net.inW) / ((float) imageInput.cols);
            detections[i].bbox.x *= xScale;
            detections[i].bbox.w *= xScale;
            float yScale = (scale * (float) net.inH) / ((float) imageInput.rows);
            detections[i].bbox.y *= yScale;
            detections[i].bbox.h *= yScale;
        }

        Mat *imagePtr = &imageInput;
        draw_detections_cv_v3((void ***) &imagePtr, detections, nrBoxes, thresh, net.names, net.alphabet,
                              net.lastDetectionLayer.classes, printDetections);
        free_detections(detections, nrBoxes);
    }

private:
    static int INITIALIZED;
    static float NMS;

    static void initialize() {
        if (YoloPostProcessor::INITIALIZED) {
            return;
        }
        YoloPostProcessor::INITIALIZED = 1;
    }
};

int YoloPostProcessor::INITIALIZED = 0;
float YoloPostProcessor::NMS = .45;  // 0.4F;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

Mat createNetSizeImage(const Mat &im, const int netW, const int netH, float &scale) {
    // for tall image
    int newH = netH;
    float s = (float) newH / (float) im.rows;
    int newW = (int) ((float) im.cols * s);
    if (newW > netW) {
        // for fat image
        newW = netW;
        s = (float) newW / (float) im.cols;
        newH = (int) ((float) im.rows * s);
    }

    scale = 1 / s;
    Rect dst_area(0, 0, newW, newH);
    Mat dst = Mat::zeros(netH, netW, CV_8UC3);
    resize(im, dst(dst_area), Size(newW, newH));
    return dst;
}

void preProcessImage(float *netInput, Mat &image, int netInW, int netInH, float &scale, bool yolo = false) {
    // cout << "Enter preProcessImage..." << endl;
    double timeBegin = getTickCount();

    // cout << "Resize image to net input..." << endl;
    // 3. resize to net input size, put scaled image on the top left
    Mat netIm = createNetSizeImage(image, netInW, netInH, scale);

    // cout << "Normalize input image..." << endl;
    // 4. normalized to float type
    cv::cvtColor(netIm, netIm, cv::COLOR_RGB2BGR);
    //*
    netIm.convertTo(netIm, CV_32F);
    /*/
    if (yolo) {
        netIm.convertTo(netIm, CV_32F, 1 / 255.f, 0.0);
    } else {
        netIm.convertTo(netIm, CV_32F, 1 / 256.f, -0.5);
    }
    //*/

    // cout << "Splitting image channels..." << endl;
    // 5. split channels
    float *netInDataPtr = netInput;
    vector<Mat> inputChannels;
    for (int i = 0; i < 3; ++i) {
        Mat channel(netInH, netInW, CV_32FC1, netInDataPtr);
        inputChannels.emplace_back(channel);
        netInDataPtr += (netInW * netInH);
    }
    split(netIm, inputChannels);

    double feeTime = (getTickCount() - timeBegin) / getTickFrequency() * 1000;
    cout << "pre-process time: " << feeTime << "ms" << endl;
}

#pragma clang diagnostic pop

class Workflow {
  public:
    explicit Workflow(DarknetNet *net, float *netInput, const char *windowName, volatile int *stop) :
            net(net), netInput(netInput), windowName(windowName), stop(stop), scale(0.0) {
        this->processStart = new volatile int(0);
        this->postProcessStart = new volatile int(0);
        this->processThread = thread(&Workflow::processThreadFunction, this);
        this->postProcessThread = thread(&Workflow::postProcessThreadFunction, this);
        namedWindow(this->windowName);
    }

    ~Workflow() {
        this->finish();
    }

    void process() {
        preProcessImage(this->netInput, *(this->inputImage), this->net->inW, this->net->inH, this->scale);

        // cout << "Feeding forward through network..." << endl;
        // 6. feed forward
        double timeBegin = getTickCount();
        this->netOutData = this->net->runNet(this->netInput);
        double feeTime = (getTickCount() - timeBegin) / getTickFrequency() * 1000;
        cout << "process time: " << feeTime << "ms" << endl;
    }

    void postProcess() {
        if (this->outputImage == nullptr) {
            return;
        }

        double timeBegin = getTickCount(), feeTime;
        OpenposePostProcessor::postProcess(*(this->net), *(this->outputImage), this->netOutData, this->scale);
        feeTime = (getTickCount() - timeBegin) / getTickFrequency() * 1000;
        cout << "Openpose post-process time: " << feeTime << "ms" << endl;
        timeBegin = getTickCount();
        YoloPostProcessor::postProcess(*(this->net), *(this->outputImage), this->netOutData, this->scale);
        feeTime = (getTickCount() - timeBegin) / getTickFrequency() * 1000;
        cout << "YOLO post-process time: " << feeTime << "ms" << endl;
    }

    void processThreadFunction() {
        while (!custom_atomic_load_int(this->stop)) {
            while (!custom_atomic_load_int(this->processStart)) {
                if (custom_atomic_load_int(this->stop)) return;
                this_thread::yield();
            }

            this->process();

            custom_atomic_store_int(this->processStart, 0);
        }
    }

    void postProcessThreadFunction() {
        while (!custom_atomic_load_int(this->stop)) {
            while (!custom_atomic_load_int(this->postProcessStart)) {
                if (custom_atomic_load_int(this->stop)) return;
                this_thread::yield();
            }

            this->postProcess();

            custom_atomic_store_int(this->postProcessStart, 0);
        }
    }

    static void threadBarrier(volatile int *threadVariable) {
        while (custom_atomic_load_int(threadVariable)) {
            std::chrono::milliseconds duration(Workflow::threadSleepTime);
            std::this_thread::sleep_for(duration);
        }
    }

    bool workflow(Mat inImage, int wait = 1) {
        this->inputImage = new Mat(std::move(inImage));
        custom_atomic_store_int(this->processStart, 1);
        custom_atomic_store_int(this->postProcessStart, 1);
        Workflow::threadBarrier(this->processStart);
        Workflow::threadBarrier(this->postProcessStart);

        bool result = true;
        if (this->outputImage != nullptr) {
            imshow(this->windowName, *(this->outputImage));
            int key = cv::waitKey(wait);
            if (key == 27 || key == 'q') {
                result = false;
            }
            delete this->outputImage;
            this->outputImage = nullptr;
        }
        this->outputImage = this->inputImage;
        return result;
    }

    void finish() {
        this->inputImage = nullptr;
        delete this->outputImage;
        this->outputImage = nullptr;
        custom_atomic_store_int(this->stop, 1);
        this->processThread.join();
        this->postProcessThread.join();
        destroyAllWindows();
    }

  private:
    thread processThread, postProcessThread;
    static const int threadSleepTime = 1;
    volatile int *stop, *processStart, *postProcessStart;
    const char *windowName;
    float *netOutData{}, *netInput, scale;
    DarknetNet *net{};
    Mat *inputImage{}, *outputImage{};
};

void cameraInput(DarknetNet &net) {
    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cout << "Error!" << endl;
        return;
    }
    Mat im;
    auto *netInData = new float[net.inW * net.inH * 3]();
    volatile int stop = 0;
    Workflow w(&net, netInData, (const char *) "demo", &stop);
    while (!custom_atomic_load_int(&stop)) {
        camera.read(im);
        if (im.empty()) {
            cout << "Image empty Error!" << endl;
            return;
        }

        if (!w.workflow(im)) {
            custom_atomic_store_int(&stop, 1);
        }
    }
    delete[] netInData;
}

void realsenseInput(DarknetNet &net) {
    try {
        rs2::pipeline pipe;
        auto config = pipe.start();
        auto profile = config.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2::align alignTo(RS2_STREAM_COLOR);
        cout << "Started RealSense pipeline!" << endl;

        unsigned long long lastFrameNumber = 0;
        auto *netInData = new float[net.inW * net.inH * 3]();
        volatile int *stop = new volatile int(0);
        // custom_atomic_store_int(&stop, 0);
        Workflow w(&net, netInData, (const char *) "demo", stop);
        while (!custom_atomic_load_int(stop)) {
            rs2::frameset currentFrame = pipe.wait_for_frames();
            currentFrame = alignTo.process(currentFrame);  // Make sure the frames are spatially aligned
            // rs2::depth_frame depth = currentFrame.get_depth_frame();
            rs2::video_frame image = currentFrame.get_color_frame();

            if (image.get_frame_number() == lastFrameNumber) continue;
            lastFrameNumber = image.get_frame_number();

            // auto depthMat = depth_frame_to_meters(depth);
            auto im = frame_to_mat(image);

            if (!w.workflow(im)) {
                custom_atomic_store_int(stop, 1);
            }
        }
        delete[] netInData;
    } catch (const rs2::error &e) {
        cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
             << "    " << e.what() << endl;
    } catch (const exception &e) {
        cerr << e.what() << endl;
    }
}

void singleImageInput(DarknetNet &net, Mat &im) {
    auto *netInData = new float[net.inW * net.inH * 3]();
    volatile int stop = 0;
    Workflow w(&net, netInData, "demo", &stop);
    w.workflow(im, 0);
    w.finish();
    delete[] netInData;
}

void selectMode(DarknetNet &net, char *mode, int argc, char **argv) {
    if (strcmp("image", mode) == 0) {
        if (argc < 6) {
            cout << "Image mode requires the image path as argument!" << endl;
            return;
        }
        Mat im = imread(argv[5]);
        if (im.empty()) {
            cout << "Failed to read image!" << endl;
            return;
        }
        singleImageInput(net, im);
    } else if (strcmp("camera", mode) == 0) {
        cameraInput(net);
    } else if (strcmp("realsense", mode) == 0) {
        realsenseInput(net);
    }
}

void workflow(char *dataPath, char *cfgPath, char *weightPath, char *mode, int argc, char **argv,
              int benchmarkLayers = 0) {
    // 1. initialize net
    DarknetNet net(cfgPath, weightPath, dataPath);
    net.initNet(benchmarkLayers);

    selectMode(net, mode, argc, argv);

    net.releaseNet();
}

int main(int ac, char **av) {
    if (ac < 5) {
        cout << "usage: ./program [data cfg] [cfg file] [weight file] [mode] [mode arguments]" << endl;
        return 1;
    }

    char *dataCfg = av[1];
    char *netCfg = av[2];
    char *weightFile = av[3];
    char *mode = av[4];
    workflow(dataCfg, netCfg, weightFile, mode, ac, av);
    return 0;
}
