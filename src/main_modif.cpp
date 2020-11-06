#include <iostream>
#include <cmath>
#include <tuple>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

#include "run_darknet.h"

#define POSE_MAX_PEOPLE 96
#define MODEL 0

#if MODEL == 0
const int MODEL_SIZE = 26;

// size = 2 * model_size
const vector<int> poseMapIndex = {
        26, 27, 40, 41, 48, 49, 42, 43, 44, 45, 50, 51, 52, 53, 32, 33, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39,
        56, 57, 58, 59, 62, 63, 60, 61, 64, 65, 46, 47, 54, 55, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
};

// size = variable
const vector<int> bodyPartPairs = {
        1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0,
        0, 15, 15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24,
};

// size = 3 * (model_size - 1)
const vector<float> colors = {
        255, 0, 85, 255, 0, 0, 255, 85, 0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 0,
        255, 0, 0, 0, 255, 85, 0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 255, 0, 170,
        170, 0, 255, 255, 0, 255, 85, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 255,
        0, 255, 255,
};
#elif MODEL == 1
const int MODEL_SIZE = 19;

// size = 2 * model_size
const vector<int> poseMapIndex = {
        31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46,
};

// size = variable
const vector<int> bodyPartPairs = {
        1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11,
        11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17,
};

// size = 3 * (model_size - 1)
const vector<float> colors = {
        255, 0, 85, 255, 0, 0, 255, 85, 0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 0, 0, 255, 85,
        0, 255, 170, 0, 255, 255, 0, 170, 255, 0, 85, 255, 0, 0, 255, 255, 0, 170, 170, 0, 255, 255, 0, 255, 85, 0, 255,
};
#endif
const int NET_OUT_CHANNELS = 3 * MODEL_SIZE;


template<typename T>
inline int intRound(const T a) {
    return lround(a);
}

template<typename T>
inline T fastMin(const T a, const T b) {
    return (a < b ? a : b);
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
void renderPoseKeyPoints(Mat &frame, const vector<float> &keyPoints, vector<int> keyShape, const float threshold,
                         float scale) {
    const int nrKeyPoints = keyShape[1];
    const int pairs_size = bodyPartPairs.size();
    const int number_colors = colors.size();

    for (int person = 0; person < keyShape[0]; ++person) {
        // Draw lines
        for (int pair = 0u; pair < pairs_size; pair += 2) {
            const int index1 = (int) (person * nrKeyPoints + bodyPartPairs[pair]) * keyShape[2];
            const int index2 = (int) (person * nrKeyPoints + bodyPartPairs[pair + 1]) * keyShape[2];
            if (keyPoints[index1 + 2] > threshold && keyPoints[index2 + 2] > threshold) {
                const int color_index = bodyPartPairs[pair + 1] * 3;
                Scalar color{colors[(color_index + 2) % number_colors],
                             colors[(color_index + 1) % number_colors],
                             colors[(color_index + 0) % number_colors]};
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
                Scalar color{colors[(color_index + 2) % number_colors],
                             colors[(color_index + 1) % number_colors],
                             colors[(color_index + 0) % number_colors]};
                Point center{intRound(keyPoints[index] * scale), intRound(keyPoints[index + 1] * scale)};
                circle(frame, center, 3, color, -1);
            }
        }
    }
}

void connectBodyparts(vector<float> &poseKeypoints, const float *const map, const float *const peaks, int mapW,
                      int mapH, const int inter_min_above_th, const float inter_th, const int min_subset_cnt,
                      const float min_subset_score, vector<int> &keypoint_shape) {
    keypoint_shape.resize(3);

    const int num_body_parts = MODEL_SIZE - 1;  // model part number
    const int num_body_part_pairs = (int) (bodyPartPairs.size() / 2);
    std::vector<std::pair<std::vector<int>, double>> subset;
    const int subset_counter_index = num_body_parts;
    const int subset_size = num_body_parts + 1;
    const int peaks_offset = 3 * (POSE_MAX_PEOPLE + 1);
    const int map_offset = mapW * mapH;

    for (unsigned int pair_index = 0u; pair_index < num_body_part_pairs; ++pair_index) {
        const int body_partA = bodyPartPairs[2 * pair_index];
        const int body_partB = bodyPartPairs[2 * pair_index + 1];
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
            const float *const mapX = map + poseMapIndex[2 * pair_index] * map_offset;
            // limb PAF y-direction heatmap
            const float *const mapY = map + poseMapIndex[2 * pair_index + 1] * map_offset;
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
                    row_vector[bodyPartPairs[0]] = indexA;
                    row_vector[bodyPartPairs[1]] = indexB;
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

void findHeatmapPeaks(const float *src, float *dst, const int SRCW, const int SRCH, const int SRC_CH,
                      const float TH) {
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

Mat createNetSizeImage(const Mat &im, const int netW, const int netH, float *scale) {
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

    *scale = 1 / s;
    Rect dst_area(0, 0, newW, newH);
    Mat dst = Mat::zeros(netH, netW, CV_8UC3);
    resize(im, dst(dst_area), Size(newW, newH));
    return dst;
}

void openposeProcessImage(Mat &image, int netInW, int netInH, int netOutW, int netOutH) {
    cout << "Resize image to net input..." << endl;
    // 3. resize to net input size, put scaled image on the top left
    float scale = 0.0f;
    Mat netIm = createNetSizeImage(image, netInW, netInH, &scale);

    cout << "Normalize input image..." << endl;
    // 4. normalized to float type
    netIm.convertTo(netIm, CV_32F, 1 / 256.f, -0.5);

    cout << "Splitting image channels..." << endl;
    // 5. split channels
    auto *netInData = new float[netInW * netInH * 3]();
    float *netInDataPtr = netInData;
    vector<Mat> inputChannels;
    for (int i = 0; i < 3; ++i) {
        Mat channel(netInH, netInW, CV_32FC1, netInDataPtr);
        inputChannels.emplace_back(channel);
        netInDataPtr += (netInW * netInH);
    }
    split(netIm, inputChannels);

    cout << "Feeding forward through network..." << endl;
    // 6. feed forward
    double timeBegin = getTickCount();
    float *netOutData = runNet(netInData);
    double feeTime = (getTickCount() - timeBegin) / getTickFrequency() * 1000;
    cout << "forward fee: " << feeTime << "ms" << endl;

    cout << "Resize net output back to input size..." << endl;
    // 7. resize net output back to input size to get heatmap
    auto *heatmap = new float[netInW * netInH * NET_OUT_CHANNELS];
    for (int i = 0; i < NET_OUT_CHANNELS; ++i) {
        cout << "At channel i = " << i << endl;
        Mat netOut(netOutH, netOutW, CV_32F, (netOutData + netOutH * netOutW * i));
        cout << netOut.size << endl;
        Mat nmsin(netInH, netInW, CV_32F, heatmap + netInH * netInW * i);
        cout << nmsin.size << endl;
        resize(netOut, nmsin, Size(netInW, netInH), 0, 0, INTER_CUBIC);
    }

    cout << "Finding heatmap peaks..." << endl;
    // 8. get heatmap peaks
    auto heatmapSize = 3 * (POSE_MAX_PEOPLE + 1) * (NET_OUT_CHANNELS - 1);
    cout << "heatmapSize = " << heatmapSize << endl;
    auto *heatmap_peaks = new float[heatmapSize];
    findHeatmapPeaks(heatmap, heatmap_peaks, netInW, netInH, NET_OUT_CHANNELS, 0.05);

    cout << "Linking parts..." << endl;
    // 9. link parts
    vector<float> keyPoints;
    vector<int> shape;
    connectBodyparts(keyPoints, heatmap, heatmap_peaks, netInW, netInH, 9, 0.05, 6, 0.4, shape);

    cout << "Keypoints:" << endl;
    for (float keyPoint : keyPoints) {
        cout << keyPoint << ", ";
    }
    cout << endl;
    for (int shapePoint : shape) {
        cout << shapePoint << ", ";
    }
    cout << endl;
    cout << "Drawing result..." << endl;
    // 10. draw result
    renderPoseKeyPoints(image, keyPoints, shape, 0.05, scale);

    cout << "Showing result..." << endl;
    // 11. show result
    cout << "people: " << shape[0] << endl;

    delete[] heatmap_peaks;
    delete[] heatmap;
    delete[] netInData;
}

void openposeCamera(int netInW, int netInH, int netOutW, int netOutH) {
    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cout << "Error!" << endl;
        return;
    }
    Mat im;
    while (true) {
        camera.read(im);
        if (im.empty()) {
            cout << "Image empty Error!" << endl;
            return;
        }

        openposeProcessImage(im, netInW, netInH, netOutW, netOutH);

        // 11. show and save result
        imshow("demo", im);
        int k = waitKey(1);
        if (k == 27 || k == 'q') {
            break;
        }
    }
}

void openposeImage(Mat *im, int netInW, int netInH, int netOutW, int netOutH) {
    openposeProcessImage(*im, netInW, netInH, netOutW, netOutH);

    imshow("demo", *im);
    waitKey(0);
}

void openposeWorkflow(char *cfgPath, char *weightPath, Mat *im) {
    // 1. initialize net
    int netInW = 0, netInH = 0, netOutW = 0, netOutH = 0;
    initNet(cfgPath, weightPath, &netInW, &netInH, &netOutW, &netOutH);

    if (im == nullptr) {
        openposeCamera(netInW, netInH, netOutW, netOutH);
    } else {
        openposeImage(im, netInW, netInH, netOutW, netOutH);
    }
}

int main(int ac, char **av) {
    if (ac != 4 && ac != 3) {
        cout << "usage 1: ./program [cfg file] [weight file]" << endl;
        cout << "usage 2: ./program [image file] [cfg file] [weight file]" << endl;
        return 1;
    }

    // 1. read args
    char *cfg_path;
    char *weight_path;
    char *im_path;
    Mat *im;
    if (ac == 3) {
        cfg_path = av[1];
        weight_path = av[2];
        im = nullptr;
    } else {
        im_path = av[1];
        cfg_path = av[2];
        weight_path = av[3];
        im = new Mat(imread(im_path));
        if (im->empty()) {
            cout << "failed to read image" << endl;
            return 1;
        }
    }

    openposeWorkflow(cfg_path, weight_path, im);

    return 0;
}
