#pragma once

#include "vo/data_loader.h"
#include "vo/matcher.h"
#include <vector>

namespace vo {

class Epipolar {
public:
    // Estimates Essential Matrix E such that x2' * E * x1 = 0
    // x1, x2 are normalized image coordinates (K^-1 * [u, v, 1]')
    static Eigen::Matrix3f estimateEssentialMatrix8Point(
        const std::vector<Eigen::Vector2f>& points1,
        const std::vector<Eigen::Vector2f>& points2);

    static Eigen::Matrix3f robustEstimateEssentialMatrix(
        const std::vector<Eigen::Vector2f>& points1,
        const std::vector<Eigen::Vector2f>& points2,
        std::vector<int>& inliers,
        float threshold = 0.01f,
        int iterations = 1000);

    // Decomposes E into R and t. 
    // Returns the solution that passes the cheirality check (points in front of both cameras)
    static void decomposeEssentialMatrix(
        const Eigen::Matrix3f& E,
        const std::vector<Eigen::Vector2f>& points1,
        const std::vector<Eigen::Vector2f>& points2,
        Eigen::Matrix3f& R,
        Eigen::Vector3f& t);
};

} // namespace vo
