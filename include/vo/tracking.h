#pragma once

#include "vo/data_loader.h"
#include "vo/matcher.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <string>

namespace vo {

class Tracker {
public:
    // Estimate camera pose (R, t) of a frame given 3D-to-2D correspondences
    // R, t are updated in-place (world to camera: X_c = R * X_w + t)
    static bool estimatePosePnP(
        const std::vector<Eigen::Vector3f>& pts_3d,
        const std::vector<Eigen::Vector2f>& pts_2d,
        const Eigen::Matrix3f& K,
        Eigen::Matrix3f& R,
        Eigen::Vector3f& t,
        int max_iterations = 20,
        float tolerance = 1e-5f);

    // Run the full Visual Odometry pipeline over a set of frame files
    // Returns a vector of camera poses as SE(3) transformation matrices
    static std::vector<Eigen::Matrix4f> runVisualOdometry(
        const std::vector<std::string>& frame_files,
        const CameraParams& cam,
        std::unordered_map<int, Eigen::Vector3f>& map_points);
};

} // namespace vo
