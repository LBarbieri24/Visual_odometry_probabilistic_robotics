#pragma once

#include <vector>
#include <Eigen/Dense>

namespace vo {

// Representation of a single 2D pixel observation of a 3D point in a camera frame
struct Observation {
    Eigen::Matrix3f R;          // Rotation from world to camera frame (X_cam = R * X_world + t)
    Eigen::Vector3f t;          // Translation from world to camera frame
    Eigen::Vector2f uv;         // Observed pixel coordinates (u, v)
};

class Triangulation {
public:
    // Performs Linear Direct Linear Transform (DLT) triangulation
    static Eigen::Vector3f triangulateDLT(
        const std::vector<Observation>& obs,
        const Eigen::Matrix3f& K);

    // Refines a 3D point using Gauss-Newton non-linear least squares
    // Minimizes the sum of squared reprojection errors in pixel space
    static Eigen::Vector3f refineGaussNewton(
        const Eigen::Vector3f& X_initial,
        const std::vector<Observation>& obs,
        const Eigen::Matrix3f& K,
        int max_iterations = 10,
        float tolerance = 1e-5f);
};

} // namespace vo
