#include "vo/triangulation.h"
#include <Eigen/SVD>
#include <iostream>

namespace vo {


Eigen::Vector3f Triangulation::refineGaussNewton(
    const Eigen::Vector3f& X_initial,
    const std::vector<Observation>& obs,
    const Eigen::Matrix3f& K,
    int max_iterations,
    float tolerance) {

    Eigen::Vector3f X = X_initial;
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        Eigen::Vector3f b = Eigen::Vector3f::Zero();

        for (const auto& o : obs) {
            // Point in camera frame: Pc = R * X + t
            Eigen::Vector3f Pc = o.R * X + o.t;
            float Xc = Pc.x(), Yc = Pc.y(), Zc = Pc.z();

            if (Zc <= 1e-4f) continue; // Skip point if too close or behind camera

            // Reprojection into image plane
            float u_proj = fx * (Xc / Zc) + cx;
            float v_proj = fy * (Yc / Zc) + cy;

            // Compute reprojection error e = proj - measurement
            float e_u = u_proj - o.uv.x();
            float e_v = v_proj - o.uv.y();

            // Jacobian of projection w.r.t. point in camera frame: Pc = (Xc, Yc, Zc)
            Eigen::Matrix<float, 2, 3> J_proj;
            J_proj << fx / Zc,   0.0f,      -fx * Xc / (Zc * Zc),
                      0.0f,      fy / Zc,   -fy * Yc / (Zc * Zc);

            // Jacobian of reprojection w.r.t. world coordinates X
            // Using Chain Rule: d_err / d_X = (d_err / d_Pc) * (d_Pc / d_X)
            // where Pc = R*X + t => d_Pc / d_X = R
            Eigen::Matrix<float, 2, 3> J = J_proj * o.R;

            // Accumulate Normal Equations: H = J^T * J, b = J^T * e
            H += J.transpose() * J;
            Eigen::Vector2f e(e_u, e_v);
            b += J.transpose() * e;
        }

        // Check if Hessian is invertible
        if (H.determinant() < 1e-8f) {
            break; 
        }

        // Solve H * dX = -b => dX = -inv(H) * b
        Eigen::Vector3f dX = H.ldlt().solve(-b);
        X += dX;

        // Convergence check
        if (dX.norm() < tolerance) {
            break;
        }
    }

    return X;
}

Eigen::Vector3f Triangulation::triangulateGaussNewtonOnly(
    const std::vector<Observation>& obs,
    const Eigen::Matrix3f& K) {

    if (obs.size() < 2) {
        return Eigen::Vector3f::Zero();
    }

    // Initialize using Grisetti's midpoint line triangulation
    Eigen::Vector3f X_initial = triangulateMidpoint(obs, K);

    // Refine using Gauss-Newton starting from the initial guess
    return refineGaussNewton(X_initial, obs, K);
}

Eigen::Vector3f Triangulation::triangulateMidpoint(
    const std::vector<Observation>& obs,
    const Eigen::Matrix3f& K) {

    if (obs.size() < 2) {
        return Eigen::Vector3f::Zero();
    }

    Eigen::Matrix3f Kinv = K.inverse();

    // 1. Ray 1 in camera 0 frame (passes through origin, direction d1)
    Eigen::Vector3f uv0_h(obs[0].uv.x(), obs[0].uv.y(), 1.0f);
    Eigen::Vector3f d1 = Kinv * uv0_h;

    // 2. Ray 2 in camera 0 frame.
    // Ray 2 in camera 1 frame passes through origin with direction d2_cam.
    Eigen::Vector3f uv1_h(obs[1].uv.x(), obs[1].uv.y(), 1.0f);
    Eigen::Vector3f d2_cam = Kinv * uv1_h;

    // Transform camera 1 center to camera 0 frame:
    // R_1_to_0 = R0 * R1^T
    // t_1_to_0 = t0 - R0 * R1^T * t1
    // The camera 1 center in camera 0 coordinates is p2 = t_1_to_0
    // The ray direction d2 in camera 0 coordinates is d2 = R_1_to_0 * d2_cam
    Eigen::Matrix3f R_1_to_0 = obs[0].R * obs[1].R.transpose();
    Eigen::Vector3f t_1_to_0 = obs[0].t - R_1_to_0 * obs[1].t;

    Eigen::Vector3f p2 = t_1_to_0;
    Eigen::Vector3f d2 = R_1_to_0 * d2_cam;

    // 3. Assemble system to find abscissas s1, s2
    // D = [-d1, d2] is a 3x2 matrix
    Eigen::Matrix<float, 3, 2> D;
    D.col(0) = -d1;
    D.col(1) = d2;

    // Solve least squares: s = -(D^T * D)^-1 * D^T * p2
    Eigen::Matrix2f DTD = D.transpose() * D;
    if (DTD.determinant() < 1e-8f) {
        return Eigen::Vector3f::Zero(); // Parallel or degenerate rays
    }
    Eigen::Vector2f s = -DTD.inverse() * (D.transpose() * p2);

    // Chirality check: depths must be positive
    if (s(0) < 0 || s(1) < 0) {
        return Eigen::Vector3f::Zero();
    }

    // Midpoint in camera 0 coordinates
    Eigen::Vector3f p1_tri = d1 * s(0);
    Eigen::Vector3f p2_tri = d2 * s(1) + p2;
    Eigen::Vector3f p_cam0 = 0.5f * (p1_tri + p2_tri);

    // Transform from camera 0 frame to world coordinate frame:
    // Since p_cam0 = R0 * X_world + t0 => X_world = R0^T * (p_cam0 - t0)
    Eigen::Vector3f X_world = obs[0].R.transpose() * (p_cam0 - obs[0].t);

    return X_world;
}

} // namespace vo
