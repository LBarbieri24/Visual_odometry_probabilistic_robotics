#include "vo/triangulation.h"
#include <Eigen/SVD>
#include <iostream>

namespace vo {

Eigen::Vector3f Triangulation::triangulateDLT(
    const std::vector<Observation>& obs,
    const Eigen::Matrix3f& K) {

    int n = obs.size();
    if (n < 2) return Eigen::Vector3f::Zero();

    Eigen::MatrixXf A(2 * n, 4);

    for (int i = 0; i < n; ++i) {
        // Construct projection matrix P = K * [R | t]
        Eigen::Matrix<float, 3, 4> P;
        P.block<3, 3>(0, 0) = K * obs[i].R;
        P.block<3, 1>(0, 3) = K * obs[i].t;

        float u = obs[i].uv.x();
        float v = obs[i].uv.y();

        Eigen::RowVector4f p1 = P.row(0);
        Eigen::RowVector4f p2 = P.row(1);
        Eigen::RowVector4f p3 = P.row(2);

        A.row(2 * i) = u * p3 - p1;
        A.row(2 * i + 1) = v * p3 - p2;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f Xh = svd.matrixV().col(3);
    
    if (std::abs(Xh(3)) < 1e-6f) {
        return Xh.head<3>(); // Prevent division by zero if point is at infinity
    }
    
    return Xh.head<3>() / Xh(3);
}

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

} // namespace vo
