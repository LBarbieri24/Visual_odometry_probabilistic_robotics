#include "vo/epipolar.h"
#include <Eigen/SVD>
#include <random>
#include <iostream>

namespace vo {

Eigen::Matrix3f Epipolar::estimateEssentialMatrix8Point(
    const std::vector<Eigen::Vector2f>& points1,
    const std::vector<Eigen::Vector2f>& points2) {
    
    int n = points1.size();
    if (n < 8) return Eigen::Matrix3f::Zero();

    // 1. Normalization
    auto normalize = [](const std::vector<Eigen::Vector2f>& pts) {
        Eigen::Vector2f centroid(0, 0);
        for (const auto& p : pts) centroid += p;
        centroid /= pts.size();

        float mean_dist = 0;
        for (const auto& p : pts) mean_dist += (p - centroid).norm();
        mean_dist /= pts.size();

        float scale = std::sqrt(2.0f) / mean_dist;
        Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
        T(0, 0) = scale;
        T(1, 1) = scale;
        T(0, 2) = -scale * centroid.x();
        T(1, 2) = -scale * centroid.y();

        std::vector<Eigen::Vector2f> normalized_pts;
        for (const auto& p : pts) {
            Eigen::Vector3f ph(p.x(), p.y(), 1.0f);
            Eigen::Vector3f pn = T * ph;
            normalized_pts.push_back(pn.head<2>());
        }
        return std::make_pair(normalized_pts, T);
    };

    auto [pts1_n, T1] = normalize(points1);
    auto [pts2_n, T2] = normalize(points2);

    // 2. Linear System A*f = 0
    Eigen::MatrixXf A(n, 9);
    for (int i = 0; i < n; ++i) {
        float x1 = pts1_n[i].x(), y1 = pts1_n[i].y();
        float x2 = pts2_n[i].x(), y2 = pts2_n[i].y();
        A.row(i) << x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0f;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXf f = svd.matrixV().col(8);
    Eigen::Matrix3f E_hat;
    E_hat << f(0), f(1), f(2),
             f(3), f(4), f(5),
             f(6), f(7), f(8);

    // 3. Forcing Rank-2 Constraint
    Eigen::JacobiSVD<Eigen::Matrix3f> svd_e(E_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f s = svd_e.singularValues();
    float avg_s = (s(0) + s(1)) / 2.0f;
    Eigen::Matrix3f diag;
    diag << avg_s, 0, 0,
            0, avg_s, 0,
            0, 0, 0;
    Eigen::Matrix3f E_normalized = svd_e.matrixU() * diag * svd_e.matrixV().transpose();

    // 4. Denormalization
    return T2.transpose() * E_normalized * T1;
}

Eigen::Matrix3f Epipolar::robustEstimateEssentialMatrix(
    const std::vector<Eigen::Vector2f>& points1,
    const std::vector<Eigen::Vector2f>& points2,
    std::vector<int>& inliers,
    float threshold,
    int iterations) {

    int n = points1.size();
    if (n < 8) return Eigen::Matrix3f::Zero();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    Eigen::Matrix3f best_E = Eigen::Matrix3f::Zero();
    int max_inliers = -1;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Eigen::Vector2f> sample1, sample2;
        std::vector<int> sample_indices;
        while (sample_indices.size() < 8) {
            int idx = dis(gen);
            if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                sample_indices.push_back(idx);
                sample1.push_back(points1[idx]);
                sample2.push_back(points2[idx]);
            }
        }

        Eigen::Matrix3f E = estimateEssentialMatrix8Point(sample1, sample2);
        
        std::vector<int> current_inliers;
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3f x1(points1[i].x(), points1[i].y(), 1.0f);
            Eigen::Vector3f x2(points2[i].x(), points2[i].y(), 1.0f);
            float error = std::abs(x2.transpose() * E * x1);
            if (error < threshold) {
                current_inliers.push_back(i);
            }
        }

        if ((int)current_inliers.size() > max_inliers) {
            max_inliers = current_inliers.size();
            best_E = E;
            inliers = current_inliers;
        }
    }

    // Refine E with all inliers
    std::vector<Eigen::Vector2f> inlier_pts1, inlier_pts2;
    for (int idx : inliers) {
        inlier_pts1.push_back(points1[idx]);
        inlier_pts2.push_back(points2[idx]);
    }
    return estimateEssentialMatrix8Point(inlier_pts1, inlier_pts2);
}

void Epipolar::decomposeEssentialMatrix(
    const Eigen::Matrix3f& E,
    const std::vector<Eigen::Vector2f>& points1,
    const std::vector<Eigen::Vector2f>& points2,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t) {

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    // Ensure U and V are rotation matrices (det = 1)
    if (U.determinant() < 0) U.col(2) *= -1.0f;
    if (V.determinant() < 0) V.col(2) *= -1.0f;

    Eigen::Matrix3f W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    Eigen::Matrix3f R1 = U * W * V.transpose();
    Eigen::Matrix3f R2 = U * W.transpose() * V.transpose();
    Eigen::Vector3f t_vec = U.col(2);

    std::vector<Eigen::Matrix3f> Rs = {R1, R1, R2, R2};
    std::vector<Eigen::Vector3f> ts = {t_vec, -t_vec, t_vec, -t_vec};

    int best_count = -1;
    int best_idx = 0;

    for (int i = 0; i < 4; ++i) {
        int count = 0;
        // Test with a few points (cheirality check)
        for (size_t j = 0; j < std::min((size_t)20, points1.size()); ++j) {
            // Simple triangulation to check depth
            // P1 = K [I | 0], P2 = K [R | t]
            // Since points are normalized, K is Identity.
            Eigen::Matrix<float, 4, 4> A;
            A.row(0) = points1[j].x() * Eigen::RowVector4f(0, 0, 1, 0) - Eigen::RowVector4f(1, 0, 0, 0);
            A.row(1) = points1[j].y() * Eigen::RowVector4f(0, 0, 1, 0) - Eigen::RowVector4f(0, 1, 0, 0);
            
            Eigen::RowVector3f r1 = Rs[i].row(0);
            Eigen::RowVector3f r2 = Rs[i].row(1);
            Eigen::RowVector3f r3 = Rs[i].row(2);
            
            Eigen::RowVector4f P1; P1 << r1, ts[i](0);
            Eigen::RowVector4f P2; P2 << r2, ts[i](1);
            Eigen::RowVector4f P3; P3 << r3, ts[i](2);

            A.row(2) = points2[j].x() * P3 - P1;
            A.row(3) = points2[j].y() * P3 - P2;

            Eigen::JacobiSVD<Eigen::Matrix4f> svd_tri(A, Eigen::ComputeFullV);
            Eigen::Vector4f X = svd_tri.matrixV().col(3);
            X /= X(3);

            // Depth in camera 1
            float depth1 = X(2);
            // Depth in camera 2
            Eigen::Vector3f X2 = Rs[i] * X.head<3>() + ts[i];
            float depth2 = X2(2);

            if (depth1 > 0 && depth2 > 0) count++;
        }

        if (count > best_count) {
            best_count = count;
            best_idx = i;
        }
    }

    R = Rs[best_idx];
    t = ts[best_idx];
}

} // namespace vo
