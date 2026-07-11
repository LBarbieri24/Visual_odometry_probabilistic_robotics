#include "vo/tracking.h"
#include "vo/triangulation.h"
#include "vo/epipolar.h"
#include <iostream>
#include <cmath>

namespace vo {

// Rodrigues' formula to convert omega (so(3) Lie Algebra vector) to rotation matrix R
Eigen::Matrix3f rodrigues(const Eigen::Vector3f& omega) {
    float theta = omega.norm();
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    if (theta < 1e-6f) {
        return R;
    }
    Eigen::Vector3f axis = omega / theta;
    Eigen::Matrix3f K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    R = Eigen::Matrix3f::Identity() + std::sin(theta) * K + (1.0f - std::cos(theta)) * K * K;
    return R;
}

bool Tracker::estimatePosePnP(
    const std::vector<Eigen::Vector3f>& pts_3d,
    const std::vector<Eigen::Vector2f>& pts_2d,
    const Eigen::Matrix3f& K,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t,
    int max_iterations,
    float tolerance) {

    int n = pts_3d.size();
    if (n < 4) {
        std::cerr << "PnP needs at least 4 points, got " << n << std::endl;
        return false;
    }

    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    float huber_threshold = 4.0f; // Huber weight threshold in pixels

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();

        for (int i = 0; i < n; ++i) {
            // Transform point to current camera frame: Pc = R * Pw + t
            Eigen::Vector3f Pc = R * pts_3d[i] + t;
            float x = Pc.x(), y = Pc.y(), z = Pc.z();

            if (z <= 1e-3f) continue; // Skip if point is behind or too close to camera

            // Project point: p_proj = K * Pc / z
            float u_proj = fx * (x / z) + cx;
            float v_proj = fy * (y / z) + cy;

            // Reprojection error: e = proj - measurement
            Eigen::Vector2f e(u_proj - pts_2d[i].x(), v_proj - pts_2d[i].y());
            float error_norm = e.norm();

            // Huber robust weight
            float weight = 1.0f;
            if (error_norm > huber_threshold) {
                weight = huber_threshold / error_norm;
            }

            // Jacobian of error w.r.t pose perturbation delta_xi = [v, omega]
            Eigen::Matrix<float, 2, 6> J;
            J << fx / z, 0.0f,      -fx * x / (z * z), -fx * x * y / (z * z), fx * (1.0f + (x * x) / (z * z)), -fx * y / z,
                 0.0f,   fy / z,    -fy * y / (z * z), -fy * (1.0f + (y * y) / (z * z)), fy * x * y / (z * z),  fy * x / z;

            H += weight * J.transpose() * J;
            b += weight * J.transpose() * e;
        }

        // Check Hessian invertibility
        if (std::abs(H.determinant()) < 1e-12f) {
            break;
        }

        // Solve H * delta_xi = -b
        Eigen::Matrix<float, 6, 1> delta_xi = H.ldlt().solve(-b);

        if (delta_xi.norm() < tolerance) {
            break;
        }

        // Update pose: R = exp(omega) * R, t = exp(omega) * t + v
        Eigen::Vector3f v = delta_xi.head<3>();
        Eigen::Vector3f omega = delta_xi.tail<3>();

        Eigen::Matrix3f R_up = rodrigues(omega);
        R = R_up * R;
        t = R_up * t + v;
    }

    return true;
}

std::vector<Eigen::Matrix4f> Tracker::runVisualOdometry(
    const std::vector<std::string>& frame_files,
    const CameraParams& cam,
    std::unordered_map<int, Eigen::Vector3f>& map_points) {

    std::vector<Eigen::Matrix4f> estimated_poses;
    if (frame_files.size() < 2) return estimated_poses;

    // 1. Load frame 0 and frame 1
    Frame f0 = DataLoader::loadFrame(frame_files[0]);
    Frame f1 = DataLoader::loadFrame(frame_files[1]);

    std::cout << "[VO] Initializing using Frame 0 (" << f0.points.size() 
              << " pts) and Frame 1 (" << f1.points.size() << " pts)..." << std::endl;

    // 2. Match frames 0 and 1
    std::vector<Match> matches = Matcher::matchFrames(f0, f1, 0.1f);
    if (matches.size() < 8) {
        std::cerr << "[VO] Error: Not enough matches to initialize!" << std::endl;
        return estimated_poses;
    }

    // 3. Prepare normalized coordinates
    Eigen::Matrix3f Kinv = cam.K.inverse();
    std::vector<Eigen::Vector2f> pts0, pts1;
    for (const auto& m : matches) {
        Eigen::Vector3f p0_h(f0.points[m.query_idx].uv.x(), f0.points[m.query_idx].uv.y(), 1.0f);
        Eigen::Vector3f p1_h(f1.points[m.train_idx].uv.x(), f1.points[m.train_idx].uv.y(), 1.0f);
        
        pts0.push_back((Kinv * p0_h).head<2>());
        pts1.push_back((Kinv * p1_h).head<2>());
    }

    // 4. Estimate Essential Matrix
    std::vector<int> inliers;
    Eigen::Matrix3f E = Epipolar::robustEstimateEssentialMatrix(pts0, pts1, inliers);

    // 5. Decompose E
    Eigen::Matrix3f R1;
    Eigen::Vector3f t1;
    Epipolar::decomposeEssentialMatrix(E, pts0, pts1, R1, t1);

    // 6. Initialize using unit-norm translation (relative scale)
    Eigen::Vector3f t1_scaled = t1.normalized();

    // Anchor observation: one reference view per landmark, used for re-refinement.
    // We use the Frame-0 observation (R=I, t=0) as the anchor since it has
    // the most accurate pose (GT-calibrated scale, no accumulated drift).
    std::unordered_map<int, vo::Observation> anchor_obs;

    // 7. Triangulate initial landmarks (in Camera 0 frame)
    for (int idx : inliers) {
        const auto& m = matches[idx];
        int landmark_id = f0.points[m.query_idx].actual_id;

        vo::Observation obs0;
        obs0.R = Eigen::Matrix3f::Identity();
        obs0.t = Eigen::Vector3f::Zero();
        obs0.uv = f0.points[m.query_idx].uv;

        vo::Observation obs1;
        obs1.R = R1;
        obs1.t = t1_scaled;
        obs1.uv = f1.points[m.train_idx].uv;

        Eigen::Vector3f X_refined = vo::Triangulation::triangulateGaussNewtonOnly({obs0, obs1}, cam.K);

        map_points[landmark_id] = X_refined;
        anchor_obs[landmark_id] = obs0; // store Frame-0 view as anchor
    }

    // Save initial poses
    Eigen::Matrix4f T0_est = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T1_est = Eigen::Matrix4f::Identity();
    T1_est.block<3, 3>(0, 0) = R1;
    T1_est.block<3, 1>(0, 3) = t1_scaled;

    estimated_poses.push_back(T0_est);
    estimated_poses.push_back(T1_est);

    std::cout << "[VO] Initialized map with " << map_points.size() << " landmarks." << std::endl;

    // 8. Track subsequent frames
    Eigen::Matrix3f R_prev = R1;
    Eigen::Vector3f t_prev = t1_scaled;
    Frame f_prev = f1;

    for (size_t k = 2; k < frame_files.size(); ++k) {
        Frame f_curr = DataLoader::loadFrame(frame_files[k]);

        // Match previous and current frame
        std::vector<Match> matches_k = Matcher::matchFrames(f_prev, f_curr, 0.1f);

        // Find 3D-to-2D correspondences
        std::vector<Eigen::Vector3f> pts_3d;
        std::vector<Eigen::Vector2f> pts_2d;

        for (const auto& m : matches_k) {
            int landmark_id = f_prev.points[m.query_idx].actual_id;
            if (map_points.find(landmark_id) != map_points.end()) {
                pts_3d.push_back(map_points[landmark_id]);
                pts_2d.push_back(f_curr.points[m.train_idx].uv);
            }
        }

        // Estimate camera pose (PnP) starting from previous frame's pose
        Eigen::Matrix3f R = R_prev;
        Eigen::Vector3f t = t_prev;
        bool ok = estimatePosePnP(pts_3d, pts_2d, cam.K, R, t);

        if (ok) {
            Eigen::Matrix4f T_k = Eigen::Matrix4f::Identity();
            T_k.block<3, 3>(0, 0) = R;
            T_k.block<3, 1>(0, 3) = t;
            estimated_poses.push_back(T_k);

            int new_points_count = 0;
            int refined_count = 0;
            for (const auto& m : matches_k) {
                int landmark_id = f_prev.points[m.query_idx].actual_id;

                if (map_points.find(landmark_id) != map_points.end()) {
                    // --- Re-refinement of existing map points on re-observation ---
                    // Build a new observation from the current (just-estimated) pose.
                    // Then re-run Gauss-Newton using anchor_obs + new_obs so that
                    // the extra view pulls the 3D position toward a better estimate.
                    if (anchor_obs.find(landmark_id) != anchor_obs.end()) {
                        vo::Observation obs_new;
                        obs_new.R   = R;
                        obs_new.t   = t;
                        obs_new.uv  = f_curr.points[m.train_idx].uv;

                        std::vector<vo::Observation> obs_ref = { anchor_obs[landmark_id], obs_new };
                        map_points[landmark_id] = vo::Triangulation::refineGaussNewton(
                            map_points[landmark_id], obs_ref, cam.K);
                        refined_count++;
                    }
                } else {
                    // --- Map expansion: triangulate new landmarks ---
                    vo::Observation obs_prev;
                    obs_prev.R  = R_prev;
                    obs_prev.t  = t_prev;
                    obs_prev.uv = f_prev.points[m.query_idx].uv;

                    vo::Observation obs_curr;
                    obs_curr.R  = R;
                    obs_curr.t  = t;
                    obs_curr.uv = f_curr.points[m.train_idx].uv;

                    Eigen::Vector3f X_refined = vo::Triangulation::triangulateGaussNewtonOnly({obs_prev, obs_curr}, cam.K);

                    map_points[landmark_id]  = X_refined;
                    anchor_obs[landmark_id]  = obs_prev; // store first observation as anchor
                    new_points_count++;
                }
            }

            R_prev = R;
            t_prev = t;
            f_prev = f_curr;
        } else {
            std::cerr << "[VO] Failed to track frame: " << frame_files[k] << "! Propagating previous pose." << std::endl;
            Eigen::Matrix4f T_k = Eigen::Matrix4f::Identity();
            T_k.block<3, 3>(0, 0) = R_prev;
            T_k.block<3, 1>(0, 3) = t_prev;
            estimated_poses.push_back(T_k);
            f_prev = f_curr;
        }
    }

    return estimated_poses;
}

} // namespace vo
