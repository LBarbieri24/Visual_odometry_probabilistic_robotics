#include <iostream>
#include <iomanip>
#include "vo/data_loader.h"
#include "vo/matcher.h"
#include "vo/epipolar.h"
#include "vo/triangulation.h"

int main() {
    using namespace vo;

    std::cout << "--- Visual Odometry Initialization ---" << std::endl;

    // 1. Load Camera
    CameraParams cam = DataLoader::loadCamera("02-VisualOdometry/data/camera.dat");
    std::cout << "Camera K:\n" << cam.K << std::endl;

    // 2. Load Frames
    Frame f0 = DataLoader::loadFrame("02-VisualOdometry/data/meas-00000.dat");
    Frame f1 = DataLoader::loadFrame("02-VisualOdometry/data/meas-00001.dat");
    std::cout << "Loaded Frame 0 with " << f0.points.size() << " points." << std::endl;
    std::cout << "Loaded Frame 1 with " << f1.points.size() << " points." << std::endl;

    // 3. Match Features
    std::vector<Match> matches = Matcher::matchFrames(f0, f1, 0.1f); // Lower threshold for synthetic data
    std::cout << "Found " << matches.size() << " matches." << std::endl;

    if (matches.size() < 8) {
        std::cerr << "Not enough matches to initialize!" << std::endl;
        return 1;
    }

    // 4. Prepare normalized coordinates
    Eigen::Matrix3f Kinv = cam.K.inverse();
    std::vector<Eigen::Vector2f> pts0, pts1;
    for (const auto& m : matches) {
        Eigen::Vector3f p0_h(f0.points[m.query_idx].uv.x(), f0.points[m.query_idx].uv.y(), 1.0f);
        Eigen::Vector3f p1_h(f1.points[m.train_idx].uv.x(), f1.points[m.train_idx].uv.y(), 1.0f);
        
        Eigen::Vector3f p0_n = Kinv * p0_h;
        Eigen::Vector3f p1_n = Kinv * p1_h;
        
        pts0.push_back(p0_n.head<2>());
        pts1.push_back(p1_n.head<2>());
    }

    // 5. Estimate Essential Matrix
    std::vector<int> inliers;
    Eigen::Matrix3f E = Epipolar::robustEstimateEssentialMatrix(pts0, pts1, inliers);
    std::cout << "Essential Matrix Estimated. Inliers: " << inliers.size() << " / " << matches.size() << std::endl;

    // 6. Decompose E
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    Epipolar::decomposeEssentialMatrix(E, pts0, pts1, R, t);

    std::cout << "\nEstimated Rotation R:\n" << R << std::endl;
    std::cout << "\nEstimated Translation t (unit scale):\n" << t.transpose() << std::endl;

    // 7. Ground Truth Comparison (Frame 0 to 1)
    // f0.gt_pose and f1.gt_pose are (x, y, theta)
    auto toSE3 = [](const Eigen::Vector3f& pose) {
        float x = pose.x(), y = pose.y(), theta = pose.z();
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(0, 0) = std::cos(theta); T(0, 1) = -std::sin(theta);
        T(1, 0) = std::sin(theta); T(1, 1) = std::cos(theta);
        T(0, 3) = x;
        T(1, 3) = y;
        return T;
    };

    Eigen::Matrix4f T0 = toSE3(f0.gt_pose);
    Eigen::Matrix4f T1 = toSE3(f1.gt_pose);
    Eigen::Matrix4f rel_GT_robot = T0.inverse() * T1;

    // Transform relative ground truth motion to the camera frame
    Eigen::Matrix4f cam_transform = cam.cam_transform;
    Eigen::Matrix4f rel_GT = cam_transform.inverse() * rel_GT_robot * cam_transform;

    std::cout << "\nGround Truth Relative Rotation (Camera Frame):\n" << rel_GT.block<3, 3>(0, 0) << std::endl;
    std::cout << "Ground Truth Relative Translation (Camera Frame):\n" << rel_GT.block<3, 1>(0, 3).transpose() << std::endl;

    // Estimate relative pose SE(3) matrix:
    Eigen::Matrix4f rel_T = Eigen::Matrix4f::Identity();
    rel_T.block<3, 3>(0, 0) = R;
    rel_T.block<3, 1>(0, 3) = t;

    // Evaluation as per README
    // 1. Translation Norm Ratio
    float t_est_norm = t.norm();
    float t_gt_norm = rel_GT.block<3, 1>(0, 3).norm();
    std::cout << "\n--- Evaluation (as per README) ---" << std::endl;
    std::cout << "Translation Norm Ratio (Scale): " << t_est_norm / t_gt_norm << std::endl;

    // 2. Rotation Error (Trace of (Identity - inv(rel_T) * rel_GT))
    Eigen::Matrix4f error_T = rel_T.inverse() * rel_GT;
    float rot_error = (Eigen::Matrix3f::Identity() - error_T.block<3, 3>(0, 0)).trace();
    std::cout << "Rotation Error (Trace): " << rot_error << std::endl;

    // Load ground truth map points
    auto gt_map = DataLoader::loadWorld("02-VisualOdometry/data/world.dat");

    float scale_factor = t_gt_norm / t_est_norm;
    double sum_sq_err = 0.0;
    int count = 0;

    std::cout << "\n--- Triangulation & Map Evaluation ---" << std::endl;

    for (int idx : inliers) {
        const auto& m = matches[idx];
        int landmark_id = f0.points[m.query_idx].actual_id;

        if (gt_map.find(landmark_id) == gt_map.end()) continue;

        // Prepare observations for DLT
        vo::Observation obs0;
        obs0.R = Eigen::Matrix3f::Identity();
        obs0.t = Eigen::Vector3f::Zero();
        obs0.uv = f0.points[m.query_idx].uv;

        vo::Observation obs1;
        obs1.R = R;
        obs1.t = t;
        obs1.uv = f1.points[m.train_idx].uv;

        std::vector<vo::Observation> obs = {obs0, obs1};

        // 1. DLT Triangulation
        Eigen::Vector3f X_dlt = vo::Triangulation::triangulateDLT(obs, cam.K);

        // 2. Gauss-Newton Refinement
        Eigen::Vector3f X_refined = vo::Triangulation::refineGaussNewton(X_dlt, obs, cam.K);

        // 3. Metric scaling
        Eigen::Vector3f X_scaled = scale_factor * X_refined;

        // 4. Ground Truth mapping
        Eigen::Vector3f X_gt_global = gt_map[landmark_id];
        Eigen::Vector4f X_gt_global_h(X_gt_global.x(), X_gt_global.y(), X_gt_global.z(), 1.0f);
        Eigen::Matrix4f T_robot_to_cam0 = cam.cam_transform.inverse() * T0.inverse();
        Eigen::Vector3f X_gt_cam0 = (T_robot_to_cam0 * X_gt_global_h).head<3>();

        float err = (X_scaled - X_gt_cam0).norm();
        sum_sq_err += err * err;
        count++;
        
        // Print first 5 points for sanity check
        if (count <= 5) {
            std::cout << "Landmark " << landmark_id << ":\n"
                      << "  GT Cam0:    " << X_gt_cam0.transpose() << "\n"
                      << "  DLT Scaled: " << (scale_factor * X_dlt).transpose() << "\n"
                      << "  Refined:    " << X_scaled.transpose() << "\n"
                      << "  Error (m):  " << err << std::endl;
        }
    }

    if (count > 0) {
        double rmse = std::sqrt(sum_sq_err / count);
        std::cout << "\nTriangulated Points Evaluated: " << count << std::endl;
        std::cout << "Mean Reprojection/Triangulation RMSE: " << rmse << " meters" << std::endl;
    } else {
        std::cout << "No matching ground truth landmarks found for inliers!" << std::endl;
    }

    return 0;
}
