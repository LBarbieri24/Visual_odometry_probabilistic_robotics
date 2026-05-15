#include <iostream>
#include <iomanip>
#include "vo/data_loader.h"
#include "vo/matcher.h"
#include "vo/epipolar.h"

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
    Eigen::Matrix4f rel_GT = T0.inverse() * T1;

    std::cout << "\nGround Truth Relative Rotation:\n" << rel_GT.block<3, 3>(0, 0) << std::endl;
    std::cout << "Ground Truth Relative Translation:\n" << rel_GT.block<3, 1>(0, 3).transpose() << std::endl;

    // Evaluation as per README
    // 1. Translation Norm Ratio
    float t_est_norm = t.norm();
    float t_gt_norm = rel_GT.block<3, 1>(0, 3).norm();
    std::cout << "\n--- Evaluation (as per README) ---" << std::endl;
    std::cout << "Translation Norm Ratio (Scale): " << t_est_norm / t_gt_norm << std::endl;

    // 2. Rotation Error (Trace)
    // Note: This might be high if axes are swapped, but follows the 'ignore cam_transform' rule.
    float rot_error = (Eigen::Matrix3f::Identity() - R.transpose() * rel_GT.block<3, 3>(0, 0)).trace();
    std::cout << "Rotation Error (Trace): " << rot_error << std::endl;

    return 0;
}
