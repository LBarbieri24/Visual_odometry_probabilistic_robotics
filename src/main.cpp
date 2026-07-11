#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "vo/data_loader.h"
#include "vo/matcher.h"
#include "vo/epipolar.h"
#include "vo/triangulation.h"
#include "vo/tracking.h"

int main(int argc, char** argv) {
    using namespace vo;

    std::string dataset_path = "02-VisualOdometry/data";
    if (argc >= 2) {
        dataset_path = argv[1];
    }

    // 1. Load Camera
    CameraParams cam = DataLoader::loadCamera(dataset_path + "/camera.dat");

    // 2. Load Frames
    Frame f0 = DataLoader::loadFrame(dataset_path + "/meas-00000.dat");
    Frame f1 = DataLoader::loadFrame(dataset_path + "/meas-00001.dat");

    // 3. Match Features
    std::vector<Match> matches = Matcher::matchFrames(f0, f1, 0.1f); 

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

    // 6. Decompose E
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    Epipolar::decomposeEssentialMatrix(E, pts0, pts1, R, t);

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

    // Estimate relative pose SE(3) matrix:
    Eigen::Matrix4f rel_T = Eigen::Matrix4f::Identity();
    rel_T.block<3, 3>(0, 0) = R;
    rel_T.block<3, 1>(0, 3) = t;

    // Evaluation
    // 1. Translation Norm Ratio
    float t_est_norm = t.norm();
    float t_gt_norm = rel_GT.block<3, 1>(0, 3).norm();

    // 2. Rotation Error (Trace of (Identity - inv(rel_T) * rel_GT))
    Eigen::Matrix4f error_T = rel_T.inverse() * rel_GT;
    float rot_error = (Eigen::Matrix3f::Identity() - error_T.block<3, 3>(0, 0)).trace();

    // Load ground truth map points
    auto gt_map = DataLoader::loadWorld(dataset_path + "/world.dat");

    float scale_factor = t_gt_norm / t_est_norm;
    double sum_sq_err = 0.0;
    int count = 0;

    for (int idx : inliers) {
        const auto& m = matches[idx];
        int landmark_id = f0.points[m.query_idx].actual_id;

        if (gt_map.find(landmark_id) == gt_map.end()) continue;

        // Prepare observations
        vo::Observation obs0;
        obs0.R = Eigen::Matrix3f::Identity();
        obs0.t = Eigen::Vector3f::Zero();
        obs0.uv = f0.points[m.query_idx].uv;

        vo::Observation obs1;
        obs1.R = R;
        obs1.t = t;
        obs1.uv = f1.points[m.train_idx].uv;

        std::vector<vo::Observation> obs = {obs0, obs1};

        // 1. Triangulate using pure Gauss-Newton (initialized with midpoint method)
        Eigen::Vector3f X_refined = vo::Triangulation::triangulateGaussNewtonOnly(obs, cam.K);

        // 2. Metric scaling
        Eigen::Vector3f X_scaled = scale_factor * X_refined;

        // 3. Ground Truth mapping
        Eigen::Vector3f X_gt_global = gt_map[landmark_id];
        Eigen::Vector4f X_gt_global_h(X_gt_global.x(), X_gt_global.y(), X_gt_global.z(), 1.0f);
        Eigen::Matrix4f T_robot_to_cam0 = cam.cam_transform.inverse() * T0.inverse();
        Eigen::Vector3f X_gt_cam0 = (T_robot_to_cam0 * X_gt_global_h).head<3>();

        float err = (X_scaled - X_gt_cam0).norm();
        sum_sq_err += err * err;
        count++;
    }

    // 1. Generate frame filenames dynamically until files do not exist
    std::vector<std::string> frame_files;
    std::vector<Frame> frames;
    for (int i = 0; ; ++i) {
        std::stringstream ss;
        ss << dataset_path << "/meas-" << std::setfill('0') << std::setw(5) << i << ".dat";
        std::ifstream test_file(ss.str());
        if (!test_file.good()) {
            break;
        }
        frame_files.push_back(ss.str());
        frames.push_back(DataLoader::loadFrame(ss.str()));
    }

    // 2. Initialize tracking map using the relative points from Phase 2 (unscaled)
    std::unordered_map<int, Eigen::Vector3f> tracking_map;
    for (int idx : inliers) {
        const auto& m = matches[idx];
        int landmark_id = f0.points[m.query_idx].actual_id;
        
        vo::Observation obs0;
        obs0.R = Eigen::Matrix3f::Identity();
        obs0.t = Eigen::Vector3f::Zero();
        obs0.uv = f0.points[m.query_idx].uv;

        vo::Observation obs1;
        obs1.R = R;
        obs1.t = t;
        obs1.uv = f1.points[m.train_idx].uv;

        std::vector<vo::Observation> obs = {obs0, obs1};
        Eigen::Vector3f X_refined = vo::Triangulation::triangulateGaussNewtonOnly(obs, cam.K);
        
        tracking_map[landmark_id] = X_refined;
    }

    // Run the rest of the sequence (from frame 1 to 120)
    std::vector<Eigen::Matrix4f> est_poses = Tracker::runVisualOdometry(frame_files, cam, tracking_map);

    // Calculate ground truth camera poses in Camera 0 frame (maps Camera i to Camera 0)
    std::vector<Eigen::Matrix4f> gt_poses;
    Eigen::Matrix4f T0_gt_robot = toSE3(frames[0].gt_pose);
    for (const auto& f : frames) {
        Eigen::Matrix4f Tk_gt_robot = toSE3(f.gt_pose);
        Eigen::Matrix4f rel_robot = T0_gt_robot.inverse() * Tk_gt_robot;
        Eigen::Matrix4f Tk_cam = cam.cam_transform.inverse() * rel_robot * cam.cam_transform;
        gt_poses.push_back(Tk_cam);
    }

    // eval_scale_factor is computed AFTER collecting all per-pair ratios below.
    // It will be set to the mean of scale_ratios (consistent with the README which
    // says to check the ratio is consistent over all poses, then use it to scale).
    float eval_scale_factor = 1.0f; // will be overwritten after scale_ratios is filled

    // 3. Trajectory Evaluation — Pass 1: collect per-pair errors and scale ratios
    double sum_sq_rotation_err = 0.0;
    std::vector<float> scale_ratios;
    int evaluated_frames = 0;

    struct EvaluationRow {
        int index;
        float rot_err;
        float ratio;
        bool has_ratio;
    };
    std::vector<EvaluationRow> eval_rows;

    for (size_t i = 0; i < est_poses.size(); ++i) {
        EvaluationRow row;
        row.index = i;
        row.rot_err = 0.0f;
        row.ratio = 0.0f;
        row.has_ratio = false;

        if (i > 0) {
            Eigen::Matrix4f rel_T  = est_poses[i-1] * est_poses[i].inverse();
            Eigen::Matrix4f rel_GT = gt_poses[i-1].inverse() * gt_poses[i];

            // SE(3) error between relative motions (rotation is scale-independent)
            Eigen::Matrix4f error_T = rel_T.inverse() * rel_GT;
            row.rot_err = (Eigen::Matrix3f::Identity() - error_T.block<3,3>(0,0)).trace();
            sum_sq_rotation_err += row.rot_err * row.rot_err;

            // Translation scale ratio: norm(rel_T_t) / norm(rel_GT_t)
            // This is the raw ratio check described in the README.
            float step_est = rel_T.block<3, 1>(0, 3).norm();
            float step_gt  = rel_GT.block<3, 1>(0, 3).norm();
            if (step_gt > 0.05f) {
                row.ratio = step_est / step_gt;
                row.has_ratio = true;
                scale_ratios.push_back(row.ratio);
            }
        }
        eval_rows.push_back(row);
        evaluated_frames++;
    }

    // Determine the final scale factor to convert unscaled estimates to metric.
    // The per-pair ratio = norm(est_step) / norm(gt_step) is printed for consistency check.
    // To map estimated coordinates -> metric: multiply by gt/est = 1/ratio.
    if (!scale_ratios.empty()) {
        double sum_r = 0.0;
        for (float r : scale_ratios) sum_r += r;
        double mean_ratio = sum_r / scale_ratios.size();
        eval_scale_factor = (mean_ratio > 1e-9) ? static_cast<float>(1.0 / mean_ratio) : 1.0f;
    } else if (est_poses.size() > 1 && gt_poses.size() > 1) {
        // Fallback: use first pair baseline if no moving pairs found
        float est_baseline = est_poses[1].inverse().block<3, 1>(0, 3).norm();
        float gt_baseline  = gt_poses[1].block<3, 1>(0, 3).norm();
        if (est_baseline > 1e-6f) eval_scale_factor = gt_baseline / est_baseline;
    }

    // Pass 2: compute absolute trajectory RMSE using the now-determined scale factor
    double sum_sq_translation_err = 0.0;
    for (size_t i = 0; i < est_poses.size(); ++i) {
        Eigen::Vector3f t_est = eval_scale_factor * est_poses[i].inverse().block<3, 1>(0, 3);
        Eigen::Vector3f t_gt  = gt_poses[i].block<3, 1>(0, 3);
        float trans_err = (t_est - t_gt).norm();
        sum_sq_translation_err += trans_err * trans_err;
    }

    // Rotation RMSE over relative steps (N-1 pairs, frame 0 has no relative error)
    int rel_frames = std::max(1, evaluated_frames - 1);
    double trans_rmse = std::sqrt(sum_sq_translation_err / evaluated_frames);
    double rot_rmse   = std::sqrt(sum_sq_rotation_err / rel_frames);

    double scale_mean = 0.0;
    for (float r : scale_ratios) scale_mean += r;
    if (!scale_ratios.empty()) scale_mean /= scale_ratios.size();

    double scale_var = 0.0;
    for (float r : scale_ratios) scale_var += (r - scale_mean) * (r - scale_mean);
    double scale_stddev = 0.0;
    if (scale_ratios.size() > 1) {
        scale_stddev = std::sqrt(scale_var / (scale_ratios.size() - 1));
    }

    // 4. Map Evaluation
    double sum_sq_map_err = 0.0;
    int map_count = 0;
    for (const auto& item : tracking_map) {
        int landmark_id = item.first;
        if (gt_map.find(landmark_id) == gt_map.end()) continue;

        Eigen::Vector3f X_est_cam0 = eval_scale_factor * item.second;
        Eigen::Vector3f X_gt_global = gt_map[landmark_id];
        Eigen::Vector4f X_gt_global_h(X_gt_global.x(), X_gt_global.y(), X_gt_global.z(), 1.0f);
        Eigen::Matrix4f T_robot_to_cam0 = cam.cam_transform.inverse() * T0.inverse();
        Eigen::Vector3f X_gt_cam0 = (T_robot_to_cam0 * X_gt_global_h).head<3>();

        float err = (X_est_cam0 - X_gt_cam0).norm();
        sum_sq_map_err += err * err;
        map_count++;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "           POSES EVALUATION             " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(12) << "Frame Pair" 
              << std::setw(28) << "Relative Rot Error (Trace)" 
              << std::setw(25) << "Translation Ratio (Raw)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (size_t i = 1; i < eval_rows.size(); ++i) {
        std::stringstream ss_pair;
        ss_pair << (i-1) << " -> " << i;
        std::string ratio_str = "N/A";
        if (eval_rows[i].has_ratio) {
            std::stringstream ss_ratio;
            ss_ratio << std::fixed << std::setprecision(6) << eval_rows[i].ratio;
            ratio_str = ss_ratio.str();
        } else {
            ratio_str = "N/A (Stationary)";
        }
        std::cout << std::left << std::setw(12) << ss_pair.str()
                  << std::setw(28) << std::scientific << eval_rows[i].rot_err
                  << std::setw(25) << ratio_str << std::endl;
    }
    std::cout << std::string(65, '-') << std::endl;
    std::cout << "Absolute Trajectory Translation RMSE:   " << trans_rmse << " meters" << std::endl;
    std::cout << "Relative Rotation RMSE:                 " << rot_rmse << std::endl;
    std::cout << "Mean Scale Ratio (Raw):                 " << scale_mean << " (std dev: " << scale_stddev << ")" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "            MAP EVALUATION              " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Scale Factor Used (mean ratio, est->GT): " << eval_scale_factor << std::endl;
    if (map_count > 0) {
        double map_rmse = std::sqrt(sum_sq_map_err / map_count);
        std::cout << "Triangulated Map Landmarks:             " << map_count << " / " << gt_map.size() << std::endl;
        std::cout << "Mean Landmark Position RMSE:            " << map_rmse << " meters" << std::endl;
    } else {
        std::cout << "No matching ground truth landmarks found in the estimated map!" << std::endl;
    }

    // --- Output files ---
    // estimated_trajectory.dat  — one line per frame:
    //   frame_id  x  y  z   (camera centre in Camera-0 frame, metric scale)
    {
        std::ofstream traj_out(dataset_path + "/estimated_trajectory.dat");
        traj_out << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < est_poses.size(); ++i) {
            Eigen::Vector3f cam_pos = eval_scale_factor * est_poses[i].inverse().block<3,1>(0,3);
            traj_out << i << " " << cam_pos.x() << " " << cam_pos.y() << " " << cam_pos.z() << "\n";
        }
        std::cout << "\nWrote: " << dataset_path << "/estimated_trajectory.dat"
                  << "  (" << est_poses.size() << " poses)" << std::endl;
    }

    // estimated_world.dat  — one line per landmark:
    //   landmark_id  x  y  z   (position in Camera-0 frame, metric scale)
    {
        std::ofstream world_out(dataset_path + "/estimated_world.dat");
        world_out << std::fixed << std::setprecision(6);
        for (const auto& item : tracking_map) {
            Eigen::Vector3f p = eval_scale_factor * item.second;
            world_out << item.first << " " << p.x() << " " << p.y() << " " << p.z() << "\n";
        }
        std::cout << "Wrote: " << dataset_path << "/estimated_world.dat"
                  << "  (" << tracking_map.size() << " landmarks)" << std::endl;
    }

    return 0;
}
