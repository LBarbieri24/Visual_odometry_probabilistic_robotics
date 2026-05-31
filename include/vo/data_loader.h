#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

namespace vo {

struct CameraParams {
    Eigen::Matrix3f K;
    Eigen::Matrix4f cam_transform;
    float z_near;
    float z_far;
    int width;
    int height;
};

struct MeasurementPoint {
    int id_in_frame;
    int actual_id;
    Eigen::Vector2f uv;
    Eigen::VectorXf descriptor;
};

struct Frame {
    int seq;
    Eigen::Vector3f gt_pose;
    Eigen::Vector3f odom_pose;
    std::vector<MeasurementPoint> points;
};

class DataLoader {
public:
    static CameraParams loadCamera(const std::string& filename);
    static Frame loadFrame(const std::string& filename);
    static std::unordered_map<int, Eigen::Vector3f> loadWorld(const std::string& filename);
};

} // namespace vo
