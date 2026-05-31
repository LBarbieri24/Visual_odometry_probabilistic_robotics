#include "vo/data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace vo {

CameraParams DataLoader::loadCamera(const std::string& filename) {
    CameraParams params;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.find("camera matrix:") != std::string::npos) {
            for (int i = 0; i < 3; ++i) {
                file >> params.K(i, 0) >> params.K(i, 1) >> params.K(i, 2);
            } 
        } else if (line.find("cam_transform:") != std::string::npos) {
            for (int i = 0; i < 4; ++i) {
                file >> params.cam_transform(i, 0) >> params.cam_transform(i, 1) >> params.cam_transform(i, 2) >> params.cam_transform(i, 3);
            }
        } else if (line.find("z_near:") != std::string::npos) {
            params.z_near = std::stof(line.substr(line.find(":") + 1));
        } else if (line.find("z_far:") != std::string::npos) {
            params.z_far = std::stof(line.substr(line.find(":") + 1));
        } else if (line.find("width:") != std::string::npos) {
            params.width = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("height:") != std::string::npos) {
            params.height = std::stoi(line.substr(line.find(":") + 1));
        }
    }
    return params;
}

Frame DataLoader::loadFrame(const std::string& filename) {
    Frame frame;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label;
        ss >> label;

        if (label == "seq:") {
            ss >> frame.seq;
        } else if (label == "gt_pose:") {
            ss >> frame.gt_pose.x() >> frame.gt_pose.y() >> frame.gt_pose.z();
        } else if (label == "odom_pose:") {
            ss >> frame.odom_pose.x() >> frame.odom_pose.y() >> frame.odom_pose.z();
        } else if (label == "point") {
            MeasurementPoint p;
            ss >> p.id_in_frame >> p.actual_id >> p.uv.x() >> p.uv.y();
            
            p.descriptor.resize(10);
            for (int i = 0; i < 10; ++i) {
                ss >> p.descriptor(i);
            }
            frame.points.push_back(p);
        }
    }
    return frame;
}

std::unordered_map<int, Eigen::Vector3f> DataLoader::loadWorld(const std::string& filename) {
    std::unordered_map<int, Eigen::Vector3f> world;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open world file: " << filename << std::endl;
        return world;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int id;
        float x, y, z;
        if (ss >> id >> x >> y >> z) {
            world[id] = Eigen::Vector3f(x, y, z);
        }
    }
    return world;
}

} // namespace vo
