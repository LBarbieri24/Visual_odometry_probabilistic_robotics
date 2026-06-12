cmake_minimum_required(VERSION 3.10)
project(VisualOdometry)

set(CMAKE_CXX_STANDARD 17)

# Find Eigen3
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(vo_main 
    src/main.cpp 
    src/data_loader.cpp 
    src/matcher.cpp 
    src/epipolar.cpp
    src/triangulation.cpp
    src/tracking.cpp
)

target_link_libraries(vo_main Eigen3::Eigen)
