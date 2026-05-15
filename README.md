# Visual Odometry Probabilistic Robotics

In this repo will be stored the code for the probabilistic robotics project on odometry retrieval through image inputs.

## Phase 1: Initialization and Epipolar Geometry
The first phase of this project implements the initialization of the visual odometry system using two-frame epipolar geometry.

### Key Features:
- **Feature Extraction and Matching**: Implemented a brute force matcher that takes every point in Frame 1 and compares it against every single point in Frame 2. Since it's computationally inefficient I'll try to find a better solution in the future. 
- **Robust Essential Matrix Estimation**: Implementation of the 8-point algorithm wrapped in a **RANSAC** loop to handle outliers and compute a reliable Essential Matrix.
- **Motion Decomposition**: SVD-based decomposition of the Essential Matrix to retrieve the relative rotation ($R$) and translation ($t$).
- **Scale Ambiguity Handling**: Setting the initial baseline to unit scale to establish a relative coordinate system.
- **Ground Truth Evaluation**: Comparison of estimated pose against ground truth to validate the geometric pipeline.
