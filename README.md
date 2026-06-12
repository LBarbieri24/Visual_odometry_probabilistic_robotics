# Visual Odometry (Probabilistic Robotics)

In this repo will be stored the code for the probabilistic robotics project on odometry retrieval through image inputs.



## Phase 1: Initialization & Epipolar Geometry

The first phase implements the initialization of the visual odometry pipeline using two-frame epipolar geometry.

### Key Features
*   **Exhaustive Feature Matching:** Implemented a brute force matcher that takes every point in Frame 1 and compares it against every single point in Frame 2. Since it's computationally inefficient I'll try to find a better solution in the future. 
*   **Robust Essential Matrix Estimation:** Implementation of the 8-point algorithm wrapped in a **RANSAC** loop to handle outliers and compute a reliable Essential Matrix.
*   **SVD Motion Decomposition:** Performs Singular Value Decomposition ($\mathbf{E} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$) to extract the 4 possible rotation/translation combinations, resolving chirality using 3D point projection tests.
*   **Scale Ambiguity Handling:** Establishes a relative coordinate system by normalizing the initial baseline translation vector ($\mathbf{t}$) to unit length.

---

## Phase 2: 3D Point Triangulation & Non-Linear Least Squares Optimization

The second phase reconstructs the 3D map coordinates of the tracked features in the environment and refines them using non-linear least squares optimization.

### Key Features
*   **Linear DLT Triangulation:** Implemented a Direct Linear Transform ($SVD$-based) solver that stacks projection equations from both cameras to compute the initial 3D positions of the inliers.
*   **Gauss-Newton Reprojection Refinement:** Formulated a robust local optimizer that minimizes the physical 2D reprojection error in pixel coordinates. This includes analytical computing of the reprojection Jacobians w.r.t the 3D landmark positions.
*   **Scale-Restored Evaluation:** Recovered metric scales using the translation baseline scale factor derived during Epipolar analysis.
*   **Rigorous Map Validation:** Projects the absolute ground truth landmarks from `world.dat` into the camera's local coordinate frame:
    $$\mathbf{X}_{gt}^{cam0} = (\mathbf{T}_C^R)^{-1} (\mathbf{T}_0)^{-1} \mathbf{X}_{gt}^{world}$$
    and compares them directly against our triangulated coordinates.

---

## Phase 3: Visual Odometry Tracking & Sequence Evaluation

The third phase implements the complete frame-to-frame visual odometry tracking loop using PnP pose estimation, dynamic map expansion, and trajectory evaluation.

### Key Features
*   **Frame-to-Frame PnP Pose Tracking:** Estimates the camera pose of the current frame $\mathbf{T}_k$ by establishing 3D-to-2D correspondences between the existing 3D map points and the new frame's 2D feature observations.
*   **Huber-Weighted Non-Linear Optimization:** Uses a robust Huber loss function inside the Gauss-Newton optimization step to weight reprojection errors and dynamically mitigate the influence of feature mismatch outliers.
*   **Dynamic Map Expansion:** When new feature matches are found between consecutive frames that do not exist in the 3D map, they are triangulated using DLT and refined, continuously building out the 3D mapping path.
*   **Trajectory & Pose Evaluation:** Evaluates absolute trajectory drift (RMSE) and relative pose consistency (scale ratio) compared to ground truth camera trajectories derived from robot odometry:
    $$\mathbf{T}_{cam}^0 = (\mathbf{T}_C^R)^{-1} (\mathbf{T}_{robot}^0)^{-1} \mathbf{T}_{robot}^k \mathbf{T}_C^R$$



