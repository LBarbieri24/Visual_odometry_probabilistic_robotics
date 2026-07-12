# Visual Odometry — Probabilistic Robotics

Implementation of a monocular visual odometry pipeline for the Probabilistic Robotics course project.

---

## Build

**Dependency:** [Eigen3](https://eigen.tuxfamily.org) — `sudo apt install libeigen3-dev`

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## Phase 1: Initialization & Epipolar Geometry

Initialization from the first two frames using epipolar geometry.

- **Feature matching:** brute-force appearance-based matcher with Lowe's ratio test and mutual cross-check to reduce false correspondences.
- **Essential matrix estimation:** 8-point algorithm inside a RANSAC loop to reject outliers.
- **Motion decomposition:** SVD of the Essential Matrix gives 4 candidate (R, t) pairs; chirality test selects the correct one.
- **Scale:** translation is normalized to unit length since monocular VO cannot recover metric scale from two views alone.

---

## Phase 2: Triangulation & Map Initialization

3D landmark positions are reconstructed from the two initialization frames.

- **Midpoint triangulation:** closed-form intersection of two camera rays, used as the initial guess.
- **Gauss-Newton refinement:** minimizes 2D reprojection error w.r.t. the 3D point position using analytically derived Jacobians.
- **Evaluation:** landmarks are scaled back to metric using the baseline ratio, then compared against ground truth from `world.dat` transformed into the Camera 0 frame:
  $$\mathbf{X}_{gt}^{cam_0} = (\mathbf{T}_C^R)^{-1} (\mathbf{T}_0^{robot})^{-1} \mathbf{X}_{gt}^{world}$$

---

## Phase 3: Tracking & Sequence Evaluation

Frame-to-frame tracking over the full sequence using PnP pose estimation and incremental map expansion.

- **PnP pose estimation:** for each new frame, 3D-to-2D correspondences between existing map points and the new frame's observations are used to estimate the camera pose via Gauss-Newton on SE(3).
- **Outlier handling:** Huber loss inside the optimization to down-weight large reprojection errors.
- **Map expansion:** feature matches with no existing 3D counterpart are triangulated and added to the map.
- **Pose evaluation:** per-pair relative rotation error `trace(I - R_err)` and translation ratio `||t_est|| / ||t_gt||` are reported for every consecutive frame pair. The ratio is expected to be consistent across the sequence (scale does not drift). Absolute trajectory RMSE is computed after rescaling estimates to metric using the mean ratio.
- **Map evaluation:** the final map is rescaled using `1 / mean_ratio` and compared against ground truth landmarks. RMSE is reported over all triangulated landmarks that appear in `world.dat`.

---

## Results

Evaluated on the provided 121-frame dataset (`02-VisualOdometry/data`).

### Pose evaluation

| Metric | Value |
|---|---|
| Absolute Trajectory RMSE | 1.368e-03 m |
| Relative Rotation RMSE | 2.076e-07 |
| Mean scale ratio `‖t_est‖ / ‖t_gt‖` | 4.988 ± 0.004 |

The scale ratio is consistent across all 121 frames (std dev < 0.1%), confirming no scale drift in the tracking.

### Map evaluation

| Metric | Value |
|---|---|
| Landmarks triangulated | 490 / 1000 |
| Landmark position RMSE | 2.786e-03 m |

Scale applied for map evaluation: `1 / 4.988 ≈ 0.2005` (GT/EST ratio, to convert to metric).

---

## Using a Different Dataset

The pipeline reads data from a directory passed as a command-line argument (defaults to `02-VisualOdometry/data`):

```bash
./build/vo_main /path/to/your/dataset
```

After running, two files are written into the dataset directory:

| File | Content |
|---|---|
| `estimated_trajectory.dat` | One line per frame: `frame_id x y z` — camera centre in Camera-0 frame, metric scale |
| `estimated_world.dat` | One line per landmark: `landmark_id x y z` — 3D position in Camera-0 frame, metric scale |

> **Note on coordinate frames:** both output files are expressed in **Camera-0 frame**, not the robot global frame used by the raw GT files. `world.dat` and `trajectory.dat` cannot be compared directly against these outputs — the GT data must first be transformed to Camera-0 frame using `cam_transform` and the first robot pose `T_0`, which is what `main.cpp` does internally for its RMSE computation.
