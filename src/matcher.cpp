#include "vo/matcher.h"
#include <limits>

namespace vo {

std::vector<Match> Matcher::matchFrames(
    const Frame& f1, const Frame& f2,
    float threshold, float ratio_threshold) {

    const size_t N1 = f1.points.size();
    const size_t N2 = f2.points.size();

    // ----------------------------------------------------------------
    // Step 1: Forward pass (f1 -> f2)
    // For each point in f1, find the best AND second-best match in f2.
    // Needed for:
    //   - Gating          (best dist < threshold)
    //   - Lonely BF test  (best dist / second-best dist < ratio)
    // ----------------------------------------------------------------
    std::vector<int>   fwd_best(N1, -1);
    std::vector<float> fwd_dist1(N1, std::numeric_limits<float>::max()); // best
    std::vector<float> fwd_dist2(N1, std::numeric_limits<float>::max()); // second-best

    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            float dist = (f1.points[i].descriptor - f2.points[j].descriptor).norm();
            if (dist < fwd_dist1[i]) {
                fwd_dist2[i] = fwd_dist1[i]; // demote current best to second
                fwd_dist1[i] = dist;
                fwd_best[i]  = (int)j;
            } else if (dist < fwd_dist2[i]) {
                fwd_dist2[i] = dist;
            }
        }
    }

    // ----------------------------------------------------------------
    // Step 2: Backward pass (f2 -> f1)
    // For each point in f2, find the single best match in f1.
    // Needed for:
    //   - Best Friends cross-check: j's best in f1 must be i
    // ----------------------------------------------------------------
    std::vector<int> bwd_best(N2, -1);
    std::vector<float> bwd_dist(N2, std::numeric_limits<float>::max());

    for (size_t j = 0; j < N2; ++j) {
        for (size_t i = 0; i < N1; ++i) {
            float dist = (f2.points[j].descriptor - f1.points[i].descriptor).norm();
            if (dist < bwd_dist[j]) {
                bwd_dist[j] = dist;
                bwd_best[j] = (int)i;
            }
        }
    }

    // ----------------------------------------------------------------
    // Step 3: Apply all three Grisetti heuristics
    // ----------------------------------------------------------------
    std::vector<Match> matches;
    for (size_t i = 0; i < N1; ++i) {
        int j = fwd_best[i];
        if (j == -1) continue;

        // Heuristic 1 — Gating: absolute distance must be below threshold
        if (fwd_dist1[i] >= threshold) continue;

        // Heuristic 2 — Lonely Best Friends (ratio test):
        // The best match must be clearly better than the second-best.
        // Guards against ambiguous matches where two candidates look similar.
        if (fwd_dist2[i] < std::numeric_limits<float>::max()) {
            if (fwd_dist1[i] / fwd_dist2[i] >= ratio_threshold) continue;
        }

        // Heuristic 3 — Best Friends (cross-check):
        // j's best match in f1 must also be i (mutual agreement).
        // Guards against many-to-one assignments.
        if (bwd_best[j] != (int)i) continue;

        matches.push_back({(int)i, j, fwd_dist1[i]});
    }

    return matches;
}

} // namespace vo
