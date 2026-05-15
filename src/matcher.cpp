#include "vo/matcher.h"
#include <limits>

namespace vo {

std::vector<Match> Matcher::matchFrames(const Frame& f1, const Frame& f2, float threshold) {
    std::vector<Match> matches;

    for (size_t i = 0; i < f1.points.size(); ++i) {
        int best_idx = -1;
        float min_dist = std::numeric_limits<float>::max();

        for (size_t j = 0; j < f2.points.size(); ++j) {
            float dist = (f1.points[i].descriptor - f2.points[j].descriptor).norm();
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }

        if (best_idx != -1 && min_dist < threshold) {
            matches.push_back({(int)i, best_idx, min_dist});
        }
    }

    return matches;
}

} // namespace vo
