#pragma once

#include "vo/data_loader.h"
#include <vector>

namespace vo {

struct Match {
    int query_idx; // Index in frame 1
    int train_idx; // Index in frame 2
    float distance;
};

class Matcher {
public:
    // threshold:       max absolute descriptor distance (Gating)
    // ratio_threshold: max ratio best/second-best (Lonely Best Friends)
    // Cross-check (Best Friends) is always applied internally.
    static std::vector<Match> matchFrames(
        const Frame& f1, const Frame& f2,
        float threshold = 0.5f,
        float ratio_threshold = 0.8f);
};

} // namespace vo
