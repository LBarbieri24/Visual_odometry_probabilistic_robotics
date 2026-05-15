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
    static std::vector<Match> matchFrames(const Frame& f1, const Frame& f2, float threshold = 0.5f);
};

} // namespace vo
