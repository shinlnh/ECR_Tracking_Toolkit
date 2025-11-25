#pragma once

#include <optional>

#include "detection/Detection.hpp"
#include "pipeline/Types.hpp"

namespace pipeline {

struct PipelineOutput {
    int frameIndex{0};
    std::optional<TrackingTarget> target;
    std::optional<BoundingBox> smoothedBox;
    std::optional<BoundingBox> rawBox;
    std::optional<detection::Detection> detection;
    bool lost{true};
    bool usedRescue{false};
    bool usedFullFrame{false};
    std::optional<std::string> rescueSource;
    std::string trackerType{"CSRT"};  // Current active tracker: "CSRT", "SiamFC", or "Detector"
};

}  // namespace pipeline
