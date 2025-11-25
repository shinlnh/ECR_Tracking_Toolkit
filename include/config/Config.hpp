#pragma once

#include "smoothing/BoxKalmanFilter.hpp"

namespace config {

struct QualityConfig {
    float minArea{400.0f};
    float minAspectRatio{0.1f};
    float maxAspectRatio{6.0f};
    float minIoU{0.05f};
};

struct RescueConfig {
    int intervalFrames{12};
    float roiScale{8.0f};  // Increased from 4.0 to 8.0 for drift-based rescue (cover 68px drift)
    float minIoU{0.0f};  // Removed filter - accept ANY detection during rescue
    int fullFrameInterval{45};
    int maxLostFrames{180};
    float apceThresholdFactor{0.15f};
    int apceConsecutiveFrames{3};
};

struct PipelineConfig {
    QualityConfig quality{};
    RescueConfig rescue{};
    smoothing::KalmanParams kalman{};
};

}  // namespace config

