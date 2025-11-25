#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pipeline/Types.hpp"

namespace detection {

struct Detection {
    std::string label;
    float score{0.0f};
    pipeline::BoundingBox box;
};

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) = 0;
    
    // Update template/model with new tracking result (optional, for adaptive detectors)
    virtual void updateTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) {
        // Default: no-op (most detectors don't need template update)
    }

    // Restore original template from frame 1 (for template-based trackers like SiamFC)
    virtual void restoreOriginalTemplate() {
        // Default: no-op (only SiamFC uses this)
    }
};

}  // namespace detection
