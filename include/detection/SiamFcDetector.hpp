#pragma once

#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "detection/Detection.hpp"

namespace detection {

class SiamFcDetector final : public Detector {
public:
    explicit SiamFcDetector(const std::string& modelPath,
                            std::string label = "target",
                            float contextAmount = 0.5f);

    std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override;

    // Official SiamFC implementation following siamfc.py exactly
    std::vector<Detection> detect_official(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt);

    // Update template with new tracking result
    void updateTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) override;

    void updateTrackedBox(const cv::Mat& frame, const pipeline::BoundingBox& box);
    
    // Restore original template (for rescue after drift)
    void restoreOriginalTemplate() override;

private:
    cv::Mat extractPatch(const cv::Mat& frame, const cv::Rect2f& rect, const cv::Size& outSize) const;
    cv::Rect2f makeTemplateRect(const pipeline::BoundingBox& box) const;
    cv::Rect clampRect(const cv::Rect2f& rect, const cv::Size& bounds) const;

    cv::dnn::Net net_;
    cv::Mat templateBlob_;              // Current template (may drift)
    cv::Mat originalTemplateBlob_;      // Original template from frame 1 (NEVER changes)
    pipeline::BoundingBox originalBox_; // Original box from frame 1
    pipeline::BoundingBox lastBox_;
    std::string label_;
    float contextAmount_;
    bool hasTemplate_{false};
    bool hasOriginalTemplate_{false};   // Flag to check if we saved original
};

}  // namespace detection
