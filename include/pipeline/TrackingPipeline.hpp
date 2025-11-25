#pragma once

#include <memory>
#include <optional>
#include <string>
#include <iostream>

#include <opencv2/core.hpp>

#include "config/Config.hpp"
#include "detection/Detection.hpp"
#include "pipeline/PipelineOutput.hpp"
#include "rescue/RescueStrategy.hpp"
#include "smoothing/BoxKalmanFilter.hpp"
#include "tracking/CsrtTracker.hpp"

namespace pipeline {

class TrackingPipeline {
public:
    explicit TrackingPipeline(config::PipelineConfig config = {});
    TrackingPipeline(detection::Detector& detector, config::PipelineConfig config = {});

    void initializeWithBox(
        const cv::Mat& frame,
        const BoundingBox& box,
        const std::string& label = "target",
        float score = 1.0f);

    bool hasTarget() const { return target_.has_value(); }
    void clear();

    PipelineOutput step(const cv::Mat& frame);

private:
    std::optional<detection::Detection> initializeTarget(const cv::Mat& frame);
    std::pair<std::optional<BoundingBox>, bool> track(const cv::Mat& frame, std::tuple<int, int> frameSize);
    std::optional<BoundingBox> smooth(const std::optional<BoundingBox>& box, bool validMeasurement);
    std::optional<BoundingBox> attemptRescue(
        const cv::Mat& frame,
        std::tuple<int, int> frameSize,
        const std::optional<BoundingBox>& rawBox,
        bool& usedFullFrameRescue,
        bool forceRescue = false);
    void reinitializeTrackers(const cv::Mat& frame, const BoundingBox& box);
    std::optional<detection::Detection> runFullFrameRecovery(const cv::Mat& frame);
    float computeConfidence(const BoundingBox& current, const BoundingBox& previous, std::tuple<int, int> frameSize) const;

    detection::Detector* detector_{nullptr};
    config::PipelineConfig config_;
    std::unique_ptr<tracking::CsrtTracker> tracker_;
    std::unique_ptr<smoothing::BoxKalmanFilter> kalman_;
    std::unique_ptr<rescue::RescueStrategy> rescue_;
    std::optional<TrackingTarget> target_;
    int frameIndex_{0};
    int lastSuccessFrame_{-1};
    int nextTrackId_{1};
    int lastFullFrameAttempt_{-1};
    float baselineConfidence_{-1.0f};
    int lowConfidenceCount_{0};
    std::optional<BoundingBox> previousFrameBox_;  // For IoU-based rescue triggering
    std::optional<BoundingBox> lastGoodBox_;  // Last known good box for drift detection
    std::optional<BoundingBox> originalTargetBox_;  // Store original target size for better rescue
    
public:
    // FOR TESTING ONLY - set ground truth IoU to trigger rescue
    void setCurrentGroundTruthIoU(float iou) { currentGroundTruthIoU_ = iou; }
    
    // FOR TESTING ONLY - set correct original target size from ground truth
    void setOriginalTargetSize(const BoundingBox& gtBox) {
        std::cout << "*** DEBUG *** Setting original target size to: " << gtBox.width << "x" << gtBox.height << std::endl;
        originalTargetBox_ = gtBox;
        if (target_) {
            target_->originalBox = gtBox;
            std::cout << "*** DEBUG *** Updated target originalBox" << std::endl;
        } else {
            std::cout << "*** DEBUG *** No target yet, only updated pipeline originalTargetBox_" << std::endl;
        }
    }
    
    // Get last tracking result for ground truth IoU calculation
    std::optional<BoundingBox> getLastTrackingBox() const { return previousFrameBox_; }
    
private:
    float currentGroundTruthIoU_{1.0f};  // For testing with ground truth
    int lastGroundTruthRescueFrame_{-10};  // Track last frame when ground truth rescue was triggered
};

}  // namespace pipeline

