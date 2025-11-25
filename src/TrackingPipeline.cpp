#include "pipeline/TrackingPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>

#include "tracking/Quality.hpp"

namespace pipeline {

TrackingPipeline::TrackingPipeline(config::PipelineConfig config)
    : detector_(nullptr),
      config_(std::move(config)),
      tracker_(std::make_unique<tracking::CsrtTracker>()),
      kalman_(std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman)) {}

TrackingPipeline::TrackingPipeline(detection::Detector& detector, config::PipelineConfig config)
    : TrackingPipeline(std::move(config)) {
    detector_ = &detector;
    rescue_ = std::make_unique<rescue::RescueStrategy>(
        *detector_,
        config_.rescue.roiScale,
        config_.rescue.intervalFrames,
        config_.rescue.minIoU);
}

void TrackingPipeline::initializeWithBox(
    const cv::Mat& frame,
    const BoundingBox& box,
    const std::string& label,
    float score) {
    target_ = TrackingTarget{nextTrackId_++, label, box, score};
    target_->originalBox = box;  // Save original target size for better rescue
    originalTargetBox_ = box;
    lastGoodBox_ = box;  // Initialize last good box for drift detection
    
    if (!tracker_) {
        tracker_ = std::make_unique<tracking::CsrtTracker>();
    } else {
        tracker_->reset();
    }
    tracker_->initialize(frame, box);

    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    kalman_->reset(box);

    lastSuccessFrame_ = frameIndex_;
    lastFullFrameAttempt_ = frameIndex_;
    baselineConfidence_ = -1.0f;
    lowConfidenceCount_ = 0;
}

void TrackingPipeline::clear() {
    target_.reset();
    if (tracker_) {
        tracker_->reset();
    }
    kalman_.reset();
    lastSuccessFrame_ = frameIndex_;
    lastFullFrameAttempt_ = frameIndex_;
    baselineConfidence_ = -1.0f;
    lowConfidenceCount_ = 0;
}

PipelineOutput TrackingPipeline::step(const cv::Mat& frame) {
    ++frameIndex_;
    const std::tuple<int, int> frameSize{frame.cols, frame.rows};

    if (!target_) {
        const auto detection = initializeTarget(frame);
        PipelineOutput output{};
        output.frameIndex = frameIndex_;
        output.target = target_;
        if (target_) {
            output.smoothedBox = target_->box;
            output.rawBox = target_->box;
        }
        output.detection = detection;
        output.lost = !target_.has_value();
        output.trackerType = detection ? "Detector" : "None";
        return output;
    }

    const auto [rawBox, valid] = track(frame, frameSize);
    auto smoothedBox = smooth(rawBox, valid);

    // RESCUE DISABLED - Pure CSRT tracking only
    bool finalValid = valid;
    auto mutableRaw = rawBox;
    bool usedRescue = false;
    bool usedFullFrame = false;

    if (finalValid && mutableRaw) {
        target_->update(*mutableRaw);
        lastSuccessFrame_ = frameIndex_;
        lowConfidenceCount_ = 0;
    }

    PipelineOutput output{};
    output.frameIndex = frameIndex_;
    output.target = target_;
    output.rawBox = mutableRaw;
    output.smoothedBox = smoothedBox;
    output.lost = !finalValid;
    output.usedRescue = usedRescue;
    output.usedFullFrame = usedFullFrame;
    
    // Set tracker type - Pure CSRT only
    output.trackerType = "CSRT";
    
    // Update previous frame box for next iteration
    if (mutableRaw) {
        previousFrameBox_ = *mutableRaw;
    }
    
    return output;
}

std::optional<detection::Detection> TrackingPipeline::initializeTarget(const cv::Mat& frame) {
    if (!detector_) {
        return std::nullopt;
    }

    auto detections = detector_->detect(frame);
    if (detections.empty()) {
        return std::nullopt;
    }
    const auto bestIt = std::max_element(
        detections.begin(),
        detections.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.score < rhs.score; });
    target_ = TrackingTarget{nextTrackId_++, bestIt->label, bestIt->box, bestIt->score};
    if (!tracker_) {
        tracker_ = std::make_unique<tracking::CsrtTracker>();
    }
    tracker_->initialize(frame, bestIt->box);
    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    kalman_->reset(bestIt->box);
    lastSuccessFrame_ = frameIndex_;
    lastFullFrameAttempt_ = frameIndex_;
    return *bestIt;
}

std::pair<std::optional<BoundingBox>, bool> TrackingPipeline::track(
    const cv::Mat& frame,
    std::tuple<int, int> frameSize) {
    const auto tracked = tracker_->update(frame);
    if (!tracked) {
        return {std::nullopt, false};
    }
    bool confidenceTooLow = false;
    if (target_) {
        const float confidence = computeConfidence(*tracked, target_->box, frameSize);
        if (baselineConfidence_ < 0.0f && confidence > 0.0f) {
            baselineConfidence_ = confidence;
        } else if (confidence > baselineConfidence_) {
            baselineConfidence_ = confidence;
        }

        if (baselineConfidence_ > 0.0f &&
            confidence < baselineConfidence_ * config_.rescue.apceThresholdFactor) {
            ++lowConfidenceCount_;
        } else {
            lowConfidenceCount_ = 0;
        }

        if (baselineConfidence_ > 0.0f &&
            lowConfidenceCount_ >= config_.rescue.apceConsecutiveFrames) {
            confidenceTooLow = true;
        }
    }

    const bool qualityAcceptable = tracking::isQualityAcceptable(
        *tracked,
        target_ ? std::optional<BoundingBox>(target_->box) : std::nullopt,
        frameSize,
        config_.quality.minArea,
        config_.quality.minAspectRatio,
        config_.quality.maxAspectRatio,
        config_.quality.minIoU);
    const bool valid = qualityAcceptable && !confidenceTooLow;
    return {tracked, valid};
}

std::optional<BoundingBox> TrackingPipeline::smooth(
    const std::optional<BoundingBox>& box,
    bool validMeasurement) {
    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    return kalman_->smooth(box, validMeasurement);
}

std::optional<BoundingBox> TrackingPipeline::attemptRescue(
    const cv::Mat& frame,
    std::tuple<int, int> frameSize,
    const std::optional<BoundingBox>& rawBox,
    bool& usedFullFrameRescue,
    bool forceRescue) {
    if (!target_) {
        return std::nullopt;
    }

    const int framesSinceSuccess = (lastSuccessFrame_ >= 0) ? (frameIndex_ - lastSuccessFrame_) : frameIndex_;

    bool roiAttemptFailed = false;
    // Always try SiamFC rescue (now searches full frame)
    if (rescue_ && (forceRescue || rescue_->shouldTrigger(frameIndex_, lastSuccessFrame_))) {
        const auto recovered = rescue_->recover(
            frame,
            *target_,
            frameIndex_,
            frameSize,
            rawBox ? std::optional<BoundingBox>(*rawBox) : std::optional<BoundingBox>(target_->box));
        if (recovered) {
            reinitializeTrackers(frame, *recovered);
            // Don't update lastSuccessFrame_ here - let the caller decide based on quality
            lastFullFrameAttempt_ = frameIndex_;
            return recovered;
        }
        roiAttemptFailed = true;
    }

    const bool shouldRunFullFrameRecovery =
        detector_ &&
        (lastFullFrameAttempt_ != frameIndex_) &&
        ((config_.rescue.fullFrameInterval > 0 && framesSinceSuccess >= config_.rescue.fullFrameInterval) ||
         roiAttemptFailed);

    if (shouldRunFullFrameRecovery) {
        lastFullFrameAttempt_ = frameIndex_;
        if (auto detection = runFullFrameRecovery(frame)) {
            target_->update(detection->box, detection->score);
            reinitializeTrackers(frame, detection->box);
            lastSuccessFrame_ = frameIndex_;
            usedFullFrameRescue = true;
            return detection->box;
        }
    }

    if (config_.rescue.maxLostFrames > 0 && framesSinceSuccess >= config_.rescue.maxLostFrames) {
        target_.reset();
        tracker_.reset();
        kalman_.reset();
        lastSuccessFrame_ = frameIndex_;
        lastFullFrameAttempt_ = frameIndex_;
        baselineConfidence_ = -1.0f;
        lowConfidenceCount_ = 0;
    }

    return std::nullopt;
}

void TrackingPipeline::reinitializeTrackers(const cv::Mat& frame, const BoundingBox& box) {
    if (!tracker_) {
        tracker_ = std::make_unique<tracking::CsrtTracker>();
    } else {
        tracker_->reset();
    }
    tracker_->initialize(frame, box);

    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    kalman_->reset(box);
    baselineConfidence_ = -1.0f;
    lowConfidenceCount_ = 0;
}

std::optional<detection::Detection> TrackingPipeline::runFullFrameRecovery(const cv::Mat& frame) {
    if (!detector_) {
        return std::nullopt;
    }

    auto detections = detector_->detect(frame);
    const detection::Detection* bestMatch = nullptr;
    for (const auto& detection : detections) {
        if (detection.label != target_->label) {
            continue;
        }
        if (!bestMatch || detection.score > bestMatch->score) {
            bestMatch = &detection;
        }
    }
    if (!bestMatch) {
        return std::nullopt;
    }
    return *bestMatch;
}

float TrackingPipeline::computeConfidence(
    const BoundingBox& current,
    const BoundingBox& previous,
    std::tuple<int, int> frameSize) const {
    const float iou = current.iou(previous);
    const auto currentCenter = current.center();
    const auto previousCenter = previous.center();
    const float dx = currentCenter[0] - previousCenter[0];
    const float dy = currentCenter[1] - previousCenter[1];
    const float frameWidth = static_cast<float>(std::get<0>(frameSize));
    const float frameHeight = static_cast<float>(std::get<1>(frameSize));
    const float diag = std::hypot(frameWidth, frameHeight);

    float centerScore = 1.0f;
    if (diag > 0.0f) {
        const float dist = std::hypot(dx, dy);
        centerScore = 1.0f - std::clamp(dist / diag, 0.0f, 1.0f);
    }

    float areaScore = 1.0f;
    const float currentArea = current.area();
    const float previousArea = previous.area();
    if (currentArea > 0.0f && previousArea > 0.0f) {
        areaScore = std::min(currentArea, previousArea) / std::max(currentArea, previousArea);
    }

    const float confidence = 0.5f * iou + 0.3f * centerScore + 0.2f * areaScore;
    return std::clamp(confidence, 0.0f, 1.0f);
}

}  // namespace pipeline


