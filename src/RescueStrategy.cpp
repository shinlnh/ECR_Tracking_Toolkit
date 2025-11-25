#include "rescue/RescueStrategy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>

namespace rescue {

RescueStrategy::RescueStrategy(
    detection::Detector& detector,
    float roiScale,
    int triggerInterval,
    float minIoU)
    : detector_(detector),
      roiScale_(roiScale),  // Increased from 2.0 to 4.0 for larger search region
      triggerInterval_(triggerInterval),
      minIoU_(minIoU) {}

bool RescueStrategy::shouldTrigger(int frameIndex, int lastSuccessFrame) {
    const int framesSinceLast = frameIndex - lastSuccessFrame;
    
    // More aggressive rescue when tracking has been lost for long time
    int effectiveInterval = triggerInterval_;
    if (framesSinceLast > 30) {
        effectiveInterval = triggerInterval_ / 2;  // Trigger twice as often
    }
    
    if (framesSinceLast < effectiveInterval) {
        return false;
    }
    if (lastTriggerFrame_ >= 0 && frameIndex - lastTriggerFrame_ < effectiveInterval) {
        return false;
    }
    lastTriggerFrame_ = frameIndex;
    return true;
}

void RescueStrategy::resetTrigger() {
    lastTriggerFrame_ = -1;
}

bool RescueStrategy::hasDrifted(const pipeline::BoundingBox& current, const pipeline::BoundingBox& reference, float maxDriftPixels) const {
    // Calculate center points
    const float currentCenterX = current.x + current.width / 2.0f;
    const float currentCenterY = current.y + current.height / 2.0f;
    const float refCenterX = reference.x + reference.width / 2.0f;
    const float refCenterY = reference.y + reference.height / 2.0f;
    
    // Calculate Euclidean distance
    const float dx = currentCenterX - refCenterX;
    const float dy = currentCenterY - refCenterY;
    const float distance = std::sqrt(dx * dx + dy * dy);
    
    return distance > maxDriftPixels;
}

std::optional<pipeline::BoundingBox> RescueStrategy::recover(
    const cv::Mat& frame,
    pipeline::TrackingTarget& target,
    int frameIndex,
    std::tuple<int, int> frameSize,
    const std::optional<pipeline::BoundingBox>& lastKnownBox) {
    static_cast<void>(frameIndex);
    const pipeline::BoundingBox reference = lastKnownBox.value_or(target.box);
    const auto [frameWidth, frameHeight] = frameSize;
    pipeline::BoundingBox roi = reference.scale(roiScale_)
                                       .clamp(static_cast<float>(frameWidth), static_cast<float>(frameHeight));

    // USE FRESH TEMPLATE for rescue detection
    // Template is continuously updated when IoU > 0.8, so it should be recent
    // NO NEED to restore original template - use current template (fresh!)
    // detector_.restoreOriginalTemplate();  // REMOVED - use fresh template instead!
    std::cout << "*** RESCUE *** Using FRESH template (last updated when IoU > 0.8)!" << std::endl;
    std::cout << "*** RESCUE *** Searching FULL FRAME (no ROI) for object!" << std::endl;

    // ALWAYS search full frame with original template (no ROI restriction)
    // This allows SiamFC to find the object anywhere in the frame, even when CSRT has drifted far
    auto detections = detector_.detect(frame, std::nullopt);  // No ROI = full frame search
    std::cout << "*** RESCUE DEBUG *** SiamFC detected " << detections.size() << " objects in full frame" << std::endl;
    
    // When searching full frame, ROI is entire frame for coordinate translation
    pipeline::BoundingBox fullFrameRoi = {0.0f, 0.0f, static_cast<float>(frameWidth), static_cast<float>(frameHeight)};
    auto candidates = filterCandidates(detections, target.label, reference, fullFrameRoi);
    std::cout << "*** RESCUE DEBUG *** After filtering: " << candidates.size() << " candidates (minIoU=" << minIoU_ << ")" << std::endl;
    
    if (candidates.empty()) {
        std::cout << "*** RESCUE DEBUG *** No valid candidates found!" << std::endl;
        return std::nullopt;
    }

    // Use combined score: 70% detection confidence + 30% IoU overlap
    const auto best = std::max_element(
        candidates.begin(),
        candidates.end(),
        [](const auto& lhs, const auto& rhs) {
            const float lhsScore = 0.7f * lhs.first.score + 0.3f * lhs.second;
            const float rhsScore = 0.7f * rhs.first.score + 0.3f * rhs.second;
            return lhsScore < rhsScore;
        });

    // When searching full frame, detection box is already in global coordinates
    pipeline::BoundingBox translated = translateBox(best->first.box, fullFrameRoi);
    
    // Use target's original size to guide rescue box size - keep SiamFC position but use proper size
    const float origWidth = target.originalBox ? target.originalBox->width : reference.width;
    const float origHeight = target.originalBox ? target.originalBox->height : reference.height;
    const float detectedWidth = translated.width;
    const float detectedHeight = translated.height;
    
    // If SiamFC detected box is significantly smaller than original target, expand it
    if (detectedWidth < origWidth * 0.7f) {  // If detected width < 70% of original
        const float centerX = translated.x + translated.width / 2.0f;
        translated.width = origWidth * 0.9f;  // Use 90% of original width
        translated.x = centerX - translated.width / 2.0f;
    }
    
    if (detectedHeight < origHeight * 0.7f) {  // If detected height < 70% of original  
        const float centerY = translated.y + translated.height / 2.0f;
        translated.height = origHeight * 0.9f;  // Use 90% of original height
        translated.y = centerY - translated.height / 2.0f;
    }
    
    // Clamp to frame boundaries
    translated = translated.clamp(static_cast<float>(frameWidth), static_cast<float>(frameHeight));
    
    std::cout << "*** RESCUE DEBUG *** Reference box (last known): (" << reference.x << "," << reference.y 
              << "," << reference.width << "," << reference.height << ")" << std::endl;
    std::cout << "*** RESCUE DEBUG *** Original target size: (" << origWidth << "x" << origHeight << ")" << std::endl;
    std::cout << "*** RESCUE DEBUG *** SiamFC detected at: (" << best->first.box.x << "," << best->first.box.y 
              << "," << best->first.box.width << "," << best->first.box.height 
              << "), confidence: " << best->first.score << std::endl;
    std::cout << "*** RESCUE DEBUG *** Final rescue box: (" << translated.x << "," << translated.y 
              << "," << translated.width << "," << translated.height << ")" << std::endl;
    std::cout << "*** RESCUE DEBUG *** Search region: FULL FRAME (" << fullFrameRoi.width << "x" << fullFrameRoi.height << ")" << std::endl;
    
    target.update(translated, best->first.score);
    return translated;
}

std::vector<std::pair<detection::Detection, float>> RescueStrategy::filterCandidates(
    const std::vector<detection::Detection>& detections,
    const std::string& label,
    const pipeline::BoundingBox& reference,
    const pipeline::BoundingBox& roi) const {
    std::vector<std::pair<detection::Detection, float>> filtered;
    filtered.reserve(detections.size());
    for (const auto& detection : detections) {
        std::cout << "*** RESCUE DEBUG *** Detection: label=" << detection.label << ", score=" << detection.score 
                  << ", box=(" << detection.box.x << "," << detection.box.y << "," << detection.box.width << "," << detection.box.height << ")" << std::endl;
        
        if (detection.label != label) {
            std::cout << "*** RESCUE DEBUG *** Skipped: wrong label (expected=" << label << ")" << std::endl;
            continue;
        }
        const pipeline::BoundingBox globalBox = translateBox(detection.box, roi);
        const float overlap = globalBox.iou(reference);
        std::cout << "*** RESCUE DEBUG *** Overlap with reference: " << overlap << " (minIoU=" << minIoU_ << ")" << std::endl;
        
        if (overlap < minIoU_) {
            std::cout << "*** RESCUE DEBUG *** Skipped: overlap too low" << std::endl;
            continue;
        }
        detection::Detection updatedDetection = detection;
        updatedDetection.box = detection.box;  // keep relative box for translation step
        filtered.emplace_back(updatedDetection, overlap);
        std::cout << "*** RESCUE DEBUG *** Accepted candidate!" << std::endl;
    }
    return filtered;
}

pipeline::BoundingBox RescueStrategy::translateBox(
    const pipeline::BoundingBox& box,
    const pipeline::BoundingBox& roi) {
    return {box.x + roi.x, box.y + roi.y, box.width, box.height};
}

}  // namespace rescue
