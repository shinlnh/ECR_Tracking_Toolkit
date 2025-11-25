#include "detection/SiamFcDetector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

#include <opencv2/imgproc.hpp>

namespace detection {
namespace {

cv::Rect2f makeSquareAround(const pipeline::BoundingBox& box, float contextAmount) {
    const float cx = box.x + box.width * 0.5f;
    const float cy = box.y + box.height * 0.5f;
    const float w = std::max(box.width, 1.0f);
    const float h = std::max(box.height, 1.0f);
    const float context = contextAmount * (w + h);
    const float size = std::sqrt((w + context) * (h + context));
    const float half = size * 0.5f;
    return {cx - half, cy - half, size, size};
}

cv::Mat toResponseMap(const cv::Mat& blob) {
    CV_Assert(blob.dims == 4);
    const int rows = blob.size[2];
    const int cols = blob.size[3];
    cv::Mat response(rows, cols, CV_32F, const_cast<float*>(blob.ptr<float>()));
    return response.clone();
}

}  // namespace

SiamFcDetector::SiamFcDetector(const std::string& modelPath,
                               std::string label,
                               float contextAmount)
    : label_(std::move(label)), contextAmount_(contextAmount) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load SiamFC ONNX model from " + modelPath);
    }
}

std::vector<Detection> SiamFcDetector::detect(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    // Use the official SiamFC implementation
    return detect_official(frame, roi);
}

std::vector<Detection> SiamFcDetector::detect_official(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    if (!hasTemplate_) {
        std::cout << "*** SIAMFC OFFICIAL *** No template available!" << std::endl;
        return {};
    }
    
    std::cout << "*** SIAMFC DEBUG *** Template available, performing search" << std::endl;

    cv::Rect searchRect;
    if (roi) {
        searchRect = clampRect(roi->toRect(), frame.size());
    } else {
        searchRect = {0, 0, frame.cols, frame.rows};
    }

    if (searchRect.width <= 2 || searchRect.height <= 2) {
        return {};
    }

    // Expand search region to be larger for better SiamFC performance
    const float expandFactor = 2.0f; // Make search region 2x larger
    const int expandedW = static_cast<int>(searchRect.width * expandFactor);
    const int expandedH = static_cast<int>(searchRect.height * expandFactor);
    const int centerX = searchRect.x + searchRect.width / 2;
    const int centerY = searchRect.y + searchRect.height / 2;
    
    cv::Rect expandedSearchRect(
        centerX - expandedW / 2,
        centerY - expandedH / 2,
        expandedW,
        expandedH
    );
    
    // Clamp to frame boundaries
    expandedSearchRect = clampRect(cv::Rect2f(expandedSearchRect), frame.size());
    
    std::cout << "*** SIAMFC DEBUG *** Original search: " << searchRect.width << "x" << searchRect.height 
              << ", Expanded: " << expandedSearchRect.width << "x" << expandedSearchRect.height << std::endl;

    // Multi-scale search like original SiamFC
    std::vector<float> scaleFactors = {0.96f, 1.0f, 1.04f}; // 3 scales
    std::vector<cv::Mat> responses;
    std::vector<double> maxVals;
    
    for (float scaleFactor : scaleFactors) {
        const int scaledW = static_cast<int>(expandedSearchRect.width * scaleFactor);
        const int scaledH = static_cast<int>(expandedSearchRect.height * scaleFactor);
        const int scaledCenterX = expandedSearchRect.x + expandedSearchRect.width / 2;
        const int scaledCenterY = expandedSearchRect.y + expandedSearchRect.height / 2;
        
        cv::Rect scaledRect(
            scaledCenterX - scaledW / 2,
            scaledCenterY - scaledH / 2,
            scaledW, scaledH
        );
        scaledRect = clampRect(cv::Rect2f(scaledRect), frame.size());
        
        cv::Mat searchPatch = extractPatch(frame, cv::Rect2f(scaledRect), {255, 255});
        if (searchPatch.empty()) continue;

        cv::Mat searchBlob = cv::dnn::blobFromImage(searchPatch, 1.0 / 255.0, {255, 255}, cv::Scalar(), false, false);
        net_.setInput(templateBlob_, "template");
        net_.setInput(searchBlob, "search");
        cv::Mat responseBlob = net_.forward("response");
        cv::Mat response = toResponseMap(responseBlob);
        
        responses.push_back(response);
        
        double maxVal;
        cv::minMaxLoc(response, nullptr, &maxVal, nullptr, nullptr);
        maxVals.push_back(maxVal);
    }
    
    if (responses.empty()) {
        return {};
    }
    
    // Find best scale (like original SiamFC)
    int bestScaleId = std::distance(maxVals.begin(), std::max_element(maxVals.begin(), maxVals.end()));
    cv::Mat response = responses[bestScaleId];
    float selectedScale = scaleFactors[bestScaleId];
    
    std::cout << "*** SIAMFC DEBUG *** Selected scale: " << selectedScale << " (ID: " << bestScaleId << ")" << std::endl;

    std::cout << "*** SIAMFC DEBUG *** Response map size: " << response.rows << "x" << response.cols << std::endl;

    std::cout << "*** SIAMFC DEBUG *** Response map size: " << response.rows << "x" << response.cols << std::endl;

    // Apply Hann window (like original SiamFC)
    const int responseSize = response.rows;
    const float windowInfluence = 0.176f; // SiamFC default
    
    // Create Hann window
    cv::Mat hannWindow = cv::Mat::zeros(responseSize, responseSize, CV_32F);
    for (int r = 0; r < responseSize; r++) {
        for (int c = 0; c < responseSize; c++) {
            float hannR = 0.5f * (1.0f - std::cos(2.0f * CV_PI * r / (responseSize - 1)));
            float hannC = 0.5f * (1.0f - std::cos(2.0f * CV_PI * c / (responseSize - 1)));
            hannWindow.at<float>(r, c) = hannR * hannC;
        }
    }
    
    // Normalize response and apply Hann window
    cv::Scalar sumResponse = cv::sum(response);
    if (sumResponse[0] > 0) {
        response = response / sumResponse[0];
    }
    response = (1.0f - windowInfluence) * response + windowInfluence * hannWindow;

    double maxVal = 0.0;
    cv::Point maxLoc;
    cv::minMaxLoc(response, nullptr, &maxVal, nullptr, &maxLoc);

    std::cout << "*** SIAMFC DEBUG *** Max response: " << maxVal << " at (" << maxLoc.x << "," << maxLoc.y << ")" << std::endl;

    // SiamFC proper coordinate transformation
    const float totalStride = 255.0f / static_cast<float>(responseSize); // SiamFC uses 255x255 search
    
    // Calculate displacement from center of response map
    const float responseCenter = (responseSize - 1) / 2.0f;
    const float deltaX = (static_cast<float>(maxLoc.x) - responseCenter) * totalStride;
    const float deltaY = (static_cast<float>(maxLoc.y) - responseCenter) * totalStride;
    
    std::cout << "*** SIAMFC DEBUG *** Stride: " << totalStride << ", Center: " << responseCenter << std::endl;
    std::cout << "*** SIAMFC DEBUG *** Delta: (" << deltaX << "," << deltaY << ") from maxLoc(" << maxLoc.x << "," << maxLoc.y << ")" << std::endl;

    // Calculate new position (center of expanded search patch + displacement)
    const float searchCenterX = expandedSearchRect.x + expandedSearchRect.width / 2.0f;
    const float searchCenterY = expandedSearchRect.y + expandedSearchRect.height / 2.0f;
    const float newCenterX = searchCenterX + deltaX;
    const float newCenterY = searchCenterY + deltaY;

    // Scale update like original SiamFC
    const float scaleLr = 0.59f; // SiamFC learning rate for scale
    const float scaleStep = (1.0f - scaleLr) * 1.0f + scaleLr * selectedScale;
    
    // Update target size based on selected scale
    const float targetW = std::max(lastBox_.width * scaleStep, 1.0f);
    const float targetH = std::max(lastBox_.height * scaleStep, 1.0f);
    
    std::cout << "*** SIAMFC DEBUG *** Scale step: " << scaleStep << ", New size: " << targetW << "x" << targetH << std::endl;

    // Convert center coordinates to top-left corner
    const float newX = newCenterX - targetW / 2.0f;
    const float newY = newCenterY - targetH / 2.0f;

    pipeline::BoundingBox detectionBox;
    if (roi) {
        // If ROI is provided, coordinates should be relative to original ROI
        detectionBox = {newX - searchRect.x, newY - searchRect.y, targetW, targetH};
    } else {
        detectionBox = {newX, newY,
            targetW,
            targetH};
    }

    Detection detection;
    detection.label = label_;
    detection.score = static_cast<float>(maxVal);
    detection.box = detectionBox;
    return {detection};
}

void SiamFcDetector::updateTrackedBox(const cv::Mat& frame, const pipeline::BoundingBox& box) {
    std::cout << "*** SIAMFC DEBUG *** updateTrackedBox called with box: (" << box.x << "," << box.y 
              << "," << box.width << "," << box.height << ")" << std::endl;
    
    if (frame.empty() || box.width <= 0.0f || box.height <= 0.0f) {
        hasTemplate_ = false;
        templateBlob_.release();
        std::cout << "*** SIAMFC DEBUG *** Invalid input - template cleared" << std::endl;
        return;
    }

    lastBox_ = box;
    const cv::Rect2f templateRect = makeTemplateRect(box);
    
    std::cout << "*** SIAMFC DEBUG *** Template rect: (" << templateRect.x << "," << templateRect.y 
              << "," << templateRect.width << "," << templateRect.height << ")" << std::endl;
    std::cout << "*** SIAMFC DEBUG *** Original box: (" << box.x << "," << box.y 
              << "," << box.width << "," << box.height << ")" << std::endl;
              
    cv::Mat patch = extractPatch(frame, templateRect, {127, 127});
    if (patch.empty()) {
        hasTemplate_ = false;
        templateBlob_.release();
        return;
    }

    // Check template patch quality
    cv::Scalar meanVal = cv::mean(patch);
    cv::Mat stdDev;
    cv::meanStdDev(patch, cv::Scalar(), stdDev);
    std::cout << "*** SIAMFC DEBUG *** Template patch - Mean: " << meanVal[0] 
              << ", StdDev: " << stdDev.at<double>(0) << std::endl;

    templateBlob_ = cv::dnn::blobFromImage(patch, 1.0 / 255.0, {127, 127}, cv::Scalar(), false, false);
    hasTemplate_ = true;
    std::cout << "*** SIAMFC DEBUG *** Template updated successfully" << std::endl;
}

cv::Mat SiamFcDetector::extractPatch(const cv::Mat& frame, const cv::Rect2f& rect, const cv::Size& outSize) const {
    if (frame.empty() || rect.width <= 1.0f || rect.height <= 1.0f) {
        return {};
    }

    const float x1 = rect.x;
    const float y1 = rect.y;
    const float x2 = rect.x + rect.width;
    const float y2 = rect.y + rect.height;

    const int leftPad = static_cast<int>(std::max(0.0f, std::ceil(0.0f - x1)));
    const int topPad = static_cast<int>(std::max(0.0f, std::ceil(0.0f - y1)));
    const int rightPad = static_cast<int>(std::max(0.0f, std::ceil(x2 - static_cast<float>(frame.cols))));
    const int bottomPad = static_cast<int>(std::max(0.0f, std::ceil(y2 - static_cast<float>(frame.rows))));

    cv::Mat padded;
    cv::copyMakeBorder(frame, padded, topPad, bottomPad, leftPad, rightPad, cv::BORDER_REPLICATE);

    const int roiX = static_cast<int>(std::floor(x1 + static_cast<float>(leftPad)));
    const int roiY = static_cast<int>(std::floor(y1 + static_cast<float>(topPad)));
    const int roiW = static_cast<int>(std::round(rect.width));
    const int roiH = static_cast<int>(std::round(rect.height));

    if (roiW <= 0 || roiH <= 0) {
        return {};
    }

    const int maxW = padded.cols - roiX;
    const int maxH = padded.rows - roiY;
    const int clampedW = std::max(1, std::min(roiW, maxW));
    const int clampedH = std::max(1, std::min(roiH, maxH));

    cv::Mat patch = padded(cv::Rect(roiX, roiY, clampedW, clampedH)).clone();
    if (patch.size() != outSize) {
        cv::resize(patch, patch, outSize);
    }
    return patch;
}

cv::Rect2f SiamFcDetector::makeTemplateRect(const pipeline::BoundingBox& box) const {
    return makeSquareAround(box, contextAmount_);
}

cv::Rect SiamFcDetector::clampRect(const cv::Rect2f& rect, const cv::Size& bounds) const {
    const float x = std::clamp(rect.x, 0.0f, static_cast<float>(bounds.width));
    const float y = std::clamp(rect.y, 0.0f, static_cast<float>(bounds.height));
    const float maxW = std::max(0.0f, static_cast<float>(bounds.width) - x);
    const float maxH = std::max(0.0f, static_cast<float>(bounds.height) - y);
    const float w = std::clamp(rect.width, 0.0f, maxW);
    const float h = std::clamp(rect.height, 0.0f, maxH);
    return {
        static_cast<int>(std::floor(x)),
        static_cast<int>(std::floor(y)),
        std::max(1, static_cast<int>(std::round(w))),
        std::max(1, static_cast<int>(std::round(h)))};
}

}  // namespace detection
