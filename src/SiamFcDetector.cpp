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
    std::cout << "*** SIAMFC *** detect() called, hasTemplate=" << hasTemplate_ 
              << ", roi=" << (roi.has_value() ? "YES" : "NO") << std::endl << std::flush;
    // Use the official SiamFC implementation
    return detect_official(frame, roi);
}

// Official SiamFC Implementation - Following siamfc.py from official repository exactly
std::vector<Detection> SiamFcDetector::detect_official(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    
    if (!hasTemplate_) {
        std::cout << "*** SIAMFC OFFICIAL *** No template available!" << std::endl << std::flush;
        return {};
    }

    // When ROI is provided (rescue mode), use ROI center as search center
    // This allows SiamFC to search in the correct region when CSRT has drifted
    bool useRoiCenter = roi.has_value();
    float searchCenterX, searchCenterY;
    
    if (useRoiCenter) {
        searchCenterX = roi->x + roi->width / 2.0f;
        searchCenterY = roi->y + roi->height / 2.0f;
        std::cout << "*** SIAMFC OFFICIAL *** RESCUE MODE: Searching in ROI center (" 
                  << searchCenterX << "," << searchCenterY << "), size: " 
                  << roi->width << "x" << roi->height << std::endl;
    } else {
        // Normal tracking mode: use lastBox_ center
        searchCenterX = lastBox_.x + lastBox_.width / 2.0f;
        searchCenterY = lastBox_.y + lastBox_.height / 2.0f;
    }

    // SiamFC Official Parameters (from siamfc.py parse_args)
    const int exemplarSz = 127;       // cfg.exemplar_sz  
    const int instanceSz = 255;       // cfg.instance_sz
    const int scaleNum = 3;           // cfg.scale_num
    const float scaleStep = 1.0375f;  // cfg.scale_step  
    const float scaleLr = 0.59f;      // cfg.scale_lr
    const float scalePenalty = 0.9745f; // cfg.scale_penalty
    const float windowInfluence = 0.176f; // cfg.window_influence
    const int responseSz = 17;        // cfg.response_sz
    const int responseUp = 16;        // cfg.response_up  
    const float totalStride = 8.0f;   // cfg.total_stride

    std::cout << "*** SIAMFC OFFICIAL *** Using parameters from siamfc.py" << std::endl;

    // Create scale factors exactly like Python code:
    // self.scale_factors = self.cfg.scale_step ** np.linspace(-(self.cfg.scale_num // 2), self.cfg.scale_num // 2, self.cfg.scale_num)
    std::vector<float> scaleFactors;
    for (int i = 0; i < scaleNum; i++) {
        int offset = i - (scaleNum / 2);  // -1, 0, 1 for scaleNum=3
        scaleFactors.push_back(std::pow(scaleStep, offset));
    }
    
    std::cout << "*** SIAMFC OFFICIAL *** Scale factors: ";
    for (float sf : scaleFactors) std::cout << sf << " ";
    std::cout << std::endl;

    // Current object center and size (following Python convention: center=[y,x], target_sz=[h,w])
    // Use ROI center when in rescue mode, otherwise use lastBox_ center
    const float centerY = searchCenterY;  // Set from ROI or lastBox_ above
    const float centerX = searchCenterX;
    const float targetH = lastBox_.height;  // Keep target size from template
    const float targetW = lastBox_.width;
    
    std::cout << "*** SIAMFC OFFICIAL *** Template target size: " << targetW << "x" << targetH << std::endl;

    // Calculate context and search size exactly like Python init():
    // context = self.cfg.context * np.sum(self.target_sz)  
    // self.z_sz = np.sqrt(np.prod(self.target_sz + context))
    // self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz
    const float context = contextAmount_ * (targetW + targetH);  // contextAmount_ = 0.5 default
    const float zSz = std::sqrt((targetW + context) * (targetH + context));
    const float xSz = zSz * instanceSz / exemplarSz;

    std::cout << "*** SIAMFC OFFICIAL *** Center: (" << centerX << "," << centerY 
              << "), Target: [" << targetW << "," << targetH << "]" << std::endl;
    std::cout << "*** SIAMFC OFFICIAL *** Context: " << context << ", zSz: " << zSz << ", xSz: " << xSz << std::endl;

    // Multi-scale search exactly like Python update():
    // x = [ops.crop_and_resize(img, self.center, self.x_sz * f, out_size=self.cfg.instance_sz, ...) for f in self.scale_factors]
    std::vector<cv::Mat> searchResponses;
    std::vector<double> maxResponseVals;

    for (size_t scaleId = 0; scaleId < scaleFactors.size(); scaleId++) {
        const float scaleFactor = scaleFactors[scaleId];
        const float searchSz = xSz * scaleFactor;
        
        // crop_and_resize equivalent - extract search patch centered at object
        cv::Rect2f searchRect(
            centerX - searchSz / 2.0f,
            centerY - searchSz / 2.0f,
            searchSz, searchSz
        );
        
        cv::Mat searchPatch = extractPatch(frame, searchRect, {instanceSz, instanceSz});
        if (searchPatch.empty()) continue;

        // Network forward pass: responses = self.net(z, x)
        cv::Mat searchBlob = cv::dnn::blobFromImage(searchPatch, 1.0 / 255.0, {instanceSz, instanceSz}, cv::Scalar(), false, false);
        net_.setInput(templateBlob_, "template");
        net_.setInput(searchBlob, "search");
        cv::Mat responseBlob = net_.forward("response");
        cv::Mat response = toResponseMap(responseBlob);
        
        searchResponses.push_back(response.clone());
        
        double maxVal;
        cv::minMaxLoc(response, nullptr, &maxVal, nullptr, nullptr);
        maxResponseVals.push_back(maxVal);
    }

    if (searchResponses.empty()) {
        return {};
    }

    // Penalize scale changes exactly like Python:
    // responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
    // responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
    for (size_t i = 0; i < static_cast<size_t>(scaleNum / 2); i++) {
        maxResponseVals[i] *= scalePenalty;
    }
    for (size_t i = static_cast<size_t>(scaleNum / 2 + 1); i < scaleNum; i++) {
        maxResponseVals[i] *= scalePenalty;  
    }

    // Peak scale exactly like Python: scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
    int peakScaleId = std::distance(maxResponseVals.begin(), 
                                   std::max_element(maxResponseVals.begin(), maxResponseVals.end()));
    cv::Mat response = searchResponses[peakScaleId];
    float selectedScaleFactor = scaleFactors[peakScaleId];
    
    std::cout << "*** SIAMFC OFFICIAL *** Peak scale ID: " << peakScaleId 
              << ", factor: " << selectedScaleFactor << std::endl;

    // Peak location processing exactly like Python:
    // response = responses[scale_id]
    // response -= response.min() 
    // response /= response.sum() + 1e-16
    double minVal, maxVal;
    cv::minMaxLoc(response, &minVal, &maxVal, nullptr, nullptr);
    response -= minVal;
    cv::Scalar sumVal = cv::sum(response);
    response = response / (sumVal[0] + 1e-16f);

    // Create Hann window exactly like Python init():
    // self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  
    // self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
    // self.hann_window /= self.hann_window.sum()
    const int upscaleSz = responseUp * responseSz;  // 16 * 17 = 272
    
    cv::Mat hannWindow = cv::Mat::zeros(upscaleSz, upscaleSz, CV_32F);
    for (int r = 0; r < upscaleSz; r++) {
        for (int c = 0; c < upscaleSz; c++) {
            float hannR = 0.5f * (1.0f - std::cos(2.0f * CV_PI * r / (upscaleSz - 1)));
            float hannC = 0.5f * (1.0f - std::cos(2.0f * CV_PI * c / (upscaleSz - 1)));
            hannWindow.at<float>(r, c) = hannR * hannC;
        }
    }
    cv::Scalar hannSum = cv::sum(hannWindow);
    hannWindow = hannWindow / hannSum[0];

    // Upsample response exactly like Python:  
    // responses = np.stack([cv2.resize(u, (self.upscale_sz, self.upscale_sz), interpolation=cv2.INTER_CUBIC) for u in responses])
    cv::Mat upsampledResponse;
    cv::resize(response, upsampledResponse, cv::Size(upscaleSz, upscaleSz), 0, 0, cv::INTER_CUBIC);

    // Apply Hann window exactly like Python:
    // response = (1 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hann_window
    upsampledResponse = (1.0f - windowInfluence) * upsampledResponse + windowInfluence * hannWindow;

    // Find peak location: loc = np.unravel_index(response.argmax(), response.shape)
    cv::Point peakLoc;
    cv::minMaxLoc(upsampledResponse, nullptr, &maxVal, nullptr, &peakLoc);

    std::cout << "*** SIAMFC OFFICIAL *** Peak at (" << peakLoc.x << "," << peakLoc.y 
              << ") in " << upscaleSz << "x" << upscaleSz << " response, confidence: " << maxVal << std::endl;

    // Locate target center exactly like Python:
    // disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
    // disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up  
    // disp_in_image = disp_in_instance * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
    // self.center += disp_in_image
    const float dispInResponseX = static_cast<float>(peakLoc.x) - (upscaleSz - 1) / 2.0f;
    const float dispInResponseY = static_cast<float>(peakLoc.y) - (upscaleSz - 1) / 2.0f;
    
    const float dispInInstanceX = dispInResponseX * totalStride / responseUp;
    const float dispInInstanceY = dispInResponseY * totalStride / responseUp;
    
    const float currentXSz = xSz * selectedScaleFactor;
    const float dispInImageX = dispInInstanceX * currentXSz / instanceSz;
    const float dispInImageY = dispInInstanceY * currentXSz / instanceSz;

    std::cout << "*** SIAMFC OFFICIAL *** Displacements - Response: (" << dispInResponseX << "," << dispInResponseY 
              << "), Instance: (" << dispInInstanceX << "," << dispInInstanceY 
              << "), Image: (" << dispInImageX << "," << dispInImageY << ")" << std::endl;

    // Update center
    const float newCenterX = centerX + dispInImageX;
    const float newCenterY = centerY + dispInImageY;

    // Update target size exactly like Python:
    // scale = (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]
    // self.target_sz *= scale
    // self.z_sz *= scale
    // self.x_sz *= scale
    const float scaleUpdate = (1.0f - scaleLr) * 1.0f + scaleLr * selectedScaleFactor;
    const float newTargetW = targetW * scaleUpdate;
    const float newTargetH = targetH * scaleUpdate;

    std::cout << "*** SIAMFC OFFICIAL *** Scale update: " << scaleUpdate 
              << ", New size: [" << newTargetW << "," << newTargetH << "]" << std::endl;
    std::cout << "*** SIAMFC OFFICIAL *** New center: (" << newCenterX << "," << newCenterY << ")" << std::endl;

    // Return bounding box exactly like Python:
    // box = np.array([
    //     self.center[1] + 1 - (self.target_sz[1] - 1) / 2,  # x = center_x - width/2  
    //     self.center[0] + 1 - (self.target_sz[0] - 1) / 2,  # y = center_y - height/2
    //     self.target_sz[1],                                   # width
    //     self.target_sz[0]                                    # height
    // ])
    // Note: Python uses 1-indexed, we use 0-indexed, so adjust accordingly
    float boxX = newCenterX - newTargetW / 2.0f;
    float boxY = newCenterY - newTargetH / 2.0f;

    // When ROI is provided (rescue mode), return box in ROI coordinates
    // RescueStrategy will translate it back to global coordinates
    if (useRoiCenter && roi.has_value()) {
        boxX -= roi->x;
        boxY -= roi->y;
        std::cout << "*** SIAMFC OFFICIAL *** Converted to ROI coordinates: (" << boxX << "," << boxY << ")" << std::endl;
    }

    pipeline::BoundingBox detectionBox = {boxX, boxY, newTargetW, newTargetH};

    // Update internal state for next frame (matches Python's self.center and self.target_sz updates)
    // Always use global coordinates for lastBox_
    lastBox_ = {newCenterX - newTargetW / 2.0f, newCenterY - newTargetH / 2.0f, newTargetW, newTargetH};

    Detection detection;
    detection.label = label_;
    detection.score = static_cast<float>(maxVal);
    detection.box = detectionBox;
    return {detection};
}

void SiamFcDetector::updateTrackedBox(const cv::Mat& frame, const pipeline::BoundingBox& box) {
    std::cout << "*** SIAMFC OFFICIAL *** updateTrackedBox called with box: (" << box.x << "," << box.y 
              << "," << box.width << "," << box.height << ")" << std::endl;
    
    if (frame.empty() || box.width <= 0.0f || box.height <= 0.0f) {
        hasTemplate_ = false;
        templateBlob_.release();
        std::cout << "*** SIAMFC OFFICIAL *** Invalid input - template cleared" << std::endl;
        return;
    }

    lastBox_ = box;
    
    // NEW STRATEGY: Extract ONLY the object bbox (no context/background)
    // This creates a pure object template for matching during rescue
    cv::Rect2f pureObjectRect(box.x, box.y, box.width, box.height);
    
    std::cout << "*** SIAMFC PURE OBJECT *** Template rect: (" << pureObjectRect.x << "," << pureObjectRect.y 
              << "," << pureObjectRect.width << "," << pureObjectRect.height << ")" << std::endl;
    std::cout << "*** SIAMFC PURE OBJECT *** Original box: (" << box.x << "," << box.y 
              << "," << box.width << "," << box.height << ")" << std::endl;
              
    cv::Mat patch = extractPatch(frame, pureObjectRect, {127, 127});
    if (patch.empty()) {
        hasTemplate_ = false;
        templateBlob_.release();
        return;
    }

    // Check template patch quality
    cv::Scalar meanVal = cv::mean(patch);
    cv::Mat stdDev;
    cv::meanStdDev(patch, cv::Scalar(), stdDev);
    std::cout << "*** SIAMFC PURE OBJECT *** Template patch - Mean: " << meanVal[0] 
              << ", StdDev: " << stdDev.at<double>(0) << std::endl;

    templateBlob_ = cv::dnn::blobFromImage(patch, 1.0 / 255.0, {127, 127}, cv::Scalar(), false, false);
    hasTemplate_ = true;
    
    // SAVE ORIGINAL PURE OBJECT TEMPLATE on first call (frame 1 - ground truth)
    // This template contains ONLY the object, no background context
    // Will be used during rescue to find the original object appearance
    if (!hasOriginalTemplate_) {
        originalTemplateBlob_ = templateBlob_.clone();
        originalBox_ = box;
        hasOriginalTemplate_ = true;
        std::cout << "*** SIAMFC ORIGINAL TEMPLATE SAVED *** Frame 1 pure object template stored for rescue!" << std::endl;
    }
    
    std::cout << "*** SIAMFC PURE OBJECT *** Template updated successfully" << std::endl;
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

void SiamFcDetector::updateTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) {
    // Delegate to updateTrackedBox - they do the same thing
    updateTrackedBox(frame, box);
}

void SiamFcDetector::restoreOriginalTemplate() {
    if (!hasOriginalTemplate_) {
        std::cout << "*** SIAMFC RESTORE *** No original template saved!" << std::endl;
        return;
    }
    
    // Restore original template from frame 1
    templateBlob_ = originalTemplateBlob_.clone();
    lastBox_ = originalBox_;
    hasTemplate_ = true;
    
    std::cout << "*** SIAMFC RESTORE *** Original template restored! Box: (" 
              << originalBox_.x << "," << originalBox_.y << "," 
              << originalBox_.width << "," << originalBox_.height << ")" << std::endl;
}

}  // namespace detection