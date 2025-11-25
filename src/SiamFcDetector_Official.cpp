// Official SiamFC Implementation based on siamfc.py from original repository
#include "detection/SiamFcDetector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

#include <opencv2/imgproc.hpp>

namespace detection {

// Official SiamFC detect implementation following siamfc.py exactly
std::vector<Detection> SiamFcDetector::detect_official(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    
    if (!hasTemplate_) {
        std::cout << "*** SIAMFC OFFICIAL *** No template available!" << std::endl;
        return {};
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

    std::cout << "*** SIAMFC OFFICIAL *** Using official parameters" << std::endl;

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
    const float centerY = lastBox_.y + lastBox_.height / 2.0f;
    const float centerX = lastBox_.x + lastBox_.width / 2.0f;
    const float targetH = lastBox_.height;
    const float targetW = lastBox_.width;

    // Calculate context and search size exactly like Python init():
    // context = self.cfg.context * np.sum(self.target_sz)  
    // self.z_sz = np.sqrt(np.prod(self.target_sz + context))
    // self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz
    const float context = contextAmount_ * (targetW + targetH);  // contextAmount_ = 0.5 default
    const float zSz = std::sqrt((targetW + context) * (targetH + context));
    const float xSz = zSz * instanceSz / exemplarSz;

    std::cout << "*** SIAMFC OFFICIAL *** Center: (" << centerX << "," << centerY 
              << "), Target: [" << targetW << "," << targetH << "]" << std::endl;
    std::cout << "*** SIAMFC OFFICIAL *** zSz: " << zSz << ", xSz: " << xSz << std::endl;

    // Multi-scale search exactly like Python update():
    // x = [ops.crop_and_resize(img, self.center, self.x_sz * f, out_size=self.cfg.instance_sz, border_value=self.avg_color) for f in self.scale_factors]
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

        // Network forward pass
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
    for (size_t i = 0; i < scaleNum / 2; i++) {
        maxResponseVals[i] *= scalePenalty;
    }
    for (size_t i = scaleNum / 2 + 1; i < scaleNum; i++) {
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

    std::cout << "*** SIAMFC OFFICIAL *** Peak location: (" << peakLoc.x << "," << peakLoc.y 
              << ") in " << upscaleSz << "x" << upscaleSz << " response, max: " << maxVal << std::endl;

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

    // Update center
    const float newCenterX = centerX + dispInImageX;
    const float newCenterY = centerY + dispInImageY;

    // Update target size exactly like Python:
    // scale = (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]
    // self.target_sz *= scale
    const float scaleUpdate = (1.0f - scaleLr) * 1.0f + scaleLr * selectedScaleFactor;
    const float newTargetW = targetW * scaleUpdate;
    const float newTargetH = targetH * scaleUpdate;

    std::cout << "*** SIAMFC OFFICIAL *** Scale update: " << scaleUpdate 
              << ", New size: [" << newTargetW << "," << newTargetH << "]" << std::endl;
    std::cout << "*** SIAMFC OFFICIAL *** New center: (" << newCenterX << "," << newCenterY << ")" << std::endl;

    // Return bounding box exactly like Python:
    // box = np.array([self.center[1] + 1 - (self.target_sz[1] - 1) / 2, self.center[0] + 1 - (self.target_sz[0] - 1) / 2, self.target_sz[1], self.target_sz[0]])
    // Note: Python uses 1-indexed, we use 0-indexed, so no +1/-1 adjustments
    const float boxX = newCenterX - newTargetW / 2.0f;
    const float boxY = newCenterY - newTargetH / 2.0f;

    pipeline::BoundingBox detectionBox = {boxX, boxY, newTargetW, newTargetH};

    // Update internal state for next frame (this matches Python's self.center and self.target_sz updates)
    lastBox_ = detectionBox;

    Detection detection;
    detection.label = label_;
    detection.score = static_cast<float>(maxVal);
    detection.box = detectionBox;
    return {detection};
}

} // namespace detection