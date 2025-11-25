#include "SiamFCONNXTracker.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace siamfc {

SiamFCONNXTracker::SiamFCONNXTracker(const std::string& model_path)
    : initialized_(false)
{
    // Load ONNX model using OpenCV DNN
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        if (net_.empty()) {
            throw std::runtime_error("Failed to load ONNX model");
        }
        
        // Use CPU backend (can be changed to CUDA if available)
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::cout << "[OK] ONNX model loaded: " << model_path << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Failed to load ONNX model: " << e.what() << std::endl;
        throw;
    }
    
    // Initialize scale factors
    scale_factors_.resize(cfg_.scale_num);
    for (int i = 0; i < cfg_.scale_num; ++i) {
        int offset = i - cfg_.scale_num / 2;
        scale_factors_[i] = std::pow(cfg_.scale_step, offset);
    }
    
    // Create Hann window
    upscale_sz_ = cfg_.response_up * cfg_.response_sz;
    hann_window_ = createHannWindow(upscale_sz_);
}

void SiamFCONNXTracker::init(const cv::Mat& image, const cv::Rect2f& bbox)
{
    // Convert box to 0-indexed and center based [y, x, h, w]
    // Input bbox is [x, y, w, h] (1-indexed, left-top based)
    center_.x = bbox.y - 1.0f + (bbox.height - 1.0f) / 2.0f;  // y
    center_.y = bbox.x - 1.0f + (bbox.width - 1.0f) / 2.0f;   // x
    target_sz_.x = bbox.height;  // h
    target_sz_.y = bbox.width;   // w
    
    // Calculate context amount
    float context = cfg_.context * (target_sz_.x + target_sz_.y);
    z_sz_ = std::sqrt((target_sz_.x + context) * (target_sz_.y + context));
    x_sz_ = z_sz_ * static_cast<float>(cfg_.instance_sz) / static_cast<float>(cfg_.exemplar_sz);
    
    // Get template patch
    cv::Scalar border_value(
        cv::mean(image)[0],
        cv::mean(image)[1],
        cv::mean(image)[2]
    );
    
    cv::Point2f img_center(center_.y, center_.x);  // [x, y] for OpenCV
    z_ = cropAndResize(image, img_center, z_sz_, cfg_.exemplar_sz, border_value);
    
    initialized_ = true;
}

cv::Rect2f SiamFCONNXTracker::update(const cv::Mat& image)
{
    if (!initialized_) {
        throw std::runtime_error("Tracker not initialized!");
    }
    
    // Get border value
    cv::Scalar border_value(
        cv::mean(image)[0],
        cv::mean(image)[1],
        cv::mean(image)[2]
    );
    
    // Multi-scale search
    std::vector<cv::Mat> responses;
    responses.reserve(cfg_.scale_num);
    
    cv::Point2f img_center(center_.y, center_.x);  // [x, y] for OpenCV
    
    for (float scale_factor : scale_factors_) {
        float current_x_sz = x_sz_ * scale_factor;
        cv::Mat x = cropAndResize(image, img_center, current_x_sz, 
                                 cfg_.instance_sz, border_value);
        
        // Run inference
        cv::Mat response = inference(z_, x);
        responses.push_back(response);
    }
    
    // Upsample responses
    std::vector<cv::Mat> upsampled_responses;
    upsampled_responses.reserve(cfg_.scale_num);
    
    for (const auto& resp : responses) {
        cv::Mat upsampled;
        cv::resize(resp, upsampled, cv::Size(upscale_sz_, upscale_sz_), 
                  0, 0, cv::INTER_CUBIC);
        upsampled_responses.push_back(upsampled);
    }
    
    // Apply scale penalty
    for (int i = 0; i < cfg_.scale_num; ++i) {
        if (i != cfg_.scale_num / 2) {
            upsampled_responses[i] *= cfg_.scale_penalty;
        }
    }
    
    // Find best scale
    int best_scale_id = 0;
    double max_response = -1e9;
    
    for (int i = 0; i < cfg_.scale_num; ++i) {
        double min_val, max_val;
        cv::minMaxLoc(upsampled_responses[i], &min_val, &max_val);
        if (max_val > max_response) {
            max_response = max_val;
            best_scale_id = i;
        }
    }
    
    // Get response at best scale
    cv::Mat response = upsampled_responses[best_scale_id].clone();
    
    // Normalize response
    double min_val, max_val;
    cv::minMaxLoc(response, &min_val, &max_val);
    response -= min_val;
    cv::Scalar sum = cv::sum(response);
    response /= (sum[0] + 1e-16);
    
    // Apply cosine window
    response = (1.0f - cfg_.window_influence) * response + 
               cfg_.window_influence * hann_window_;
    
    // Find peak location
    cv::Point max_loc;
    cv::minMaxLoc(response, nullptr, nullptr, nullptr, &max_loc);
    
    // Calculate displacement
    cv::Point2f disp_in_response(
        max_loc.x - (upscale_sz_ - 1) / 2.0f,
        max_loc.y - (upscale_sz_ - 1) / 2.0f
    );
    
    cv::Point2f disp_in_instance = disp_in_response * 
        (static_cast<float>(cfg_.total_stride) / static_cast<float>(cfg_.response_up));
    
    cv::Point2f disp_in_image = disp_in_instance * 
        (x_sz_ * scale_factors_[best_scale_id] / static_cast<float>(cfg_.instance_sz));
    
    // Update center (remember: center_ is [y, x])
    center_.x += disp_in_image.y;  // y
    center_.y += disp_in_image.x;  // x
    
    // Update scale
    float scale = (1.0f - cfg_.scale_lr) * 1.0f + 
                  cfg_.scale_lr * scale_factors_[best_scale_id];
    target_sz_.x *= scale;  // h
    target_sz_.y *= scale;  // w
    z_sz_ *= scale;
    x_sz_ *= scale;
    
    // Return bounding box [x, y, width, height] (1-indexed, left-top based)
    // Convert from [y, x, h, w] to [x, y, w, h]
    cv::Rect2f bbox;
    bbox.x = center_.y + 1.0f - (target_sz_.y - 1.0f) / 2.0f;  // x
    bbox.y = center_.x + 1.0f - (target_sz_.x - 1.0f) / 2.0f;  // y
    bbox.width = target_sz_.y;   // w
    bbox.height = target_sz_.x;  // h
    
    return bbox;
}

cv::Mat SiamFCONNXTracker::cropAndResize(const cv::Mat& image,
                                         const cv::Point2f& center,
                                         float size,
                                         int out_size,
                                         const cv::Scalar& border_value)
{
    int size_i = static_cast<int>(std::round(size));
    
    // Calculate corners (0-indexed)
    cv::Point2f top_left(
        std::round(center.x - (size - 1.0f) / 2.0f),
        std::round(center.y - (size - 1.0f) / 2.0f)
    );
    
    cv::Point2f bottom_right = top_left + cv::Point2f(size_i, size_i);
    
    // Calculate padding needed
    int pad_left = std::max(0, -static_cast<int>(top_left.x));
    int pad_top = std::max(0, -static_cast<int>(top_left.y));
    int pad_right = std::max(0, static_cast<int>(bottom_right.x) - image.cols);
    int pad_bottom = std::max(0, static_cast<int>(bottom_right.y) - image.rows);
    
    // Pad image if necessary
    cv::Mat padded_image;
    if (pad_left > 0 || pad_top > 0 || pad_right > 0 || pad_bottom > 0) {
        cv::copyMakeBorder(image, padded_image, 
                          pad_top, pad_bottom, pad_left, pad_right,
                          cv::BORDER_CONSTANT, border_value);
        
        top_left.x += pad_left;
        top_left.y += pad_top;
    } else {
        padded_image = image;
    }
    
    // Crop patch
    cv::Rect crop_rect(
        static_cast<int>(top_left.x),
        static_cast<int>(top_left.y),
        size_i, size_i
    );
    
    // Ensure crop_rect is within bounds
    crop_rect.x = std::max(0, crop_rect.x);
    crop_rect.y = std::max(0, crop_rect.y);
    crop_rect.width = std::min(crop_rect.width, padded_image.cols - crop_rect.x);
    crop_rect.height = std::min(crop_rect.height, padded_image.rows - crop_rect.y);
    
    cv::Mat patch = padded_image(crop_rect);
    
    // Resize to output size
    cv::Mat resized;
    cv::resize(patch, resized, cv::Size(out_size, out_size), 0, 0, cv::INTER_LINEAR);
    
    return resized;
}

cv::Mat SiamFCONNXTracker::inference(const cv::Mat& template_patch, 
                                     const cv::Mat& search_patch)
{
    // Convert BGR to RGB and normalize to [0, 255] float
    cv::Mat template_rgb, search_rgb;
    cv::cvtColor(template_patch, template_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(search_patch, search_rgb, cv::COLOR_BGR2RGB);
    
    template_rgb.convertTo(template_rgb, CV_32F);
    search_rgb.convertTo(search_rgb, CV_32F);
    
    // Create blobs (CHW format, normalized to [0, 255])
    cv::Mat template_blob = cv::dnn::blobFromImage(template_rgb, 1.0, 
                                                   cv::Size(cfg_.exemplar_sz, cfg_.exemplar_sz),
                                                   cv::Scalar(0, 0, 0), false, false);
    
    cv::Mat search_blob = cv::dnn::blobFromImage(search_rgb, 1.0,
                                                 cv::Size(cfg_.instance_sz, cfg_.instance_sz),
                                                 cv::Scalar(0, 0, 0), false, false);
    
    // Set inputs
    net_.setInput(template_blob, "template");
    net_.setInput(search_blob, "search");
    
    // Forward pass
    cv::Mat response_blob = net_.forward("response");
    
    // Extract response map (remove batch dimension)
    // response_blob shape: [1, 1, 17, 17]
    cv::Mat response(cfg_.response_sz, cfg_.response_sz, CV_32F);
    
    const float* data = reinterpret_cast<const float*>(response_blob.data);
    for (int i = 0; i < cfg_.response_sz; ++i) {
        for (int j = 0; j < cfg_.response_sz; ++j) {
            response.at<float>(i, j) = data[i * cfg_.response_sz + j];
        }
    }
    
    return response;
}

cv::Mat SiamFCONNXTracker::createHannWindow(int size)
{
    cv::Mat hann1d = cv::Mat::zeros(1, size, CV_32F);
    for (int i = 0; i < size; ++i) {
        hann1d.at<float>(0, i) = 0.5f - 0.5f * std::cos(2.0f * CV_PI * i / (size - 1));
    }
    
    cv::Mat hann2d = hann1d.t() * hann1d;
    hann2d /= cv::sum(hann2d)[0];
    
    return hann2d;
}

float SiamFCONNXTracker::calculateIoU(const cv::Rect2f& rect1, const cv::Rect2f& rect2)
{
    float inter_x1 = std::max(rect1.x, rect2.x);
    float inter_y1 = std::max(rect1.y, rect2.y);
    float inter_x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float inter_y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * 
                       std::max(0.0f, inter_y2 - inter_y1);
    
    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    float union_area = area1 + area2 - inter_area;
    
    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

} // namespace siamfc
