#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace siamfc {

/**
 * @brief Configuration for SiamFC tracker
 */
struct Config {
    int exemplar_sz = 127;      // Template size
    int instance_sz = 255;      // Search region size
    float context = 0.5f;       // Context amount
    int scale_num = 3;          // Number of scales
    float scale_step = 1.0375f; // Scale step
    float scale_lr = 0.59f;     // Scale learning rate
    float scale_penalty = 0.9745f; // Scale penalty factor
    float window_influence = 0.176f; // Window influence
    int response_sz = 17;       // Response map size
    int response_up = 16;       // Response upsampling factor
    int total_stride = 8;       // Total stride of backbone
};

/**
 * @brief SiamFC tracker using ONNX model
 */
class SiamFCONNXTracker {
public:
    /**
     * @brief Constructor
     * @param model_path Path to ONNX model file
     */
    explicit SiamFCONNXTracker(const std::string& model_path);
    
    /**
     * @brief Initialize tracker with first frame
     * @param image First frame (BGR format)
     * @param bbox Initial bounding box [x, y, width, height]
     */
    void init(const cv::Mat& image, const cv::Rect2f& bbox);
    
    /**
     * @brief Update tracker with new frame
     * @param image New frame (BGR format)
     * @return Updated bounding box [x, y, width, height]
     */
    cv::Rect2f update(const cv::Mat& image);
    
    /**
     * @brief Get tracker configuration
     */
    const Config& getConfig() const { return cfg_; }
    
private:
    /**
     * @brief Crop and resize image patch
     * @param image Input image
     * @param center Center point [x, y]
     * @param size Patch size
     * @param out_size Output size
     * @param border_value Border padding value
     * @return Resized patch
     */
    cv::Mat cropAndResize(const cv::Mat& image, 
                         const cv::Point2f& center,
                         float size, 
                         int out_size,
                         const cv::Scalar& border_value);
    
    /**
     * @brief Run inference on ONNX model
     * @param template_patch Template patch (127x127x3)
     * @param search_patch Search patch (255x255x3)
     * @return Response map (17x17)
     */
    cv::Mat inference(const cv::Mat& template_patch, const cv::Mat& search_patch);
    
    /**
     * @brief Create Hann window for cosine window weighting
     * @param size Window size
     * @return Hann window
     */
    static cv::Mat createHannWindow(int size);
    
    /**
     * @brief Calculate IoU between two rectangles
     * @param rect1 First rectangle
     * @param rect2 Second rectangle
     * @return IoU value [0, 1]
     */
    static float calculateIoU(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

private:
    cv::dnn::Net net_;              // ONNX model
    Config cfg_;                    // Tracker configuration
    
    // Tracking state
    cv::Point2f center_;            // Current center [y, x] (image coordinates)
    cv::Point2f target_sz_;         // Current target size [h, w]
    float z_sz_;                    // Exemplar size
    float x_sz_;                    // Search region size
    cv::Mat z_;                     // Template features (stored for tracking)
    cv::Mat hann_window_;           // Hann window for response weighting
    std::vector<float> scale_factors_; // Scale factors for multi-scale search
    int upscale_sz_;                // Upscaled response size
    
    bool initialized_;              // Initialization flag
};

} // namespace siamfc
