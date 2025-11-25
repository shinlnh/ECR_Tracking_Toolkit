#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace otb {

/**
 * @brief OTB100 dataset sequence
 */
struct Sequence {
    std::string name;
    std::vector<std::string> image_files;
    std::vector<cv::Rect2f> ground_truth;
    
    /**
     * @brief Check if sequence is valid
     */
    bool isValid() const {
        return !image_files.empty() && 
               !ground_truth.empty() && 
               image_files.size() == ground_truth.size();
    }
    
    /**
     * @brief Get number of frames
     */
    size_t size() const {
        return image_files.size();
    }
};

/**
 * @brief OTB100 dataset loader
 */
class OTBDataset {
public:
    /**
     * @brief Constructor
     * @param root_dir Path to OTB100 dataset directory
     */
    explicit OTBDataset(const std::string& root_dir);
    
    /**
     * @brief Load a specific sequence
     * @param seq_name Sequence name (e.g., "Basketball")
     * @return Sequence data
     */
    Sequence loadSequence(const std::string& seq_name);
    
    /**
     * @brief Get all sequence names
     * @return Vector of sequence names
     */
    std::vector<std::string> getSequenceNames() const;
    
    /**
     * @brief Get number of sequences
     */
    size_t size() const {
        return sequence_names_.size();
    }

private:
    /**
     * @brief Load ground truth from file
     * @param gt_file Path to groundtruth_rect.txt
     * @return Vector of bounding boxes
     */
    std::vector<cv::Rect2f> loadGroundTruth(const std::string& gt_file);
    
    /**
     * @brief Find image files in directory
     * @param img_dir Path to img directory
     * @return Vector of image file paths
     */
    std::vector<std::string> findImageFiles(const std::string& img_dir);

private:
    std::string root_dir_;
    std::vector<std::string> sequence_names_;
};

/**
 * @brief Evaluation metrics for tracking
 */
struct EvaluationMetrics {
    double average_iou = 0.0;
    double success_rate = 0.0;  // Percentage of frames with IoU > 0.5
    double average_fps = 0.0;
    int total_frames = 0;
    
    /**
     * @brief Print metrics
     */
    void print(const std::string& sequence_name = "") const;
};

/**
 * @brief Calculate IoU between two rectangles
 */
float calculateIoU(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

} // namespace otb
