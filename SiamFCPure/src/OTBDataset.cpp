#include "OTBDataset.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

namespace otb {

OTBDataset::OTBDataset(const std::string& root_dir)
    : root_dir_(root_dir)
{
    // Scan for sequence directories
    try {
        for (const auto& entry : fs::directory_iterator(root_dir_)) {
            if (entry.is_directory()) {
                sequence_names_.push_back(entry.path().filename().string());
            }
        }
        
        std::sort(sequence_names_.begin(), sequence_names_.end());
        
        std::cout << "[OK] Found " << sequence_names_.size() 
                  << " sequences in " << root_dir_ << std::endl;
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Failed to scan dataset directory: " << e.what() << std::endl;
        throw;
    }
}

Sequence OTBDataset::loadSequence(const std::string& seq_name)
{
    Sequence seq;
    seq.name = seq_name;
    
    fs::path seq_dir = fs::path(root_dir_) / seq_name;
    
    // Check for nested structure (e.g., Basketball/Basketball/)
    fs::path nested_dir = seq_dir / seq_name;
    if (fs::exists(nested_dir) && fs::is_directory(nested_dir)) {
        seq_dir = nested_dir;
    }
    
    // Load ground truth
    fs::path gt_file = seq_dir / "groundtruth_rect.txt";
    if (!fs::exists(gt_file)) {
        std::cerr << "[ERROR] Ground truth file not found: " << gt_file << std::endl;
        return seq;
    }
    
    seq.ground_truth = loadGroundTruth(gt_file.string());
    
    // Load image files
    fs::path img_dir = seq_dir / "img";
    if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) {
        std::cerr << "[ERROR] Image directory not found: " << img_dir << std::endl;
        return seq;
    }
    
    seq.image_files = findImageFiles(img_dir.string());
    
    if (seq.image_files.size() != seq.ground_truth.size()) {
        std::cerr << "[WARNING] Number of images (" << seq.image_files.size() 
                  << ") != number of ground truth boxes (" << seq.ground_truth.size() 
                  << ") for sequence " << seq_name << std::endl;
    }
    
    return seq;
}

std::vector<std::string> OTBDataset::getSequenceNames() const
{
    return sequence_names_;
}

std::vector<cv::Rect2f> OTBDataset::loadGroundTruth(const std::string& gt_file)
{
    std::vector<cv::Rect2f> boxes;
    std::ifstream file(gt_file);
    
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open ground truth file: " << gt_file << std::endl;
        return boxes;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Replace commas with spaces for easier parsing
        std::replace(line.begin(), line.end(), ',', ' ');
        
        std::istringstream iss(line);
        float x, y, w, h;
        
        if (iss >> x >> y >> w >> h) {
            boxes.emplace_back(x, y, w, h);
        }
    }
    
    file.close();
    return boxes;
}

std::vector<std::string> OTBDataset::findImageFiles(const std::string& img_dir)
{
    std::vector<std::string> image_files;
    
    try {
        for (const auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_files.push_back(entry.path().string());
                }
            }
        }
        
        std::sort(image_files.begin(), image_files.end());
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Failed to scan image directory: " << e.what() << std::endl;
    }
    
    return image_files;
}

void EvaluationMetrics::print(const std::string& sequence_name) const
{
    std::cout << "================================================================================\n";
    if (!sequence_name.empty()) {
        std::cout << "Sequence: " << sequence_name << "\n";
    }
    std::cout << "Evaluation Results:\n";
    std::cout << "  Total frames:     " << total_frames << "\n";
    std::cout << "  Average IoU:      " << std::fixed << std::setprecision(4) << average_iou << "\n";
    std::cout << "  Success Rate:     " << std::fixed << std::setprecision(1) << success_rate << "%\n";
    std::cout << "  Average FPS:      " << std::fixed << std::setprecision(2) << average_fps << "\n";
    std::cout << "================================================================================\n";
}

float calculateIoU(const cv::Rect2f& rect1, const cv::Rect2f& rect2)
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

} // namespace otb
