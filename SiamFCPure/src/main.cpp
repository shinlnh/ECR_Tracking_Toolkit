#include "SiamFCONNXTracker.hpp"
#include "OTBDataset.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace std::chrono;

int main(int argc, char** argv)
{
    std::cout << "================================================================================\n";
    std::cout << "SiamFC ONNX Tracker - OTB100 Evaluation\n";
    std::cout << "================================================================================\n\n";
    
    // Paths
    std::string model_path = "../models/siamfc/siamfc.onnx";
    std::string dataset_path = "../otb100/OTB-dataset/OTB100";
    
    // Allow command line arguments to override paths
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        dataset_path = argv[2];
    }
    
    std::cout << "Model path:   " << model_path << "\n";
    std::cout << "Dataset path: " << dataset_path << "\n\n";
    
    try {
        // Load tracker
        std::cout << "[1] Loading SiamFC ONNX model...\n";
        siamfc::SiamFCONNXTracker tracker(model_path);
        std::cout << "\n";
        
        // Load dataset
        std::cout << "[2] Loading OTB100 dataset...\n";
        otb::OTBDataset dataset(dataset_path);
        std::cout << "\n";
        
        // Get sequence names
        auto sequence_names = dataset.getSequenceNames();
        
        // Ask user which sequence to test
        std::string seq_name = "Basketball";  // Default
        
        if (sequence_names.empty()) {
            std::cerr << "[ERROR] No sequences found in dataset!\n";
            return 1;
        }
        
        std::cout << "[3] Select sequence to test (default: Basketball):\n";
        std::cout << "    Available sequences: ";
        for (size_t i = 0; i < std::min(size_t(10), sequence_names.size()); ++i) {
            std::cout << sequence_names[i];
            if (i < std::min(size_t(10), sequence_names.size()) - 1) {
                std::cout << ", ";
            }
        }
        if (sequence_names.size() > 10) {
            std::cout << ", ... (" << sequence_names.size() - 10 << " more)";
        }
        std::cout << "\n";
        std::cout << "    Enter sequence name (or press Enter for Basketball): ";
        
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty()) {
            seq_name = input;
        }
        
        std::cout << "\n[4] Loading sequence: " << seq_name << "...\n";
        otb::Sequence sequence = dataset.loadSequence(seq_name);
        
        if (!sequence.isValid()) {
            std::cerr << "[ERROR] Failed to load sequence: " << seq_name << "\n";
            return 1;
        }
        
        std::cout << "    Frames: " << sequence.size() << "\n";
        std::cout << "    Initial box: [" << sequence.ground_truth[0].x << ", "
                  << sequence.ground_truth[0].y << ", "
                  << sequence.ground_truth[0].width << ", "
                  << sequence.ground_truth[0].height << "]\n\n";
        
        // Track
        std::cout << "[5] Starting tracking...\n";
        std::cout << "================================================================================\n";
        
        std::vector<cv::Rect2f> predicted_boxes;
        std::vector<float> ious;
        std::vector<double> frame_times;
        
        predicted_boxes.reserve(sequence.size());
        ious.reserve(sequence.size());
        frame_times.reserve(sequence.size());
        
        // Initialize with first frame
        cv::Mat frame = cv::imread(sequence.image_files[0]);
        if (frame.empty()) {
            std::cerr << "[ERROR] Failed to read first frame: " << sequence.image_files[0] << "\n";
            return 1;
        }
        
        auto start_time = high_resolution_clock::now();
        tracker.init(frame, sequence.ground_truth[0]);
        auto end_time = high_resolution_clock::now();
        
        double init_time = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
        
        predicted_boxes.push_back(sequence.ground_truth[0]);
        ious.push_back(1.0f);
        frame_times.push_back(init_time);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Frame    1/" << std::setw(4) << sequence.size() 
                  << " | Time: " << std::setw(7) << init_time << "ms"
                  << " | FPS: " << std::setw(6) << (1000.0 / init_time)
                  << " | IoU: " << std::setprecision(4) << ious[0]
                  << " | Box: [" << std::setw(7) << std::setprecision(2) << predicted_boxes[0].x
                  << ", " << std::setw(7) << predicted_boxes[0].y
                  << ", " << std::setw(7) << predicted_boxes[0].width
                  << ", " << std::setw(7) << predicted_boxes[0].height << "]\n";
        
        // Track remaining frames
        for (size_t i = 1; i < sequence.size(); ++i) {
            frame = cv::imread(sequence.image_files[i]);
            if (frame.empty()) {
                std::cerr << "[WARNING] Failed to read frame " << i << ": " 
                          << sequence.image_files[i] << "\n";
                continue;
            }
            
            start_time = high_resolution_clock::now();
            cv::Rect2f bbox = tracker.update(frame);
            end_time = high_resolution_clock::now();
            
            double frame_time = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
            float iou = otb::calculateIoU(bbox, sequence.ground_truth[i]);
            
            predicted_boxes.push_back(bbox);
            ious.push_back(iou);
            frame_times.push_back(frame_time);
            
            // Print every frame
            std::cout << "Frame " << std::setw(4) << (i + 1) << "/" << std::setw(4) << sequence.size()
                      << " | Time: " << std::setw(7) << frame_time << "ms"
                      << " | FPS: " << std::setw(6) << (1000.0 / frame_time)
                      << " | IoU: " << std::setprecision(4) << iou
                      << " | Box: [" << std::setw(7) << std::setprecision(2) << bbox.x
                      << ", " << std::setw(7) << bbox.y
                      << ", " << std::setw(7) << bbox.width
                      << ", " << std::setw(7) << bbox.height << "]\n";
            
            // Print progress every 10% or every 50 frames
            if ((i + 1) % std::max(size_t(1), sequence.size() / 10) == 0 || 
                (i + 1) % 50 == 0) {
                double avg_time = std::accumulate(frame_times.begin() + 1, frame_times.end(), 0.0) / 
                                 (frame_times.size() - 1);
                double avg_iou = std::accumulate(ious.begin(), ious.end(), 0.0f) / ious.size();
                
                std::cout << "\n--- Progress: " << std::fixed << std::setprecision(1) 
                          << (100.0 * (i + 1) / sequence.size()) << "% (" << (i + 1) 
                          << "/" << sequence.size() << ") | Avg FPS: " 
                          << std::setprecision(2) << (1000.0 / avg_time)
                          << " | Avg IoU: " << std::setprecision(4) << avg_iou << " ---\n\n";
            }
        }
        
        // Calculate metrics
        std::cout << "\n";
        std::cout << "================================================================================\n";
        std::cout << "Tracking completed!\n";
        std::cout << "================================================================================\n";
        
        otb::EvaluationMetrics metrics;
        metrics.total_frames = static_cast<int>(ious.size());
        metrics.average_iou = std::accumulate(ious.begin(), ious.end(), 0.0) / ious.size();
        
        int success_count = 0;
        for (float iou : ious) {
            if (iou > 0.5f) {
                success_count++;
            }
        }
        metrics.success_rate = 100.0 * success_count / ious.size();
        
        double avg_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
        metrics.average_fps = 1000.0 / avg_time;
        
        metrics.print(seq_name);
        
        std::cout << "\n[OK] Evaluation completed successfully!\n\n";
        
    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
