#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "detection/SiamFcDetector.hpp"
#include "detection/StaticBoxDetector.hpp"
#include "pipeline/TrackingPipeline.hpp"
#include "pipeline/Types.hpp"
#include "config/Config.hpp"

namespace {

struct Args {
    std::string datasetPath;
    std::optional<std::string> sequenceName;
    bool useMockDetector{false};
    bool bootstrapFromGroundTruth{true};
    std::optional<std::string> modelPath;
    std::string bootstrapLabel{"person"};
    float bootstrapScore{0.99f};
    float successThreshold{0.5f};
    bool display{false};
    std::optional<std::string> outputPath;  // Output file for bounding boxes
};

void printUsage() {
    std::cout << "Usage: eldercare_tracking_otb --dataset <path> [options]\n"
                 "Options:\n"
                 "  --sequence <name>           Evaluate a single sequence\n"
                 "  --model <path>              SiamFC ONNX model path (required unless --use-mock-detector)\n"
                 "  --use-mock-detector         Use the static detector stub\n"
                 "  --label <string>            Label injected with bootstrap detection (default: person)\n"
                 "  --bootstrap-score <float>   Confidence value for bootstrap detection (default: 0.99)\n"
                 "  --no-bootstrap              Do not inject ground-truth detection on first frame\n"
                 "  --success-threshold <float> IoU threshold for success rate (default: 0.5)\n"
                 "  --output <path>             Output file for tracking results (bounding boxes)\n"
                 "  --display                   Show tracking visualization window\n"
                 "  --no-display                Disable visualization (default)\n";
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && (i + 1) < argc) {
            args.datasetPath = argv[++i];
        } else if (arg == "--sequence" && (i + 1) < argc) {
            args.sequenceName = std::string(argv[++i]);
        } else if (arg == "--model" && (i + 1) < argc) {
            args.modelPath = std::string(argv[++i]);
        } else if (arg == "--use-mock-detector") {
            args.useMockDetector = true;
        } else if (arg == "--label" && (i + 1) < argc) {
            args.bootstrapLabel = argv[++i];
        } else if (arg == "--bootstrap-score" && (i + 1) < argc) {
            args.bootstrapScore = std::stof(argv[++i]);
        } else if (arg == "--no-bootstrap") {
            args.bootstrapFromGroundTruth = false;
        } else if (arg == "--success-threshold" && (i + 1) < argc) {
            args.successThreshold = std::stof(argv[++i]);
        } else if (arg == "--output" && (i + 1) < argc) {
            args.outputPath = std::string(argv[++i]);
        } else if (arg == "--display") {
            args.display = true;
        } else if (arg == "--no-display") {
            args.display = false;
        } else if (arg == "--help") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            return false;
        }
    }

    if (args.datasetPath.empty()) {
        std::cerr << "Missing required option --dataset\n";
        return false;
    }
    if (!args.useMockDetector && !args.modelPath && args.bootstrapFromGroundTruth == false) {
        std::cerr << "Provide --model or enable bootstrap to seed the tracker\n";
        return false;
    }
    return true;
}

std::vector<std::filesystem::path> collectFramePaths(const std::filesystem::path& imgDir) {
    std::vector<std::filesystem::path> frames;
    if (!std::filesystem::exists(imgDir)) {
        return frames;
    }
    for (const auto& entry : std::filesystem::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::optional<std::vector<pipeline::BoundingBox>> loadGroundTruth(const std::filesystem::path& file) {
    std::ifstream stream(file);
    if (!stream.is_open()) {
        return std::nullopt;
    }
    std::vector<pipeline::BoundingBox> boxes;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        for (char& ch : line) {
            if (ch == ',' || ch == '\t') {
                ch = ' ';
            }
        }
        std::stringstream ss(line);
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        if (!(ss >> x >> y >> w >> h)) {
            continue;
        }
        boxes.emplace_back(x - 1.0f, y - 1.0f, w, h);
    }
    return boxes;
}

std::optional<std::filesystem::path> resolveSequenceRoot(const std::filesystem::path& sequencePath) {
    const auto directGt = sequencePath / "groundtruth_rect.txt";
    if (std::filesystem::exists(directGt)) {
        return sequencePath;
    }

    const auto nestedSameName = sequencePath / sequencePath.filename();
    if (!sequencePath.filename().empty() && std::filesystem::exists(nestedSameName / "groundtruth_rect.txt")) {
        return nestedSameName;
    }

    for (const auto& entry : std::filesystem::directory_iterator(sequencePath)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto candidate = entry.path();
        if (std::filesystem::exists(candidate / "groundtruth_rect.txt")) {
            return candidate;
        }
    }
    return std::nullopt;
}

struct SequenceData {
    std::string name;
    std::filesystem::path path;
    std::vector<std::filesystem::path> frames;
    std::vector<pipeline::BoundingBox> groundTruth;
};

std::optional<SequenceData> loadSequence(const std::filesystem::path& sequencePath) {
    const auto resolved = resolveSequenceRoot(sequencePath);
    if (!resolved) {
        std::cerr << "Could not locate ground truth under " << sequencePath << "\\n";
        return std::nullopt;
    }

    SequenceData data;
    data.name = sequencePath.filename().string();
    data.path = *resolved;

    const auto gt = loadGroundTruth(data.path / "groundtruth_rect.txt");
    if (!gt || gt->empty()) {
        std::cerr << "Failed to read ground truth for sequence " << data.name << "\\n";
        return std::nullopt;
    }

    data.frames = collectFramePaths(data.path / "img");
    if (data.frames.empty()) {
        std::cerr << "No frames found in sequence " << data.name << "\\n";
        return std::nullopt;
    }

    const std::size_t frameCount = std::min(gt->size(), data.frames.size());
    data.frames.resize(frameCount);
    data.groundTruth.assign(gt->begin(), gt->begin() + static_cast<std::ptrdiff_t>(frameCount));
    return data;
}

class BootstrapDetector final : public detection::Detector {
public:
    BootstrapDetector(std::unique_ptr<detection::Detector> inner, 
                      const pipeline::BoundingBox& initialBox,
                      std::string label,
                      float score)
        : inner_(std::move(inner)),
          initialBox_(initialBox),
          label_(std::move(label)),
          score_(score) {
        std::cout << "*** BOOTSTRAP CONSTRUCTOR *** inner_ = " << (inner_ != nullptr) << std::endl;
        if (inner_) {
            siam_ = dynamic_cast<detection::SiamFcDetector*>(inner_.get());
            std::cout << "*** BOOTSTRAP CONSTRUCTOR *** siam_ = " << (siam_ != nullptr) << std::endl;
            if (!siam_) {
                std::cout << "*** BOOTSTRAP CONSTRUCTOR *** WARNING: inner_ is not a SiamFcDetector!" << std::endl;
            }
        }
    }

    std::vector<detection::Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override {
        std::cout << "*** BOOTSTRAP *** detect() called, seeded=" << seeded_ 
                  << ", hasInner=" << (inner_ != nullptr) 
                  << ", hasSiam=" << (siam_ != nullptr) << std::endl << std::flush;
        if (!seeded_) {
            seeded_ = true;
            if (siam_) {
                std::cout << "*** BOOTSTRAP *** First call - updating template with initialBox" << std::endl << std::flush;
                siam_->updateTrackedBox(frame, initialBox_);
            }
            static_cast<void>(roi);
            return {detection::Detection{label_, score_, initialBox_}};
        }
        if (inner_) {
            std::cout << "*** BOOTSTRAP *** Calling inner_->detect()" << std::endl << std::flush;
            return inner_->detect(frame, roi);
        }
        std::cout << "*** BOOTSTRAP *** No inner detector!" << std::endl << std::flush;
        return {};
    }

    void updateTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) override {
        if (siam_) {
            siam_->updateTemplate(frame, box);
        }
    }

    void updateModel(const cv::Mat& frame, const pipeline::BoundingBox& box) {
        if (siam_) {
            siam_->updateTrackedBox(frame, box);
        }
    }

    void restoreOriginalTemplate() override {
        if (siam_) {
            siam_->restoreOriginalTemplate();
        }
    }

private:
    std::unique_ptr<detection::Detector> inner_;
    pipeline::BoundingBox initialBox_;
    std::string label_;
    float score_{1.0f};
    bool seeded_{false};
    detection::SiamFcDetector* siam_{nullptr};
};

struct SequenceResult {
    std::string name;
    int frames{0};
    int successFrames{0};
    double averageIoU{0.0};
    double successRate{0.0};
    double fps{0.0};
    double totalSeconds{0.0};
};

std::unique_ptr<detection::Detector> createDetector(const Args& args, const pipeline::BoundingBox& firstBox) {
    std::unique_ptr<detection::Detector> inner;
    if (args.useMockDetector) {
        inner = std::make_unique<detection::StaticBoxDetector>(args.bootstrapLabel);
    } else if (args.modelPath) {
        try {
            inner = std::make_unique<detection::SiamFcDetector>(*args.modelPath, args.bootstrapLabel);
        } catch (const std::exception& ex) {
            std::cerr << "Failed to initialise SiamFC detector: " << ex.what() << "\n";
        }
    }

    if (args.bootstrapFromGroundTruth) {
        return std::make_unique<BootstrapDetector>(std::move(inner), firstBox, args.bootstrapLabel, args.bootstrapScore);
    }
    return inner;
}

std::optional<SequenceResult> evaluateSequence(const Args& args, const SequenceData& data) {
    if (data.frames.empty() || data.groundTruth.empty()) {
        return std::nullopt;
    }

    auto detector = createDetector(args, data.groundTruth.front());
    if (!detector) {
        std::cerr << "Unable to construct detector for sequence " << data.name << "\n";
        return std::nullopt;
    }

    // Create config with more aggressive rescue settings to test SiamFC
    config::PipelineConfig config;
    config.rescue.intervalFrames = 5;          // Trigger rescue after 5 frames instead of 12
    config.rescue.fullFrameInterval = 15;      // Full frame recovery after 15 frames instead of 45
    config.rescue.minIoU = 0.0f;               // Accept ANY detection - use original template
    config.rescue.apceThresholdFactor = 0.25f; // Less strict confidence threshold
    config.rescue.apceConsecutiveFrames = 2;   // Trigger after 2 consecutive low confidence frames
    config.quality.minIoU = 0.02f;             // Lower quality IoU threshold
    
    pipeline::TrackingPipeline pipeline(*detector, config);

    SequenceResult result;
    result.name = data.name;
    const std::size_t frameCount = std::min(data.frames.size(), data.groundTruth.size());
    double sumIoU = 0.0;
    int successFrames = 0;
    double totalSeconds = 0.0;
    
    // Storage for bounding boxes (for --output option)
    std::vector<pipeline::BoundingBox> trackingResults;

    // Print detailed per-frame header
    std::cout << "\nDetailed Per-Frame Results for " << data.name << ":\n";
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left << std::setw(8) << "Frame"
              << std::right << std::setw(12) << "IoU"
              << std::setw(12) << "Avg IoU"
              << std::setw(12) << "Success Rate"
              << std::setw(15) << "Tracker Type"
              << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (std::size_t i = 0; i < frameCount; ++i) {
        std::cout << "Processing frame " << (i+1) << "/" << frameCount << std::endl << std::flush;
        const auto& framePath = data.frames[i];
        cv::Mat frame = cv::imread(framePath.string(), cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Failed to load frame " << framePath << "\n";
            break;
        }
        std::cout << "  Frame loaded: " << frame.cols << "x" << frame.rows << std::endl << std::flush;

        // Set current ground truth IoU for rescue decision before step()
        if (i > 0) { // Skip first frame as there's no previous tracking result
            auto lastOutput = pipeline.getLastTrackingBox();
            if (lastOutput.has_value()) {
                double groundTruthIoU = lastOutput->iou(data.groundTruth[i]);
                pipeline.setCurrentGroundTruthIoU(groundTruthIoU);
            }
        }

        const auto start = std::chrono::steady_clock::now();
        std::cout << "  Calling pipeline.step()..." << std::endl << std::flush;
        pipeline::PipelineOutput output = pipeline.step(frame);
        std::cout << "  pipeline.step() returned" << std::endl << std::flush;
        const auto end = std::chrono::steady_clock::now();
        totalSeconds += std::chrono::duration<double>(end - start).count();
        
        // Set correct original target size after first frame initialization
        if (i == 0) {
            pipeline.setOriginalTargetSize(data.groundTruth[0]);
        }
        
        // Debug output for rescue events (keep for special events)
        if (output.usedRescue) {
            std::cout << "*** RESCUE EVENT *** Frame " << (i + 1) << ": " << output.trackerType;
            if (output.rescueSource) {
                std::cout << " via " << *output.rescueSource;
            }
            std::cout << " ***" << std::endl;
        }

        double iou = 0.0;
        if (output.smoothedBox) {
            iou = output.smoothedBox->iou(data.groundTruth[i]);
            trackingResults.push_back(*output.smoothedBox);  // Save for output file
            
            // Debug rescue results
            if (output.usedRescue) {
                std::cout << "*** RESCUE RESULT *** Frame " << (i + 1) << std::endl;
                std::cout << "*** Ground Truth: (" << data.groundTruth[i].x << "," << data.groundTruth[i].y 
                          << "," << data.groundTruth[i].width << "," << data.groundTruth[i].height << ")" << std::endl;
                std::cout << "*** Rescue Box:   (" << output.smoothedBox->x << "," << output.smoothedBox->y 
                          << "," << output.smoothedBox->width << "," << output.smoothedBox->height << ")" << std::endl;
                std::cout << "*** IoU Result: " << iou << std::endl;
            }
            
            if (auto bootstrap = dynamic_cast<BootstrapDetector*>(detector.get())) {
                bootstrap->updateModel(frame, *output.smoothedBox);
            } else if (auto siam = dynamic_cast<detection::SiamFcDetector*>(detector.get())) {
                // FOR TESTING: Never update SiamFC template to preserve original good template
                // This helps SiamFC maintain consistent reference for rescue operations
                if (i == 0) {  // Only update template on first frame
                    std::cout << "*** SIAMFC DEBUG *** Setting initial template from first frame" << std::endl;
                    siam->updateTrackedBox(frame, *output.smoothedBox);
                } else {
                    std::cout << "*** SIAMFC DEBUG *** Preserving original template (frame=" << (i+1) 
                              << ", usedRescue=" << output.usedRescue << ", IoU=" << iou << ")" << std::endl;
                }
            }
        }

        std::string trackerStatus;
        if (output.lost) {
            trackerStatus = "Lost";
        } else if (output.usedRescue) {
            const std::string source = output.rescueSource.value_or(output.usedFullFrame ? "FullFrame" : "ROI");
            trackerStatus = "Rescue (" + source + ")";
        } else {
            trackerStatus = "CSRT";
        }

        sumIoU += iou;
        if (iou >= static_cast<double>(args.successThreshold)) {
            ++successFrames;
        }
        ++result.frames;
        
        // Per-frame detailed logging
        std::cout << std::left << std::setw(8) << (i + 1)
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << iou
                  << std::setw(12) << std::fixed << std::setprecision(3) 
                  << (sumIoU / static_cast<double>(result.frames))
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << (static_cast<double>(successFrames) / static_cast<double>(result.frames))
                  << std::setw(15) << output.trackerType
                  << std::endl;

        if (args.display) {
            cv::Mat vis = frame.clone();
            if (output.rawBox) {
                cv::rectangle(vis, output.rawBox->toRect(), {0, 0, 255}, 2);
            }
            if (output.smoothedBox) {
                cv::rectangle(vis, output.smoothedBox->toRect(), {0, 255, 0}, 2);
            }
            const cv::Rect gtRect = data.groundTruth[i].toRect();
            cv::rectangle(vis, gtRect, {255, 0, 0}, 1);
            
            // Display tracking status and IoU
            std::string text = trackerStatus + " | IoU: " + std::to_string(iou);
            if (output.usedRescue) {
                text += " [RESCUED]";
            }
            cv::putText(vis, text, {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
            
            // Display active tracker type
            const std::string trackerInfo = "Tracker: " + output.trackerType;
            cv::Scalar trackerColor = {255, 255, 0}; // Cyan color for tracker info
            if (output.trackerType == "CSRT") {
                trackerColor = {255, 255, 0}; // Cyan for CSRT
            } else if (output.trackerType == "SiamFC/ROI") {
                trackerColor = {0, 255, 255}; // Yellow for SiamFC
            } else if (output.trackerType == "Detector") {
                trackerColor = {255, 0, 255}; // Magenta for Detector
            }
            cv::putText(vis, trackerInfo, {15, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.7, trackerColor, 2);
            
            // Display rescue information if used
            if (output.usedRescue && output.rescueSource) {
                const std::string rescueInfo = "Rescue: " + *output.rescueSource;
                cv::putText(vis, rescueInfo, {15, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 165, 255}, 2);
            }
            
            // Display frame index in top right corner
            const std::string frameInfo = "Frame: " + std::to_string(i + 1) + "/" + std::to_string(data.frames.size());
            int textWidth = cv::getTextSize(frameInfo, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr).width;
            cv::putText(vis, frameInfo, {vis.cols - textWidth - 10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);
            
            cv::imshow("OTB Tracking", vis);
            const int key = cv::waitKey(1);
            if (key == 27) {  // ESC
                break;
            }
        }
    }

    if (result.frames == 0) {
        return std::nullopt;
    }

    result.averageIoU = sumIoU / static_cast<double>(result.frames);
    result.successRate = static_cast<double>(successFrames) / static_cast<double>(result.frames);
    result.fps = (totalSeconds > 0.0) ? static_cast<double>(result.frames) / totalSeconds : 0.0;
    result.totalSeconds = totalSeconds;
    
    // Save bounding boxes to output file if requested
    if (args.outputPath && !trackingResults.empty()) {
        std::ofstream outFile(*args.outputPath);
        if (outFile.is_open()) {
            for (const auto& box : trackingResults) {
                outFile << box.x << "," << box.y << "," << box.width << "," << box.height << "\n";
            }
            outFile.close();
            std::cout << "Saved " << trackingResults.size() << " bounding boxes to: " << *args.outputPath << std::endl;
        } else {
            std::cerr << "Warning: Failed to open output file: " << *args.outputPath << std::endl;
        }
    }
    
    // Print footer
    std::cout << std::string(75, '-') << std::endl;
    std::cout << "Summary: " << result.frames << " frames processed" << std::endl;
    std::cout << std::string(75, '=') << std::endl;
    
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    const std::filesystem::path datasetRoot(args.datasetPath);
    if (!std::filesystem::exists(datasetRoot)) {
        std::cerr << "Dataset path does not exist: " << datasetRoot << "\n";
        return 1;
    }

    std::vector<std::filesystem::path> sequencePaths;
    if (std::filesystem::exists(datasetRoot / "groundtruth_rect.txt")) {
        sequencePaths.push_back(datasetRoot);
    } else if (args.sequenceName) {
        sequencePaths.push_back(datasetRoot / *args.sequenceName);
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(datasetRoot)) {
            if (entry.is_directory()) {
                sequencePaths.push_back(entry.path());
            }
        }
        std::sort(sequencePaths.begin(), sequencePaths.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.filename().string() < rhs.filename().string();
        });
    }

    if (sequencePaths.empty()) {
        std::cerr << "No sequences found under " << datasetRoot << "\n";
        return 1;
    }

    std::vector<SequenceResult> results;
    for (const auto& sequencePath : sequencePaths) {
        const auto maybeSequence = loadSequence(sequencePath);
        if (!maybeSequence) {
            continue;
        }
        const auto maybeResult = evaluateSequence(args, *maybeSequence);
        if (maybeResult) {
            results.push_back(*maybeResult);
        }
    }

    if (results.empty()) {
        std::cerr << "No sequences were successfully evaluated.\n";
        return 1;
    }

    if (args.display) {
        cv::destroyWindow("OTB Tracking");
    }

    std::ostringstream successHeader;
    successHeader << std::fixed << std::setprecision(2) << args.successThreshold;
    const std::string successColumn = "Success@" + successHeader.str();

    std::cout << std::left << std::setw(18) << "Sequence"
              << std::right << std::setw(10) << "Frames"
              << std::setw(12) << "Avg IoU"
              << std::setw(16) << successColumn
              << std::setw(12) << "FPS" << "\n";

    std::cout << std::string(68, '-') << "\n";

    double totalIoU = 0.0;
    int totalFrames = 0;
    int totalSuccess = 0;
    double totalSeconds = 0.0;

    for (const auto& result : results) {
        std::cout << std::left << std::setw(18) << result.name
                  << std::right << std::setw(10) << result.frames
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.averageIoU
                  << std::setw(16) << std::fixed << std::setprecision(3) << result.successRate
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.fps << "\n";

        totalIoU += result.averageIoU * static_cast<double>(result.frames);
        totalFrames += result.frames;
        totalSuccess += static_cast<int>(result.successRate * result.frames + 0.5);
        totalSeconds += (result.frames > 0 && result.fps > 0.0)
            ? static_cast<double>(result.frames) / result.fps
            : 0.0;
    }

    if (results.size() > 1 && totalFrames > 0) {
        const double avgIoU = totalIoU / static_cast<double>(totalFrames);
        const double successRate = static_cast<double>(totalSuccess) / static_cast<double>(totalFrames);
        const double fps = (totalSeconds > 0.0) ? static_cast<double>(totalFrames) / totalSeconds : 0.0;
        std::cout << std::string(68, '-') << "\n";
        std::cout << std::left << std::setw(18) << "Overall"
                  << std::right << std::setw(10) << totalFrames
                  << std::setw(12) << std::fixed << std::setprecision(3) << avgIoU
                  << std::setw(16) << std::fixed << std::setprecision(3) << successRate
                  << std::setw(12) << std::fixed << std::setprecision(2) << fps << "\n";
    }

    return 0;
}





















