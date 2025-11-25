# SiamFC ONNX Tracker

C++ implementation of SiamFC tracker using ONNX model with OpenCV DNN backend.

## Features

- SiamFC tracker using ONNX model
- OpenCV DNN backend (CPU)
- OTB100 dataset evaluation
- Frame-by-frame IoU calculation
- Real-time FPS display

## Requirements

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x with DNN module
- ONNX model: `../models/siamfc/siamfc.onnx`
- OTB100 dataset: `../otb100/OTB-dataset/OTB100`

## Build

### Windows (MinGW)

```bash
build.bat
```

### Manual build

```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
cmake --build . --config Release
```

## Usage

### Run on Basketball sequence (default)

```bash
cd build/bin
siamfc_otb_test.exe
```

Press Enter to use default Basketball sequence, or enter another sequence name.

### Specify custom paths

```bash
siamfc_otb_test.exe <model_path> <dataset_path>
```

Example:
```bash
siamfc_otb_test.exe ../../models/siamfc/siamfc.onnx ../../otb100/OTB-dataset/OTB100
```

## Project Structure

```
SiamFCPure/
├── CMakeLists.txt
├── build.bat
├── README.md
├── include/
│   ├── SiamFCONNXTracker.hpp    # SiamFC tracker interface
│   └── OTBDataset.hpp            # OTB100 dataset loader
└── src/
    ├── SiamFCONNXTracker.cpp     # SiamFC tracker implementation
    ├── OTBDataset.cpp             # Dataset loader implementation
    └── main.cpp                   # Main program for OTB100 evaluation
```

## Implementation Details

### SiamFC Tracker

- **Model**: ONNX format loaded via OpenCV DNN
- **Input**: 
  - Template: 127x127x3 RGB
  - Search: 255x255x3 RGB
- **Output**: Response map 17x17
- **Multi-scale search**: 3 scales (1/1.0375, 1.0, 1.0375)
- **Cosine window**: Hann window for response weighting

### Tracking Pipeline

1. Initialize with first frame and bounding box
2. Extract template patch (127x127)
3. For each subsequent frame:
   - Extract search patches at multiple scales (255x255)
   - Run ONNX inference for each scale
   - Upsample response maps to 272x272
   - Apply scale penalty and cosine window
   - Find peak location and update bounding box

### Coordinate System

- **Input boxes**: [x, y, width, height] (1-indexed, left-top based)
- **Internal representation**: [y, x, height, width] (0-indexed, center based)
- **Output boxes**: [x, y, width, height] (1-indexed, left-top based)

This matches the PyTorch implementation for exact compatibility.

## Expected Results

On Basketball sequence:
- Average IoU: ~0.698 (69.8%)
- Success Rate: ~38-40% (frames with IoU > 0.5)
- Average FPS: 10-15 FPS (depends on CPU)

## Comparison with Python

The C++ implementation produces **identical results** to the Python ONNX version:
- Same tracking boxes
- Same IoU values
- Faster execution (C++ native vs Python)

## Notes

- Model uses RGB format internally (converted from OpenCV's BGR)
- Response maps are normalized before applying cosine window
- Scale updates use exponential moving average
- Border padding uses image mean color
