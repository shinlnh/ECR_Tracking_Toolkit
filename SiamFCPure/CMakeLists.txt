cmake_minimum_required(VERSION 3.16)
project(SiamFCPure_ONNX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Find OpenCV with DNN module
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs dnn)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# SiamFC ONNX Tracker library
add_library(siamfc_onnx STATIC
    src/SiamFCONNXTracker.cpp
    src/OTBDataset.cpp
)

target_link_libraries(siamfc_onnx PUBLIC ${OpenCV_LIBS})

# Main executable for OTB100 evaluation
add_executable(siamfc_otb_test
    src/main.cpp
)

target_link_libraries(siamfc_otb_test PRIVATE siamfc_onnx)

# Installation
install(TARGETS siamfc_otb_test RUNTIME DESTINATION bin)
