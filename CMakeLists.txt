cmake_minimum_required(VERSION 3.16)
project(ECR_Tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure all build outputs land directly in the build directory regardless of generator.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config}" CONFIG_UPPER)
        set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}" "${CMAKE_BINARY_DIR}")
        set("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}" "${CMAKE_BINARY_DIR}")
        set("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER}" "${CMAKE_BINARY_DIR}")
    endforeach()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio tracking video dnn)

# Print OpenCV information for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

set(PIPELINE_SOURCES
    src/TrackingPipeline.cpp
    src/RescueStrategy.cpp
    src/CsrtTracker.cpp
    src/Quality.cpp
    src/BoxKalmanFilter.cpp
    src/NCNNSsdDetector.cpp
    src/StaticBoxDetector.cpp
    src/TemplateMatcherDetector.cpp
    src/OnnxSsdDetector.cpp
    src/SiamFcDetector.cpp)

add_library(eldercare_pipeline STATIC ${PIPELINE_SOURCES})

target_include_directories(eldercare_pipeline PUBLIC 
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(eldercare_pipeline PUBLIC ${OpenCV_LIBS})

add_executable(ECR_Tracking
    src/main.cpp)

target_include_directories(ECR_Tracking PRIVATE 
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ECR_Tracking PRIVATE eldercare_pipeline)

add_executable(ECR_Tracking_otb
    src/otb_evaluator.cpp)

target_include_directories(ECR_Tracking_otb PRIVATE 
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ECR_Tracking_otb PRIVATE eldercare_pipeline)
