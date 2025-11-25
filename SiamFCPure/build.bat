@echo off
echo ================================================================================
echo Building SiamFC ONNX Tracker
echo ================================================================================
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure CMake
echo [1] Configuring CMake...
cmake -G "MinGW Makefiles" ..
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed!
    cd ..
    exit /b 1
)

echo.
echo [2] Building project...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    cd ..
    exit /b 1
)

echo.
echo ================================================================================
echo Build completed successfully!
echo ================================================================================
echo Executable: build\bin\siamfc_otb_test.exe
echo.
echo To run: cd build\bin ^&^& siamfc_otb_test.exe
echo.

cd ..
