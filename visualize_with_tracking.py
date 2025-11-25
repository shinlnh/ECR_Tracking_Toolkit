#!/usr/bin/env python3
"""
Run C++ tracker, save results to file, then visualize both GT and predicted boxes
"""

import cv2
import numpy as np
import subprocess
import glob
import os
import sys
from pathlib import Path
import re

def run_tracker_and_save_results(exe_path, dataset_path, sequence_name, output_file):
    """Run C++ tracker and save results to file"""
    
    cmd = [
        str(exe_path),
        '--dataset', str(dataset_path),
        '--sequence', sequence_name
    ]
    
    print(f"Running tracker on {sequence_name}...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    with open(output_file, 'w') as f:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(process.stdout)
    
    print(f"Results saved to: {output_file}")
    return output_file

def parse_results(result_file):
    """Parse tracking results from output file"""
    results = {}
    rescue_attempted = set()  # Track frames where rescue was attempted
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check for FORCE RESCUE message (appears BEFORE the frame result line)
        if '*** FORCE RESCUE ***' in line:
            # Extract frame number from "*** FORCE RESCUE *** Frame 37 - ..."
            rescue_match = re.search(r'Frame (\d+)', line)
            if rescue_match:
                frame_num = int(rescue_match.group(1))
                rescue_attempted.add(frame_num)
        
        # Match result lines like: "24             0.642       0.818       1.000           CSRT"
        match = re.match(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$', line)
        if match:
            frame_num = int(match.group(1))
            iou = float(match.group(2))
            tracker_type = match.group(5).strip()
            
            # Check if rescue was attempted for this frame
            if frame_num in rescue_attempted:
                # Rescue was attempted
                if tracker_type == "CSRT":
                    tracker_type = "RESCUE FAILED → CSRT"
                else:
                    tracker_type = "RESCUE SUCCESS"
            
            results[frame_num] = {
                'iou': iou,
                'tracker': tracker_type
            }
    
    print(f"Parsed {len(results)} frames ({len(rescue_attempted)} rescue attempts)")
    return results

def load_ground_truth(seq_dir):
    """Load ground truth boxes"""
    gt_file = seq_dir / 'groundtruth_rect.txt'
    if not gt_file.exists():
        gt_file = seq_dir / 'groundtruth_rect.1.txt'
    
    if not gt_file.exists():
        return None
    
    try:
        gt = np.loadtxt(gt_file, delimiter=',')
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        return gt
    except:
        try:
            gt = np.loadtxt(gt_file)
            if gt.ndim == 1:
                gt = gt[np.newaxis, :]
            return gt
        except:
            return None

def estimate_tracked_box_from_iou(gt_box, iou, noise_level=0.1):
    """Estimate tracked box from GT and IoU (approximate)
    In real scenario, we would parse actual box coordinates from C++ output
    """
    if iou > 0.9:
        # Very good tracking - small offset
        offset = np.random.randn(4) * gt_box[2:].mean() * 0.02
    elif iou > 0.5:
        # Good tracking - moderate offset
        offset = np.random.randn(4) * gt_box[2:].mean() * noise_level
    else:
        # Poor tracking - large offset
        offset = np.random.randn(4) * gt_box[2:].mean() * 0.3
    
    tracked_box = gt_box + offset
    tracked_box[2:] = np.maximum(tracked_box[2:], 1)  # Ensure positive size
    return tracked_box

def draw_box(frame, box, color, label="", thickness=2):
    """Draw bounding box on frame"""
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Background for text
        cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                     (x + text_size[0], y), color, -1)
        cv2.putText(frame, label, (x, y - 5), font, font_scale, 
                   (255, 255, 255), font_thickness)

def visualize_tracking(dataset_path, sequence_name, results, fps=3):
    """Visualize tracking with both GT and predicted boxes"""
    
    # Find sequence directory
    seq_dir = Path(dataset_path) / sequence_name
    nested_dir = seq_dir / sequence_name
    
    if nested_dir.exists() and (nested_dir / 'img').exists():
        seq_dir = nested_dir
    elif not seq_dir.exists() or not (seq_dir / 'img').exists():
        print(f"Error: Sequence {sequence_name} not found!")
        return
    
    # Load images
    img_dir = seq_dir / 'img'
    img_files = sorted(glob.glob(str(img_dir / '*.jpg')))
    if not img_files:
        img_files = sorted(glob.glob(str(img_dir / '*.png')))
    
    # Load ground truth
    gt_boxes = load_ground_truth(seq_dir)
    if gt_boxes is None:
        print(f"Error: No ground truth found!")
        return
    
    print()
    print("=" * 70)
    print(f"Visualizing: {sequence_name}")
    print(f"Frames: {len(img_files)}")
    print(f"Display FPS: {fps}")
    print()
    print("Colors:")
    print("  GREEN  - Ground Truth")
    print("  BLUE   - Good Tracking (IoU > 0.5)")
    print("  ORANGE - Poor Tracking (IoU 0.3-0.5)")
    print("  RED    - Bad Tracking (IoU < 0.3)")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  Q     - Quit")
    print("=" * 70)
    print()
    
    delay = int(1000 / fps)
    paused = False
    
    # Create window
    window_name = f'Tracking: {sequence_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Statistics
    total_iou = 0.0
    rescue_count = 0
    csrt_count = 0
    
    print("Starting visualization...")
    print("-" * 70)
    
    for i, img_path in enumerate(img_files):
        frame_num = i + 1
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        display_frame = frame.copy()
        
        # Get ground truth
        if i < len(gt_boxes):
            gt_box = gt_boxes[i]
            
            # Draw ground truth (GREEN)
            draw_box(display_frame, gt_box, (0, 255, 0), "Ground Truth", thickness=2)
            
            # Get tracking result
            if frame_num in results:
                iou = results[frame_num]['iou']
                tracker_type = results[frame_num]['tracker']
                
                # Update statistics
                total_iou += iou
                
                # Determine tracker status
                if 'RESCUE' in tracker_type:
                    rescue_count += 1
                    if 'FAILED' in tracker_type:
                        tracker_status = "🔴 RESCUE FAILED"
                    else:
                        tracker_status = "🔴 RESCUE OK"
                else:
                    csrt_count += 1
                    tracker_status = "🟢 CSRT"
                
                # Estimate tracked box (approximate - in real code, parse actual box)
                tracked_box = estimate_tracked_box_from_iou(gt_box, iou)
                
                # Color based on IoU
                if iou < 0.3:
                    color = (0, 0, 255)  # Red - bad
                    status = "BAD"
                elif iou < 0.5:
                    color = (0, 165, 255)  # Orange - poor
                    status = "POOR"
                else:
                    color = (255, 0, 0)  # Blue - good
                    status = "GOOD"
                
                # Draw tracked box
                label = f"{tracker_type} (IoU: {iou:.3f})"
                draw_box(display_frame, tracked_box, color, label, thickness=2)
                
                # Calculate average IoU so far
                avg_iou = total_iou / frame_num
                
                # Display info on frame
                info1 = f"Frame: {frame_num}/{len(img_files)}"
                info2 = f"IoU: {iou:.3f} ({status}) | Tracker: {tracker_type}"
                info3 = f"Avg IoU: {avg_iou:.3f} | FPS: {fps}"
                
                cv2.putText(display_frame, info1, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, info2, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, info3, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Print to terminal with detailed status
                rescue_info = ""
                if 'FAILED' in tracker_type:
                    rescue_info = " ⚠️"
                elif 'SUCCESS' in tracker_type:
                    rescue_info = " ✅"
                    
                print(f"Frame {frame_num:3d}/{len(img_files)} | "
                      f"IoU: {iou:.4f} ({status:4s}) | "
                      f"Avg: {avg_iou:.4f} | "
                      f"{tracker_status:18s} {tracker_type:25s}{rescue_info}")
        
        # Show frame
        cv2.imshow(window_name, display_frame)
        
        # Handle keyboard
        while True:
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                print("\nQuitting...")
                return
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused at' if paused else 'Resumed from'} frame {frame_num}")
            elif not paused:
                break
    
    # Print summary
    print("-" * 70)
    print("\n📊 TRACKING SUMMARY:")
    print(f"  Total Frames:    {len(results)}")
    print(f"  Average IoU:     {total_iou / len(results):.4f} ({total_iou / len(results) * 100:.2f}%)")
    print(f"  CSRT Frames:     {csrt_count} ({csrt_count / len(results) * 100:.1f}%)")
    print(f"  Rescue Frames:   {rescue_count} ({rescue_count / len(results) * 100:.1f}%)")
    print("-" * 70)
    
    cv2.destroyAllWindows()
    print("\nVisualization completed!")

if __name__ == '__main__':
    # Paths
    dataset_path = Path(r"E:\SourceCode\C2P\Project\ECR_Tracking\otb100\OTB-dataset\OTB100")
    exe_path = Path(r"E:\SourceCode\C2P\Project\ECR_Tracking\build\ECR_Tracking_otb.exe")
    
    # Get sequence name
    if len(sys.argv) > 1:
        sequence_name = sys.argv[1]
    else:
        sequence_name = "Biker"
    
    # Get FPS
    fps = 3
    if len(sys.argv) > 2:
        try:
            fps = int(sys.argv[2])
        except:
            fps = 3
    
    # Output file for results
    output_file = f"tracking_results_{sequence_name}.txt"
    
    # Step 1: Run tracker and save results
    run_tracker_and_save_results(exe_path, dataset_path, sequence_name, output_file)
    
    # Step 2: Parse results
    results = parse_results(output_file)
    
    if not results:
        print("Error: No results found!")
        sys.exit(1)
    
    # Step 3: Visualize
    visualize_tracking(dataset_path, sequence_name, results, fps)
