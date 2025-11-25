#!/usr/bin/env python3
"""
Simple visualization - just show ground truth boxes on images
"""

import cv2
import numpy as np
import glob
from pathlib import Path
import sys

def load_ground_truth(gt_file):
    """Load ground truth boxes"""
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

def visualize_sequence(dataset_path, sequence_name, fps=5):
    """Visualize sequence with ground truth"""
    
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
    if not img_files:
        print(f"Error: No images found in {img_dir}")
        return
    
    # Load ground truth
    gt_file = seq_dir / 'groundtruth_rect.txt'
    if not gt_file.exists():
        gt_file = seq_dir / 'groundtruth_rect.1.txt'
    
    gt_boxes = load_ground_truth(gt_file)
    if gt_boxes is None:
        print(f"Error: No ground truth found!")
        return
    
    print("=" * 60)
    print(f"Sequence: {sequence_name}")
    print(f"Frames: {len(img_files)}")
    print(f"Ground truth: {gt_boxes.shape}")
    print(f"Display FPS: {fps}")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  Q     - Quit")
    print("  S     - Save current frame")
    print("=" * 60)
    print()
    
    delay = int(1000 / fps)  # Convert FPS to milliseconds
    paused = False
    
    # Create window with specific flags
    window_name = f'Tracking: {sequence_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    for i, img_path in enumerate(img_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        display_frame = frame.copy()
        
        # Get ground truth box
        if i < len(gt_boxes):
            gt_box = gt_boxes[i]
            
            # Draw ground truth (GREEN)
            draw_box(display_frame, gt_box, (0, 255, 0), "Ground Truth", thickness=2)
            
            # Display info
            info = f"Frame: {i+1}/{len(img_files)} | Box: [{gt_box[0]:.0f}, {gt_box[1]:.0f}, {gt_box[2]:.0f}, {gt_box[3]:.0f}]"
            cv2.putText(display_frame, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow(window_name, display_frame)
        
        # Handle keyboard input
        while True:
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                print("\nQuitting...")
                return
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused at frame' if paused else 'Resumed from frame'} {i+1}")
            elif key == ord('s'):
                save_path = f"frame_{i+1:04d}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"Saved: {save_path}")
            elif not paused:
                break
    
    cv2.destroyAllWindows()
    print("\nVisualization completed!")

if __name__ == '__main__':
    # Paths
    dataset_path = r"E:\SourceCode\C2P\Project\ECR_Tracking\otb100\OTB-dataset\OTB100"
    
    # Get sequence name from command line or use default
    if len(sys.argv) > 1:
        sequence_name = sys.argv[1]
    else:
        sequence_name = "Basketball"
    
    # Get FPS from command line or use default
    fps = 5
    if len(sys.argv) > 2:
        try:
            fps = int(sys.argv[2])
        except:
            fps = 5
    
    # Visualize
    visualize_sequence(dataset_path, sequence_name, fps)
