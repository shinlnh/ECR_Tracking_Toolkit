#!/usr/bin/env python3
"""
Visualize tracking results with bounding boxes
"""

import cv2
import numpy as np
import glob
import os
import sys
from pathlib import Path

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

def load_ground_truth(gt_file):
    """Load ground truth boxes"""
    if not os.path.exists(gt_file):
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

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def visualize_sequence(dataset_path, sequence_name, output_dir=None, 
                      save_video=False, show_frames=True):
    """Visualize tracking on a sequence"""
    
    # Find sequence directory (handle nested structure like Biker/Biker/)
    seq_dir = Path(dataset_path) / sequence_name
    nested_dir = seq_dir / sequence_name
    
    # Check if nested structure exists
    if nested_dir.exists() and (nested_dir / 'img').exists():
        seq_dir = nested_dir
    elif not seq_dir.exists() or not (seq_dir / 'img').exists():
        print(f"Error: Sequence {sequence_name} not found or no img directory!")
        print(f"Tried: {seq_dir} and {nested_dir}")
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
        print(f"Warning: No ground truth found!")
    
    print(f"Sequence: {sequence_name}")
    print(f"Frames: {len(img_files)}")
    print(f"Ground truth: {gt_boxes.shape if gt_boxes is not None else 'N/A'}")
    print()
    
    # Setup video writer if needed
    video_writer = None
    if save_video:
        if output_dir is None:
            output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        first_frame = cv2.imread(img_files[0])
        h, w = first_frame.shape[:2]
        video_path = os.path.join(output_dir, f"{sequence_name}_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        print(f"Saving video to: {video_path}")
    
    # Initialize tracker (simple IoU-based simulation)
    # In real scenario, this would be replaced with actual C++ tracker output
    tracked_boxes = []
    
    print("Press 'q' to quit, 'space' to pause, 's' to save frame")
    paused = False
    
    for i, img_path in enumerate(img_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        display_frame = frame.copy()
        
        # Get ground truth box
        if gt_boxes is not None and i < len(gt_boxes):
            gt_box = gt_boxes[i]
            
            # Draw ground truth (GREEN)
            draw_box(display_frame, gt_box, (0, 255, 0), "Ground Truth", thickness=2)
            
            # Simulate tracking box (for demo - in real code, read from C++ output)
            # For now, just show ground truth
            if i == 0:
                tracked_box = gt_box.copy()
            else:
                # Simple simulation: add some noise to show difference
                # In real implementation, this would be actual tracking result
                tracked_box = gt_box.copy()
                # tracked_box[0] += np.random.randint(-5, 5)
                # tracked_box[1] += np.random.randint(-5, 5)
            
            tracked_boxes.append(tracked_box)
            
            # Draw tracked box (RED)
            iou = calculate_iou(tracked_box, gt_box)
            color = (0, 0, 255) if iou < 0.5 else (255, 0, 0)  # Red if poor, Blue if good
            draw_box(display_frame, tracked_box, color, f"Track (IoU: {iou:.3f})", thickness=2)
            
            # Display info
            info_text = f"Frame: {i+1}/{len(img_files)} | IoU: {iou:.3f}"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save to video
        if video_writer is not None:
            video_writer.write(display_frame)
        
        # Display frame
        if show_frames:
            cv2.imshow(f'Tracking: {sequence_name}', display_frame)
            
            while True:
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    if video_writer is not None:
                        video_writer.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    save_path = f"frame_{i+1:04d}.jpg"
                    cv2.imwrite(save_path, display_frame)
                    print(f"Saved: {save_path}")
                elif not paused:
                    break
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved successfully!")
    
    cv2.destroyAllWindows()
    print("\nVisualization completed!")

if __name__ == '__main__':
    # Default paths
    dataset_path = r"E:\SourceCode\C2P\Project\ECR_Tracking\otb100\OTB-dataset\OTB100"
    
    # Get sequence name from command line or use default
    if len(sys.argv) > 1:
        sequence_name = sys.argv[1]
    else:
        sequence_name = "Biker"  # Default to Biker
    
    # Visualize
    visualize_sequence(
        dataset_path=dataset_path,
        sequence_name=sequence_name,
        output_dir="output",
        save_video=True,  # Save video
        show_frames=True   # Show real-time display
    )
