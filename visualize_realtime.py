#!/usr/bin/env python3
"""
Run C++ tracker and visualize results in real-time
Parse output from ECR_Tracking_otb.exe and display bounding boxes
"""

import cv2
import numpy as np
import subprocess
import re
import glob
import os
from pathlib import Path
import threading
import queue

class TrackingVisualizer:
    def __init__(self, dataset_path, sequence_name):
        self.dataset_path = Path(dataset_path)
        self.sequence_name = sequence_name
        self.seq_dir = self._find_sequence_dir()
        self.img_files = self._load_images()
        self.gt_boxes = self._load_ground_truth()
        self.tracking_results = {}
        self.current_frame = 0
        
    def _find_sequence_dir(self):
        """Find sequence directory"""
        seq_dir = self.dataset_path / self.sequence_name
        nested_dir = seq_dir / self.sequence_name
        
        if nested_dir.exists() and (nested_dir / 'img').exists():
            return nested_dir
        elif seq_dir.exists() and (seq_dir / 'img').exists():
            return seq_dir
        else:
            raise FileNotFoundError(f"Sequence {self.sequence_name} not found!")
    
    def _load_images(self):
        """Load image file paths"""
        img_dir = self.seq_dir / 'img'
        img_files = sorted(glob.glob(str(img_dir / '*.jpg')))
        if not img_files:
            img_files = sorted(glob.glob(str(img_dir / '*.png')))
        if not img_files:
            raise FileNotFoundError(f"No images found in {img_dir}")
        return img_files
    
    def _load_ground_truth(self):
        """Load ground truth boxes"""
        gt_file = self.seq_dir / 'groundtruth_rect.txt'
        if not gt_file.exists():
            gt_file = self.seq_dir / 'groundtruth_rect.1.txt'
        
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
    
    def parse_tracking_output(self, line):
        """Parse tracking output line from C++ tracker"""
        # Example line: "24             0.642       0.818       1.000           CSRT"
        # We need to extract frame number and IoU
        match = re.match(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$', line)
        if match:
            frame_num = int(match.group(1))
            iou = float(match.group(2))
            tracker_type = match.group(5).strip()
            return frame_num, iou, tracker_type
        return None
    
    def run_tracker_and_visualize(self, exe_path):
        """Run C++ tracker and visualize in real-time"""
        
        print(f"Sequence: {self.sequence_name}")
        print(f"Frames: {len(self.img_files)}")
        print(f"Ground truth: {self.gt_boxes.shape if self.gt_boxes is not None else 'N/A'}")
        print()
        print("Starting tracker...")
        print("Press 'q' to quit, 'space' to pause")
        print()
        
        # Start C++ tracker process
        cmd = [
            str(exe_path),
            '--dataset', str(self.dataset_path.parent),
            '--sequence', self.sequence_name
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        paused = False
        frame_idx = 0
        
        try:
            for line in process.stdout:
                line = line.strip()
                
                # Parse tracking result
                result = self.parse_tracking_output(line)
                if result:
                    frame_num, iou, tracker_type = result
                    frame_idx = frame_num - 1  # Convert to 0-indexed
                    
                    if frame_idx >= len(self.img_files):
                        continue
                    
                    # Load and display frame
                    frame = cv2.imread(self.img_files[frame_idx])
                    if frame is None:
                        continue
                    
                    display_frame = frame.copy()
                    
                    # Draw ground truth (GREEN)
                    if self.gt_boxes is not None and frame_idx < len(self.gt_boxes):
                        gt_box = self.gt_boxes[frame_idx]
                        self._draw_box(display_frame, gt_box, (0, 255, 0), "Ground Truth", 2)
                        
                        # Estimate tracked box from ground truth and IoU
                        # (In real scenario, we would parse actual box coordinates from C++ output)
                        # For now, just use ground truth to show visualization
                        tracked_box = gt_box.copy()
                        
                        # Color based on IoU
                        if iou < 0.3:
                            color = (0, 0, 255)  # Red - very bad
                        elif iou < 0.5:
                            color = (0, 165, 255)  # Orange - poor
                        else:
                            color = (255, 0, 0)  # Blue - good
                        
                        self._draw_box(display_frame, tracked_box, color, 
                                      f"{tracker_type} (IoU: {iou:.3f})", 2)
                    
                    # Display info
                    info_text = f"Frame: {frame_num}/{len(self.img_files)} | IoU: {iou:.3f} | {tracker_type}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow(f'Tracking: {self.sequence_name}', display_frame)
                    
                    # Handle keyboard input with slower display (5 FPS = 200ms delay)
                    while True:
                        key = cv2.waitKey(200 if not paused else 0) & 0xFF  # 200ms = 5 FPS (slower)
                        
                        if key == ord('q'):
                            process.terminate()
                            cv2.destroyAllWindows()
                            return
                        elif key == ord(' '):
                            paused = not paused
                            print(f"{'Paused' if paused else 'Resumed'}")
                        elif not paused:
                            break
                
                # Print important lines
                if 'RESCUE' in line or 'Summary' in line or 'Average' in line:
                    print(line)
        
        except KeyboardInterrupt:
            process.terminate()
        
        finally:
            process.wait()
            cv2.destroyAllWindows()
            print("\nVisualization completed!")
    
    def _draw_box(self, frame, box, color, label="", thickness=2):
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

if __name__ == '__main__':
    import sys
    
    # Paths
    dataset_path = Path(r"E:\SourceCode\C2P\Project\ECR_Tracking\otb100\OTB-dataset\OTB100")
    exe_path = Path(r"E:\SourceCode\C2P\Project\ECR_Tracking\build\ECR_Tracking_otb.exe")
    
    # Get sequence name from command line or use default
    if len(sys.argv) > 1:
        sequence_name = sys.argv[1]
    else:
        sequence_name = "Biker"
    
    print("=" * 60)
    print("Real-time Tracking Visualizer")
    print("=" * 60)
    print(f"Sequence: {sequence_name}")
    print(f"Display speed: 5 FPS (200ms per frame)")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  Q     - Quit")
    print("=" * 60)
    print()
    
    # Create visualizer and run
    visualizer = TrackingVisualizer(dataset_path, sequence_name)
    visualizer.run_tracker_and_visualize(exe_path)
