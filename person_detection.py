#!/usr/bin/env python3
"""
Real-time Person Detection using YOLOv8m
Detects persons in video frames, draws bounding boxes, and logs detections to file.
"""

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import sys


def main():
    # Initialize YOLOv8m model
    print("Loading YOLOv8m model...")
    model = YOLO('yolov8m.pt')
    
    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    print("Webcam opened successfully")
    print("Press 'q' to quit")
    
    # Open log file for writing detections
    log_file = open('person_detections.txt', 'a')
    log_file.write(f"\n=== Detection Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # Target loop interval (3 milliseconds = 0.003 seconds)
    target_interval = 0.003
    
    try:
        while True:
            loop_start_time = time.time()
            
            # Capture frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Run YOLOv8 inference on the frame
            # conf=0.5 sets confidence threshold to 50%
            # classes=0 filters for person class only (class 0 in COCO dataset)
            results = model(frame, conf=0.5, classes=[0], verbose=False)
            
            # Process results
            person_count = 0
            
            for result in results:
                # Get bounding boxes for detected persons
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence score
                    confidence = float(box.conf[0])
                    
                    # Draw green rectangle around person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"Person {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    person_count += 1
            
            # Add person count to frame
            count_text = f"Persons detected: {person_count}"
            cv2.putText(frame, count_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add timestamp to frame
            timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            cv2.putText(frame, timestamp_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Log detection to file
            log_entry = f"{timestamp_text} - Persons detected: {person_count}\n"
            log_file.write(log_entry)
            log_file.flush()  # Ensure immediate write to disk
            
            # Display the frame
            cv2.imshow('Person Detection', frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
            
            # Calculate time to sleep to maintain 3ms loop interval
            elapsed_time = time.time() - loop_start_time
            sleep_time = target_interval - elapsed_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If processing took longer than 3ms, log it
                actual_interval_ms = elapsed_time * 1000
                if actual_interval_ms > target_interval * 1000 * 1.5:  # More than 1.5x target
                    print(f"Warning: Loop took {actual_interval_ms:.2f}ms (target: 3ms)")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        log_file.write(f"=== Detection Session Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        log_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released successfully")


if __name__ == "__main__":
    main()
