#!/usr/bin/env python3
"""
Real-time Person Detection and Tracking using YOLOv8m
Detects persons, tracks individuals, and logs time spent in frame.
"""

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import sys
import numpy as np


class PersonTracker:
    """Tracks individual persons across frames"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_person_id = 1
        self.persons = {}  # person_id: {centroid, first_seen, last_seen, total_time}
        self.disappeared = {}  # person_id: frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new person"""
        person_id = self.next_person_id
        self.persons[person_id] = {
            'centroid': centroid,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'total_time': 0
        }
        self.disappeared[person_id] = 0
        self.next_person_id += 1
        return person_id
    
    def deregister(self, person_id):
        """Deregister a person who has left the frame"""
        person_info = self.persons.pop(person_id)
        self.disappeared.pop(person_id)
        return person_info
    
    def update(self, detections):
        """
        Update tracked persons with new detections
        detections: list of bounding boxes [(x1, y1, x2, y2), ...]
        Returns: dict of {person_id: (x1, y1, x2, y2)}
        """
        current_time = time.time()
        
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            person_ids_to_remove = []
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    person_ids_to_remove.append(person_id)
            
            return {}, person_ids_to_remove
        
        # Calculate centroids for new detections
        input_centroids = []
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
        
        # If no persons are being tracked, register all detections
        if len(self.persons) == 0:
            result = {}
            for i, (cx, cy) in enumerate(input_centroids):
                person_id = self.register((cx, cy))
                result[person_id] = detections[i]
            return result, []
        
        # Match existing persons to new detections
        person_ids = list(self.persons.keys())
        existing_centroids = [self.persons[pid]['centroid'] for pid in person_ids]
        
        # Calculate distance matrix
        distances = np.zeros((len(existing_centroids), len(input_centroids)))
        for i, (ex, ey) in enumerate(existing_centroids):
            for j, (ix, iy) in enumerate(input_centroids):
                distances[i, j] = np.sqrt((ex - ix) ** 2 + (ey - iy) ** 2)
        
        # Match persons using minimum distance
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        result = {}
        
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            
            if distances[row, col] > self.max_distance:
                continue
            
            person_id = person_ids[row]
            self.persons[person_id]['centroid'] = input_centroids[col]
            self.persons[person_id]['last_seen'] = current_time
            self.persons[person_id]['total_time'] = current_time - self.persons[person_id]['first_seen']
            self.disappeared[person_id] = 0
            
            result[person_id] = detections[col]
            used_rows.add(row)
            used_cols.add(col)
        
        # Register new persons for unmatched detections
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            person_id = self.register(input_centroids[col])
            result[person_id] = detections[col]
        
        # Mark disappeared persons
        unused_rows = set(range(len(existing_centroids))) - used_rows
        person_ids_to_remove = []
        for row in unused_rows:
            person_id = person_ids[row]
            self.disappeared[person_id] += 1
            if self.disappeared[person_id] > self.max_disappeared:
                person_ids_to_remove.append(person_id)
        
        return result, person_ids_to_remove


def format_duration(seconds):
    """Format duration in seconds to readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} min {secs:.2f} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hr {minutes} min {secs:.2f} sec"


def main():
    # Initialize YOLOv8m model
    print("Loading YOLOv8m model...")
    model = YOLO('yolov8m.pt')
    
    # Initialize person tracker
    tracker = PersonTracker(max_disappeared=10, max_distance=100)
    
    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    print("Webcam opened successfully")
    print("Press 'q' to quit")
    
    # Open log files
    detection_log = open('person_detections.txt', 'a')
    tracking_log = open('person_tracking.txt', 'a')
    
    session_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detection_log.write(f"\n=== Detection Session Started at {session_start} ===\n")
    tracking_log.write(f"\n=== Tracking Session Started at {session_start} ===\n")
    
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
            results = model(frame, conf=0.5, classes=[0], verbose=False)
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((int(x1), int(y1), int(x2), int(y2)))
            
            # Update tracker
            tracked_persons, departed_persons = tracker.update(detections)
            
            # Log departed persons
            for person_id in departed_persons:
                person_info = tracker.deregister(person_id)
                duration = person_info['total_time']
                first_seen = datetime.fromtimestamp(person_info['first_seen']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                last_seen = datetime.fromtimestamp(person_info['last_seen']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                log_entry = (f"Person ID {person_id} left the frame\n"
                           f"  First seen: {first_seen}\n"
                           f"  Last seen: {last_seen}\n"
                           f"  Duration: {format_duration(duration)}\n\n")
                tracking_log.write(log_entry)
                tracking_log.flush()
                print(f"Person {person_id} departed - Duration: {format_duration(duration)}")
            
            # Draw bounding boxes and labels
            for person_id, (x1, y1, x2, y2) in tracked_persons.items():
                # Draw green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate current duration
                duration = time.time() - tracker.persons[person_id]['first_seen']
                
                # Add label with person ID and duration
                label = f"Person {person_id} ({duration:.1f}s)"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add person count to frame
            person_count = len(tracked_persons)
            count_text = f"Persons in frame: {person_count}"
            cv2.putText(frame, count_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add timestamp to frame
            timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            cv2.putText(frame, timestamp_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Log detection to file
            log_entry = f"{timestamp_text} - Persons in frame: {person_count}\n"
            detection_log.write(log_entry)
            detection_log.flush()
            
            # Display the frame
            cv2.imshow('Person Detection & Tracking', frame)
            
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
                actual_interval_ms = elapsed_time * 1000
                if actual_interval_ms > target_interval * 1000 * 1.5:
                    print(f"Warning: Loop took {actual_interval_ms:.2f}ms (target: 3ms)")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Log all remaining tracked persons
        print("\nLogging remaining persons...")
        for person_id in list(tracker.persons.keys()):
            person_info = tracker.persons[person_id]
            duration = time.time() - person_info['first_seen']
            first_seen = datetime.fromtimestamp(person_info['first_seen']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            log_entry = (f"Person ID {person_id} was in frame at session end\n"
                       f"  First seen: {first_seen}\n"
                       f"  Duration: {format_duration(duration)}\n\n")
            tracking_log.write(log_entry)
            print(f"Person {person_id} - Duration: {format_duration(duration)}")
        
        # Cleanup
        session_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detection_log.write(f"=== Detection Session Ended at {session_end} ===\n\n")
        tracking_log.write(f"=== Tracking Session Ended at {session_end} ===\n\n")
        
        detection_log.close()
        tracking_log.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released successfully")


if __name__ == "__main__":
    main()
