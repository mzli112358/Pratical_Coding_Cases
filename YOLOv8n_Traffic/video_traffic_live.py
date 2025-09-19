import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
from scipy import stats
import time


class VehicleTracker:
    def __init__(self, model_path='../models/yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracks = {}  # track_id -> {'positions': [], 'last_seen': frame_num, 'class': 'car'}
        self.next_id = 1
        self.max_disappeared = 30  # frames before removing track
        self.trajectory_length = 50  # max trajectory points to keep
        self.counting_line_y = None  # horizontal line for counting
        self.vehicle_count = 0
        self.passed_vehicles = set()  # track_ids that have crossed the line
        
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_tracks(self, detections, frame_num):
        """Update vehicle tracks with new detections"""
        # Get car detections
        cars = []
        for detection in detections:
            if detection['class'] == 'car' and detection['confidence'] > 0.5:
                center_x = int((detection['x1'] + detection['x2']) / 2)
                center_y = int((detection['y1'] + detection['y2']) / 2)
                cars.append((center_x, center_y, detection))
        
        # Update existing tracks or create new ones
        used_tracks = set()
        
        for car_x, car_y, detection in cars:
            best_track_id = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                if len(track['positions']) > 0:
                    last_pos = track['positions'][-1]
                    distance = self.calculate_distance((car_x, car_y), last_pos)
                    
                    if distance < 100 and distance < best_distance:  # max distance threshold
                        best_distance = distance
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['positions'].append((car_x, car_y))
                self.tracks[best_track_id]['last_seen'] = frame_num
                used_tracks.add(best_track_id)
            else:
                # Create new track
                track_id = self.next_id
                self.tracks[track_id] = {
                    'positions': [(car_x, car_y)],
                    'last_seen': frame_num,
                    'class': 'car'
                }
                self.next_id += 1
                used_tracks.add(track_id)
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if frame_num - track['last_seen'] > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def fit_trajectory_line(self, positions):
        """Fit a straight line to vehicle trajectory"""
        if len(positions) < 3:
            return None, None
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, y_coords)
            return slope, intercept
        except:
            return None, None
    
    def check_line_crossing(self, track_id, positions):
        """Check if vehicle crossed the counting line"""
        if self.counting_line_y is None or len(positions) < 2:
            return False
        
        # Check if vehicle crossed the line
        for i in range(1, len(positions)):
            y1, y2 = positions[i-1][1], positions[i][1]
            if (y1 < self.counting_line_y < y2) or (y2 < self.counting_line_y < y1):
                if track_id not in self.passed_vehicles:
                    self.passed_vehicles.add(track_id)
                    self.vehicle_count += 1
                    return True
        return False
    
    def draw_trajectories(self, frame):
        """Draw vehicle trajectories and regression lines"""
        for track_id, track in self.tracks.items():
            positions = track['positions']
            if len(positions) < 2:
                continue
            
            # Draw trajectory points
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i-1], positions[i], (0, 255, 255), 2)
            
            # Draw regression line
            slope, intercept = self.fit_trajectory_line(positions)
            if slope is not None and intercept is not None:
                h, w = frame.shape[:2]
                x1, x2 = 0, w
                y1 = int(slope * x1 + intercept)
                y2 = int(slope * x2 + intercept)
                
                # Only draw line if it's within frame bounds
                if 0 <= y1 < h and 0 <= y2 < h:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Draw track ID
            if len(positions) > 0:
                cv2.putText(frame, f"ID:{track_id}", 
                           (positions[-1][0], positions[-1][1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_counting_line(self, frame):
        """Draw the counting line"""
        if self.counting_line_y is not None:
            h, w = frame.shape[:2]
            cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y), (0, 0, 255), 3)
            cv2.putText(frame, "Counting Line", (10, self.counting_line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_video(video_path, model_path='../models/yolov8n.pt'):
    """Process video with vehicle tracking and counting"""
    # Initialize tracker
    tracker = VehicleTracker(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set counting line to middle of frame
    tracker.counting_line_y = height // 2
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print("Press 'q' to quit, 'r' to reset counting")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = tracker.model(frame)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = tracker.model.names[class_id]
                    
                    detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': confidence,
                        'class': class_name
                    })
        
        # Update tracks
        tracker.update_tracks(detections, frame_count)
        
        # Check line crossings
        for track_id, track in tracker.tracks.items():
            tracker.check_line_crossing(track_id, track['positions'])
        
        # Draw everything
        tracker.draw_trajectories(frame)
        tracker.draw_counting_line(frame)
        
        # Draw detection boxes
        for detection in detections:
            if detection['class'] == 'car':
                x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"car: {detection['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw vehicle count
        cv2.putText(frame, f"Vehicle Count: {tracker.vehicle_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(tracker.tracks)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display frame
        try:
            cv2.imshow('Traffic Analysis', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.vehicle_count = 0
                tracker.passed_vehicles.clear()
                print("Reset counting")
        except cv2.error as e:
            if "not implemented" in str(e) or "cvShowImage" in str(e):
                print("OpenCV GUI not available. Saving frames instead...")
                # Save frame every 30 frames (about 2.5 seconds at 12 FPS)
                if frame_count % 30 == 0:
                    output_path = f"traffic_analysis_frame_{frame_count:06d}.png"
                    cv2.imwrite(output_path, frame)
                    print(f"Saved frame: {output_path}")
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(0.1)
                
                # Check for keyboard interrupt
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'q':
                            break
                        elif key == b'r':
                            tracker.vehicle_count = 0
                            tracker.passed_vehicles.clear()
                            print("Reset counting")
                except:
                    pass
            else:
                raise e
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final vehicle count: {tracker.vehicle_count}")


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to script directory
    model_path = os.path.join(os.path.dirname(script_dir), 'models', 'yolov8n.pt')
    video_file = os.path.join(script_dir, "traffic_cctv.mp4")
    
    # First download the model if it doesn't exist
    if not os.path.exists(model_path):
        print("Downloading YOLOv8n model...")
        from download_yolov8n import download_yolov8n
        download_yolov8n()
    
    # Process video
    if os.path.exists(video_file):
        print(f"Processing video: {video_file}")
        process_video(video_file, model_path)
    else:
        print(f"Video file {video_file} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
