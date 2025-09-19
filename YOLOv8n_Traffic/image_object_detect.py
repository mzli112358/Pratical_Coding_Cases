import os
import cv2
import numpy as np
from ultralytics import YOLO
import glob


def detect_objects_and_crop_cars(image_path, model_path='../models/yolov8n.pt', output_dir='detections'):
    """
    Detect objects in image and crop cars specifically
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run detection
    results = model(image)
    
    # Get image name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Draw bounding boxes and crop cars
    annotated_image = image.copy()
    car_count = 0
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Crop and save cars
                if class_name == 'car':
                    car_count += 1
                    car_crop = image[y1:y2, x1:x2]
                    if car_crop.size > 0:
                        car_filename = os.path.join(output_dir, f"{base_name}_car_{car_count}.png")
                        cv2.imwrite(car_filename, car_crop)
                        print(f"Saved car crop: {car_filename}")
    
    # Save annotated image
    annotated_filename = os.path.join(output_dir, f"{base_name}_annotated.png")
    cv2.imwrite(annotated_filename, annotated_image)
    print(f"Saved annotated image: {annotated_filename}")
    print(f"Found {car_count} cars in {image_path}")


def process_all_frames(frames_dir='frames', model_path='../models/yolov8n.pt', output_dir='detections'):
    """Process all extracted frames"""
    frame_files = glob.glob(os.path.join(frames_dir, "test*.png"))
    frame_files.sort()
    
    print(f"Processing {len(frame_files)} frames...")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    for frame_file in frame_files:
        print(f"\nProcessing: {frame_file}")
        detect_objects_and_crop_cars(frame_file, model_path, output_dir)


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to script directory
    model_path = os.path.join(os.path.dirname(script_dir), 'models', 'yolov8n.pt')
    
    # First download the model if it doesn't exist
    if not os.path.exists(model_path):
        print("Downloading YOLOv8n model...")
        from download_yolov8n import download_yolov8n
        download_yolov8n()
    
    # Process all frames
    frames_dir = os.path.join(script_dir, 'frames')
    output_dir = os.path.join(script_dir, 'detections')
    
    print(f"Script directory: {script_dir}")
    print(f"Frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")
    
    process_all_frames(frames_dir=frames_dir, model_path=model_path, output_dir=output_dir)
