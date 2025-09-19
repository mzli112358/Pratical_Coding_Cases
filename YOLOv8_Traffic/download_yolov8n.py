import os
from ultralytics import YOLO


def download_yolov8n():
    """Download YOLOv8n model to current directory"""
    try:
        # Download YOLOv8n model
        model = YOLO('yolov8n.pt')
        print(f"YOLOv8n model downloaded successfully!")
        print(f"Model saved to: {os.path.abspath('yolov8n.pt')}")
        return model
    except Exception as e:
        print(f"Error downloading YOLOv8n model: {e}")
        return None


if __name__ == "__main__":
    model = download_yolov8n()
