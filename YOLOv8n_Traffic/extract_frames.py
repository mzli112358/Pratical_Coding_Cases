import os
import cv2


def extract_one_frame_per_minute(video_path: str, output_dir: str) -> None:

	if not os.path.exists(video_path):
		raise FileNotFoundError(f"Video not found: {video_path}")

	os.makedirs(output_dir, exist_ok=True)

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 0
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	if fps <= 0:
		cap.release()
		raise RuntimeError("Unable to read FPS from video.")

	frames_per_minute = int(round(fps * 60))
	if frames_per_minute <= 0:
		frames_per_minute = int(fps) if fps > 0 else 1

	frame_indices = list(range(0, frame_count, frames_per_minute))
	if len(frame_indices) == 0:
		frame_indices = [0]

	for idx, frame_index in enumerate(frame_indices, start=1):
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
		success, frame = cap.read()
		if not success or frame is None:
			continue
		filename = os.path.join(output_dir, f"test{idx}.png")
		cv2.imwrite(filename, frame)

	cap.release()


if __name__ == "__main__":
	# Get the directory where this script is located
	script_dir = os.path.dirname(os.path.abspath(__file__))
	
	# Define paths relative to script directory
	video = os.path.join(script_dir, "traffic_cctv.mp4")
	output = os.path.join(script_dir, "frames")
	
	print(f"Looking for video: {video}")
	print(f"Output directory: {output}")
	
	extract_one_frame_per_minute(video, output)
	print(f"Saved frames to: {os.path.abspath(output)}")


