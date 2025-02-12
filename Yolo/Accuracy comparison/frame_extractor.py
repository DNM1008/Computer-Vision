import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    success, frame = video_capture.read()

    while success:
        # Save the current frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")

        # Read the next frame
        success, frame = video_capture.read()
        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Finished extracting {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    video_file = "video.mp4"
    output_directory = "output/screenshots"
    extract_frames(video_file, output_directory)
