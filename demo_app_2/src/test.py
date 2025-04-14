import time

import cv2
from ultralytics import YOLO

from tqdm import tqdm

# Timing accumulators
detected_times = []
not_detected_times = []
# Load models
custom_model = YOLO("../conf/big_model.pt")
people_model = YOLO("../conf/yolov8m")

# Define object classes
PEOPLE_CLASS_ID = 0  # YOLOv8 class ID for 'person'

# Video input/output
video_path = "../../videos/raw/01_16_2025 4_05_59 PM (UTC+07_00).mkv"
output_path = "../../videos/results/01_16_2025 4_05_59 PM_2_stages_full.mkv"


cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
custom_class_names = custom_model.model.names

# Progress bar
progress_bar = tqdm(desc="Processing video", unit="frame")

# Begin processing video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Run custom model
    custom_results = custom_model.predict(source=frame, conf=0.5, verbose=False)[0]
    special_boxes = custom_results.boxes
    special_filtered = [box for box in special_boxes]
    special_detected = len(special_filtered) > 0

    # Annotate frame
    annotated_frame = frame.copy()

    # Draw special object boxes with confidence
    for box in special_filtered:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = f'{custom_class_names.get(class_id, f"Class {class_id}")}: {conf:.2f}'
        cv2.rectangle(annotated_frame, xyxy[:2], xyxy[2:], (0, 0, 255), 2)
        cv2.putText(
            annotated_frame,
            label,
            (xyxy[0], xyxy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # If an object is detected
    if special_detected:
        people_results = people_model.predict(source=frame, conf=0.4, verbose=False)[0]
        people_boxes = people_results.boxes
        people_filtered = people_boxes[people_boxes.cls == PEOPLE_CLASS_ID]

        for box in people_filtered:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                "Person",
                xyxy[:2],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        people_count = len(people_filtered)
        cv2.putText(
            annotated_frame,
            f"People Count: {people_count} (YOLOv8m)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            annotated_frame,
            "Custom model: No objects detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Measure frame time
    frame_time = time.time() - start_time

    frame_time_ms = frame_time * 1000
    fps = 1000 / frame_time_ms if frame_time_ms > 0 else 0

    cv2.putText(
        annotated_frame,
        f"Frame Time: {frame_time_ms:.1f} ms | FPS: {fps:.1f}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    if special_detected:
        detected_times.append(frame_time)
    else:
        not_detected_times.append(frame_time)

    out.write(annotated_frame)
    progress_bar.update()

progress_bar.close()

cap.release()
out.release()
print("✅ Finished! Output saved to:", output_path)

# Convert seconds to milliseconds and calculate FPS
if detected_times:
    avg_detected_ms = sum(detected_times) / len(detected_times) * 1000
    fps_detected = 1000 / avg_detected_ms
    print(
        f"📈 Avg frame time (objects detected): {avg_detected_ms:.2f} ms ({fps_detected:.2f} FPS)"
    )
else:
    print("📈 No frames where any object was detected.")

if not_detected_times:
    avg_not_detected_ms = sum(not_detected_times) / len(not_detected_times) * 1000
    fps_not_detected = 1000 / avg_not_detected_ms
    print(
        f"📉 Avg frame time (no objects): {avg_not_detected_ms:.2f} ms ({fps_not_detected:.2f} FPS)"
    )
else:
    print("📉 No frames without objects.")


print("✅ Finished!")
