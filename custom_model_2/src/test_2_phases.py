"""
Mock test of 2 phase model


"""

import cv2
import time
from ultralytics import YOLO

# Constants
big_model = "../data/results/big_model_11/weights/best_m.pt"
people_counter_model = YOLO("../conf/source_model/yolo11m.pt")
ALERT_CLASSES = ["cassette", "atm"]
PEOPLE_CLASS = "person"
PEOPLE_THRESHOLD = 2
ALERT_FRAME_COUNT = 5
PEOPLE_COUNT_TIMEOUT = 5  # seconds


def draw_text_with_background(
    img,
    text,
    position,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.8,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=2,
    padding=5,
):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = position
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + padding),
        bg_color,
        -1,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def process_video(input_video_path, output_video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = output_video_path.replace(".mkv", ".mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: VideoWriter failed to initialize.")
        return

    # State
    over_threshold_counter = 0
    detected_counter = 0
    draw_red_border = False
    counting_active = False
    both_last_seen_time = None

    # Timing trackers
    alert_frame_time = 0.0
    alert_frame_count = 0
    empty_frame_time = 0.0
    empty_frame_count = 0

    frame_index = 0

    # Frame time tracking
    frame_processing_time_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        current_time = time.time()

        results = model(frame, conf=0.2)
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

        # Detection flags
        detected = all(cls in detected_classes for cls in ALERT_CLASSES)

        # Update counting status
        if detected:
            detected_counter += 1
            if detected_counter >= 20:
                counting_active = True
                both_last_seen_time = current_time
        elif counting_active:
            if both_last_seen_time is not None and (
                current_time - both_last_seen_time > PEOPLE_COUNT_TIMEOUT
            ):
                counting_active = False
                over_threshold_counter = 0
                detected_counter = 0
        else:
            detected_counter = 0

        person_count = 0
        person_boxes = []

        if counting_active:
            person_results = people_counter_model(frame, conf=0.3)
            for box, cls_id in zip(
                person_results[0].boxes.xyxy, person_results[0].boxes.cls
            ):
                class_name = person_results[0].names[int(cls_id)]
                if class_name == PEOPLE_CLASS:
                    person_boxes.append(box.cpu().numpy())
                    person_count += 1

            # Threshold logic
            if person_count != PEOPLE_THRESHOLD:
                over_threshold_counter += 1
            else:
                over_threshold_counter = 0

            draw_red_border = over_threshold_counter > ALERT_FRAME_COUNT
            alert_frame_time += time.time() - frame_start_time
            alert_frame_count += 1
        else:
            draw_red_border = False
            empty_frame_time += time.time() - frame_start_time
            empty_frame_count += 1

        # Annotate main model detections
        annotated_frame = results[0].plot()

        # Calculate processing time for current frame
        frame_processing_time_ms = (time.time() - frame_start_time) * 1000

        # Draw person boxes
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw person count
        if counting_active:
            cv2.putText(
                annotated_frame,
                f"People: {person_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Red border
        if draw_red_border:
            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (frame_width - 1, frame_height - 1),
                (0, 0, 255),
                10,
            )

        # Define overlay text
        status_text = "No triggers detected"
        model_status = "Running: big_model"

        if counting_active:
            status_text = "Counting people"
            model_status += " + people_counter"

        # Draw status text
        draw_text_with_background(
            annotated_frame, status_text, (10, frame_height - 60), bg_color=(0, 100, 0)
        )

        # Draw model info
        cv2.putText(
            annotated_frame,
            model_status,
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw frame processing time
        draw_text_with_background(
            annotated_frame,
            f"{frame_processing_time_ms:.1f} ms",
            (frame_width - 180, 35),
            bg_color=(0, 0, 0),
        )

        out.write(annotated_frame)

        print(
            f"Frame {frame_index} written | People: {person_count} | Counting: {counting_active}"
        )
        frame_index += 1

    # Release
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved as {output_video_path}")

    # Timing summary
    if empty_frame_count > 0:
        print(
            f"Avg idle frame time: {empty_frame_time * 1000 /
                                      empty_frame_count:.4f} ms"
        )
    if alert_frame_count > 0:
        print(
            f"Avg alert frame time: {alert_frame_time * 1000 /
                                       alert_frame_count:.4f} ms"
        )


input_video = "../../videos/raw/source_3.mp4"
output_video = "../../videos/results/test_7_2_phases_yolo_11_m.mp4"
process_video(input_video, output_video, big_model)
