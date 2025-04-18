"""
Mock test of 2 phase model


"""

import cv2
import time
from ultralytics import YOLO
from tqdm import tqdm
import csv

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # Logging results to csv
    csv_file = open("../conf/frame_object_count.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame_id",
            "atm_count",
            "cassette_count",
            "person_count",
            "counting_active",
            "compliance",
        ]
    )

    # Preparing for loop
    frame_index = 0
    compliance = None

    # Frame time tracking
    frame_processing_time_ms = 0

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        current_time = time.time()

        results = model(frame, conf=0.05)

        # Class-specific confidence thresholds
        class_conf_thresholds = {
            "atm": 0.7,
            "cassette": 0.2,
        }

        # Filter boxes based on custom thresholds
        filtered_boxes = []
        filtered_classes = []

        for box, cls_id, conf in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        ):
            class_name = model.names[int(cls_id)]
            if class_name in ALERT_CLASSES:
                if conf >= class_conf_thresholds.get(class_name, 0.2):
                    filtered_boxes.append(box)
                    filtered_classes.append(class_name)

        # Update detected_classes to reflect filtered ones
        detected_classes = filtered_classes

        # Logging object detection
        atm_count = detected_classes.count("atm")
        cassette_count = detected_classes.count("cassette")

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
            compliance = not draw_red_border
            alert_frame_time += time.time() - frame_start_time
            alert_frame_count += 1
        else:
            draw_red_border = False
            empty_frame_time += time.time() - frame_start_time
            empty_frame_count += 1

        # Clone original names
        annotated_frame = frame.copy()  # Start with original frame

        for box, class_name in zip(filtered_boxes, filtered_classes):
            cls_id = list(model.names.keys())[
                list(model.names.values()).index(class_name)
            ]
            conf = (
                results[0].boxes.conf[results[0].boxes.cls == cls_id][0].item()
            )  # Gets the confidence for drawing

            x1, y1, x2, y2 = map(int, box)

            label = "opened atm" if class_name == "atm" else class_name
            label_text = f"{label} {conf:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1
            )
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

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
            # Alternate red and green border every other frame
            border_color = (0, 0, 255) if frame_index % 2 == 0 else (0, 255, 0)

            # Draw blinking border
            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (frame_width - 1, frame_height - 1),
                border_color,
                10,
            )

            # Add "Compliance Error!" label in the center
            label_text = "Compliance Error!"
            font_scale = 2.0
            thickness = 4
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_width, text_height), _ = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            center_x = (frame_width - text_width) // 2
            center_y = (frame_height + text_height) // 2

            # Background for text
            cv2.rectangle(
                annotated_frame,
                (center_x - 20, center_y - text_height - 20),
                (center_x + text_width + 20, center_y + 20),
                (0, 0, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label_text,
                (center_x, center_y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
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

        if atm_count > 0 or cassette_count > 0 or person_count > 0:
            csv_writer.writerow(
                [
                    frame_index,
                    atm_count,
                    cassette_count,
                    person_count,
                    counting_active,
                    compliance,
                ]
            )

        frame_index += 1

    # Release
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    csv_file.close()

    print(f"Output video saved as {output_video_path}.")
    print("Object count logged.")

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


input_video = "../../videos/raw/source_full.mp4"
output_video = "../../videos/results/results_full_custom_conf.mp4"
process_video(input_video, output_video, big_model)
