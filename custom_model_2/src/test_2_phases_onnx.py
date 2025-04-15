"""
Mock test of 2 phase model


"""

import cv2
import time
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

# Constants
big_model = "../data/results/big_model_11/weights/best_m.onnx"
people_counter_model = YOLO("../conf/source_model/yolo11m.pt")
ALERT_CLASSES = ["cassette", "atm"]
PEOPLE_CLASS = "person"
PEOPLE_THRESHOLD = 3
ALERT_FRAME_COUNT = 5
PEOPLE_COUNT_TIMEOUT = 2  # seconds


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


def preprocess_frame_for_onnx(frame, input_shape=(640, 640)):
    resized = cv2.resize(frame, input_shape)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(transposed, axis=0)  # Add batch dim
    return input_tensor


def process_video(input_video_path, output_video_path, model_path):
    # ONNX Runtime session
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

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

    over_threshold_counter = 0
    draw_red_border = False
    counting_active = False
    both_last_seen_time = None
    alert_frame_time = 0.0
    alert_frame_count = 0
    empty_frame_time = 0.0
    empty_frame_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        current_time = time.time()

        # Preprocess and run ONNX model
        input_tensor = preprocess_frame_for_onnx(frame)
        outputs = session.run(None, {input_name: input_tensor})

        # You may need to decode outputs depending on your model
        # Here we assume you're using a YOLO-like model; if you used Ultralytics export, use:
        # boxes, scores, classes = decode_yolo_outputs(outputs[0])

        # You need to modify this part based on your ONNX model output format
        detected_classes = []  # Placeholder
        # TODO: decode ONNX output and populate `detected_classes`

        detected = all(cls in detected_classes for cls in ALERT_CLASSES)

        if detected:
            counting_active = True
            both_last_seen_time = current_time
        elif counting_active:
            if both_last_seen_time is not None and (
                current_time - both_last_seen_time > PEOPLE_COUNT_TIMEOUT
            ):
                counting_active = False
                over_threshold_counter = 0

        person_count = 0
        person_boxes = []

        if counting_active:
            person_results = people_counter_model(frame)
            for box, cls_id in zip(
                person_results[0].boxes.xyxy, person_results[0].boxes.cls
            ):
                class_name = person_results[0].names[int(cls_id)]
                if class_name == PEOPLE_CLASS:
                    person_boxes.append(box.cpu().numpy())
                    person_count += 1

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

        # Annotate frame manually (since plot() is not available here)
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if counting_active:
            cv2.putText(
                frame,
                f"People: {person_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if draw_red_border:
            cv2.rectangle(
                frame,
                (0, 0),
                (frame_width - 1, frame_height - 1),
                (0, 0, 255),
                10,
            )

        status_text = "No triggers detected"
        model_status = "Running: big_model (ONNX)"

        if counting_active:
            status_text = "Counting people"
            model_status += " + people_counter"

        draw_text_with_background(
            frame, status_text, (10, frame_height - 60), bg_color=(0, 100, 0)
        )
        cv2.putText(
            frame,
            model_status,
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame_processing_time_ms = (time.time() - frame_start_time) * 1000
        draw_text_with_background(
            frame,
            f"{frame_processing_time_ms:.1f} ms",
            (frame_width - 180, 35),
            bg_color=(0, 0, 0),
        )

        out.write(frame)
        print(
            f"Frame {frame_index} written | People: {person_count} | Counting: {counting_active}"
        )
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved as {output_video_path}")
    if empty_frame_count > 0:
        print(
            f"Avg idle frame time: {empty_frame_time * 1000 / empty_frame_count:.4f} ms"
        )
    if alert_frame_count > 0:
        print(
            f"Avg alert frame time: {alert_frame_time * 1000 / alert_frame_count:.4f} ms"
        )


input_video = "../../videos/raw/01_16_2025 4_05_59 PM (UTC+07_00).mkv"
output_video = (
    "../../videos/results/01_16_2025 4_05_59 PM (UTC+07_00)_2_phases_yolo_11_m.mp4"
)
process_video(input_video, output_video, big_model)
