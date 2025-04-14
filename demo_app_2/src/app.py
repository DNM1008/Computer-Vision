"""
YOLO Real-Time Object Detection with PyQt5 UI and RTSP Video Feed

This application captures video from an RTSP stream, processes each frame using two YOLO models:
1. A custom model for specialized object detection.
2. The YOLOv8m model to detect people.

Detections are rendered and displayed in a PyQt5 GUI with annotated bounding boxes, labels, and FPS information.
"""

import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from ultralytics import YOLO


# --- Worker Thread for Detection ---
class DetectionWorker(QThread):
    """
    Worker thread to capture video frames and perform detection using YOLO models.

    Attributes:
        frame_updated (pyqtSignal): Signal emitted with the annotated frame, elapsed time, and FPS.
    """

    frame_updated = pyqtSignal(np.ndarray, float, float)

    def __init__(self, source=0):
        """
        Initialize the detection worker with video source and models.

        Args:
            source (str|int): Path to the video stream or device index.
        """
        super().__init__()
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.running = True
        self.custom_model = YOLO("../conf/big_model.pt").to("cuda")
        self.people_model = YOLO("../conf/yolov8m.pt").to("cuda")
        self.PEOPLE_CLASS_ID = 0
        self.custom_class_names = self.custom_model.model.names

    def run(self):
        """
        Continuously reads video frames, performs detection, and emits annotated results.
        """
        while self.running and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            annotated = frame.copy()

            # Resize frame before detection (e.g., 640x640)
            resized_frame = cv2.resize(frame, (640, 640))

            # --- Custom Model Detection ---
            custom_results = self.custom_model.predict(
                resized_frame, conf=0.3, verbose=False
            )[0]
            special_boxes = custom_results.boxes
            special_detected = len(special_boxes) > 0

            for box in special_boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = f'{self.custom_class_names.get(class_id, f"Class {class_id}")}: {conf:.2f}'
                cv2.rectangle(annotated, xyxy[:2], xyxy[2:], (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    label,
                    (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            if special_detected:
                people_results = self.people_model.predict(
                    resized_frame, conf=0.4, verbose=False
                )[0]
                people_boxes = people_results.boxes
                people_filtered = people_boxes[people_boxes.cls == self.PEOPLE_CLASS_ID]

                for box in people_filtered:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(annotated, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        "Person",
                        xyxy[:2],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    annotated,
                    f"People Count: {len(people_filtered)} (YOLOv8m)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    annotated,
                    "Custom model: No objects detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Frame timing
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated,
                f"Frame Time: {elapsed*1000:.1f} ms | FPS: {fps:.1f}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            self.frame_updated.emit(annotated, elapsed, fps)

        self.cap.release()

    def stop(self):
        """
        Stop the detection thread.
        """
        self.running = False
        self.wait()


# --- PyQt Main Window ---
class VideoApp(QWidget):
    """
    PyQt5 application window that displays live YOLO detection from an RTSP stream.
    """

    def __init__(self):
        """
        Initialize the application UI and detection worker.
        """
        super().__init__()
        self.setWindowTitle("YOLO Live Detection")
        self.label = QLabel("Loading...")
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.worker = DetectionWorker(
            "rtsp://admin:Vcb!2025@192.168.1.100:554/profile2/media.smp"
        )
        self.worker.frame_updated.connect(self.update_frame)
        self.worker.start()

    def update_frame(self, frame, elapsed, fps):
        """
        Update the displayed video frame in the UI.

        Args:
            frame (np.ndarray): The annotated frame.
            elapsed (float): Time taken to process the frame.
            fps (float): Frames per second value.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """
        Handle window close event by stopping the detection thread.
        """
        self.worker.stop()
        event.accept()


# --- App Start ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec_())
