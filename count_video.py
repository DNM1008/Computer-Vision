import json
import sys
import time
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
)
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


class StreamRedirect:
    """
    Redirects stdout and stderr to a QTextEdit widget for live logging.

    Attributes:
        text_widget (QTextEdit): The widget where the console output is displayed.
    """

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        """
        Initializes the stream redirector.

        Args:
            text_widget (QTextEdit): The widget to display redirected console output.
        """
        self.text_widget.moveCursor(self.text_widget.textCursor().End)
        self.text_widget.insertPlainText(message)
        self.text_widget.ensureCursorVisible()
        QApplication.processEvents()  # Forces UI update

    def flush(self):
        """
        Handles flush calls to prevent output buffering.
        """
        QApplication.processEvents()


class PolygonDetectionApp(QMainWindow):
    """PyQt5 GUI for detecting people inside a defined polygon in a video using YOLOv8."""

    def __init__(self):
        """
        Initializes the application window and UI elements.
        """
        super().__init__()

        self.setWindowTitle("YOLOv8 Polygon Detection")
        self.setGeometry(100, 100, 1300, 750)  # Increased window size

        self.video_path = None
        self.video_capture = None
        self.polygon = []
        self.centers = []
        self.confidence_threshold = 0.7  # Choose confidence threshold
        self.model = YOLO("yolov8l.pt")  # Load YOLO model

        # Layouts
        self.main_layout = QHBoxLayout()  # Horizontal: Video | Console
        self.left_layout = QVBoxLayout()  # Left: Video & Buttons
        self.right_layout = QVBoxLayout()  # Right: Console Output

        # Video Display
        self.video_label = QLabel(self)
        self.left_layout.addWidget(self.video_label)

        # Buttons
        self.import_video_button = QPushButton("Import Video", self)
        self.import_video_button.clicked.connect(self.import_video)
        self.left_layout.addWidget(self.import_video_button)

        self.import_json_button = QPushButton("Import JSON", self)
        self.import_json_button.clicked.connect(self.import_json)
        self.left_layout.addWidget(self.import_json_button)

        self.start_detection_button = QPushButton("Start Detection", self)
        self.start_detection_button.clicked.connect(self.start_detection)
        self.left_layout.addWidget(self.start_detection_button)

        self.result_label = QLabel(self)
        self.left_layout.addWidget(self.result_label)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_app)
        self.left_layout.addWidget(self.close_button)

        # Console Output (styled like a terminal)
        self.console_output = QTextEdit(self)
        self.console_output.setReadOnly(True)
        self.console_output.setMinimumWidth(450)
        self.console_output.setStyleSheet(
            "background-color: black; color: lime; font-family: monospace;"
        )
        self.right_layout.addWidget(self.console_output)

        # Combine layouts
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # Set main widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # Timer for video processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

        # Redirect stdout & stderr to console
        sys.stdout = StreamRedirect(self.console_output)
        sys.stderr = StreamRedirect(self.console_output)

    def import_video(self):
        """
        Opens file dialog to select a video file.
        """
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            options=options,
        )
        if file:
            self.video_path = file
            self.video_capture = cv2.VideoCapture(file)
            print(f"[INFO] Loaded video: {file}")

    def import_json(self):
        """
        Opens file dialog to select a JSON file containing polygon coordinates.
        """
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON File",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options,
        )
        if file:
            with open(file, "r") as f:
                self.polygon = json.load(f)
            print(f"[INFO] Loaded polygon coordinates: {self.polygon}")

    def start_detection(self):
        """
        Starts the object detection process.
        """
        if self.video_capture and self.polygon:
            print("[INFO] Starting detection...")
            self.timer.start(30)

    def process_frame(self):
        """
        Processes each frame, detects people, and logs results in terminal format.
        """
        if not self.video_capture.isOpened():
            self.timer.stop()
            print("[INFO] Video capture ended.")
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.video_capture.release()
            print("[INFO] Video processing completed.")
            return

        # Start timer for benchmarking
        start_time = time.time()

        # Run YOLO detection
        results = self.model(frame)
        preprocess_time = (time.time() - start_time) * 1000  # ms
        boxes = results[0].boxes.data.cpu().numpy()
        self.centers = []

        # Parse detected objects
        detected_objects = {}
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            # if conf >= self.confidence_threshold:
            if conf >= self.confidence_threshold and int(cls) == 0:
                class_name = self.model.names[int(cls)]
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                )
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                self.centers.append(center)

        # Draw polygon
        if self.polygon:
            polygon_points = np.array(self.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2
            )

        # Count people inside polygon
        count = self.count_centers_in_polygon()
        inference_time = (time.time() - start_time) * 1000 - preprocess_time
        postprocess_time = (
            (time.time() - start_time) * 1000 - inference_time - preprocess_time
        )

        # Log formatted output
        objects_str = ", ".join([f"{v} {k}" for k, v in detected_objects.items()])
        print(
            f"0: {frame.shape[0]}x{frame.shape[1]} {objects_str}, {inference_time:.1f}ms"
        )
        print(
            f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess per image\n"
        )

        # Display result
        self.result_label.setText(f"People counted: {count}")
        self.display_frame(frame)

    def display_frame(self, frame):
        """
        Converts & displays OpenCV frame in PyQt5.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap(qimg))
        self.video_label.setAlignment(Qt.AlignCenter)

    def count_centers_in_polygon(self):
        """
        Counts people inside the defined polygon.
        """
        if not self.polygon:
            return 0
        polygon_shape = Polygon(self.polygon)
        return sum(
            1 for center in self.centers if polygon_shape.contains(Point(center))
        )

    def close_app(self):
        """
        Stops the video & closes the app.
        """
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonDetectionApp()
    window.show()
    sys.exit(app.exec_())
