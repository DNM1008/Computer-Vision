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
    """Redirects stdout & stderr to a QTextEdit widget for real-time logging."""

    def __init__(self, text_widget):
        """
        Initializes StreamRedirect.

        Args:
            text_widget (QTextEdit): Widget to display redirected text output.
        """
        self.text_widget = text_widget

    def write(self, message):
        """
        Writes console output to QTextEdit.

        Args:
            message (str): Console output message.
        """
        self.text_widget.moveCursor(self.text_widget.textCursor().End)
        self.text_widget.insertPlainText(message)
        self.text_widget.ensureCursorVisible()
        QApplication.processEvents()

    def flush(self):
        """Handles flush calls to prevent output buffering."""
        QApplication.processEvents()


class PolygonDetectionApp(QMainWindow):
    """GUI application for detecting people inside a polygon using YOLOv8."""

    def __init__(self):
        """Initializes the application window and UI components."""
        super().__init__()

        self.setWindowTitle("YOLOv8 Polygon Detection")
        self.setGeometry(100, 100, 1300, 750)

        self.video_path = None
        self.video_capture = None
        self.polygon = []
        self.centers = []
        self.confidence_threshold = 0.7
        self.model = YOLO("yolov8l.pt")
        self.high_count_frames = 0  # Track frames where people count > 3

        # Layouts
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

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
        self.result_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; font-family: Arial; color: black;"
        )
        self.left_layout.addWidget(self.result_label)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_app)
        self.left_layout.addWidget(self.close_button)

        # Console Output
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
        """Opens file dialog to select a video file."""
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
        """Opens file dialog to select a JSON file containing polygon coordinates."""
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
        """Starts the object detection process if a video and polygon are available."""
        if self.video_capture and self.polygon:
            print("[INFO] Starting detection...")
            self.timer.start(30)

    def process_frame(self):
        """Processes each frame, detects people, and updates UI accordingly."""
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

        start_time = time.time()

        # Run YOLO detection
        results = self.model(frame)
        boxes = results[0].boxes.data.cpu().numpy()
        self.centers = []

        detected_objects = {}
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if (
                conf >= self.confidence_threshold and int(cls) == 0
            ):  # Only detect people
                detected_objects["person"] = detected_objects.get("person", 0) + 1
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                self.centers.append(center)

                # Draw bounding box
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    "Person",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Draw polygon
        if self.polygon:
            polygon_points = np.array(self.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2
            )

        # Count people inside polygon
        count = self.count_centers_in_polygon()

        # Update frame counter if count > 3
        if count > 3:
            self.high_count_frames += 1
        else:
            self.high_count_frames = 0  # Reset if count drops

        # Change text color if condition is met
        if self.high_count_frames > 10:
            self.result_label.setStyleSheet(
                "font-size: 120px; font-weight: bold; font-family: Arial; color: red;"
            )
        else:
            self.result_label.setStyleSheet(
                "font-size: 120px; font-weight: bold; font-family: Arial; color: black;"
            )

        self.result_label.setText(f"People counted: {count}")

        self.display_frame(frame)

    def display_frame(self, frame):
        """Converts and displays OpenCV frame in PyQt5 UI.

        Args:
            frame (numpy.ndarray): Processed video frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap(qimg))
        self.video_label.setAlignment(Qt.AlignCenter)

    def count_centers_in_polygon(self):
        """Counts the number of detected people inside the defined polygon.

        Returns:
            int: Count of people within the polygon.
        """
        if not self.polygon:
            return 0
        polygon_shape = Polygon(self.polygon)
        return sum(
            1 for center in self.centers if polygon_shape.contains(Point(center))
        )

    def close_app(self):
        """Stops the video and closes the application."""
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonDetectionApp()
    window.show()
    sys.exit(app.exec_())
