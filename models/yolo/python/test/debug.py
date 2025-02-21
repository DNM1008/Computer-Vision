import json
import sys
import time
import torch
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
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
    QMessageBox,
    QSpinBox,
    QColorDialog,
)
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


class StreamRedirect:
    """
    Redirects stdout & stderr to both QTextEdit and terminal with thread safety.

    A utility class that captures standard output and error streams and redirects them
    to both a QTextEdit widget and the terminal, ensuring thread-safe operation.

    Attributes:
        text_widget (QTextEdit): The text widget where output will be displayed.
        _buffer (list): Internal buffer for storing messages.
    """

    def __init__(self, text_widget: QTextEdit) -> None:
        """
        Initialize the stream redirector.

        Args:
            text_widget: QTextEdit widget where output will be displayed.
        """
        self.text_widget = text_widget
        self._buffer = []

    def write(self, message: str) -> None:
        """
        Write a message to both the text widget and terminal.

        Args:
            message: The string message to be written.
        """
        if message.strip():
            self.text_widget.append(message.rstrip())
            sys.__stdout__.write(message)
            sys.__stdout__.flush()

    def flush(self) -> None:
        """
        Flush the output stream.

        Ensures all pending output is processed and displayed.
        """
        QApplication.processEvents()
        sys.__stdout__.flush()


class VideoProcessor:
    """
    Handles video processing operations using YOLOv8 for object detection.

    This class manages video frame processing, object detection, and visualization
    of detection results including polygon zones and detected persons.

    Attributes:
        model: YOLO model instance for object detection.
        confidence_threshold: Minimum confidence score for valid detections.
        frame_skip: Number of frames to skip between detections.
        frame_count: Counter for processed frames.
        polygon_color: RGB color tuple for polygon visualization.
        polygon_thickness: Line thickness for polygon drawing.
    """

    def __init__(self, model_path: str = "yolov8l.pt"):
        """
        Initialize the video processor.

        Args:
            model_path: Path to the YOLO model weights file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.confidence_threshold = 0.7
        self.frame_skip = 1  # Process every nth frame
        self.frame_count = 0
        self.polygon_color = (255, 153, 153)  # Default light red color for polygon
        self.polygon_thickness = 2

    def draw_polygon(
        self, frame: np.ndarray, polygon_points: List[List[int]]
    ) -> np.ndarray:
        """
        Draw the detection zone polygon on the frame.

        Args:
            frame: Input video frame.
            polygon_points: List of [x, y] coordinates defining the polygon vertices.

        Returns:
            Modified frame with polygon overlay.
        """
        if not polygon_points:
            return frame

        # Convert points to numpy array of integers
        points = np.array(polygon_points, np.int32)
        points = points.reshape((-1, 1, 2))

        # Draw the polygon
        cv2.polylines(frame, [points], True, self.polygon_color, self.polygon_thickness)

        # Add semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (*self.polygon_color, 128))

        # Blend the overlay with the original frame
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    def process_frame(
        self, frame: np.ndarray, polygon_points: List[List[int]]
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], float, int]:
        """
        Process a single frame with YOLO detection.

        Args:
            frame: Input video frame.
            polygon_points: List of [x, y] coordinates defining the detection zone.

        Returns:
            Tuple containing:
                - Processed frame with visualizations
                - List of detected person center points
                - Inference time in milliseconds
                - Count of detected persons
        """
        self.frame_count += 1

        # Draw polygon on every frame
        frame = self.draw_polygon(frame.copy(), polygon_points)

        # Only run YOLO detection every nth frame
        if self.frame_count % self.frame_skip == 0:
            start_time = time.time()
            results = self.model(frame)
            inference_time = (time.time() - start_time) * 1000

            boxes = results[0].boxes.data.cpu().numpy()
            centers = []
            person_count = 0

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if (
                    conf >= self.confidence_threshold
                    and self.model.names[int(cls)] == "person"
                ):
                    person_count += 1
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    centers.append(center)

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )

                    # Draw center point
                    cv2.circle(frame, center, 4, (255, 0, 0), -1)

            # Store last detection results
            self._last_results = (centers, inference_time, person_count)
        else:
            # Use last detection results for skipped frames
            centers, inference_time, person_count = getattr(
                self, "_last_results", ([], 0.0, 0)
            )

        return frame, centers, inference_time, person_count


class PolygonDetectionApp(QMainWindow):
    """
    GUI application for detecting people inside a polygon using YOLOv8.

    This class implements a complete application for real-time person detection
    within a defined polygon zone using YOLOv8 object detection.

    Attributes:
        video_path: Path to the input video file.
        video_capture: OpenCV video capture object.
        polygon: List of points defining the detection zone.
        processor: VideoProcessor instance for frame processing.
        high_count_frames: Counter for frames with high person count.
        max_high_count_frames: Threshold for triggering warnings.
    """

    def __init__(self):
        """Initialize the application window and setup UI components."""
        super().__init__()
        self.setWindowTitle("YOLOv8 Polygon Detection")
        self.setGeometry(1000, 100, 1600, 900)

        # Initialize state
        self.video_path: Optional[str] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.polygon: List[List[int]] = []
        self.processor = VideoProcessor()
        self.high_count_frames = 0
        self.max_high_count_frames = 10

        self._setup_ui()
        self._setup_signals()
        self.apply_theme()

    def _setup_ui(self) -> None:
        """
        Initialize and setup UI components.

        Creates and arranges all UI elements including video display,
        control buttons, and output panels.
        """
        # Main layouts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel (video and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        left_layout.addWidget(self.video_label)

        # Controls
        controls_layout = QHBoxLayout()

        self.import_video_btn = QPushButton("Import Video")
        self.import_json_btn = QPushButton("Import coordinates")
        self.change_color_btn = QPushButton("Change Polygon Color")
        self.start_detection_btn = QPushButton("Start Detection")
        self.start_detection_btn.setStyleSheet(
            "background-color: #a6da95; color: #181926; font-weight: bold;"
        )  # Start button highlight
        self.stop_detection_btn = QPushButton("Stop")
        self.stop_detection_btn.setStyleSheet(
            "background-color: #eed49f; color: #181926; font-weight: bold;"
        )  # Start button highlight

        for btn in [
            self.import_video_btn,
            self.import_json_btn,
            self.change_color_btn,
            self.start_detection_btn,
            self.stop_detection_btn,
        ]:
            controls_layout.addWidget(btn)

        left_layout.addLayout(controls_layout)

        # Configuration
        config_layout = QHBoxLayout()
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(1, 10)
        self.threshold_spinbox.setValue(3)
        config_layout.addWidget(QLabel("Person Threshold:"))
        config_layout.addWidget(self.threshold_spinbox)
        left_layout.addLayout(config_layout)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet(
            "font-size: 120px; font-weight: bold; font-family: Arial;"
        )
        left_layout.addWidget(self.status_label)

        main_layout.addWidget(left_panel)

        # Right panel (logs)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet(
            "background-color: #181926; color: #a6da95; font-family: monospace;"
            # Hacker style
        )
        right_layout.addWidget(QLabel("Console Output:"))
        right_layout.addWidget(self.console_output)

        # Detection log
        self.detection_log = QTextEdit()
        self.detection_log.setReadOnly(True)
        self.detection_log.setStyleSheet(
            "background-color: #181926; color: #a6da95; font-family: monospace;"
            # Hacker style
        )
        right_layout.addWidget(QLabel("Detection Log:"))
        right_layout.addWidget(self.detection_log)

        main_layout.addWidget(right_panel)

        # Redirect stdout/stderr
        sys.stdout = StreamRedirect(self.console_output)
        sys.stderr = StreamRedirect(self.console_output)

        # Processing timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        # Add close button
        self.close_btn = QPushButton("Close Program")
        self.close_btn.setStyleSheet(
            "background-color: #ed8796; color: #181926; font-weight: bold;"
        )
        controls_layout.addWidget(self.close_btn)

    def apply_theme(self):
        """Apply Catppuccin Macchiato color scheme."""
        self.setStyleSheet(
            """
            QWidget {
                background-color: #24273A;
                color: #CAD3F5;
                font-size: 25px;
                font-family: "Segoe UI", "Arial", sans-serif;
            }
    
            QPushButton {
                background-color: #363A4F;
                color: #CAD3F5;
                border-radius: 6px;
                padding: 6px;
                border: 1px solid #494D64;
            }
            QPushButton:hover {
                background-color: #494D64;
            }
            QPushButton:pressed {
                background-color: #5B6078;
            }
    
            QLabel {
                color: #CAD3F5;
                font-weight: bold;
            }
    
            QLineEdit, QTextEdit {
                background-color: #1E2030;
                border: 1px solid #494D64;
                border-radius: 4px;
                padding: 4px;
                color: #CAD3F5;
            }
    
            QSpinBox, QComboBox {
                background-color: #1E2030;
                border: 1px solid #494D64;
                color: #CAD3F5;
            }
    
            QSlider::groove:horizontal {
                background: #494D64;
                height: 6px;
                border-radius: 3px;
            }
    
            QSlider::handle:horizontal {
                background: #8AADF4;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
    
            QMessageBox {
                background-color: #24273A;
                color: #CAD3F5;
            }
        """
        )

    def _setup_signals(self) -> None:
        """
        Connect UI signals to their respective slots.

        This method establishes connections between UI elements (buttons) and their
        corresponding event handler methods, enabling user interaction with the application.

        Connected Signals:
            - `import_video_btn.clicked` → `import_video()`: Opens a dialog to import a video file.
            - `import_json_btn.clicked` → `import_json()`: Loads polygon coordinates from a JSON file.
            - `start_detection_btn.clicked` → `start_detection()`: Begins object detection in the video.
            - `stop_detection_btn.clicked` → `stop_detection()`: Stops the detection process.
            - `change_color_btn.clicked` → `change_polygon_color()`: Opens a color picker to change the polygon color.
            - `close_btn.clicked` → `close()`: Closes the application.
        """
        self.import_video_btn.clicked.connect(self.import_video)
        self.import_json_btn.clicked.connect(self.import_json)
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        self.change_color_btn.clicked.connect(self.change_polygon_color)
        self.close_btn.clicked.connect(self.close)

    @pyqtSlot()
    def change_polygon_color(self) -> None:
        """
        Open color dialog to change polygon color.

        Allows user to select a new color for the detection zone polygon.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            # Convert Qt color to OpenCV BGR format
            self.processor.polygon_color = (color.blue(), color.green(), color.red())

    @pyqtSlot()
    def import_video(self) -> None:
        """
        Opens a file dialog to import a video file and initializes video capture.

        This method allows the user to select a video file using a file dialog.
        It then attempts to load the selected video using OpenCV. If a video is
        already loaded, it releases the previous video capture before opening the new one.

        Error handling is included to manage file selection issues or video loading failures.

        Raises:
            RuntimeError: If the selected video file cannot be opened.

        UI Elements:
            - Opens a QFileDialog for selecting a video file.
            - Displays a QMessageBox in case of an error.

        Supported Formats:
            - MP4 (*.mp4)
            - AVI (*.avi)
            - MOV (*.mov)
            - MKV (*.mkv)
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Video File",
                "",
                "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            )

            if not file_path:
                return

            if self.video_capture:
                self.video_capture.release()

            self.video_capture = cv2.VideoCapture(
                "rtsp://admin:Vcb!2025@192.168.1.100:554/profile2/media.smp"
            )
            if not self.video_capture.isOpened():
                raise RuntimeError("Failed to open video file")

            self.video_path = file_path
            print(f"[INFO] Loaded video: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
            self.video_capture = None

    @pyqtSlot()
    def import_json(self) -> None:
        """
        Opens a file dialog to import polygon coordinates from a JSON file.

        This method allows the user to select a JSON file containing polygon data.
        It loads and validates the data to ensure it represents a valid polygon
        structure. The polygon must be a list of at least three points, where
        each point is represented as a list of two numerical values [x, y].

        If the JSON data is invalid, an error message is displayed, and the
        polygon data is reset.

        Raises:
            ValueError: If the JSON data is not a valid polygon format.
            JSONDecodeError: If the file cannot be parsed as valid JSON.
            OSError: If there are issues opening the file.

        UI Elements:
            - Opens a QFileDialog for selecting a JSON file.
            - Displays a QMessageBox in case of an error.

        JSON Format Example:
            [
                [100, 200],
                [150, 250],
                [200, 200]
            ]
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open JSON File",
                "",
                "JSON Files (*.json);;All Files (*)",
            )

            if not file_path:
                return

            with open(file_path, "r") as f:
                polygon_data = json.load(f)

            # Validate polygon data
            if not isinstance(polygon_data, list) or len(polygon_data) < 3:
                raise ValueError(
                    "Invalid polygon data: must be a list of at least 3 points"
                )

            for point in polygon_data:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Invalid polygon point: must be [x, y]")

            self.polygon = polygon_data
            print(f"[INFO] Loaded polygon with {len(self.polygon)} points")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load polygon: {str(e)}")
            self.polygon = []

    @pyqtSlot()
    def start_detection(self) -> None:
        """
        Initiates the detection process if all prerequisites are met.

        Ensures that a video file has been imported and that polygon
        coordinates are available before starting detection. If either
        requirement is missing, a warning message is displayed.

        Once started, the detection process runs at approximately 30 FPS.

        UI Elements:
            - Displays a QMessageBox warning if prerequisites are not met.
            - Disables the "Start Detection" button and enables the "Stop Detection" button.
        """
        if not self.video_capture:
            QMessageBox.warning(self, "Warning", "Please import a video first")
            return

        if not self.polygon:
            QMessageBox.warning(
                self, "Warning", "Please import polygon coordinates first"
            )
            return

        print("[INFO] Starting detection...")
        self.timer.start(30)  # ~30 FPS
        self.start_detection_btn.setEnabled(False)
        self.stop_detection_btn.setEnabled(True)

    @pyqtSlot()
    def stop_detection(self) -> None:
        """
        Stops the ongoing detection process.

        Halts the detection timer, re-enables the "Start Detection" button,
        and disables the "Stop Detection" button. This method ensures that
        processing stops cleanly.

        UI Elements:
            - Stops the detection timer.
            - Enables the "Start Detection" button and disables the "Stop Detection" button.
        """
        self.timer.stop()
        self.start_detection_btn.setEnabled(True)
        self.stop_detection_btn.setEnabled(False)
        print("[INFO] Detection stopped")

    @pyqtSlot()
    def process_frame(self) -> None:
        """
        Processes a single video frame and updates detection results.

        This method:
            - Reads a frame from the video source.
            - Passes the frame to the YOLO-based processor for detection.
            - Counts detected people and checks if they are within the polygon zone.
            - Updates the UI with the processed frame and detection statistics.

        Handles the end of the video by stopping detection when no more frames
        are available.

        Raises:
            Exception: If frame processing encounters an error.

        UI Elements:
            - Calls `_update_ui()` to update the displayed frame and detection results.
            - Prints error messages in case of processing failure.
        """
        try:
            if not self.video_capture or not self.video_capture.isOpened():
                self.stop_detection()
                return

            ret, frame = self.video_capture.read()
            if not ret:
                self.stop_detection()
                print("[INFO] Video processing completed")
                return

            # Process frame with YOLO
            processed_frame, centers, inference_time, person_count = (
                self.processor.process_frame(frame, self.polygon)
            )

            # Count people in polygon
            polygon = Polygon(self.polygon)
            people_in_zone = sum(
                1 for center in centers if polygon.contains(Point(center))
            )

            # Update UI
            self._update_ui(
                processed_frame, people_in_zone, inference_time, person_count
            )

        except Exception as e:
            print(f"[ERROR] Frame processing failed: {str(e)}")
            self.stop_detection()

    def _update_ui(
        self, frame: np.ndarray, count: int, inference_time: float, total_count: int
    ) -> None:
        """
        Updates UI elements with detection results and visual feedback.

        This method:
            - Updates the displayed frame in the UI.
            - Adjusts the status label color based on the detected count.
            - Logs detection results for debugging and tracking.

        Args:
            frame (np.ndarray): The processed video frame with visualizations.
            count (int): Number of people detected within the defined zone.
            inference_time (float): YOLO model inference time in milliseconds.
            total_count (int): Total number of people detected in the frame.

        UI Updates:
            - Changes the status label color to red if the detected count exceeds
              the defined threshold for a sustained duration.
            - Displays the number of people in the zone and the total count.
                - Changes the colour to red if the number of people exceeds the
                  threshold for at least 10 seconds, could implement other
                  functions here.
            - Logs frame detection details including count and inference time.
            - Updates the displayed video frame using QImage and QPixmap.

        """
        # Update count and warning status
        threshold = self.threshold_spinbox.value()
        if count > threshold:
            self.high_count_frames += 1
            if self.high_count_frames > self.max_high_count_frames:
                self.status_label.setStyleSheet(
                    "font-size: 50px; font-weight: bold; font-family: Arial; color: red;"
                )  # Warning that there are more people in the zone
                # Could run special functions here, for example a function to
                # send emails to whoever in charge
        else:
            self.high_count_frames = 0
            self.status_label.setStyleSheet(
                "font-size: 50px; font-weight: bold; font-family: Arial; color: black;"
            )  # Text in normal state

        self.status_label.setText(
            f"People in zone: {count} (Total detected: {total_count})"
        )

        # Update detection log
        log_entry = (
            f"Frame: {self.processor.frame_count} | "
            f"In Zone: {count} | Total: {total_count} | "
            f"Inference: {inference_time:.1f}ms\n"
        )
        self.detection_log.append(log_entry)

        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def close_event(self, event) -> None:
        """
        Handles application close event by releasing resources.

        Ensures that timers and video capture resources are properly stopped
        before the application exits.

        Args:
            event: The close event object triggered when the application is closing.

        Cleanup Actions:
            - Stops the detection timer.
            - Releases the video capture object if it is in use.
            - Calls the superclass close event to ensure proper shutdown.
        """
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
        super().close_event(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonDetectionApp()
    window.show()
    sys.exit(app.exec_())
