"""
Import the modules to ensure that the app can take in json files
as well as handle multi-thread

"""

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
import smtplib
import sys
import time
import threading
from threading import Lock
from typing import List, Optional, Tuple
import torch

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


class ConnectionDialog(QDialog):
    """
    Dialog that prompts the user to connect to the ip camera

    Attributes:
        ip_input: IP Address of the camera
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connect to IP Camera")

        layout = QFormLayout()

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("e.g., 192.168.1.100")
        layout.addRow("IP Address:", self.ip_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_ip(self):
        """
        Gets the ip address of the video source

        Returns:
            str: the ip address
        """
        return self.ip_input.text()


class CredentialsDialog(QDialog):
    """
    Dialog where the user can enter their username, their password, as well as chose their profile

    Attributes:
        username_input: username
        password_input: password, appears as '***' to hide the password
        profile_combo: the video profile. Profile 1 has the highest quality, 3 has the lowest.
        Might need to look in to if this is exclusive to Hanwha cameras
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter Credentials")

        layout = QFormLayout()

        self.username_input = QLineEdit()
        layout.addRow("Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(
            QLineEdit.Password
        )  # Replace password chars with '*'
        layout.addRow("Password:", self.password_input)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["profile3", "profile2", "profile1"])
        layout.addRow("Profile:", self.profile_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_credentials(self):
        """
        Collects the credentials to login the camera, this is put together to
        form a string that is the url that the program will use to to connect
        to the camera

        Returns:
            username_input (str): The username, typically 'admin'
            password_input (str): The password, visually concealed
            profile_combo (str): A choice between profile 1, 2, or 3, gradually
                                decreasing in quality but increasing in
                                smoothnees

        """
        return (
            self.username_input.text(),
            self.password_input.text(),
            self.profile_combo.currentText(),
        )


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

    def __init__(self, model_path: str = "../data/yolo12l.pt"):
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

    def set_confidence(self, confidence_threshold):
        """
        This function sets the confidence_threshold in which the model would
        use to determine a n object. In layment's term, this is how confident
        the model is that an object is of which class (in this case humans).

        Confidence should be from 0 to 1, with 0 meaning no confidence and 1
        meaning absolute confidence.

        For example, the model is 70% sure that this object is a person.

        Args:
            confidence_threshold (float): the confidence_threshold
        """
        self.confidence_threshold = confidence_threshold

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

        Each nth frame is put through the model, where it detects and classifies
        objects. If an object fits the classifcation and has its centre within
        the bounderies of the polygon, they're considered "in the zone"

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

                    # Draw bounding box and confidence
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (166, 218, 149),
                        2,
                    )

                    confidence_text = f"{conf:.2f}"
                    cv2.putText(
                        frame,
                        confidence_text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (166, 218, 149),
                        2,
                        cv2.LINE_AA,
                    )

                    # Draw center point
                    cv2.circle(frame, center, 4, (138, 173, 244), -1)

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
        self.setGeometry(1000, 100, 3200, 1800)

        # Initialize state
        self.video_path: Optional[str] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.polygon: List[List[int]] = []
        self.processor = VideoProcessor()
        self.high_count_frames = 0
        self.max_high_count_frames = 10
        self.email_enabled = False
        self.max_people_exceeded = False
        self.email_cooldown = 60  # Email cool down time
        self.last_email_time = None  # Set to None initially
        self.email_lock = Lock()

        self._setup_ui()
        self._setup_signals()

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
        self.video_label.setMinimumSize(1920, 1080)
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

        # Person count threshold
        self.person_count_threshold_spinbox = QSpinBox()
        self.person_count_threshold_spinbox.setRange(1, 10)
        self.person_count_threshold_spinbox.setValue(3)
        config_layout.addWidget(QLabel("Person Threshold:"))
        config_layout.addWidget(self.person_count_threshold_spinbox)

        # Confidence threshhold
        self.confidence_threshold_spinbox = (
            QDoubleSpinBox()
        )  # Spin box that accepts float values
        self.confidence_threshold_spinbox.setRange(0, 1.0)
        self.confidence_threshold_spinbox.setValue(0.7)
        config_layout.addWidget(QLabel("Confidence Threshold:"))
        config_layout.addWidget(self.confidence_threshold_spinbox)

        # Email Toggle Checkbox
        self.toggle_email_checkbox = QCheckBox("Send Emails", self)
        self.toggle_email_checkbox.setChecked(False)  # Default to false
        self.toggle_email_checkbox.stateChanged.connect(self.update_email_state)
        config_layout.addWidget(self.toggle_email_checkbox)

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

    def update_email_state(self, state):
        """
        Updates the email state.

        This function toggles the email state, if it's set to false, then the
        program won't send emails even if there are more people than the
        threshold in the polygon.

        If it's set to true then it would only send emails if there are more
        people than the threshold in the polygon.

        """
        self.email_enabled = state == Qt.Checked

    def send_email(self, subject, body):
        """
        Sends an email notifying the admin if there are more people in the area.

        As of now sends email from gmail to outlook.

        Args:
            subject (str): subject of the email, this should be in capslock to
            attract attention.
            body (str): body of the email.
        """
        msg = MIMEMultipart()
        msg["From"] = "dungnguyen10082000@gmail.com"
        msg["To"] = "dungnm2.ho@vietcombank.com.vn"
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()  # Use TLS
                server.login("dungnguyen10082000@gmail.com", "cqivsivdvasduedn")
                server.send_message(msg)
                print("Email sent!")
        except Exception as e:
            print(f"Failed to send email: {e}")

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
        Open a (series of) of dialogs to import a video stream into the program

        This method allows the user to enter an ip Address, then their username
        and password. The information would then be used to create the rtsp url
        rtsp://{username}:{password}@{ip_address}:554/{profile}/media.smp. The
        url is then used to load the live feed.

        It also prints the outputs to the console output

        Raises:
            RuntimeError: If the video stream fails to open (e.g., incorrect credentials,
                          invalid IP address, or connection issues).
            Exception: For any other unexpected errors during the process.

        Side Effects:
            - Opens input dialogs for IP address and credentials.
            - Initializes or reinitializes `self.video_capture` with the new video stream.
            - Sets `self.video_path` to the constructed RTSP URL.
            - Displays error messages via `QMessageBox` in case of failures.
            - Prints connection info to the console.

        Returns:
        None
        """
        try:
            # IP Address
            ip_dialog = ConnectionDialog()
            if ip_dialog.exec_() != QDialog.Accepted:
                return
            ip_address = ip_dialog.get_ip()

            # Credentials
            cred_dialog = CredentialsDialog()
            if cred_dialog.exec_() != QDialog.Accepted:
                return
            username, password, profile = cred_dialog.get_credentials()

            # Construct RTSP URL
            rtsp_url = (
                f"rtsp://{username}:{password}@{ip_address}:554/{profile}/media.smp"
            )

            # Release previous video capture if exists
            if self.video_capture:
                self.video_capture.release()

            # Initialize video capture
            self.video_capture = cv2.VideoCapture(rtsp_url)
            if not self.video_capture.isOpened():
                raise RuntimeError("Failed to open video stream")

            self.video_path = rtsp_url
            print(f"[INFO] Loaded video from: {rtsp_url}")

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
            - Sends an email to the supervisor every minute in a separate thread
            - Logs detection results for debugging and tracking.
            - The user can update the person count and confidence_threshold here

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
        person_count_threshold = self.person_count_threshold_spinbox.value()

        # Update confidence_threshold
        confidence_threshold = self.confidence_threshold_spinbox.value()
        self.processor.set_confidence(confidence_threshold)

        if count > person_count_threshold:
            # If there are more people in the polygon than the threshold
            self.high_count_frames += 1

            if self.high_count_frames > self.max_high_count_frames:
                self.status_label.setStyleSheet(
                    "font-size: 50px;\n"
                    "font-weight: bold;\n"
                    "font-family: Arial;\n"
                    "color: #ed8796;"
                )

                # Could run special functions here, for example a function to
                # send emails to whoever in charge
                # self.send_email(
                #     "ALERT: THE MAXIMUM NUMBER OF PEOPLE BREACHED!",
                #     "There are too many people in the designated area!",
                # )
            current_time = time.time()
            with self.email_lock:  # Ensure thread safety
                if (
                    self.last_email_time is None
                    or current_time - self.last_email_time > self.email_cooldown
                ):
                    self.last_email_time = current_time
                    if self.email_enabled is True:
                        threading.Thread(
                            target=self.send_email,
                            args=(
                                "ALERT: THE MAXIMUM NUMBER OF PEOPLE BREACHED!",
                                "There are too many people in the designated area!",
                            ),
                            daemon=True,
                        ).start()
        else:
            self.high_count_frames = 0
            self.status_label.setStyleSheet(
                "font-size: 50px; font-weight: bold; font-family: Arial; color: #cad3f5;"
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

    # Set app theme
    with open("../conf/theme.qss", "r", encoding="utf-8") as file:
        theme = file.read()
    app.setStyleSheet(theme)

    # Show window
    window = PolygonDetectionApp()
    window.show()
    sys.exit(app.exec_())
