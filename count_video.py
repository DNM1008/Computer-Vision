"""

import the necessary modules for the Qt application
"""

import json
import sys

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
    QWidget,
)
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


class PolygonDetectionApp(QMainWindow):
    """

    Attributes:
        video_path: path to the video file
        video_capture: capture the frames
        polygon: polygon that defines the area in which the program will capture
                people to count
        centers: number of box centers in the polygons, this is the number of
                peole that the program recognises
        model: the model that the program uses, this case YOLOv8l
        layout: dimensions of the app window
        video_label: video label
        import_video_button: import the video into the program
        import_json_button: import the coordinates
        start_detection_button: start the detection process
        result_label: print result on the app
        close_button: quit the program
        timer: time into the video
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Polygon Detection App - Video")
        self.setGeometry(100, 100, 800, 600)

        self.video_path = None
        self.video_capture = None
        self.polygon = []
        self.centers = []

        # Load YOLOv8 model for people detection
        self.model = YOLO("yolov8l.pt")  # Ensure you have this model downloaded

        # UI setup
        self.layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.import_video_button = QPushButton("Import Video", self)
        self.import_video_button.clicked.connect(self.import_video)
        self.layout.addWidget(self.import_video_button)

        self.import_json_button = QPushButton("Import JSON", self)
        self.import_json_button.clicked.connect(self.import_json)
        self.layout.addWidget(self.import_json_button)

        self.start_detection_button = QPushButton("Start Detection", self)
        self.start_detection_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_detection_button)

        self.result_label = QLabel(self)
        self.layout.addWidget(self.result_label)

        # Close Button
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_app)
        self.layout.addWidget(self.close_button)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Timer for processing video frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

    def import_video(self):
        """import the video into the app"""
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

    def import_json(self):
        """import the coordinates into the app"""
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

    def start_detection(self):
        """start the detection process"""
        if self.video_capture and self.polygon:
            self.timer.start(30)  # Process frames every 30 ms (~33 FPS)

    def process_frame(self):
        """count the number in the area in each frame"""
        if not self.video_capture.isOpened():
            self.timer.stop()
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.video_capture.release()
            return

        # Detect people in the frame
        results = self.model(frame)
        boxes = results[0].boxes.data.cpu().numpy()
        self.centers = []

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Class 0 is "person" in COCO dataset
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                )
                # Calculate center of the bounding box
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                self.centers.append(center)

        # Draw the polygon on the frame
        if self.polygon:
            polygon_points = np.array(self.polygon, np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
            cv2.polylines(
                frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2
            )

        # Count people inside the polygon
        count = self.count_centers_in_polygon()

        # Display count on the frame
        cv2.putText(
            frame,
            f"People in area: {count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Update result label
        self.result_label.setText(f"People counted: {count}")

        # Display the frame in PyQt
        self.display_frame(frame)

    def display_frame(self, frame):
        """
        display the frame onto the app
        Args:
            frame (numpy.ndarray): the frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        convert_to_Qt_format = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap(convert_to_Qt_format)
        self.video_label.setPixmap(pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

    def count_centers_in_polygon(self):
        """
        count the number of people inside the area. Each person detected would
        have a bouncing box around them, the centers of those boxes are used to
        determine if they should be counted.

        Returns:
           count (int): The number of centers inside the polygon
        """
        if not self.polygon:
            return 0

        # Convert polygon points into a Shapely Polygon
        polygon_shape = Polygon(self.polygon)

        # Count centers inside the polygon using Shapely
        count = sum(
            1 for center in self.centers if polygon_shape.contains(Point(center))
        )

        return count

    def close_app(self):
        """closes the application when the close button is pressed."""
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonDetectionApp()
    window.show()
    sys.exit(app.exec_())
