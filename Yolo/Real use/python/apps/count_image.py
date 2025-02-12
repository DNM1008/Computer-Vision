import json
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
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
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Polygon Detection App")
        self.setGeometry(100, 100, 800, 600)

        self.image = None
        self.polygon = []
        self.centers = []

        # Load YOLOv8 model for people detection
        self.model = YOLO("yolov8l.pt")  # Ensure you have this model downloaded

        # UI setup
        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.import_image_button = QPushButton("Import Image", self)
        self.import_image_button.clicked.connect(self.import_image)
        self.layout.addWidget(self.import_image_button)

        self.import_json_button = QPushButton("Import JSON", self)
        self.import_json_button.clicked.connect(self.import_json)
        self.layout.addWidget(self.import_json_button)

        self.detect_button = QPushButton("Detect People", self)
        self.detect_button.clicked.connect(self.detect_people)
        self.layout.addWidget(self.detect_button)

        self.result_label = QLabel(self)
        self.layout.addWidget(self.result_label)

        # Close Button
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_app)
        self.layout.addWidget(self.close_button)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def import_image(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg);;All Files (*)",
            options=options,
        )
        if file:
            self.image = cv2.imread(file)
            self.display_image()

    def import_json(self):
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
            self.draw_polygon()

    def display_image(self):
        if self.image is not None:
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            convert_to_Qt_format = QImage(
                self.image.data, width, height, bytes_per_line, QImage.Format_BGR888
            )
            pixmap = QPixmap(convert_to_Qt_format)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)

    def draw_polygon(self):
        if self.image is not None and self.polygon:
            # Convert polygon to numpy array
            polygon_points = np.array(self.polygon, np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))

            # Draw only the polygon outline (no overlay)
            cv2.polylines(
                self.image,
                [polygon_points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            self.display_image()

    def detect_people(self):
        if self.image is not None:
            # Perform YOLOv8 inference on the image
            results = self.model(self.image)

            # Extract bounding boxes
            boxes = results[0].boxes.data.cpu().numpy()
            self.centers = []

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 0:  # Class 0 is "person" in COCO dataset
                    cv2.rectangle(
                        self.image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),
                        2,
                    )
                    # Calculate center of the bounding box
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    self.centers.append(center)

            # Redraw polygon outline after detection
            self.draw_polygon()

            # Count the number of centers inside the polygon
            count = self.count_centers_in_polygon()
            self.result_label.setText(f"People counted: {count}")
            self.display_image()

    def count_centers_in_polygon(self):
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
        """Closes the application when the close button is pressed."""
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonDetectionApp()
    window.show()
    sys.exit(app.exec_())
