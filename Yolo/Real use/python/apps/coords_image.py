"""

import necessary modules for the Qt applications
"""

import csv
import json
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ImagePointSelector(QWidget):
    """

    Attributes:
        layout: dimension of the window
        load_button: run the function that loads an image
        save_json_button: save the dots' coordinates to a json file
        save_csv_button: save the dots' coordinates to a csv file
        undo_button: undo last click
        close_button: close the application
        label: prompts the user to load the image
        cursor_label: current position of the cursor
        scene: Qt scene
        view: Qtwindow
        image_path: path to the image
        points: dots defined by the coordinates in the json file
        pixmap_item: pixmap on the original image
        current_pixmap: pixmap on the current, which now has dots on it
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Point Selector")
        self.setGeometry(100, 100, 900, 700)

        # UI Components
        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.save_json_button = QPushButton("Save as JSON", self)
        self.save_json_button.clicked.connect(self.save_points_json)
        self.layout.addWidget(self.save_json_button)

        self.save_csv_button = QPushButton("Save as CSV", self)
        self.save_csv_button.clicked.connect(self.save_points_csv)
        self.layout.addWidget(self.save_csv_button)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_last_point)
        self.layout.addWidget(self.undo_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.close_button)

        self.label = QLabel("Load an image to start", self)
        self.layout.addWidget(self.label)

        # Cursor position label
        self.cursor_label = QLabel("Cursor: (X, Y)", self)
        self.layout.addWidget(self.cursor_label)

        # Graphics View setup for displaying images
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)  # Enables real-time tracking
        self.layout.addWidget(self.view)

        self.setLayout(self.layout)

        self.image_path = None
        self.points = []

        self.pixmap_item = None
        self.current_pixmap = None

        self.view.viewport().setMouseTracking(True)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

    def load_image(self):
        """

        load the image into the program
        can take png, jpg, jpeg, bmp, and gif file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.image_path = file_path
            self.display_image()

    def display_image(self):
        """display image onto the app"""
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            if pixmap.isNull():
                self.label.setText("Error loading image")
                return

            self.current_pixmap = pixmap

            if self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)

            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.clear()
            self.scene.addItem(self.pixmap_item)
            self.view.setScene(self.scene)
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def map_to_image_coordinates(self, pos):
        """Converts a viewport position to image-relative coordinates."""
        scene_pos = self.view.mapToScene(pos)
        x, y = (
            scene_pos.x() - self.pixmap_item.x(),
            scene_pos.y() - self.pixmap_item.y(),
        )
        return x, y

    def eventFilter(self, obj, event):
        """Capture mouse movement and clicks in the image area."""
        if self.pixmap_item:
            if event.type() == event.MouseMove:
                x, y = self.map_to_image_coordinates(event.pos())
                if (
                    0 <= x <= self.current_pixmap.width()
                    and 0 <= y <= self.current_pixmap.height()
                ):
                    self.cursor_label.setText(f"Cursor: ({int(x)}, {int(y)})")

            elif (
                event.type() == event.MouseButtonPress
                and event.button() == Qt.LeftButton
            ):
                x, y = self.map_to_image_coordinates(event.pos())
                if (
                    0 <= x <= self.current_pixmap.width()
                    and 0 <= y <= self.current_pixmap.height()
                ):
                    self.points.append((x, y))
                    self.draw_point(x, y)
                    self.cursor_label.setText(f"Clicked: ({int(x)}, {int(y)})")

        return super().eventFilter(obj, event)

    def draw_point(self, x, y):
        """Draw a red dot at the given coordinates."""
        dot = self.scene.addEllipse(x - 5, y - 5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
        dot.setZValue(1)  # Ensure it appears on top

    def undo_last_point(self):
        """remove the last dot"""
        if self.points:
            self.points.pop()
            self.display_image()
            for x, y in self.points:
                self.draw_point(x, y)

    def save_points_json(self):
        """
        save the dots' coordinates into a json file in the following format:
        {
            [X1, Y1],
            [X2, Y2],
            ...
        }
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Points as JSON", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.points, f, indent=4)

    def save_points_csv(self):
        """
        save the dots' coordinates into a csv file in the following format:
        X, Y
        X1, Y1
        X2, Y2
        ...
        WARNING: This is not meant to be used in this project since the next
        steps doesn't have csv functionality built in
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Points as CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["X", "Y"])
                writer.writerows(self.points)

    def close_application(self):
        """close the application"""
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = ImagePointSelector()
    selector.show()
    sys.exit(app.exec_())
