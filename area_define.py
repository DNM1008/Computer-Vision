"""

import necessary modules for the Qt app
"""

import json
import sys

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPainter, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PolygonDrawer(QMainWindow):
    """

    Attributes:
        json_data: the list of the dots that the program will draw on top of the
                    image
        image_path: path to the image
        label: label
        image_label: image label
        load_image_button: run the function to load the image
        load_json_button: run the function to load the coordinates
        draw_polygon_button: run the function to draw the polygon made up of the
                            coordinates on top of the image
        close_button:
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polygon Drawer")
        self.setGeometry(100, 100, 800, 600)

        self.json_data = None
        self.image_path = None

        self.initUI()

    def initUI(self):
        """

        initiialise the UI
        """
        layout = QVBoxLayout()

        self.label = QLabel("Select JSON and Image to Draw Polygon", self)
        layout.addWidget(self.label)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        self.load_json_button = QPushButton("Load JSON", self)
        self.load_json_button.clicked.connect(self.load_json)
        layout.addWidget(self.load_json_button)

        self.draw_polygon_button = QPushButton("Draw Polygon", self)
        self.draw_polygon_button.clicked.connect(self.draw_polygon)
        layout.addWidget(self.draw_polygon_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_json(self):
        """

        try to load the json file which contains the coordinates

        Raises:
            ValueError: if the file doesn't have the coordinates as required
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON File",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options,
        )
        if file_name:
            try:
                with open(file_name, "r") as f:
                    self.json_data = json.load(f)
                if not isinstance(self.json_data, list) or not all(
                    isinstance(p, list) and len(p) == 2 for p in self.json_data
                ):
                    raise ValueError(
                        "Invalid JSON format. Expected a list of [x, y] pairs."
                    )
                self.label.setText(f"Loaded JSON: {file_name}")
            except Exception as e:
                self.label.setText(f"Error loading JSON: {str(e)}")

    def load_image(self):
        """try to load the image into the app"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg);;All Files (*)",
            options=options,
        )
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap)
            self.label.setText(f"Loaded Image: {file_name}")

    def draw_polygon(self):
        """

        try to draw the polygon on top of the image
        the program will start at the first dot defined in the json file, then go
        sequentially through the dots in the json file, this way it can also
        draw concave polygons
        """
        if not self.json_data or not self.image_path:
            self.label.setText("Load both JSON and Image first!")
            return

        pixmap = QPixmap(self.image_path)
        label_width, label_height = self.image_label.width(), self.image_label.height()
        pixmap = pixmap.scaled(label_width, label_height)  # Scale image to fit QLabel

        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 0, 0))  # Red outline
        painter.setBrush(
            QColor(255, 0, 0, 100)
        )  # Semi-transparent red fill (alpha=100)

        polygon = QPolygonF()

        for point in self.json_data:
            x, y = point
            polygon.append(QPointF(x, y))

        if len(polygon) > 2:
            painter.drawPolygon(polygon)

        painter.end()
        self.image_label.setPixmap(pixmap)
        self.label.setText("Polygon highlighted on image")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolygonDrawer()
    window.show()
    sys.exit(app.exec_())
