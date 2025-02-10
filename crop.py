import json
import os
import sys

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ImageCropperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Cropper")

        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Buttons
        self.import_image_button = QPushButton("Import Image")
        self.import_json_button = QPushButton("Load JSON Coordinates")

        self.import_image_button.clicked.connect(self.import_image)
        self.import_json_button.clicked.connect(self.import_json)

        layout.addWidget(self.import_image_button)
        layout.addWidget(self.import_json_button)

        # Image preview
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Variables to hold image and JSON data
        self.image_path = None
        self.coordinates = None

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            )

    def import_json(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open JSON File", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.coordinates = data if isinstance(data, list) else None
                    if not self.coordinates:
                        raise ValueError("Invalid JSON format.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")
                return

            self.process_coordinates()

    def process_coordinates(self):
        if not self.coordinates or not all(
            isinstance(coord, list) and len(coord) == 2 for coord in self.coordinates
        ):
            QMessageBox.critical(
                self,
                "Error",
                "Invalid coordinates format in JSON. Must be a list of [x, y].",
            )
            return

        try:
            x_values = [coord[0] for coord in self.coordinates]
            y_values = [coord[1] for coord in self.coordinates]

            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            self.crop_image(x_min, x_max, y_min, y_max)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process coordinates: {e}")

    def crop_image(self, x_min, x_max, y_min, y_max):
        try:
            image = Image.open(self.image_path)
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Cropped Image",
                os.path.splitext(self.image_path)[0] + "_cropped.png",
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
            if save_path:
                cropped_image.save(save_path)
                QMessageBox.information(
                    self, "Success", f"Cropped image saved to {save_path}."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to crop and save image: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageCropperApp()
    ex.resize(800, 600)
    ex.show()
    sys.exit(app.exec_())
