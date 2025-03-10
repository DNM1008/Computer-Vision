"""
Import necessary modules:
    Qt5 for the UI
    PIL to handle images
    cv2 to handle computer vision things


"""

import os
import re
import sys
import cv2
from ultralytics import YOLO
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QRect

# Fix for OpenCV-Qt plugin conflicts
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QApplication.libraryPaths()[0]


class Canvas(QLabel):
    """
    A custom QLabel-based widget that allows users to draw bounding boxes by clicking and dragging.

    This widget enables users to define bounding boxes on an image or blank canvas.
    It supports mouse events to track user input and dynamically updates the rectangle
    as the user drags the mouse. When the mouse button is released, the bounding box
    coordinates are passed to the parent widget (if the parent implements an `end_draw` method).

    Attributes:
        drawing (bool): Indicates whether the user is currently drawing a rectangle.
        start_x (int | None): The x-coordinate where the mouse press event started.
        start_y (int | None): The y-coordinate where the mouse press event started.
        temp_rect (QRect | None): A temporary rectangle drawn while dragging.
        parent (QWidget | None): The parent widget, which may handle completed bounding boxes.

    Methods:
        mousePressEvent(event): Starts drawing a rectangle when the left mouse button is pressed.
        mouseMoveEvent(event): Updates the temporary rectangle while the mouse is dragged.
        mouseReleaseEvent(event): Finalizes the rectangle and sends the bounding box coordinates to the parent.
        paintEvent(event): Renders the temporary rectangle on the widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(1800, 1600)
        self.setMouseTracking(True)
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.temp_rect = None
        self.parent = parent

    def mousePressEvent(self, event):
        """
        Handles the mouse press event to start drawing a bounding box.

        This method is triggered when the user clicks on the canvas. If the left mouse
        button is pressed, it initializes the starting coordinates for drawing.

        Args:
            event (QMouseEvent): The mouse event containing information about the click,
                including the button pressed and cursor position.
        """
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_x = event.x()
            self.start_y = event.y()

    def mouseMoveEvent(self, event):
        """
        Handles the "dragging" event while drawing a bounding box

        This method is triggered when the user move the mouse after clicking but
        before releasing. When initialised, it draws a bounding box that has the
        line between the start of the drag to the current location as the
        diagonal line through the rectangle.

        Args:
            event (QMouseEvent): The mouse event containing the information
            about the movement, including the starting point and the current
            cursor location
        """
        if self.drawing:
            self.temp_rect = QRect(
                self.start_x,
                self.start_y,
                event.x() - self.start_x,
                event.y() - self.start_y,
            )
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Handdles the release event at the end of the bounding box drawing
        process

        This method is triggered when the user releases the mouse button after
        drawing the box, finishing the box drawing process. The box has start_x,
        start_y and end_x, end_y as the points defining the triangle.

        Args:
            event (QMouseEvent): The mouse event containing the information
            about the release, including the button pressed and the cursor
            position
        """
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            end_x = event.x()
            end_y = event.y()
            bbox = [self.start_x, self.start_y, end_x, end_y]
            if hasattr(self.parent, "end_draw"):
                self.parent.end_draw(bbox)
            self.temp_rect = None
            self.update()

    def paintEvent(self, event):
        """
        Handles the paint event to render the bounding box on the canvas.

        This method is called whenever the widget needs to be repainted.
        If the user is in the process of drawing a bounding box, it renders
        the temporary rectangle using a blue outline.

        Args:
            event (QPaintEvent): The paint event triggered when the widget
                needs to be updated.
        """
        super().paintEvent(event)
        painter = QPainter(self)

        # Draw temporary rectangle while dragging
        if self.temp_rect:
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            painter.drawRect(self.temp_rect)


class ImageLabeller(QMainWindow):
    """
    A GUI application for labelling images with AI-assisted bounding box detection.

    This application allows users to load images, automatically detect objects
    using a YOLO model, manually adjust bounding boxes, and export labels in
    the YOLO format. Users can navigate through images, edit detected objects,
    and save the labelled data.

    Attributes:
        model (YOLO): The YOLO model for object detection.
        image_files (list): List of image file paths loaded from a directory.
        detections (dict): Dictionary storing bounding boxes for each image.
        current_index (int): Index of the currently displayed image.
        boxes (list): List of bounding boxes for the current image.
        current_image (PIL.Image): The current image being displayed.
        pixmap (QPixmap): The current image converted for display in Qt.
    """

    def __init__(self):
        """
        Initializes the ImageLabeller application.

        Sets up the main window, loads the YOLO model, and initializes the UI.
        """
        super().__init__()
        self.setWindowTitle("AI Image Labeller")
        self.model = YOLO("../data/yolov8n.pt")
        self.image_files = []
        self.detections = {}
        self.current_index = 0
        self.boxes = []
        self.current_image = None
        self.pixmap = None

        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface, including widgets and layouts.

        The UI consists of:
        - A canvas for displaying images.
        - A list box for detected object classes.
        - Buttons for loading images, navigation, exporting labels, and closing the application.
        """
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Top section with canvas and list
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        # Canvas for displaying images
        self.canvas = Canvas(self)

        # Class listbox
        self.class_listbox = QListWidget()
        self.class_listbox.setFixedWidth(400)
        self.class_listbox.itemDoubleClicked.connect(self.delete_selected_box)

        top_layout.addWidget(self.canvas)
        top_layout.addWidget(self.class_listbox)

        # Filename label
        self.label_filename = QLabel("No image loaded")
        self.label_filename.setAlignment(Qt.AlignCenter)
        self.label_filename.setStyleSheet("font-size: 12pt;")

        # Buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)

        self.btn_load = QPushButton("Load Images")
        self.btn_load.clicked.connect(self.load_images)

        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self.previous_image)
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)

        self.btn_export = QPushButton("Export Labels")
        self.btn_export.clicked.connect(self.export_labels)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)

        buttons_layout.addWidget(self.btn_load)
        buttons_layout.addWidget(self.btn_prev)
        buttons_layout.addWidget(self.btn_next)
        buttons_layout.addWidget(self.btn_export)
        buttons_layout.addWidget(self.btn_close)

        # Add all components to main layout
        main_layout.addWidget(top_widget)
        main_layout.addWidget(self.label_filename)
        main_layout.addWidget(buttons_widget)

        self.setCentralWidget(main_widget)
        self.resize(1050, 700)

    def load_images(self):
        """
        Opens a file dialog to select a directory containing images.

        Loads all valid image files (JPG, PNG) from the selected folder,
        sorts them numerically, and prepares the first image for display.
        """
        folder_selected = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder_selected:
            return

        def numerical_sort(filename):
            return [
                int(text) if text.isdigit() else text
                for text in re.split(r"(\d+)", filename)
            ]

        self.image_files = sorted(
            [
                os.path.join(folder_selected, f)
                for f in os.listdir(folder_selected)
                if f.endswith((".jpg", ".png"))
            ],
            key=lambda x: numerical_sort(os.path.basename(x)),
        )
        if not self.image_files:
            QMessageBox.critical(
                self, "Error", "No images found in the selected folder."
            )
            return

        self.btn_next.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.current_index = 0
        self.load_image()

    def pil_to_pixmap(self, pil_image):
        """
        Converts a PIL image to a QPixmap for display in Qt.

        Args:
            pil_image (PIL.Image): The image to convert.

        Returns:
            QPixmap: The converted pixmap.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert PIL image to QImage
        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(
            data,
            pil_image.width,
            pil_image.height,
            pil_image.width * 3,
            QImage.Format_RGB888,
        )

        # Create pixmap from QImage
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def load_image(self):
        """
        Loads and displays the current image from `image_files`.

        If detections exist for the image, it retrieves and displays them.
        Otherwise, it runs object detection using YOLO.
        """
        if not self.image_files:
            return

        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1

        image_path = self.image_files[self.current_index]
        self.label_filename.setText(f"Current file: {os.path.basename(image_path)}")

        try:
            # Load image with PIL and convert to QPixmap
            self.current_image = Image.open(image_path)
            self.current_image.thumbnail((800, 600))
            self.pixmap = self.pil_to_pixmap(self.current_image)
            self.canvas.setPixmap(self.pixmap)

            # Check if we already have detections for this image
            if os.path.basename(image_path) not in self.detections:
                self.run_detection(image_path)
            else:
                self.boxes = self.detections[os.path.basename(image_path)]
                self.update_ui()
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading Image", f"Failed to load image: {str(e)}"
            )

    def run_detection(self, image_path):
        """
        Runs object detection using the YOLO model on the given image.

        Extracts bounding boxes, confidence scores, and class IDs,
        storing them in `detections`.

        Args:
            image_path (str): The file path of the image to process.
        """
        try:
            results = self.model(image_path)
            self.boxes = []
            for result in results:
                for box, conf, cls in zip(
                    result.boxes.xyxy, result.boxes.conf, result.boxes.cls
                ):
                    bbox = box.tolist()
                    self.boxes.append(
                        {"bbox": bbox, "confidence": float(conf), "class": int(cls)}
                    )
            self.detections[os.path.basename(image_path)] = self.boxes
            self.update_ui()
        except Exception as e:
            QMessageBox.warning(
                self, "Detection Error", f"Error running model: {str(e)}"
            )
            self.boxes = []
            self.detections[os.path.basename(image_path)] = self.boxes

    def update_ui(self):
        """
        Updates the UI by redrawing bounding boxes on the image
        and refreshing the class list box.

        This method ensures that all annotations are properly displayed
        after detection or manual modifications.
        """
        if self.current_image is None or self.pixmap is None:
            return

        # Clear listbox
        self.class_listbox.clear()

        # Create a new pixmap from the current image
        self.pixmap = self.pil_to_pixmap(self.current_image)

        # Create a painter to draw on the pixmap
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(QColor(255, 0, 0), 2))

        # Draw all boxes and add to listbox
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box["bbox"]]
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Draw class text
            text_y = y1 - 10 if y1 > 20 else y1 + 15
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawText(int((x1 + x2) / 2), int(text_y), str(box["class"]))
            painter.setPen(QPen(QColor(255, 0, 0), 2))

            # Add to listbox
            self.class_listbox.addItem(f"Class {box['class']}: {box['bbox']}")

        painter.end()
        self.canvas.setPixmap(self.pixmap)

    def end_draw(self, bbox):
        """
        Handles the completion of a manually drawn bounding box.

        Prompts the user to assign a class ID and adds the bounding box
        to the list of detections.

        Args:
            bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        """
        new_class, ok = QInputDialog.getInt(
            self, "New Class", "Enter class ID for new box:"
        )
        if ok:
            self.boxes.append({"bbox": bbox, "confidence": 1.0, "class": new_class})
            self.detections[os.path.basename(self.image_files[self.current_index])] = (
                self.boxes
            )
            self.update_ui()

    def delete_selected_box(self):
        """
        Deletes the currently selected bounding box from the class list.

        Updates the UI to reflect the removal.
        """
        selected_items = self.class_listbox.selectedItems()
        if selected_items:
            index = self.class_listbox.row(selected_items[0])
            del self.boxes[index]
            self.detections[os.path.basename(self.image_files[self.current_index])] = (
                self.boxes
            )
            self.update_ui()

    def previous_image(self):
        """
        Loads the previous image in the dataset, if available.

        Ensures the index does not go below zero.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        """
        Loads the next image in the dataset, if available.

        Ensures the index does not exceed the number of loaded images.
        """
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def export_labels(self):
        """
        Exports all labelled bounding boxes in the YOLO format.

        Saves labels as text files with the same name as the corresponding image.
        Displays a message indicating success or errors encountered.
        """
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded to export labels.")
            return

        exported_count = 0
        error_count = 0

        for image_path in self.image_files:
            image_name = os.path.basename(image_path)
            label_file = os.path.join(
                os.path.dirname(image_path), f"{os.path.splitext(image_name)[0]}.txt"
            )

            try:
                image = Image.open(image_path)
                width, height = image.size

                if image_name in self.detections:
                    with open(label_file, "w") as f:
                        for box in self.detections[image_name]:
                            x1, y1, x2, y2 = box["bbox"]
                            x_center = ((x1 + x2) / 2) / width
                            y_center = ((y1 + y2) / 2) / height
                            box_width = (x2 - x1) / width
                            box_height = (y2 - y1) / height
                            f.write(
                                f"{box['class']} {x_center} {y_center} {box_width} {box_height}\n"
                            )
                    exported_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error exporting {image_name}: {str(e)}")

        if error_count > 0:
            QMessageBox.warning(
                self,
                "Export Partial",
                f"Labels exported: {exported_count}\nErrors: {error_count}",
            )
        else:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Labels exported in YOLO format! ({exported_count} files)",
            )


if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Set app theme
    with open("../conf/theme.qss", "r", encoding="utf-8") as file:
        theme = file.read()
    app.setStyleSheet(theme)
    try:
        window = ImageLabeller()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"The application encountered a fatal error:\n\n{str(e)}",
        )
        sys.exit(1)
