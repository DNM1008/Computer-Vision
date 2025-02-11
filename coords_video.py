"""

"""

import json
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QImage, QPen, QPixmap, QPainter, QKeySequence
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
    QSlider,
    QHBoxLayout,
    QShortcut,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
)


class VideoPointSelector(QWidget):
    """
    A PyQt5 application for selecting points on images or videos.
    Includes a slider for video navigation, zooming, and panning.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video & Image Point Selector")
        self.setGeometry(100, 100, 2560, 1440)

        # UI Components
        self.layout = QVBoxLayout()

        # Buttons
        self.load_button = QPushButton("Load Image/Video", self)
        self.load_button.clicked.connect(self.load_media)
        self.layout.addWidget(self.load_button)

        self.save_json_button = QPushButton("Save as JSON", self)
        self.save_json_button.clicked.connect(self.save_points_json)
        self.layout.addWidget(self.save_json_button)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_last_point)
        self.layout.addWidget(self.undo_button)

        self.play_pause_button = QPushButton("Play/Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.layout.addWidget(self.play_pause_button)
        self.play_pause_button.setEnabled(False)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.close_button)

        # Labels
        self.label = QLabel("Load an image or video to start", self)
        self.layout.addWidget(self.label)

        self.cursor_label = QLabel("Cursor: (X, Y)", self)
        self.layout.addWidget(self.cursor_label)

        # Video Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.set_video_position)
        self.layout.addWidget(self.slider)

        # Graphics View and Scene
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.layout.addWidget(self.view)

        self.setLayout(self.layout)

        # Media Attributes
        self.media_path = None
        self.points = []  # Store points globally across frames
        self.current_frame = 0

        # Graphics Attributes
        self.pixmap_item = None
        self.current_pixmap = None
        self.view.viewport().installEventFilter(self)

        # Video Attributes
        self.video = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

        # Zoom Attributes
        self.zoom_factor = 1.0

    def load_media(self):
        """
        Open a file dialog to load an image or video.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image or Video",
            "",
            "Images/Videos (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.media_path = file_path
            if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.load_video()
            else:
                self.load_image()

    def load_image(self):
        """
        Load and display an image.
        """
        pixmap = QPixmap(self.media_path)
        if pixmap.isNull():
            self.label.setText("Error loading image")
            return
        self.display_pixmap(pixmap)

    def load_video(self):
        """
        Load and initialize video playback.
        """
        self.video = cv2.VideoCapture(self.media_path)
        if not self.video.isOpened():
            self.label.setText("Error loading video")
            return

        # Enable video controls
        self.play_pause_button.setEnabled(True)
        self.slider.setEnabled(True)
        self.slider.setMaximum(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        self.next_frame()

    def next_frame(self):
        """
        Read and display the next frame in the video.
        """
        if self.video:
            ret, frame = self.video.read()
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            self.current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(self.current_frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.display_pixmap(pixmap)

    def set_video_position(self, position):
        """
        Set the video to a specific frame based on the slider position.
        """
        if self.video:
            self.timer.stop()  # Prevent frame updates while seeking
            self.video.set(cv2.CAP_PROP_POS_FRAMES, position)

            # Only update the frame if it's different from the current frame
            if self.current_frame != position:
                self.current_frame = position
                self.next_frame()

            self.timer.start(30)  # Resume if playing

    def toggle_playback(self):
        """
        Toggle video playback between play and pause.
        """
        if self.playing:
            self.timer.stop()  # Stop playback
        else:
            self.timer.start(30)  # Adjust frame rate
        self.playing = not self.playing

    def display_pixmap(self, pixmap):
        """
        Update the displayed image or video frame and redraw any marked points.
        """
        self.view.viewport().removeEventFilter(self)  # Disable event filter temporarily
        self.current_pixmap = pixmap

        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.clear()
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        # Redraw all saved points
        for x, y in self.points:
            self.draw_point(x, y)

        self.view.viewport().installEventFilter(self)  # Re-enable event filter

    def map_to_image_coordinates(self, pos):
        """
        Convert the cursor position to image coordinates.
        """
        scene_pos = self.view.mapToScene(pos)
        x, y = (
            scene_pos.x() - self.pixmap_item.x(),
            scene_pos.y() - self.pixmap_item.y(),
        )
        return x, y

    def eventFilter(self, obj, event):
        """
        Handle mouse events for cursor tracking, point marking, and closing the shape on double-click.
        """
        if obj == self.view.viewport() and self.pixmap_item:
            if self.playing:  # Ignore events during video playback
                return False

            if event.type() == event.MouseMove:
                x, y = self.map_to_image_coordinates(event.pos())
                new_text = f"Cursor: ({int(x)}, {int(y)})"

                if self.cursor_label.text() != new_text:
                    self.cursor_label.setText(new_text)

                return True  # Stop recursion

            elif event.type() == event.MouseButtonPress:
                x, y = self.map_to_image_coordinates(event.pos())

                if (
                    0 <= x <= self.current_pixmap.width()
                    and 0 <= y <= self.current_pixmap.height()
                ):
                    if event.button() == Qt.LeftButton:
                        # Add a new point
                        self.points.append((x, y))
                        self.draw_point(x, y)
                        self.cursor_label.setText(f"Clicked: ({int(x)}, {int(y)})")
                        return True  # Stop recursion

                    elif event.button() == Qt.RightButton:
                        # Remove the closest point
                        self.remove_nearest_point(x, y)
                        return True  # Stop recursion

            elif event.type() == event.MouseButtonDblClick:
                # If double-click detected, close the shape
                if len(self.points) > 1:
                    first_x, first_y = self.points[0]  # First point
                    last_x, last_y = self.points[-1]  # Last point
                    self.scene.addLine(
                        last_x, last_y, first_x, first_y, QPen(Qt.red, 2)
                    )
                    return True  # Stop recursion

        return False  # Allow normal event propagation

    def draw_point(self, x, y):
        """
        Draw a red dot at the specified coordinates and connect it to the previous point with a line.
        """
        # Draw the red dot
        dot = self.scene.addEllipse(x - 5, y - 5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
        dot.setZValue(1)

        # Draw a line to the previous point if it exists
        if len(self.points) > 1:
            last_x, last_y = self.points[-2]  # Get the previous point
            line = self.scene.addLine(
                last_x, last_y, x, y, QPen(Qt.red, 2)
            )  # Draw a red line
            line.setZValue(0)  # Make sure the line is below the dots

    def undo_last_point(self):
        """
        Undo the last added point.
        - If the shape is closed (last point connects to the first), remove only the closing line.
        - If the shape is not closed, remove the last point and its connecting line.
        """
        if not self.points:
            return  # No points to undo

        last_point = self.points[-1]
        first_point = self.points[0]

        # Find the closing line (if it exists)
        closing_line = None
        for item in self.scene.items():
            if isinstance(item, QGraphicsLineItem):
                line = item.line()
                if (
                    (round(line.p1().x()), round(line.p1().y()))
                    == (round(first_point[0]), round(first_point[1]))
                    and (round(line.p2().x()), round(line.p2().y()))
                    == (round(last_point[0]), round(last_point[1]))
                ) or (
                    (round(line.p1().x()), round(line.p1().y()))
                    == (round(last_point[0]), round(last_point[1]))
                    and (round(line.p2().x()), round(line.p2().y()))
                    == (round(first_point[0]), round(first_point[1]))
                ):
                    closing_line = item
                    break

        if closing_line:
            # Case 1: Shape is closed -> Remove only the closing line
            self.scene.removeItem(closing_line)
        else:
            # Case 2: Shape is NOT closed -> Remove last line and last point
            if len(self.points) > 1:
                second_last_point = self.points[-2]
                last_line = None

                for item in self.scene.items():
                    if isinstance(item, QGraphicsLineItem):
                        line = item.line()
                        if (
                            (round(line.p1().x()), round(line.p1().y()))
                            == (
                                round(second_last_point[0]),
                                round(second_last_point[1]),
                            )
                            and (round(line.p2().x()), round(line.p2().y()))
                            == (round(last_point[0]), round(last_point[1]))
                        ) or (
                            (round(line.p1().x()), round(line.p1().y()))
                            == (round(last_point[0]), round(last_point[1]))
                            and (round(line.p2().x()), round(line.p2().y()))
                            == (
                                round(second_last_point[0]),
                                round(second_last_point[1]),
                            )
                        ):
                            last_line = item
                            break

                if last_line:
                    self.scene.removeItem(last_line)

            # Remove the last point from the list
            self.points.pop()

            # Remove only the last point (red dot) from the scene
            for item in self.scene.items():
                if isinstance(item, QGraphicsEllipseItem):  # Red dot
                    if round(item.rect().center().x()) == round(
                        last_point[0]
                    ) and round(item.rect().center().y()) == round(last_point[1]):
                        self.scene.removeItem(item)
                        break

    def remove_nearest_point(self, x, y, threshold=10):
        """
        Remove the nearest point to the given (x, y) position if within a threshold.
        """
        if not self.points:
            return

        # Find the nearest point
        nearest_index = None
        min_distance = float("inf")

        for i, (px, py) in enumerate(self.points):
            distance = np.hypot(px - x, py - y)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                nearest_index = i

        # Remove the point if a valid one was found
        if nearest_index is not None:
            del self.points[nearest_index]
            self.display_pixmap(self.current_pixmap)  # Redraw without the removed point

    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts for common actions.
        """
        if event.key() == Qt.Key_Space:
            self.toggle_playback()  # Space: Play/Pause
        elif event.key() == Qt.Key_Z:
            self.undo_last_point()  # Z: Undo last point
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_points_json()  # Ctrl + S: Save JSON
        else:
            super().keyPressEvent(event)  # Pass other events to the default handler

    def save_points_json(self):
        """
        Save the marked points to a JSON file.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Points as JSON", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.points, f, indent=4)

    def close_application(self):
        """
        Close the application and release resources.
        """
        if self.video:
            self.video.release()
        self.close()

    def wheelEvent(self, event):
        """
        Handle mouse wheel events for zooming.
        """
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        """
        Zoom in on the image or video.
        """
        self.view.scale(1.2, 1.2)
        self.zoom_factor *= 1.2

    def zoom_out(self):
        """
        Zoom out on the image or video.
        """
        self.view.scale(1 / 1.2, 1 / 1.2)
        self.zoom_factor /= 1.2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = VideoPointSelector()
    selector.show()
    sys.exit(app.exec_())
