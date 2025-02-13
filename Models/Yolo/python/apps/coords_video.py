"""
import Qt modules
"""

import json
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPointF, QLineF
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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

        # Main Layout (Horizontal)
        self.main_layout = QHBoxLayout(self)

        # Left Panel (Video/Image + Slider)
        self.left_panel = QVBoxLayout()

        # Graphics View and Scene
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.left_panel.addWidget(self.view)

        # Video Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.set_video_position)
        self.left_panel.addWidget(self.slider)

        self.main_layout.addLayout(self.left_panel)  # Add left panel to main layout

        # Right Panel (Controls)
        self.right_panel = QVBoxLayout()

        # Buttons
        self.load_button = QPushButton("Load Image/Video", self)
        self.load_button.clicked.connect(self.load_media)
        self.right_panel.addWidget(self.load_button)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_last_point)
        self.right_panel.addWidget(self.undo_button)

        self.play_pause_button = QPushButton("Play/Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.play_pause_button.setEnabled(False)
        self.right_panel.addWidget(self.play_pause_button)

        self.save_json_button = QPushButton("Save as JSON", self)
        self.save_json_button.clicked.connect(self.save_points_json)
        self.right_panel.addWidget(self.save_json_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_application)
        self.right_panel.addWidget(self.close_button)

        # Choose number of points
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(1, 10)
        self.threshold_spinbox.setValue(5)

        self.right_panel.addWidget(QLabel("Number of dots:"))
        self.right_panel.addWidget(self.threshold_spinbox)

        # Labels
        self.label = QLabel("Load an image or video to start", self)
        self.right_panel.addWidget(self.label)

        self.cursor_label = QLabel("Cursor: (X, Y)", self)
        self.right_panel.addWidget(self.cursor_label)

        # Edit Mode Toggle
        self.toggle_edit_button = QCheckBox("Edit Mode", self)
        self.toggle_edit_button.stateChanged.connect(self.toggle_edit_mode)
        self.right_panel.addWidget(self.toggle_edit_button)

        # Add right panel to main layout
        self.main_layout.addLayout(self.right_panel)

        self.setLayout(self.main_layout)

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

        # Polygon Attributes
        self.polygon_closed = False  # Keeps track of whether the polygon is closed

        # Zoom Attributes
        self.zoom_factor = 1.0

        # Viewing mode checkbox
        self.edit_mode = False

    def toggle_edit_mode(self, state):
        """
        Toggle Edit mode. If the program is in edit mode, the user can
        add/remove dots, if not then they can only view them.
        Args:
            state (bool): whether the program is in edit mode or not
        """
        self.edit_mode = state == Qt.Checked
        self.undo_button.setEnabled(self.edit_mode)  # Enable undo only in edit mode

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
        Load and display a video, could either be mp4, avi, mov, or mkv. The
        video is paused by default

        For now, it is assumed that the the user would always load a video
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
        Displays the given pixmap in the scene, along with any drawn points and
        polygons. If the polygon is closed, it is redrawn and filled.

        This method:
        - Clears the current scene and removes any existing pixmap.
        - Adds the new pixmap to the scene.
        - Redraws previously stored points.
        - If the polygon is closed, it redraws the polygon.
        - Adjusts the view to fit the new pixmap.
        - Temporarily disables and then reinstalls the event filter.

        Args:
            pixmap (QPixmap): The image to be displayed in the scene.
        """
        self.view.viewport().removeEventFilter(self)  # Temporarily disable event filter
        self.current_pixmap = pixmap

        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.clear()  # Clear scene before re-adding items
        self.scene.addItem(self.pixmap_item)

        # Redraw points
        polygon_points = []
        for x, y in self.points:
            self.draw_point(x, y)
            polygon_points.append(QPointF(x, y))

        # Redraw the polygon if it's closed
        if len(polygon_points) > 2 and self.polygon_closed:
            self.draw_polygon(polygon_points)

        self.view.setScene(self.scene)
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.view.viewport().installEventFilter(self)  # Re-enable event filter

    def draw_polygon(self, polygon_points):
        """
        Draws the polygon from the points

        This method draws the edges from a list of coordinates, if the polygon is
        closed (the last and first points are 1), it shades the area.
        Args:
            polygon_points (list of list of float): list of coordinates of the
            vertices
        """
        if len(polygon_points) < 2:
            return  # Not enough points to form a line

        # Remove previously drawn items (to avoid duplicates)
        for item in self.scene.items():
            if isinstance(item, (QGraphicsLineItem, QGraphicsPolygonItem)):
                self.scene.removeItem(item)

        pen = QPen(Qt.red, 2)
        polygon = QPolygonF(polygon_points)  # Convert to QPolygonF

        # Draw the edges
        for i in range(len(polygon_points) - 1):
            line = QGraphicsLineItem(QLineF(polygon_points[i], polygon_points[i + 1]))
            line.setPen(pen)
            self.scene.addItem(line)

        if self.polygon_closed:
            # Draw the closing edge
            closing_line = QGraphicsLineItem(
                QLineF(polygon_points[-1], polygon_points[0])
            )
            closing_line.setPen(pen)
            self.scene.addItem(closing_line)

            # Fill the polygon with transparency (add this LAST)
            brush = QBrush(QColor(255, 0, 0, 80))  # Red with 80/255 alpha transparency
            polygon_item = QGraphicsPolygonItem(polygon)
            polygon_item.setBrush(brush)
            polygon_item.setPen(QPen(Qt.NoPen))  # No border lines on the filled area
            self.scene.addItem(polygon_item)  # Add the filled polygon last
        if len(polygon_points) < 2:
            return  # Not enough points to form a line

        # Remove previously drawn items (to avoid duplicates)
        for item in self.scene.items():
            if isinstance(item, (QGraphicsLineItem, QGraphicsPolygonItem)):
                self.scene.removeItem(item)

        pen = QPen(Qt.red, 2)
        polygon = QPolygonF(polygon_points)  # Convert to QPolygonF

        # Draw the edges
        for i in range(len(polygon_points) - 1):
            line = QGraphicsLineItem(QLineF(polygon_points[i], polygon_points[i + 1]))
            line.setPen(pen)
            self.scene.addItem(line)

        if self.polygon_closed:
            # Draw the closing edge
            closing_line = QGraphicsLineItem(
                QLineF(polygon_points[-1], polygon_points[0])
            )
            closing_line.setPen(pen)
            self.scene.addItem(closing_line)

            # Fill the polygon with transparency (add this LAST)
            brush = QBrush(QColor(255, 0, 0, 80))  # Red with 80/255 alpha transparency
            polygon_item = QGraphicsPolygonItem(polygon)
            polygon_item.setBrush(brush)
            polygon_item.setPen(QPen(Qt.NoPen))  # No border lines on the filled area
            self.scene.addItem(polygon_item)  # Add the filled polygon last

    def close_polygon(self):
        """
        Closes the polygon by connecting the last point to the first.

        This method:
        - Ensures that there are at least three points before closing the polygon.
        - Sets the `polygon_closed` flag to `True`.
        - Adds the first point to the end of the `polygon_points` list to form a closed shape.
        - Calls `draw_polygon` to render the closed polygon.

        """
        if len(self.polygon_points) > 2:  # Need at least 3 points to close
            self.polygon_closed = True
            self.polygon_points.append(self.polygon_points[0])  # Connect last to first
            self.draw_polygon(self.polygon_points)

    def map_to_image_coordinates(self, pos):
        """
        Converts the cursor position in the view to image coordinates.

        This method maps a given cursor position from the view to the corresponding
        coordinates on the displayed image.

        Args:
            pos (QPoint): The cursor position in the view's coordinate system.

        Returns:
            tuple: A tuple (x, y) representing the corresponding coordinates
                   on the image.
        """
        scene_pos = self.view.mapToScene(pos)
        x, y = (
            scene_pos.x() - self.pixmap_item.x(),
            scene_pos.y() - self.pixmap_item.y(),
        )
        return x, y

    def eventFilter(self, obj, event):
        """
        Handles mouse events for cursor tracking, point marking, and closing the polygon.

        This method processes mouse interactions within the viewport, allowing the user to:
        - Track the cursor position and update the label dynamically.
        - Mark points on the image with left-click.
        - Close the polygon on a double-click if there are enough points.

        Args:
            obj (QObject): The object receiving the event (typically the viewport).
            event (QEvent): The event being processed (mouse movement, click, etc.).

        Returns:
            bool: True if the event is handled and should not propagate further, False otherwise.
        """
        threshold = self.threshold_spinbox.value()
        if obj == self.view.viewport() and self.pixmap_item:
            if (
                self.playing or not self.edit_mode
            ):  # Ignore events during video playback
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
                        if len(self.points) < threshold:
                            self.points.append((x, y))
                            self.draw_point(x, y)
                            self.cursor_label.setText(f"Clicked: ({int(x)}, {int(y)})")
                        if len(self.points) == threshold:
                            self.points.append(self.points[0])
                            self.polygon_closed = True
                            self.display_pixmap(self.current_pixmap)  # Redraw the scene

                        return True  # Stop recursion

                    # elif event.button() == Qt.RightButton:
                    #     # Remove the closest point
                    #     self.remove_nearest_point(x, y)
                    #     return True  # Stop recursion

            elif event.type() == event.MouseButtonDblClick:
                if len(self.points) > 2:
                    # Close the polygon by connecting the last point to the first
                    self.points.append(self.points[0])
                    self.polygon_closed = True
                    self.display_pixmap(self.current_pixmap)  # Redraw the scene

        return False  # Allow normal event propagation

    def draw_point(self, x, y):
        """
        Draw a red dot at the specified coordinates and connect it to the previous point with a line.

        This method draws a red dot at the position of the cursor when there's a
        left click. The coordinates takes the upper left corner of the image as
        (0,0)

        Args:
            x (float): the horizontal coordinate of the cursor
            y (float): the vertical coordinate of the cursor
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
        Undo the last added point and its associated line.

        This method handles two cases:
        1. If the polygon is closed (i.e., the last point connects to the first), only the closing line is removed.
        2. If the polygon is not closed, the last point and the line connecting it to the previous point are
        removed.

        Functionality:
        - Identifies and removes the last drawn line if applicable.
        - Removes the corresponding graphical point representation (red dot).
        - Updates the internal list of points.

        Returns:
            None
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

    # def remove_nearest_point(self, x, y, threshold=10):
    #     """
    #     Remove the nearest point to the given (x, y) position if within a threshold.
    #     (This function is not used)
    #     Args:
    #         x (float): horizontal position of the cursor at current position
    #         y (float): vertical position of the cursor at current position
    #         threshold (int = 10): how far can a dot be to be considered in this
    #         function (if the nearest point is still too far away, it might not
    #         be what the user is looking to remove)
    #     """
    #     if not self.points:
    #         return
    #
    #     # Find the nearest point
    #     nearest_index = None
    #     min_distance = float("inf")
    #
    #     for i, (px, py) in enumerate(self.points):
    #         distance = np.hypot(px - x, py - y)
    #         if distance < min_distance and distance < threshold:
    #             min_distance = distance
    #             nearest_index = i
    #
    #     # Remove the point if a valid one was found
    #     if nearest_index is not None:
    #         del self.points[nearest_index]
    #         self.display_pixmap(self.current_pixmap)  # Redraw without the removed point

    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts for common actions.

        Supported Key Bindings:
        - Spacebar: Toggles video playback (Play/Pause).
        - 'Z': Undoes the last added point.
        - 'Ctrl + S': Saves the drawn points as a JSON file.

        Args:
            event (QKeyEvent): The key press event triggered by user input.

        Returns:
            None
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
        Handle mouse wheel events for zooming in and out.

        - Scroll up (positive delta): Zooms in.
        - Scroll down (negative delta): Zooms out.

        Args:
            event (QWheelEvent): The wheel event triggered by the mouse scroll.

        Returns:
            None
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
        Zoom out on the image or video using the scroll wheel
        """
        self.view.scale(1 / 1.2, 1 / 1.2)
        self.zoom_factor /= 1.2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = VideoPointSelector()
    selector.show()
    sys.exit(app.exec_())
