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

        # Polygon Toggle Checkbox
        self.toggle_polygon_checkbox = QCheckBox("Show Polygon", self)
        self.toggle_polygon_checkbox.setChecked(False)  # Default to visible
        self.toggle_polygon_checkbox.stateChanged.connect(self.toggle_polygon)
        self.right_panel.addWidget(self.toggle_polygon_checkbox)

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
        Toggle Edit mode.

        If the program is in edit mode, the user can
        add/remove dots, if not then they can only view them.

        Args:
            state (bool): whether the program is in edit mode or not

        """
        self.edit_mode = state == Qt.Checked
        self.undo_button.setEnabled(self.edit_mode)  # Enable undo only in edit mode

    def toggle_polygon(self, state):
        """
        Toggles the visibility of the polygon.

        If the checkbox is checked, the polygon is drawn using the currently marked points.
        If unchecked, the polygon is removed from the scene.

        Args:
            state (Qt.CheckState): The state of the checkbox (checked or unchecked).

        """
        if state == Qt.Checked:
            self.draw_polygon([QPointF(x, y) for x, y in self.points])
        else:
            if hasattr(self, "polygon_item") and self.polygon_item:
                self.scene.removeItem(self.polygon_item)
                self.polygon_item = None  # Remove reference

    def load_media(self):
        """
        Opens a file dialog to load an image or video.

        Behavior:
            - Allows the user to select an image (.png, .jpg, .jpeg, .bmp, .gif)
              or video (.mp4, .avi, .mov, .mkv).
            - Determines whether the selected file is an image or video.
            - Calls `load_video()` if a video is selected.
            - Calls `load_image()` if an image is selected.

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
        Loads and displays an image from the selected file.

        - Reads the image file and converts it into a QPixmap.
        - If the image fails to load, an error message is displayed.
        - Calls `display_pixmap()` to render the image in the scene.

        """
        pixmap = QPixmap(self.media_path)
        if pixmap.isNull():
            self.label.setText("Error loading image")
            return
        self.display_pixmap(pixmap)

    def load_video(self):
        """
        Loads and initializes video playback.

        - Opens the selected video file and sets it for playback.
        - If the video fails to load, an error message is displayed.
        - Enables video controls (play/pause button and slider).
        - Calls `next_frame()` to display the first frame of the video.

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
        Reads and displays the next frame of the video.

        - Advances to the next frame in the video.
        - If the end of the video is reached, it loops back to the start.
        - Converts the frame from OpenCV format (BGR) to Qt format (RGB).
        - Calls `display_pixmap()` to render the frame.
        - If the polygon toggle is enabled, redraws the polygon on the frame.

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

            # Ensure the polygon is redrawn on new frames
            if self.toggle_polygon_checkbox.isChecked() and len(self.points) > 1:
                self.draw_polygon([QPointF(x, y) for x, y in self.points])

    def set_video_position(self, position):
        """
        Sets the video to a specific frame based on the slider position.

        - Stops the timer to prevent automatic playback while seeking.
        - Moves the video to the specified frame position.
        - Calls `next_frame()` to display the new frame.
        - Resumes playback if the video was previously playing.

        Args:
            position (int): The frame index to seek to.

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
        Toggles video playback between play and pause.

        - If the video is currently playing, it stops playback.
        - If the video is paused, it starts playback with a frame update every 30 ms.

        """
        if self.playing:
            self.timer.stop()  # Stop playback
        else:
            self.timer.start(30)  # Adjust frame rate
        self.playing = not self.playing

    def display_pixmap(self, pixmap):
        """
        Displays the given pixmap in the scene while preserving drawn points and polygons.

        This function updates the scene with the provided pixmap, ensuring that any previously
        drawn points or polygons remain visible. If the polygon toggle is enabled and the
        required number of points exist, the polygon is redrawn.

        Args:
            pixmap (QPixmap): The image to be displayed in the scene.

        Behavior:
            - Removes the existing pixmap item (if any) and replaces it with the new one.
            - Redraws previously added points.
            - If the polygon toggle is enabled and at least two points exist, the polygon is redrawn.
            - Ensures event filters are properly managed for the view.
        """
        self.view.viewport().removeEventFilter(self)  # Temporarily disable event filter
        self.current_pixmap = pixmap

        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)  # Do NOT clear the scene anymore

        # Redraw points
        polygon_points = []
        for x, y in self.points:
            self.draw_point(x, y)
            polygon_points.append(QPointF(x, y))

        # Redraw polygon if enabled
        if self.toggle_polygon_checkbox.isChecked() and len(polygon_points) > 1:
            self.draw_polygon(polygon_points)

        self.view.setScene(self.scene)
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.view.viewport().installEventFilter(self)  # Re-enable event filter

    def draw_polygon(self, polygon_points):
        """
        Draws a polygon based on the given points if the polygon toggle is enabled.

        This function either draws an open or closed polygon depending on the state
        of `self.polygon_closed`. If the polygon is closed, it is filled with a
        semi-transparent red color. Otherwise, only the edges are drawn.

        Args:
            polygon_points (list of QPointF): A list of points defining the polygon.

        Behavior:
            - If fewer than two points are provided or the polygon toggle is off, the function exits.
            - Removes any previously drawn polygon before drawing a new one.
            - If `self.polygon_closed` is `True`, a filled polygon is drawn.
            - If `self.polygon_closed` is `False`, edges between points are drawn.
            - If the polygon is open, the final edge is not drawn unless `self.polygon_closed` is `True`.
        """
        if len(polygon_points) < 2 or not self.toggle_polygon_checkbox.isChecked():
            return  # Not enough points OR checkbox is off

        # Remove previous polygon if it exists
        if hasattr(self, "polygon_item") and self.polygon_item:
            self.scene.removeItem(self.polygon_item)

        polygon = QPolygonF(polygon_points)
        pen = QPen(Qt.red, 2)

        if self.polygon_closed:
            # Create and store the filled polygon
            brush = QBrush(QColor(255, 0, 0, 80))  # Red with transparency
            self.polygon_item = QGraphicsPolygonItem(polygon)
            self.polygon_item.setBrush(brush)
            self.polygon_item.setPen(pen)
            self.scene.addItem(self.polygon_item)
        else:
            # Draw the edges
            for i in range(len(polygon_points) - 1):
                line = QGraphicsLineItem(
                    QLineF(polygon_points[i], polygon_points[i + 1])
                )
                line.setPen(pen)
                self.scene.addItem(line)

            if self.polygon_closed:
                closing_line = QGraphicsLineItem(
                    QLineF(polygon_points[-1], polygon_points[0])
                )
                closing_line.setPen(pen)
                self.scene.addItem(closing_line)

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
                            self.polygon_closed = True

                        return True  # Stop recursion

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

    def undo_last_point(self):
        """
        Removes the last drawn point and updates the scene accordingly.

        This function deletes the most recently added point from `self.points`, clears
        the scene (except for the video frame), and then redraws the remaining points
        and polygon (if applicable).

        Behavior:
            - If no points exist, the function exits.
            - The last point in `self.points` is removed.
            - The scene is cleared of all drawn points and polygons while keeping the video frame.
            - Remaining points are redrawn.
            - If polygon mode is enabled and at least two points remain, the polygon is redrawn.

        """
        if not self.points:
            return  # Nothing to undo

        # Remove last point
        self.points.pop()

        # Clear everything except the video frame
        for item in self.scene.items():
            if isinstance(item, QGraphicsEllipseItem) or isinstance(
                item, QGraphicsPolygonItem
            ):
                self.scene.removeItem(item)

        # Redraw remaining points
        polygon_points = []
        for x, y in self.points:
            self.draw_point(x, y)
            polygon_points.append(QPointF(x, y))

        # Redraw polygon if needed
        if self.toggle_polygon_checkbox.isChecked() and len(polygon_points) > 1:
            self.draw_polygon(polygon_points)

        self.view.setScene(self.scene)

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
        Saves the marked points to a JSON file.

        This function opens a file dialog to let the user select a location and filename
        to save the points. The marked points are then written to the file in JSON format.

        Behavior:
            - Opens a QFileDialog for the user to choose the save location.
            - Saves `self.points` as a JSON file with indentation for readability.
            - If no file path is selected, the function does nothing.

        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Points as JSON", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.points, f, indent=4)

    def close_application(self):
        """
        Closes the application and releases resources.

        If a video is currently open, this function releases it before closing
        the application window.

        Behavior:
            - Releases `self.video` if it is active.
            - Closes the application window.

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
        Zooms in on the image or video.

        This function increases the zoom level by scaling the view by a factor of 1.2.
        The zoom factor is also updated to track the current zoom level.

        """
        self.view.scale(1.2, 1.2)
        self.zoom_factor *= 1.2

    def zoom_out(self):
        """
        Zooms out on the image or video.

        This function decreases the zoom level by scaling the view down by a factor of 1.2.
        The zoom factor is also updated accordingly.

        """
        self.view.scale(1 / 1.2, 1 / 1.2)
        self.zoom_factor /= 1.2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = VideoPointSelector()
    selector.show()
    sys.exit(app.exec_())
