import json
import os
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
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ConnectionDialog(QDialog):

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
        self.apply_theme()

    def get_ip(self):
        return self.ip_input.text()

    def apply_theme(self):
        """
        Apply Catppuccin Macchiato color scheme.

        This method looks at ../conf/catppuccin_macchiato.qss and apply the theme
        """

        qss_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "catppuccin_macchiato.qss"
        )
        qss_path = os.path.abspath(qss_path)
        try:
            with open(qss_path, "r") as file:
                self.setStyleSheet(file.read())
                print("Stylesheet applied successfully.")
        except FileNotFoundError:
            print(f"Stylesheet not found at {qss_path}")


class CredentialsDialog(QDialog):

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
        self.profile_combo.addItems(["profile1", "profile2", "profile3"])
        layout.addRow("Profile:", self.profile_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self.apply_theme()

    def get_credentials(self):
        return (
            self.username_input.text(),
            self.password_input.text(),
            self.profile_combo.currentText(),
        )

    def apply_theme(self):
        """
        Apply Catppuccin Macchiato color scheme.

        This method looks at ../conf/catppuccin_macchiato.qss and apply the theme
        """

        qss_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "catppuccin_macchiato.qss"
        )
        qss_path = os.path.abspath(qss_path)
        try:
            with open(qss_path, "r") as file:
                self.setStyleSheet(file.read())
                print("Stylesheet applied successfully.")
        except FileNotFoundError:
            print(f"Stylesheet not found at {qss_path}")


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
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        self.right_panel.addWidget(self.load_button)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_last_point)
        self.right_panel.addWidget(self.undo_button)

        self.play_pause_button = QPushButton("Play/Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.play_pause_button.setEnabled(False)
        self.right_panel.addWidget(self.play_pause_button)

        self.save_json_button = QPushButton("Save coordinates", self)
        self.save_json_button.clicked.connect(self.save_points_json)
        self.right_panel.addWidget(self.save_json_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_application)
        self.right_panel.addWidget(self.close_button)

        # Choose number of points
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(1, 10)
        self.threshold_spinbox.setValue(5)

        self.right_panel.addWidget(QLabel("Maximum vertices threshold:"))
        self.right_panel.addWidget(self.threshold_spinbox)

        # Labels
        self.label = QLabel("Load a video to start", self)
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
        self.video_capture: Optional[cv2.VideoCapture] = None
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

        # Apply Catppuccin Macchiato Theme
        self.apply_theme()

    def apply_theme(self):
        """
        Apply Catppuccin Macchiato color scheme.

        This method looks at ../conf/catppuccin.qss and apply the theme
        """

        qss_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "catppuccin_macchiato.qss"
        )
        qss_path = os.path.abspath(qss_path)
        try:
            with open(qss_path, "r") as file:
                self.setStyleSheet(file.read())
                print("Stylesheet applied successfully.")
        except FileNotFoundError:
            print(f"Stylesheet not found at {qss_path}")

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

    def load_video(self):
        """
        Loads and initializes video playback.

        - Opens the selected video file and sets it for playback.
        - If the video fails to load, an error message is displayed.
        - Enables video controls (play/pause button and slider).
        - Calls `next_frame()` to display the first frame of the video.

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
        self.timer.start()
        self.next_frame()

    def next_frame(self):
        if self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            self.current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
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
        if self.video:
            self.timer.stop()  # Prevent frame updates while seeking
            self.video.set(cv2.CAP_PROP_POS_FRAMES, position)

            # Only update the frame if it's different from the current frame
            if self.current_frame != position:
                self.current_frame = position
                self.next_frame()

            self.timer.start(30)  # Resume if playing

    def toggle_playback(self):
        if self.playing:
            self.timer.stop()  # Stop playback
        else:
            self.timer.start(30)  # Adjust frame rate
        self.playing = not self.playing

    def display_pixmap(self, pixmap):
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
        scene_pos = self.view.mapToScene(pos)
        x, y = (
            scene_pos.x() - self.pixmap_item.x(),
            scene_pos.y() - self.pixmap_item.y(),
        )
        return x, y

    def eventFilter(self, obj, event):
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
        # Draw the red dot
        dot = self.scene.addEllipse(x - 5, y - 5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
        dot.setZValue(1)

    def undo_last_point(self):
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
        if event.key() == Qt.Key_Space:
            self.toggle_playback()  # Space: Play/Pause
        elif event.key() == Qt.Key_Z:
            self.undo_last_point()  # Z: Undo last point
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_points_json()  # Ctrl + S: Save JSON
        else:
            super().keyPressEvent(event)  # Pass other events to the default handler

    def save_points_json(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Points as JSON", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.points, f, indent=4)

    def close_application(self):
        if self.video:
            self.video.release()
        self.close()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self.view.scale(1.2, 1.2)
        self.zoom_factor *= 1.2

    def zoom_out(self):
        self.view.scale(1 / 1.2, 1 / 1.2)
        self.zoom_factor /= 1.2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = VideoPointSelector()
    selector.show()
    sys.exit(app.exec_())
