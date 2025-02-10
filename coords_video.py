"""

import necessary modules for the Qt application
"""

import json
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QImage, QPen, QPixmap
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


class VideoPointSelector(QWidget):
    """

    Attributes:
        layout: dimension of the window
        load_button: load video/image
        save_json_button: run save_points_jason()
        undo_button: run undo_last_point()
        play_pause_button: run toggle_playback()
        close_button: run close_application()
        label: prompts the user to load a video to start
        cursor_label: the cursor's current position
        scene: background to the window
        view: the window
        media_path: path to the video/image
        points: the dots the the user clicks on the scene
        current_frame: the current frame/image being processed
        pixmap_item: the pixmap of the image/video
        current_pixmap: the pixmap of the current frame
        video: the video file
        timer: time into the video
        playing: if the video is playing
        current_pixmap: video's pixmap at currnet state
        pixmap_item: video's original pixmap
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video & Image Point Selector")
        self.setGeometry(100, 100, 1280, 800)

        # UI Components
        self.layout = QVBoxLayout()

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

        self.label = QLabel("Load an image or video to start", self)
        self.layout.addWidget(self.label)

        self.cursor_label = QLabel("Cursor: (X, Y)", self)
        self.layout.addWidget(self.cursor_label)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        self.layout.addWidget(self.view)

        self.setLayout(self.layout)

        self.media_path = None
        self.points = []  # Store points globally across frames
        self.current_frame = 0

        self.pixmap_item = None
        self.current_pixmap = None
        self.view.viewport().installEventFilter(self)

        self.video = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

    def load_media(self):
        """

        detect if the media is image or video, then calls the function to
        load them into the program
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
        """load the image into the program"""
        pixmap = QPixmap(self.media_path)
        if pixmap.isNull():
            self.label.setText("Error loading image")
            return
        self.display_pixmap(pixmap)

    def load_video(self):
        """load the video into the program"""
        self.video = cv2.VideoCapture(self.media_path)
        if not self.video.isOpened():
            self.label.setText("Error loading video")
            return
        self.play_pause_button.setEnabled(True)
        self.next_frame()

    def next_frame(self):
        """read and display the next frame in the video"""
        if self.video:
            ret, frame = self.video.read()
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            self.current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.display_pixmap(pixmap)

    def toggle_playback(self):
        """play the video is paused, pause the video if playing"""
        if self.playing:
            self.timer.stop()  # stop playback
        else:
            self.timer.start(30)  # Adjust frame rate
        self.playing = not self.playing

    def display_pixmap(self, pixmap):
        """
        updates the displayed image in the QGraphicsView and redraws previously marked
        points.

        clears the existing image from the QGraphicsScene, replaces it
        with a new QPixmap, and ensures it fits within the view while maintaining the
        aspect ratio. It also redraws any previously marked points to retain user
        interactions.

        Args:
            pixmap (QPixmap): The new image to be displayed in the QGraphicsView.
        """
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

    def map_to_image_coordinates(self, pos):
        """
        convert the position of the cursor to its coordinates according to the
        image
        Args:
            pos (tuple): the cursor's current coordinate

        Returns:
            a tuple: the cursor's current coordinate with regards to the image"""
        scene_pos = self.view.mapToScene(pos)
        x, y = (
            scene_pos.x() - self.pixmap_item.x(),
            scene_pos.y() - self.pixmap_item.y(),
        )
        return x, y

    def eventFilter(self, obj, event):
        """
        handles mouse events to track cursor movement and register clicks on the image.

        filter mouse events within the QGraphicsView to:
        - update the cursor position label when the mouse moves over the image.
        - register left mouse clicks and store the clicked coordinates.
        - draw a red dot at each clicked position.

        Args:
            obj (QObject): The object that received the event.
            event (QEvent): The event being processed.

        Returns:
            bool: Returns the result of the superclass's eventFilter method to allow
            normal event processing.
        """
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
        """
        draw the dot on the frame
        Args:
            x (float): x position of the dot
            y (float): y position of the dot
        """
        dot = self.scene.addEllipse(x - 5, y - 5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
        dot.setZValue(1)

    def undo_last_point(self):
        """remove last dot from the list"""
        if self.points:
            self.points.pop()
            self.display_pixmap(self.current_pixmap)

    def save_points_json(self):
        """

        save the dots' coordinate to a json file in the following format:
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

    def close_application(self):
        """quits the application"""
        if self.video:
            self.video.release()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = VideoPointSelector()
    selector.show()
    sys.exit(app.exec_())
