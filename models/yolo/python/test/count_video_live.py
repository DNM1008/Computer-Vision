import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QDialog,
    QFormLayout,
    QComboBox,
    QDialogButtonBox,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


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

    def get_ip(self):
        return self.ip_input.text()


class CredentialsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter Credentials")

        layout = QFormLayout()

        self.username_input = QLineEdit()
        layout.addRow("Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addRow("Password:", self.password_input)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["profile1", "profile2", "profile3"])
        layout.addRow("Profile:", self.profile_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_credentials(self):
        return (
            self.username_input.text(),
            self.password_input.text(),
            self.profile_combo.currentText(),
        )


class IPCameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IP Camera Viewer - PyQt5")

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # QLabel to show video
        self.video_label = QLabel("No video feed")
        self.video_label.setScaledContents(True)
        self.layout.addWidget(self.video_label)

        # Connect button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_to_camera)
        self.layout.addWidget(self.connect_button)

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def connect_to_camera(self):
        # Step 1: Get IP Address
        ip_dialog = ConnectionDialog()
        if ip_dialog.exec_() != QDialog.Accepted:
            return
        ip_address = ip_dialog.get_ip()

        # Step 2: Get Credentials
        cred_dialog = CredentialsDialog()
        if cred_dialog.exec_() != QDialog.Accepted:
            return
        username, password, profile = cred_dialog.get_credentials()

        # Step 3: Build RTSP URL
        rtsp_url = f"rtsp://{username}:{password}@{ip_address}:554/{profile}/media.smp"

        # Step 4: Connect to Camera
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            self.video_label.setText("Failed to connect to camera.")
            return

        self.video_label.setText("Connected. Streaming...")
        self.timer.start(30)  # ~30 FPS
        self.connect_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            else:
                self.video_label.setText("Failed to read frame.")

    def stop_stream(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.setText("Stream stopped.")
        self.connect_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = IPCameraViewer()
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec_())
