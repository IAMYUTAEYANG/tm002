import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from keras.models import load_model
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Add start button
        self.start_button = QPushButton("Start")
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_camera)

        # Initialize webcam
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Load the model
        self.model = load_model("keras_Model.h5", compile=False)

        # Load the labels
        with open("labels.txt", "r") as file:
            self.labels = [line.strip() for line in file.readlines()]

    def start_camera(self):
        if not self.timer.isActive():
            self.timer.start(100)  # Update frame every 100 milliseconds
            self.start_button.setText("Stop")
        else:
            self.timer.stop()
            self.start_button.setText("Start")       

    def update_frame(self):
        # Read frame from webcam
        ret, frame = self.camera.read()
        if ret:
            # Convert BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to match model input size
            resized_frame = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)

            # Normalize frame
            normalized_frame = resized_frame.astype(np.float32) / 255.0

            # Predict using model
            prediction = self.model.predict(np.expand_dims(normalized_frame, axis=0))
            index = np.argmax(prediction)
            class_name = self.labels[index]
            confidence_score = prediction[0][index]

            # Display prediction result on the frame
            cv2.putText(frame_rgb, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"Confidence: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert numpy array to QImage
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display image on QLabel
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

            # Update QLabel with class name
            self.setWindowTitle(f"Webcam Viewer - Class: {class_name}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Webcam Viewer with Prediction")
    window.setGeometry(100, 100, 640, 480)
    window.show()
    sys.exit(app.exec_())
 