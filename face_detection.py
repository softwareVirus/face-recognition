import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import os
from datetime import datetime
import time  # Import the time module

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Specify the folder to save the cropped faces
faces_folder = r"C:\Users\dogan\OneDrive\Desktop\Bitirme Projesi Dataset\Face Dataset"
os.makedirs(faces_folder, exist_ok=True)


def scale_image(image, scale_factor=1.5):
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)


def main_loop():
    last_time = time.time()  # Initialize the last_time variable

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            current_time = time.time()  # Get the current time
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

            # Scale up the screenshot
            scaled_screenshot = scale_image(screenshot, scale_factor=1.5)

            # Detect faces
            results = face_detection.process(cv2.cvtColor(scaled_screenshot, cv2.COLOR_BGR2RGB))

            if results.detections and current_time - last_time >= 0.1:  # Check if 50ms have passed
                for idx, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = scaled_screenshot.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)

                    # Adjust coordinates for original image size
                    x, y, w, h = int(x / 1.5), int(y / 1.5), int(w / 1.5), int(h / 1.5)

                    cropped_face = screenshot[max(0, y):min(y + h, ih), max(0, x):min(x + w, iw)];

                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                last_time = current_time  # Update the last_time variable

                # Now draw the rectangle around the face on the original screenshot for visualization
                cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the screenshot with detected faces
            cv2.imshow('Real-time Face Detection', screenshot)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
