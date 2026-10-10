import cv2
import json
import numpy as np

from pathlib import Path
from collections import deque

try:
    from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    load_model = None


# 1. Find the saved model and labels
PROJECT_PATH = Path(__file__).resolve().parent

model = load_model(PROJECT_PATH / "emotion_cnn.keras")

with open(
    PROJECT_PATH / "emotion_classes.json",
    "r",
    encoding="utf-8"
) as file:
    emotion_classes = json.load(file)


# 2. Load the face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load face detector.")


# 3. Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


# Keep recent predictions to reduce label flickering
recent_predictions = deque(maxlen=7)

frame_count = 0

print("Custom CNN webcam started. Press Q to quit.")


# 4. Process webcam frames
while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    frame_count += 1

    # Convert camera image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast for face detection
    detector_image = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        detector_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        recent_predictions.clear()

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    for (x, y, w, h) in faces:

        # Draw rectangle around the detected face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Run prediction every 5 frames
        if frame_count % 5 == 0:
            face = gray[y:y + h, x:x + w]

            # CNN was trained using 48x48 grayscale images
            face = cv2.resize(face, (48, 48))

            # Shape becomes (1, 48, 48, 1)
            face = face.astype("float32")
            face = face[np.newaxis, ..., np.newaxis]

            # Predict the seven expression categories
            probabilities = model.predict(
                face,
                verbose=0
            )[0]

            recent_predictions.append(probabilities)

        if recent_predictions:
            # Average recent predictions for smoother labels
            average_probabilities = np.mean(
                recent_predictions,
                axis=0
            )

            best_index = int(
                np.argmax(average_probabilities)
            )

            emotion = emotion_classes[best_index]
            confidence = average_probabilities[best_index]

            label = (
                f"{emotion}: {confidence * 100:.1f}%"
            )
        else:
            label = "Classifying..."

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Custom CNN Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 5. Release camera resources
cap.release()
cv2.destroyAllWindows()