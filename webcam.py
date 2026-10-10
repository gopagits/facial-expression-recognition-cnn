import cv2
#try:
import deepface as DeepFace  # pyright: ignore[reportMissingImports]
#except ImportError:
    #DeepFace = None

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    print("ERROR: Face detector did not load!")
    exit()
else:
    print("Face detector loaded successfully!")

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
emotion = "Detecting..."

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    frame_count += 1

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(30, 30)
    )

    if len(faces) == 0:
        cv2.putText(
            frame, "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        # Analyze emotion every 15 frames
        if frame_count % 15 == 0:
            face_image = frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(
                    img_path=face_image,
                    actions=["emotion"],
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(result, list):
                    result = result[0]

                emotion = result["dominant_emotion"]

            except Exception as error:
                print("Emotion analysis error:", error)
                emotion = "Analysis failed"

        # Display the predicted emotion
        cv2.putText(
            frame, emotion,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()