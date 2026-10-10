from pathlib import Path
import cv2


#try:
import deepface as DeepFace  # pyright: ignore[reportMissingImports]
#except ImportError:
    #DeepFace = None



def detect_emotion(image_path):
    if DeepFace is None:
        print("Error: 'deepface' is not installed. Install it with: pip install deepface")
        return

    image_path = Path(image_path)

    # 1. Check whether the image exists.
    if not image_path.exists() or not image_path.is_file():
        print("Error: Image file not found.")
        return

    # 2. Read the original image.
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: Could not read this image.")
        return

    try:
        # 3. Analyze the image using DeepFace.
        results = DeepFace.analyze(
            img_path=str(image_path),
            actions=["emotion"],
            enforce_detection=True
        )

        if isinstance(results, dict):
            results = [results]

        # 4. Draw a box and label for every detected face.
        for index, face in enumerate(results, start=1):
            emotion = face["dominant_emotion"]
            scores = face["emotion"]

            region = face["region"]

            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]

            # Draw a rectangle around the face.
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Write the predicted expression.
            label = f"{emotion} ({index})"

            cv2.putText(
                image,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(f"\nFace {index}")
            print("Predicted expression:", emotion)

            print("Expression scores:")
            for name, score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            ):
                print(f"{name}: {score:.2f}%")

        # 5. Save the annotated image.
        output_path = image_path.with_name(
            image_path.stem + "_result.jpg"
        )

        saved = cv2.imwrite(str(output_path), image)

        if saved:
            print("\nResult image saved to:", output_path)
        else:
            print("\nError: Could not save the result image.")

    except Exception as error:
        print("Error during emotion detection:", error)


if __name__ == "__main__":
    image_path = input(
        "Enter the image path: "
    ).strip().strip('"')

    detect_emotion(image_path)
