import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is required to run this evaluation script. "
        "Install the project dependencies first."
    ) from exc

from sklearn.metrics import classification_report, confusion_matrix


# Project paths
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "emotion_cnn.keras"
LABELS_PATH = BASE_DIR / "emotion_classes.json"

# Load saved emotion labels
with open(LABELS_PATH, "r", encoding="utf-8") as file:
    emotion_classes = json.load(file)

# Load the trained CNN
model = tf.keras.models.load_model(MODEL_PATH)

# Load test images without shuffling
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=emotion_classes,
    color_mode="grayscale",
    image_size=(48, 48),
    batch_size=64,
    shuffle=False
)

# Get the actual labels
actual_labels = np.concatenate([
    labels.numpy() for _, labels in test_data
])

# Predict the expressions
probabilities = model.predict(test_data, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

# Print evaluation metrics
print("\nCLASSIFICATION REPORT\n")

report = classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(len(emotion_classes))),
    target_names=emotion_classes,
    zero_division=0
)

print(report)

# Save the report
with open(
    BASE_DIR / "evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# Create the confusion matrix
matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(emotion_classes)))
)

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(matrix, cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set_xticks(range(len(emotion_classes)))
ax.set_yticks(range(len(emotion_classes)))

ax.set_xticklabels(emotion_classes, rotation=45, ha="right")
ax.set_yticklabels(emotion_classes)

ax.set_xlabel("Predicted expression")
ax.set_ylabel("Actual expression")
ax.set_title("CNN Confusion Matrix")

for row in range(len(emotion_classes)):
    for column in range(len(emotion_classes)):
        ax.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
            fontsize=8
        )

fig.tight_layout()
fig.savefig(
    BASE_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved evaluation_report.txt")
print("Saved confusion_matrix.png")