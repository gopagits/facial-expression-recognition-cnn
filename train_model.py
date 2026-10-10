import json
from pathlib import Path

import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore


# --------------------------------------------------
# 1. Basic settings
# --------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_PATH / "dataset"

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 10
SEED = 42

print("TensorFlow version:", tf.__version__)
print("Loading dataset...")


# --------------------------------------------------
# 2. Check dataset folders
# --------------------------------------------------

if not TRAIN_PATH.is_dir() or not TEST_PATH.is_dir():
    raise FileNotFoundError(
        "Could not find dataset/train and dataset/test."
    )

class_names = sorted(
    folder.name
    for folder in TRAIN_PATH.iterdir()
    if folder.is_dir()
)

if len(class_names) != 7:
    raise ValueError(
        f"Expected 7 emotion folders, found {class_names}"
    )

print("Emotion classes:", class_names)


# --------------------------------------------------
# 3. Load training and validation images
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=SEED
)


# --------------------------------------------------
# 4. Load the separate test dataset
# --------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Dataset loaded successfully!")


# --------------------------------------------------
# 5. Calculate class weights
# --------------------------------------------------

# Some expressions have fewer images than others.
# Class weights help the model pay more attention
# to categories with fewer training examples.

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class_counts = []

for class_name in class_names:
    folder = TRAIN_PATH / class_name

    count = sum(
        1
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
    )

    class_counts.append(count)

total_images = sum(class_counts)
num_classes = len(class_names)

class_weights = {
    i: total_images / (num_classes * class_counts[i])
    for i in range(num_classes)
}

print("Class weights:", class_weights)


# --------------------------------------------------
# 6. Build the CNN model
# --------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    # Convert pixel values from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # First convolution block
    layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    # Second convolution block
    layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.20),

    # Third convolution block
    layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.40),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(PROJECT_PATH / "emotion_cnn.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# --------------------------------------------------
# 8. Evaluate on unseen test images
# --------------------------------------------------

print("\nEvaluating the model on test images...")

best_model = tf.keras.models.load_model(
    PROJECT_PATH / "emotion_cnn.keras"
)

test_loss, test_accuracy = best_model.evaluate(test_ds)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# --------------------------------------------------
# 9. Save the emotion labels
# --------------------------------------------------

with open(
    PROJECT_PATH / "emotion_classes.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(class_names, file, indent=4)

print("\nTraining finished!")
print("Saved model: emotion_cnn.keras")
print("Saved labels: emotion_classes.json")