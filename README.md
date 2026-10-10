# Real-Time Facial Expression Recognition Using CNN

A Python computer-vision project that detects a face from a webcam frame and classifies the visible facial expression into one of seven categories: **angry, disgust, fear, happy, neutral, sad,** or **surprise**.

> **Project status:** The webcam expression display is working. A custom CNN has been trained and evaluated. Current test accuracy is **36.46%**, so treat this as an educational prototype—not a reliable real-world emotion detector.

## Features

- Webcam capture and face localization with OpenCV.
- A custom convolutional neural network (CNN) trained on images organised into seven class folders.
- Saved Keras model and label mapping for inference.
- Test-set evaluation, per-class precision/recall/F1, and a confusion matrix.
- A real-time label display in the webcam window.

## Project structure

```text
emotion-detection/
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                 # Same seven class names
├── train_model.py            # Train and save the CNN
├── webcam_cnn.py             # Run webcam with the custom CNN
├── webcam.py                 # Optional earlier DeepFace baseline, if retained
├── evaluate_model.py         # Evaluate saved model on test images
├── emotion_cnn.keras         # Trained CNN checkpoint
├── emotion_classes.json     # Class-label order
├── evaluation_report.txt    # Per-class metrics
├── confusion_matrix.png     # Confusion matrix figure
├── requirements.txt
├── PROJECT_REPORT.docx
└── README.md
```

The dataset and virtual environment are not included in this documentation pack. Keep your existing `dataset/`, `.venv/`, Python files, `emotion_cnn.keras`, and `emotion_classes.json` in your project folder. The `webcam.py` file is optional if you want to keep the earlier DeepFace version for comparison.

## Requirements

- Windows 10/11
- Python 3.11 (the project was tested with Python 3.11.9)
- A working webcam
- Sufficient free RAM and disk space for TensorFlow and the dataset

## Setup

Open the project folder in VS Code and use a PowerShell terminal in that folder.

Create a virtual environment if you do not already have a working one:

```powershell
py -3.11 -m venv .venv
```

Install the packages required for the custom-CNN application:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select the interpreter at `.venv\Scripts\python.exe` using **Python: Select Interpreter**. Run the scripts through that interpreter rather than the Code Runner extension's generic `Run Code` command.

## Run the webcam application

```powershell
.\.venv\Scripts\python.exe webcam_cnn.py
```

Face boxes and predicted expression labels should appear in a window. Press **Q** to close the window.

## Train the model again (optional)

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Training reads `dataset/train` and `dataset/test`, creates a validation split from the training folder, trains the CNN, evaluates it on the test set, and saves `emotion_cnn.keras` and `emotion_classes.json`. It can take several minutes on a CPU. Do not run another training process at the same time.

## Evaluate the saved model

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

This creates or updates `evaluation_report.txt` and `confusion_matrix.png` from the test set.

## Dataset summary

| Split | Images |
|---|---:|
| Training folder | 28,709 |
| Validation subset (15% of training folder) | 4,306 |
| Actual training subset | 24,403 |
| Test folder | 7,178 |
| Total train + test folders | 35,887 |

Counts by class:

| Class | Train | Test |
|---|---:|---:|
| angry | 3,995 | 958 |
| disgust | 436 | 111 |
| fear | 4,097 | 1,024 |
| happy | 7,215 | 1,774 |
| neutral | 4,965 | 1,233 |
| sad | 4,830 | 1,247 |
| surprise | 3,171 | 831 |

## Current evaluation

- **Test accuracy:** 36.46%
- **Macro F1-score:** 0.28
- **Weighted F1-score:** 0.30
- Strongest recall in the current run: happy (0.79), followed by surprise (0.57).
- Low recall in the current run: angry (0.05), fear (0.09), and sad (0.04).

The confusion matrix shows that many angry, fear, neutral, and sad samples are classified as happy. This limitation must be stated in the project report and presentation. Do not present the system as highly accurate.

## Limitations and ethical note

The model classifies visual patterns in a face into dataset labels. A label is a model prediction—not proof of a person's actual internal emotional state. Results may vary with lighting, pose, camera quality, demographic representation, and expressions that resemble one another. Do not use this prototype for consequential decisions about people.

## Troubleshooting

- **`ModuleNotFoundError: cv2`:** select the project's `.venv` interpreter and run the program with ` .\.venv\Scripts\python.exe webcam_cnn.py`.
- **Webcam GUI/display error:** use the GUI-enabled `opencv-python` package specified in `requirements.txt`; avoid simultaneously installing multiple OpenCV package variants in the same environment.
- **Model file not found:** check that `emotion_cnn.keras` and `emotion_classes.json` are in the same folder as `webcam_cnn.py`.
- **Dataset path error:** confirm the exact structure `dataset/train/<class>/` and `dataset/test/<class>/`.
- **TensorFlow oneDNN or CPU messages:** these are generally informational messages, not fatal errors.

## Suggested presentation outline

1. Problem and objective
2. Dataset and class distribution
3. CNN architecture and training pipeline
4. Live webcam demonstration
5. Test results and confusion matrix
6. Limitations and future improvements

## Project report

See [`PROJECT_REPORT.docx`](PROJECT_REPORT.docx). Fill in the editable student, department, institution, and supervisor fields on the cover page before submitting.
