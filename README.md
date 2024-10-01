# Computer Vision Project - DLBAIPCV01

This is the code repository to support the project report for module: **DLBAIPCV01 – Project: Computer Vision** at **I.U. International University of Applied Sciences**.

## Overview

In this project, we utilize various object detection models, including YOLOv8, Faster R-CNN, and Single Shot Detector, to perform predictions on custom datasets and videos.

## Repository Contents

### Scripts

- **`object_detector.py`**: Contains the code to make predictions on an input video using the YOLOv8 model.
- **`torchvision_models_evaluator.py`**: Contains the code to make predictions on a dataset using torchvision's pretrained models.
- **`yolo_evaluator.py`**: A simple script to evaluate the YOLOv8 model on a custom dataset. The **`evaluation.yaml`** file points to the custom dataset.
- **`calculate_metrics.py`**: A simple script which utilizes the `pycocotools` library to compute the key metrics used for evaluation.

### Media Files

- **`desk.mp4`**: An example video used for prediction.
- **`output_video.mp4`**: The result video with the annotated frames.

### Folders

- **`evaluation/`**: Holds the custom dataset, formatted according to YOLO standards.
- **`evaluation_results/`**: Contains the results of evaluation for the three models (YOLOv8, Faster R-CNN, Single Shot Detector).
- **`resources/`**: Contains the ground truth of the COCO 2017 validation dataset as a JSON file. A modified version with unnecessary items removed (e.g., licenses, info) is also present.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- `torchvision`
- YOLOv8 (`ultralytics` package)
- `pycocotools`

### Dataset

The evaluation/ folder contains the custom dataset formatted according to YOLO standards. Ensure that your dataset follows the same structure if you plan to use your own data.

### Evaluation Results

The evaluation_results/ folder contains the results of evaluation for the three models:

- YOLOv8
- Faster R-CNN
- Single Shot Detector

### Resources

The resources/ folder contains:

- The ground truth of the COCO 2017 validation dataset (instances_val2017.json).
- A modified version with unnecessary items removed (e.g., licenses, info).

### Acknowledgments
- I.U. International University of Applied Sciences
- PyTorch
- Ultralytics YOLOv8
