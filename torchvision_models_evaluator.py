import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import json
import time
from tqdm import tqdm


# Load the pre-trained Faster R-CNN or SSD (Single Shot Detector) model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

transform = T.Compose([
    T.ToTensor()
])


def predict(model, img_path):
    # Open and transform the image
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)
    end_time = time.time()
    inference_time = end_time - start_time

    results = []
    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()  # Bounding boxes
    labels = prediction['labels'].cpu().numpy()  # Category labels
    scores = prediction['scores'].cpu().numpy()  # Confidence scores

    for i in range(len(boxes)):
        if scores[i] >= 0.5:
            x_min, y_min, x_max, y_max = boxes[i]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            results.append({
                "bbox": [float(x) for x in bbox],
                "category_id": int(labels[i]),
                "score": float(scores[i])
            })

    return results, inference_time


def run_inference_on_folder(model, image_folder, output_json="predictions.json", time_json="inference_times.json"):
    predictions_results = []
    inference_times = []


    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]


    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for filename in image_files:
            img_path = os.path.join(image_folder, filename)


            image_id = int(os.path.splitext(filename)[0])


            predictions, inference_time = predict(model, img_path)


            for pred in predictions:
                pred.update({"image_id": image_id})
                predictions_results.append(pred)


            inference_times.append({
                "image_id": image_id,
                "inference_time": inference_time
            })


            pbar.update(1)


    with open(output_json, "w") as f:
        json.dump(predictions_results, f, indent=4)


    with open(time_json, "w") as f:
        json.dump(inference_times, f, indent=4)

    print(f"Predictions saved to {output_json}")
    print(f"Inference times saved to {time_json}")


# Path to the folder containing images
# image_folder = r"C:\Users\kast3\OneDrive\Documents\Python Scripts\project_Computer_Vision\test_eval"


# run_inference_on_folder(model, image_folder, output_json="predictions.json", time_json="inference_times.json")
