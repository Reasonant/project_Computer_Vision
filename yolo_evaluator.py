from ultralytics import YOLO

model = YOLO("yolov8m.pt")

evaluation_file = 'evaluation.yaml'

results = model.val(data=evaluation_file, save_json=True, plots=True)

# print(results.results_dict)
