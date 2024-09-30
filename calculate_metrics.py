from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Load the dataset annotations in COCO format
coco_gt = COCO('resources/modified_val2017.json')


# Load model predictions in COCO format
coco_dt = coco_gt.loadRes('torchvision_models_results/fasterrcnn/predictions.json')


# Initialize COCOeval object and run evaluation
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()