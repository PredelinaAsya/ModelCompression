import mlflow
from ultralytics import YOLO

from model_metrics import get_model_size, get_yolo_metrics

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

with mlflow.start_run(run_name="initial") as run:
    # Use the model
    mlflow.log_param("compression_type", "none")
    mlflow.log_metric("model_size", get_model_size(model))
    mlflow.log_param("device", "cpu")
    mlflow.log_param("model_name", "yolov8n")

    metrics = model.val(data="coco128.yaml")  # evaluate model performance on the validation set
    metrics = get_yolo_metrics(metrics)
    for key, val in metrics.items():
        mlflow.log_metric(key, val)