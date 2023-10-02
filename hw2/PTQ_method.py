from ultralytics import YOLO
import mlflow

from model_metrics import get_model_size, get_yolo_metrics

model = YOLO('yolov8n.pt')
results = model.export(format='onnx', half=True)
# Use the model
quantized_model = YOLO('yolov8n.onnx')

with mlflow.start_run(run_name="quantization") as run:
    # Use the model
    mlflow.log_param("compression_type", "quantization")
    mlflow.log_metric("model_size", get_model_size(model))
    mlflow.log_param("device", "cpu")
    mlflow.log_param("model_name", "yolov8n")

    metrics = quantized_model.val(data="coco128.yaml")  # evaluate model performance on the validation set
    metrics = get_yolo_metrics(metrics)
    for key, val in metrics.items():
        mlflow.log_metric(key, val)