import os
import torch
from torch.nn.utils import prune
from ultralytics import YOLO
import mlflow

from model_metrics import get_model_size, get_model_sparsity, get_yolo_metrics

for amount in [0.025, 0.05, 0.1, 0.2, 0.5]:
    for pruning_method in [prune.l1_unstructured, prune.random_unstructured]:
        with mlflow.start_run(run_name="local pruning") as run:
            model = YOLO("yolov8n.pt")
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    pruning_method(module, name="weight", amount=amount)

            mlflow.log_param("compression_type", "pruning")
            mlflow.log_param("pruning_amount", amount)
            mlflow.log_param("pruning_method", str(pruning_method))
            mlflow.log_param("device", "cpu")
            mlflow.log_param("model_name", "yolov8n")


            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.remove(module, 'weight')

            mlflow.log_metric("model_size", get_model_size(model))
            mlflow.log_metric("model_sparsity", get_model_sparsity(model))

            metrics = model.val(data="coco128.yaml", device="cpu")
            metrics = get_yolo_metrics(metrics)
            for key, val in metrics.items():
                mlflow.log_metric(key, val)