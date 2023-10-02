import os
import torch
from torch.nn.utils import prune
from ultralytics import YOLO
import mlflow

from model_metrics import get_model_size, get_model_sparsity, get_yolo_metrics


mlflow.set_tracking_uri('https://dagshub.com/PredelinaAsya/ModelCompression.mlflow')

for amount in [0.025, 0.05, 0.1, 0.2, 0.5]:
    for pruning_method in [prune.L1Unstructured, prune.RandomUnstructured]:
        with mlflow.start_run(run_name="global pruning") as run:
            model = YOLO("yolov8n.pt")
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, 'weight'))

            mlflow.log_param("compression_type", "pruning")
            mlflow.log_param("pruning_amount", amount)
            mlflow.log_param("pruning_method", str(pruning_method))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=pruning_method,
                amount=amount,
            )
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.remove(module, 'weight')

            mlflow.log_metric("model_size", get_model_size(model))
            mlflow.log_metric("model_sparsity", get_model_sparsity(model))

            metrics = model.val(data="coco128.yaml")
            metrics = get_yolo_metrics(metrics)
            for key, val in metrics.items():
                mlflow.log_metric(key, val)
