import torch


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2


def get_model_sparsity(model):
    zeros = 0
    n_elements = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            zeros += float(torch.sum(module.weight == 0))
            n_elements += float(module.weight.nelement())

    return 100. * zeros / n_elements


def get_yolo_metrics(metrics):
    speed_metrics = metrics.speed
    result_metrics = metrics.results_dict

    result = {"inference": speed_metrics["inference"], "mAP .5-.95": result_metrics["metrics/mAP50-95(B)"],
              "mAP .5": result_metrics["metrics/mAP50(B)"], "recall": result_metrics["metrics/recall(B)"],
              "precision": result_metrics["metrics/precision(B)"]}

    return result
