from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.export(format='onnx', half=True)

# Use the model
quantized_model = YOLO('yolov8n.onnx')
metrics = quantized_model.val(data="coco128.yaml")  # evaluate model performance on the validation set
