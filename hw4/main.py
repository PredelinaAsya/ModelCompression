from ultralytics import YOLO
from hw4.knowledge_distil_yolo_training import KnowledgeDistillationDetectionTraining

student_model = YOLO("yolov8m.pt")
student_model.train(trainer=KnowledgeDistillationDetectionTraining)


