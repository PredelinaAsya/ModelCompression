from ultralytics import YOLO
from hw4.knowledge_distil_yolo_training import KnowledgeDistillationDetectionTraining


if __name__ == '__main__':
    student_model = YOLO("yolov8m.pt")
    student_model.train(trainer=KnowledgeDistillationDetectionTraining)
    student_model.val(data="coco128.yaml")

