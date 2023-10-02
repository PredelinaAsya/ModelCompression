# ModelCompression

Репозиторий для выполнения домашних работ в рамках курса "Методы компрессии нейросетевых моделей" командой Model Compression Club.

## Модель и данные

Для дальнейших экспериментов с различными методами компрессии мы выбрали детекционную модель yolov8n, обученную на датасете COCO.

Ссылка на репозиторий ultralytics, где реализована данная модель: https://github.com/ultralytics/ultralytics/tree/main.

## Структура репозитория

Для каждой домашней работы будет создаваться отдельная папка, внутри которой будут сохраняться ноутбуки с кодом или команды для запуска, необходимое окружение, скриншоты с запусками.

## Трекинг экспериментов

Ниже будет общая таблица с замерами метрик производительности модели для разных экспериментов, она будет регулярно обновляться. Метрики считаются в google colab.

| **Эксперимент** 	    | **Precision** 	 | **Recall** 	 | **mAP:.5** 	| **mAP:.5-.95** 	| **Inference time (cpu)** 	 |
|----------------------|-----------|---------|--	|--	|------------------------|
| Исходная модель 	    |  0.639	        | 0.536 	 |  0.604	|  0.445	| 250 ms 	               |
| PTQ, float16         |  	 0.650       | 	   0.532 |  0.605	|  0.452	| 230 ms 	               |
| Global pruning 0.05% | 0.632	   | 0.538	  | 0.606 	| 0.446 	| 255 ms 	               |
| Weight clustering    | 0.655 	   | 0.496	   | 0.591 	| 0.432 	| 260 ms                 |

Также все эксперименты заносятся сюда https://dagshub.com/PredelinaAsya/ModelCompression/experiments с помощью mlflow.