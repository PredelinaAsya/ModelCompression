# Optimum

## Окружение

Необходимые пакеты для установки вынесены в файл ```requirements.txt```.

## Базовая модель-трансформер

В качестве базовой модели была выбрана детекционная модель [YOLOS-base](https://huggingface.co/hustvl/yolos-base). Замеры метрик качества производились на изображениях из валидационной части датасета [COCO 2017](https://huggingface.co/datasets/rafaelpadilla/coco2017).

## Оптимизация базовой модели и замеры метрик

Для оптимизации исходной модели методом BetterTransformer, а также подсчёта метрик качества и производительности можно воспользоваться ```Optimum.ipynb```.

## Запуск в google colab

Также можно воспользоваться [Google Colab notebook](https://colab.research.google.com/drive/1sowOz_6sGlPEh_boKblP9QpoDv6tj7ve?usp=sharing).

## Таблица с результатами экспериментов

| **Эксперимент** 	      | **mAP** 	 | **Inference time (gpu, T4)** 	 | **Model size** |
|------------------------|-----------------|----------------------------|----------------|
| YOLOS-base 	      | 0.380	          | 17 ms 	                   | 487.512          | 
| YOLOS-base, BetterTransformer          | 	 0.380         | 7 ms 	                   | 487.512          | 
