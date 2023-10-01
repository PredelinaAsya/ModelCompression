# Кластеризация весов

## Установка зависимостей

Для работы notebook требуется создать виртуальное окружение и установить в него зависимости из `reuiqrements.txt`:

```pip install -qr requirements.txt```

Так же можно воспользоваться [Google Colab notebook](https://colab.research.google.com/drive/17V4DALphQNwRwbB_-TcK_7QqmLZhAJ9-?usp=sharing)

## Результирующие метрики
| **Models** 	      | **Precision** 	| **Recall** 	| **mAP:.5** 	| **mAP:.5-.95** 	| **Inference** 	|
|-------------------|---	|---	|---	|---	|---	|
| Original model 	  | 0.638 	|  0.536	| 0.607 	| 0.448 	| 274.8 ms 	|
| Clustering model	 | 0.617	|  0.515	| 0.58	| 0.434	| 314.4 ms 	|