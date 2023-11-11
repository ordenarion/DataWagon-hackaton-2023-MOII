# DataWagon-hackaton-2023-MOII

**Трек 2: Прогнозирование отправления вагонов в ремонт**

*Решение команды МОИИ*

При прогнозировании вероятности отцепки вагона в плановый ремонт необходимо учитывать как информацию о самом вагоне, так и о его более предшествующих перемещениях, ранее перевозимых грузах, а также о его текущем техническом состоянии. Предварительный анализ представленных в рамках хакатона данных и проверка гипотез приведены в разделе
[notebooks/data_analysis](notebooks/data_analysis).

В результате проведенного анализа было сформировано признаковое описание, которое далее подавалось на вход моделям машинного обучения. Результаты работы моделей приведены в разделе [notebooks/models_result](notebooks/models_result).

Для запуска предсказательной модели необходимо вызвать скрипт ```predict.py wagnum_lst forecast_month```. 

Например, ```predict.py [0, 1, 2] '2023-02-01'```
