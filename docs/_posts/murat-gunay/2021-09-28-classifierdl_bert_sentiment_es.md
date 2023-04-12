---
layout: model
title: Sentiment Analysis of Spanish texts
author: John Snow Labs
name: classifierdl_bert_sentiment
date: 2021-09-28
tags: [es, spanish, sentiment, open_source]
task: Sentiment Analysis
language: es
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies the sentiments (neutral, positive or negative) in Spanish texts.

## Predicted Entities

`NEUTRAL`, `POSITIVE`, `NEGATIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_es_3.3.0_2.4_1632820716491.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_es_3.3.0_2.4_1632820716491.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

embeddings = BertSentenceEmbeddings\
    .pretrained('labse', 'xx') \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "es") \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

fr_sentiment_pipeline = Pipeline(stages=[document, embeddings, sentimentClassifier])

light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result1 = light_pipeline.annotate("Estoy seguro de que esta vez pasará la entrevista.")

result2 = light_pipeline.annotate("Soy una persona que intenta desayunar todas las mañanas sin falta.")

result3 = light_pipeline.annotate("No estoy seguro de si mi salario mensual es suficiente para vivir.")

print(result1["class"], result2["class"], sep = "\n")
```
```scala
val document = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val embeddings = BertSentenceEmbeddings
    .pretrained("labse", "xx")
    .setInputCols(Array("document"))
    .setOutputCol("sentence_embeddings")

val sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "es")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val fr_sentiment_pipeline = new Pipeline().setStages(Array(document, embeddings, sentimentClassifier))

val light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result1 = light_pipeline.annotate("Estoy seguro de que esta vez pasará la entrevista.")

val result2 = light_pipeline.annotate("Soy una persona que intenta desayunar todas las mañanas sin falta.")

val result3 = light_pipeline.annotate("No estoy seguro de si mi salario mensual es suficiente para vivir.")
```
</div>

## Results

```bash
['POSITIVE']
['NEUTRAL']
['NEGATIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_sentiment|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|es|

## Data Source

https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis

## Benchmarking

```bash
              precision    recall  f1-score   support

    NEGATIVE       0.73      0.81      0.77      1837
     NEUTRAL       0.80      0.71      0.75      2222
    POSITIVE       0.80      0.81      0.80      2187

    accuracy                           0.78      6246
   macro avg       0.77      0.78      0.77      6246
weighted avg       0.78      0.78      0.77      6246
```