---
layout: model
title: Sentiment Analysis of Turkish texts
author: John Snow Labs
name: classifierdl_use_sentiment
date: 2021-10-19
tags: [tr, sentiment, use, classification, open_source]
task: Sentiment Analysis
language: tr
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies the sentiments (positive or negative) in Turkish texts.

## Predicted Entities

`POSITIVE`, `NEGATIVE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_TR/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_SENTIMENT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_sentiment_tr_3.3.0_2.4_1634634525008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\
    .setCleanupMode("shrink")

embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_use_sentiment", "tr") \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

fr_sentiment_pipeline = Pipeline(stages=[document, embeddings, sentimentClassifier])

light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result1 = light_pipeline.annotate("Bu sıralar kafam çok karışık.")

result2 = light_pipeline.annotate("Sınavımı geçtiğimi öğrenince derin bir nefes aldım.")

print(result1["class"], result2["class"], sep = "\n")
```
```scala
val document = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

val sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "tr")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val fr_sentiment_pipeline = new Pipeline().setStages(Array(document, embeddings, sentimentClassifier))

val light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result1 = light_pipeline.annotate("Bu sıralar kafam çok karışık.")

val result2 = light_pipeline.annotate("Sınavımı geçtiğimi öğrenince derin bir nefes aldım.")
```
</div>

## Results

```bash
['NEGATIVE']
['POSITIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_sentiment|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|tr|

## Data Source

https://raw.githubusercontent.com/gurkandyilmaz/sentiment/master/data/

## Benchmarking

```bash
              precision    recall  f1-score   support

    NEGATIVE       0.86      0.88      0.87     19967
    POSITIVE       0.88      0.85      0.86     19826

    accuracy                           0.87     39793
   macro avg       0.87      0.87      0.87     39793
weighted avg       0.87      0.87      0.87     39793
```