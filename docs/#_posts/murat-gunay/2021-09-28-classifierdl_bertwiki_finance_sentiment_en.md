---
layout: model
title: Sentiment Analysis of Financial news
author: John Snow Labs
name: classifierdl_bertwiki_finance_sentiment
date: 2021-09-28
tags: [finance, en, sentiment, open_source]
task: Sentiment Analysis
language: en
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies the sentiments (positive, negative or neutral) in financial news.

## Predicted Entities

`neutral`, `positive`, `negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bertwiki_finance_sentiment_en_3.3.0_2.4_1632818954596.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

embeddings = BertSentenceEmbeddings\
    .pretrained('sent_bert_wiki_books_sst2', 'en') \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bertwiki_finance_sentiment", "en") \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

fr_sentiment_pipeline = Pipeline(stages=[document, embeddings, sentimentClassifier])

light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result1 = light_pipeline.annotate("As interest rates have increased, housing rents have also increased.")

result2 = light_pipeline.annotate("Unemployment rates have skyrocketed this month.")

result3 = light_pipeline.annotate("Tax rates on durable consumer goods were reduced.")

print(result1["class"], result2["class"], result3["class"], sep = "\n")
```
```scala
val document = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val embeddings = BertSentenceEmbeddings
    .pretrained("sent_bert_wiki_books_sst2", "en")
    .setInputCols(Array("document"))
    .setOutputCol("sentence_embeddings")

val sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bertwiki_finance_sentiment", "en")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val fr_sentiment_pipeline = new Pipeline().setStages(Array(document, embeddings, sentimentClassifier))

val light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result1 = light_pipeline.annotate("As interest rates have increased, housing rents have also increased.")

val result2 = light_pipeline.annotate("Unemployment rates have skyrocketed this month.")

val result3 = light_pipeline.annotate("Tax rates on durable consumer goods were reduced.")
```
</div>

## Results

```bash
['neutral']
['negative']
['positive']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bertwiki_finance_sentiment|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|

## Data Source

A custom dataset curated in John Snow Labs.

## Benchmarking

```bash
              precision    recall  f1-score   support

    negative       0.90      0.76      0.83       125
     neutral       0.88      0.90      0.89       537
    positive       0.77      0.79      0.78       258

    accuracy                           0.85       920
   macro avg       0.85      0.82      0.83       920
weighted avg       0.85      0.85      0.85       920
```