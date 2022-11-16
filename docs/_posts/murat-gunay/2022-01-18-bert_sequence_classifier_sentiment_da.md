---
layout: model
title: Sentiment Analysis of Danish texts
author: John Snow Labs
name: bert_sequence_classifier_sentiment
date: 2022-01-18
tags: [danish, sentiment, da, open_source]
task: Sentiment Analysis
language: da
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/DaNLP/da-bert-tone-sentiment-polarity)) and it's been finetuned for Danish language, leveraging Danish Bert embeddings and `BertForSequenceClassification` for text classification purposes.

## Predicted Entities

`negative`, `positive`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_sentiment_da_3.3.4_3.0_1642495726064.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
      .pretrained('bert_sequence_classifier_sentiment', 'da') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['Protester over hele landet ledet af utilfredse civilsamfund på grund af den danske regerings COVID-19 lockdown-politik er kommet ud af kontrol.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_sentiment", "da")
      .setInputCols("document", "token")
      .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["Protester over hele landet ledet af utilfredse civilsamfund på grund af den danske regerings COVID-19 lockdown-politik er kommet ud af kontrol."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
['negative']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sentiment|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|da|
|Size:|415.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## Data Source

[https://huggingface.co/DaNLP/da-bert-tone-sentiment-polarity#training-data](https://huggingface.co/DaNLP/da-bert-tone-sentiment-polarity#training-data)
