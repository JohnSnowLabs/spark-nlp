---
layout: model
title: BERT Sequence Classification Multilingual Sentiment
author: John Snow Labs
name: bert_sequence_classifier_multilingual_sentiment
date: 2021-11-03
tags: [en, nl, de, fr, it, es, sequence_classification, bert, multilingual, sentiment, xx, open_source]
task: Text Classification
language: xx
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).

This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above, or for further finetuning on related sentiment analysis tasks.

## Training data

Here is the number of product reviews we used for finetuning the model:

| Language | Number of reviews |
| -------- | ----------------- |
| English  | 150k           |
| Dutch    | 80k            |
| German   | 137k           |
| French   | 140k           |
| Italian  | 72k            |
| Spanish  | 50k            |

## Predicted Entities

`1 star`, `2 stars`, `3 stars`, `4 stars`, `5 stars`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_multilingual_sentiment_xx_3.3.2_3.0_1635934569723.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('bert_sequence_classifier_multilingual_sentiment', 'xx') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(False) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

example = spark.createDataFrame([['I really liked that movie!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_multilingual_sentiment", "xx")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_multilingual_sentiment|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|xx|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## Benchmarking

```bash
The finetuned model obtained the following accuracy on 5,000 held-out product reviews in each of the languages:

- Accuracy (exact) is the exact match on the number of stars.
- Accuracy (off-by-1) is the percentage of reviews where the number of stars the model predicts differs by a maximum of 1 from the number given by the human reviewer.


| Language | Accuracy (exact) | Accuracy (off-by-1) |
| -------- | ---------------------- | ------------------- |
| English  | 67%                 | 95%
| Dutch    | 57%                 | 93%
| German   | 61%                 | 94%
| French   | 59%                 | 94%
| Italian  | 59%                 | 95%
| Spanish  | 58%                 | 95%

```
