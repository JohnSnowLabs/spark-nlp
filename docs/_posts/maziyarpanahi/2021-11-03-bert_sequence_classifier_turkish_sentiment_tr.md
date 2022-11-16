---
layout: model
title: BERT Sequence Classification - Turkish Sentiment (bert_sequence_classifier_turkish_sentiment)
author: John Snow Labs
name: bert_sequence_classifier_turkish_sentiment
date: 2021-11-03
tags: [tr, turkish, sentiment, open_source, sequence_classification, bert, berturk]
task: Text Classification
language: tr
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

# Bert-base Turkish Sentiment Model

This model is used for Sentiment Analysis, which is based on BERTurk for Turkish Language https://huggingface.co/dbmdz/bert-base-turkish-cased

## Dataset

The dataset is taken from the studies [[2]](#paper-2) and [[3]](#paper-3), and merged.

* The study [2] gathered movie and product reviews. The products are book, DVD, electronics, and kitchen.
The movie dataset is taken from a cinema Web page ([Beyazperde](www.beyazperde.com)) with
5331 positive and 5331 negative sentences. Reviews in the Web page are marked in
scale from 0 to 5 by the users who made the reviews. The study considered a review
sentiment positive if the rating is equal to or bigger than 4, and negative if it is less
or equal to 2. They also built Turkish product review dataset from an online retailer
Web page. They constructed benchmark dataset consisting of reviews regarding some
products (book, DVD, etc.). Likewise, reviews are marked in the range from 1 to 5,
and majority class of reviews are 5. Each category has 700 positive and 700 negative
reviews in which average rating of negative reviews is 2.27 and of positive reviews
is 4.5. This dataset is also used by the study [[1]](#paper-1).

* The study [[3]](#paper-3) collected tweet dataset. They proposed a new approach for automatically classifying the sentiment of microblog messages. The proposed approach is based on utilizing robust feature representation and fusion.

*Merged Dataset*

| *size*   | *data* |
|--------|----|
|   8000 |dev.tsv|
|   8262 |test.tsv|
|  32000 |train.tsv|
|  *48290* |*total*|

### The dataset is used by following papers

<a id="paper-1">[1]</a> Yildirim, SavasÌ§. (2020). Comparing Deep Neural Networks to Traditional Models for Sentiment Analysis in Turkish Language. 10.1007/978-981-15-1216-2_12.

<a id="paper-2">[2]</a> Demirtas, Erkin and Mykola Pechenizkiy. 2013. Cross-lingual polarity detection with machine translation. In Proceedings of the Second International Workshop on Issues of Sentiment
Discovery and Opinion Mining (WISDOM â€™13)

<a id="paper-3">[3]</a> Hayran, A.,   Sert, M. (2017), "Sentiment Analysis on Microblog Data based on Word Embedding and Fusion Techniques", IEEE 25th Signal Processing and Communications Applications Conference (SIU 2017), Belek, Turkey

## Predicted Entities

`negative`, `positive`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_turkish_sentiment_tr_3.3.2_3.0_1635935288807.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('bert_sequence_classifier_turkish_sentiment', 'tr') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

example = spark.createDataFrame([['bu telefon modelleri çok kaliteli , her parçası çok özel bence']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_turkish_sentiment", "tr")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("bu telefon modelleri çok kaliteli , her parçası çok özel bence").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_turkish_sentiment|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|tr|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/savasy/bert-base-turkish-sentiment-cased](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased)