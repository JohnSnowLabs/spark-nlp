---
layout: model
title: BERT Sequence Classification - Russian Sentiment Analysis (bert_sequence_classifier_rubert_sentiment)
author: John Snow Labs
name: bert_sequence_classifier_rubert_sentiment
date: 2024-09-02
tags: [ru, russian, bert, sentiment, rubert, open_source, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

# RuBERT for Sentiment Analysis
Short Russian texts sentiment classification

This is a [DeepPavlov/rubert-base-cased-conversational](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational) model trained on aggregated corpus of 351.797 texts.

## Predicted Entities

`NEUTRAL`, `POSITIVE`, `NEGATIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_rubert_sentiment_ru_5.5.0_3.0_1725293870033.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_rubert_sentiment_ru_5.5.0_3.0_1725293870033.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
      .pretrained('bert_sequence_classifier_rubert_sentiment', 'ru') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(False) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

example = spark.createDataFrame([['Ты мне нравишься. Я тебя люблю']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_rubert_sentiment", "ru")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Ты мне нравишься. Я тебя люблю").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_rubert_sentiment|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ru|
|Size:|664.5 MB|