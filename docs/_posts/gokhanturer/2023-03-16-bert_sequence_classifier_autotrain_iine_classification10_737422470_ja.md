---
layout: model
title: Japanese BertForSequenceClassification Cased model (from vabadeh213)
author: John Snow Labs
name: bert_sequence_classifier_autotrain_iine_classification10_737422470
date: 2023-03-16
tags: [ja, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: ja
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autotrain-iine_classification10-737422470` is a Japanese model originally trained by `vabadeh213`.

## Predicted Entities

`0`, `1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autotrain_iine_classification10_737422470_ja_4.3.1_3.0_1678986097705.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autotrain_iine_classification10_737422470_ja_4.3.1_3.0_1678986097705.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autotrain_iine_classification10_737422470","ja") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autotrain_iine_classification10_737422470","ja")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_autotrain_iine_classification10_737422470|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ja|
|Size:|415.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/vabadeh213/autotrain-iine_classification10-737422470