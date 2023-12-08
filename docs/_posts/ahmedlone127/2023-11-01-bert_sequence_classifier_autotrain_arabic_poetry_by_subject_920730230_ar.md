---
layout: model
title: Arabic BertForSequenceClassification Cased model (from zenkri)
author: John Snow Labs
name: bert_sequence_classifier_autotrain_arabic_poetry_by_subject_920730230
date: 2023-11-01
tags: [ar, open_source, bert, sequence_classification, ner, onnx]
task: Named Entity Recognition
language: ar
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autotrain-Arabic_Poetry_by_Subject-920730230` is a Arabic model originally trained by `zenkri`.

## Predicted Entities

`اعتذار`, `مدح`, `صبر`, `سياسية`, `وطنيه`, `رثاء`, `عامه`, `ابتهال`, `قصيره`, `رومنسيه`, `حزينه`, `عتاب`, `رحمة`, `الاناشيد`, `المعلقات`, `فراق`, `هجاء`, `نصيحة`, `جود`, `حكمة`, `شوق`, `دينية`, `عدل`, `غزل`, `ذم`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autotrain_arabic_poetry_by_subject_920730230_ar_5.1.4_3.4_1698801427269.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autotrain_arabic_poetry_by_subject_920730230_ar_5.1.4_3.4_1698801427269.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autotrain_arabic_poetry_by_subject_920730230","ar") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autotrain_arabic_poetry_by_subject_920730230","ar")
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
|Model Name:|bert_sequence_classifier_autotrain_arabic_poetry_by_subject_920730230|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ar|
|Size:|466.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/zenkri/autotrain-Arabic_Poetry_by_Subject-920730230