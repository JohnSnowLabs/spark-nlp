---
layout: model
title: English Named Entity Recognition (from CouchCat)
author: John Snow Labs
name: distilbert_ner_ma_ner_v7_distil
date: 2022-05-16
tags: [distilbert, ner, token_classification, en, open_source]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ma_ner_v7_distil` is a English model orginally trained by `CouchCat`.

## Predicted Entities

`MATR`, `PERS`, `TIME`, `MISC`, `PAD`, `PROD`, `BRND`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_ner_ma_ner_v7_distil_en_3.4.2_3.0_1652721967576.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_ner_ma_ner_v7_distil_en_3.4.2_3.0_1652721967576.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_ner_ma_ner_v7_distil","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_ner_ma_ner_v7_distil","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_ner_ma_ner_v7_distil|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|244.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/CouchCat/ma_ner_v7_distil