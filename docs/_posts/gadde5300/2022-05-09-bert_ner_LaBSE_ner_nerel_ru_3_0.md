---
layout: model
title: Russian Named Entity Recognition (from surdan)
author: John Snow Labs
name: bert_ner_LaBSE_ner_nerel
date: 2022-05-09
tags: [bert, ner, token_classification, ru, open_source]
task: Named Entity Recognition
language: ru
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `LaBSE_ner_nerel` is a Russian model orginally trained by `surdan`.

## Predicted Entities

`DATE`, `NATIONALITY`, `LAW`, `PERSON`, `PERCENT`, `FACILITY`, `PROFESSION`, `NUMBER`, `RELIGION`, `DISTRICT`, `WORK_OF_ART`, `LANGUAGE`, `LOCATION`, `AGE`, `AWARD`, `IDEOLOGY`, `COUNTRY`, `TIME`, `FAMILY`, `MONEY`, `CRIME`, `ORDINAL`, `EVENT`, `PRODUCT`, `CITY`, `ORGANIZATION`, `STATE_OR_PROVINCE`, `DISEASE`, `PENALTY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_LaBSE_ner_nerel_ru_3.4.2_3.0_1652098988587.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_LaBSE_ner_nerel_ru_3.4.2_3.0_1652098988587.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_LaBSE_ner_nerel","ru") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Я люблю Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_LaBSE_ner_nerel","ru") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Я люблю Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_LaBSE_ner_nerel|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ru|
|Size:|481.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/surdan/LaBSE_ner_nerel