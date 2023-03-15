---
layout: model
title: English BertForTokenClassification Uncased model (from ghadeermobasher)
author: John Snow Labs
name: bert_ner_BC5CDR_Disease_Modified_bluebert_pubmed_uncased_L_12_H_768_A_12_latest
date: 2022-07-09
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `BC5CDR-Disease-Modified_bluebert_pubmed_uncased_L-12_H-768_A-12_latest` is a English model originally trained by `ghadeermobasher`.

## Predicted Entities

`Disease`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_BC5CDR_Disease_Modified_bluebert_pubmed_uncased_L_12_H_768_A_12_latest_en_4.0.0_3.0_1657351950002.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_BC5CDR_Disease_Modified_bluebert_pubmed_uncased_L_12_H_768_A_12_latest_en_4.0.0_3.0_1657351950002.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_BC5CDR_Disease_Modified_bluebert_pubmed_uncased_L_12_H_768_A_12_latest","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_BC5CDR_Disease_Modified_bluebert_pubmed_uncased_L_12_H_768_A_12_latest","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_BC5CDR_Disease_Modified_bluebert_pubmed_uncased_L_12_H_768_A_12_latest|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.7 MB|
|Case sensitive:|false|
|Max sentence length:|128|

## References

- https://huggingface.co/ghadeermobasher/BC5CDR-Disease-Modified_bluebert_pubmed_uncased_L-12_H-768_A-12_latest