---
layout: model
title: Laws Spanish Named Entity Recognition (from `hackathon-pln-es`)
author: John Snow Labs
name: roberta_ner_jurisbert_finetuning_ner
date: 2022-05-20
tags: [roberta, ner, token_classification, es, open_source]
task: Named Entity Recognition
language: es
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `jurisbert-finetuning-ner` is a Spanish model orginally trained by `hackathon-pln-es`.

## Predicted Entities

`TRAT_INTL`, `LEY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_jurisbert_finetuning_ner_es_3.4.4_3.0_1653046369327.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_jurisbert_finetuning_ner","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Me encanta Spark PNL"]]).toDF("text")

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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_jurisbert_finetuning_ner","es") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Me encanta Spark PNL").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_jurisbert_finetuning_ner|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|464.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://huggingface.co/hackathon-pln-es/jurisbert-finetuning-ner
