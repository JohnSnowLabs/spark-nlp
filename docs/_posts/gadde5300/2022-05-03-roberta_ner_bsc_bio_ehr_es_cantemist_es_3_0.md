---
layout: model
title: Spanish Named Entity Recognition (from PlanTL-GOB-ES)
author: John Snow Labs
name: roberta_ner_bsc_bio_ehr_es_cantemist
date: 2022-05-03
tags: [roberta, ner, open_source, es]
task: Named Entity Recognition
language: es
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bsc-bio-ehr-es-cantemist` is a Spanish model orginally trained by `PlanTL-GOB-ES`.

## Predicted Entities

`MORFOLOGIA_NEOPLASIA`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_bsc_bio_ehr_es_cantemist_es_3.4.2_3.0_1651593711611.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_bsc_bio_ehr_es_cantemist","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Amo Spark NLP"]]).toDF("text")

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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_bsc_bio_ehr_es_cantemist","es") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_bsc_bio_ehr_es_cantemist|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|435.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/PlanTL-GOB-ES/bsc-bio-ehr-es-cantemist
- https://arxiv.org/abs/1907.11692
- https://temu.bsc.es/cantemist/
- https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es
- https://paperswithcode.com/sota?task=token-classification&dataset=cantemist-ner