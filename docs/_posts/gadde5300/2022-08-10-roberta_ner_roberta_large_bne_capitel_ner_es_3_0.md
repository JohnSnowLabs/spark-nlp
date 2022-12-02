---
layout: model
title: Spanish RobertaForTokenClassification Large Cased model (from BSC-TeMU)
author: John Snow Labs
name: roberta_ner_roberta_large_bne_capitel_ner
date: 2022-08-10
tags: [bert, ner, open_source, es]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-large-bne-capitel-ner` is a Spanish model originally trained by `BSC-TeMU`.

## Predicted Entities

`S_OTH`, `S_PER`, `ORG`, `E_OTH`, `PER`, `S_LOC`, `LOC`, `E_LOC`, `S_ORG`, `E_ORG`, `OTH`, `E_PER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_roberta_large_bne_capitel_ner_es_4.1.0_3.0_1660140529822.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("roberta_ner_roberta_large_bne_capitel_ner","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

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

val tokenClassifier = BertForTokenClassification.pretrained("roberta_ner_roberta_large_bne_capitel_ner","es") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_roberta_large_bne_capitel_ner|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/BSC-TeMU/roberta-large-bne-capitel-ner
- http://www.bne.es/en/Inicio/index.html
- https://arxiv.org/abs/2107.07253
- https://github.com/PlanTL-SANIDAD/lm-spanish
- https://arxiv.org/abs/1907.11692
- https://sites.google.com/view/capitel2020