---
layout: model
title: Spanish BertForTokenClassification Cased model (from rjuez00)
author: John Snow Labs
name: bert_ner_meddocan_beto_ner
date: 2022-08-03
tags: [bert, ner, open_source, es]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `meddocan-beto-ner` is a Spanish model originally trained by `rjuez00`.

## Predicted Entities

`CALLE`, `NUMERO_FAX`, `FECHAS`, `CENTRO_SALUD`, `INSTITUCION`, `PROFESION`, `ID_EMPLEO_PERSONAL_SANITARIO`, `SEXO_SUJETO_ASISTENCIA`, `PAIS`, `FAMILIARES_SUJETO_ASISTENCIA`, `EDAD_SUJETO_ASISTENCIA`, `CORREO_ELECTRONICO`, `NUMERO_TELEFONO`, `HOSPITAL`, `ID_CONTACTO_ASISTENCIAL`, `ID_ASEGURAMIENTO`, `OTROS_SUJETO_ASISTENCIA`, `NOMBRE_SUJETO_ASISTENCIA`, `ID_SUJETO_ASISTENCIA`, `NOMBRE_PERSONAL_SANITARIO`, `ID_TITULACION_PERSONAL_SANITARIO`, `TERRITORIO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_meddocan_beto_ner_es_4.1.0_3.0_1659514838770.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_meddocan_beto_ner","es") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_meddocan_beto_ner","es") 
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
|Model Name:|bert_ner_meddocan_beto_ner|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|410.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/rjuez00/meddocan-beto-ner