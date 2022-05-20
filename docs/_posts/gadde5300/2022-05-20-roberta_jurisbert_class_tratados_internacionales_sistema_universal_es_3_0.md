---
layout: model
title: Spanish Named Entity Recognition (from `hackathon-pln-es`)
author: John Snow Labs
name: roberta_jurisbert_class_tratados_internacionales_sistema_universal
date: 2022-05-20
tags: [roberta, ner, token_classification, es, open_source]
task: Named Entity Recognition
language: es
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `jurisbert-class-tratados-internacionales-sistema-universal ` is a Spanish model orginally trained by `hackathon-pln-es`.

## Predicted Entities

`Convención sobre la Eliminación de todas las formas de Discriminación contra la Mujer`, `Convención sobre los Derechos de las Personas con Discapacidad`, `Convención Internacional Sobre la Eliminación de Todas las Formas de Discriminación Racial`, `Convención contra la Tortura y otros Tratos o Penas Crueles, Inhumanos o Degradantes`, `Convención Internacional sobre la Protección de los Derechos de todos los Trabajadores Migratorios y de sus Familias`, `Convención de los Derechos del Niño`, `Pacto Internacional de Derechos Económicos, Sociales y Culturales`, `Pacto Internacional de Derechos Civiles y Políticos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_jurisbert_class_tratados_internacionales_sistema_universal_es_3.4.4_3.0_1653047814525.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \        .setInputCol("text") \        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_jurisbert_class_tratados_internacionales_sistema_universal","es") \
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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_jurisbert_class_tratados_internacionales_sistema_universal","es") 
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
|Model Name:|roberta_jurisbert_class_tratados_internacionales_sistema_universal|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|464.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://huggingface.co/hackathon-pln-es/jurisbert-class-tratados-internacionales-sistema-universal