---
layout: model
title: French BertForSequenceClassification Cased model (from inovex)
author: John Snow Labs
name: bert_classifier_multi2convai_quality
date: 2022-09-07
tags: [fr, open_source, bert, sequence_classification, classification]
task: Text Classification
language: fr
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `multi2convai-quality-fr-bert` is a French model originally trained by `inovex`.

## Predicted Entities

`neo.getriebedeckel`, `neo.buerstentraegerdefekt`, `neo.motor.anchor`, `neo.hello`, `neo.help`, `neo.zahnradklein`, `neo.magnet.magnet`, `neo.gearbox`, `neo.magnetklammern`, `neo.magnet`, `neo.schraube`, `neo.motor.housing`, `neo.einpressen`, `neo.anlaufscheibe`, `undefined`, `neo.zusammenfuehrung`, `neo.cancel`, `neo.back`, `neo.buerstentraegerfalschmontiert`, `neo.start`, `neo.motor.worm`, `neo.magnetisierung`, `neo.motor`, `neo.motor.brushcollar`, `neo.gehause`, `neo.anker`, `neo.verschaubung`, `neo.zahnradgross`, `neo.sinterbuchse`, `neo.yes`, `neo.no`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_multi2convai_quality_fr_4.1.0_3.0_1662514209226.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_multi2convai_quality_fr_4.1.0_3.0_1662514209226.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_multi2convai_quality","fr") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_multi2convai_quality","fr") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_multi2convai_quality|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|415.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/inovex/multi2convai-quality-fr-bert
- https://multi2convai/en/blog/use-cases
- https://github.com/inovex/multi2convai