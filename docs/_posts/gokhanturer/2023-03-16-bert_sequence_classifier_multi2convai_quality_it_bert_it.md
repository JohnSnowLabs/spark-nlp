---
layout: model
title: Italian BertForSequenceClassification Cased model (from inovex)
author: John Snow Labs
name: bert_sequence_classifier_multi2convai_quality_it_bert
date: 2023-03-16
tags: [it, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: it
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `multi2convai-quality-it-bert` is a Italian model originally trained by `inovex`.

## Predicted Entities

`neo.buerstentraegerfalschmontiert`, `neo.getriebedeckel`, `neo.gehause`, `neo.start`, `neo.magnet`, `neo.schraube`, `undefined`, `neo.help`, `neo.motor.housing`, `neo.motor.anchor`, `neo.magnetisierung`, `neo.motor.brushcollar`, `neo.anlaufscheibe`, `neo.gearbox`, `neo.verschaubung`, `neo.zusammenfuehrung`, `neo.yes`, `neo.magnetklammern`, `neo.motor`, `neo.motor.worm`, `neo.back`, `neo.cancel`, `neo.hello`, `neo.sinterbuchse`, `neo.no`, `neo.zahnradklein`, `neo.buerstentraegerdefekt`, `neo.zahnradgross`, `neo.anker`, `neo.magnet.magnet`, `neo.einpressen`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_multi2convai_quality_it_bert_it_4.3.1_3.0_1678986951308.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_multi2convai_quality_it_bert_it_4.3.1_3.0_1678986951308.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_multi2convai_quality_it_bert","it") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_multi2convai_quality_it_bert","it")
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
|Model Name:|bert_sequence_classifier_multi2convai_quality_it_bert|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|it|
|Size:|412.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/inovex/multi2convai-quality-it-bert
- https://multi2conv.ai
- https://multi2convai/en/blog/use-cases
- https://multi2convai/en/blog/use-cases
- https://multi2conv.ai
- https://github.com/inovex/multi2convai