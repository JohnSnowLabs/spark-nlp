---
layout: model
title: English BertForSequenceClassification Cased model (from Smone55)
author: John Snow Labs
name: bert_sequence_classifier_autonlp_au_topics_452311620
date: 2023-11-01
tags: [en, open_source, bert, sequence_classification, ner, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-au_topics-452311620` is a English model originally trained by `Smone55`.

## Predicted Entities

`113`, `112`, `24`, `98`, `54`, `62`, `114`, `69`, `68`, `15`, `47`, `45`, `107`, `34`, `14`, `37`, `96`, `9`, `81`, `51`, `83`, `79`, `111`, `27`, `50`, `4`, `95`, `101`, `61`, `56`, `64`, `104`, `10`, `78`, `41`, `55`, `103`, `87`, `124`, `120`, `80`, `25`, `53`, `22`, `90`, `1`, `5`, `29`, `20`, `97`, `86`, `32`, `16`, `85`, `94`, `105`, `91`, `93`, `88`, `48`, `102`, `13`, `35`, `40`, `121`, `49`, `23`, `63`, `72`, `39`, `2`, `109`, `122`, `125`, `12`, `21`, `66`, `11`, `67`, `30`, `0`, `43`, `74`, `58`, `73`, `75`, `108`, `38`, `116`, `6`, `33`, `123`, `100`, `65`, `77`, `19`, `106`, `117`, `44`, `8`, `46`, `92`, `57`, `115`, `118`, `70`, `31`, `17`, `7`, `60`, `82`, `110`, `26`, `28`, `71`, `59`, `42`, `119`, `99`, `18`, `3`, `-1`, `84`, `36`, `76`, `89`, `52`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autonlp_au_topics_452311620_en_5.1.4_3.4_1698798115513.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_autonlp_au_topics_452311620_en_5.1.4_3.4_1698798115513.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autonlp_au_topics_452311620","en") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_autonlp_au_topics_452311620","en")
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
|Model Name:|bert_sequence_classifier_autonlp_au_topics_452311620|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/Smone55/autonlp-au_topics-452311620