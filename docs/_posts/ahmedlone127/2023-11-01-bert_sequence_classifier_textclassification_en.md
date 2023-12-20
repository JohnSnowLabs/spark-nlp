---
layout: model
title: English BertForSequenceClassification Cased model (from palakagl)
author: John Snow Labs
name: bert_sequence_classifier_textclassification
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert_TextClassification` is a English model originally trained by `palakagl`.

## Predicted Entities

`alarm_remove`, `transport_query`, `email_addcontact`, `general_praise`, `general_dontcare`, `takeaway_query`, `email_query`, `transport_traffic`, `iot_wemo_off`, `weather_query`, `iot_hue_lightchange`, `calendar_query`, `iot_wemo_on`, `email_sendemail`, `general_negate`, `qa_currency`, `general_joke`, `alarm_query`, `alarm_set`, `general_repeat`, `datetime_convert`, `transport_taxi`, `lists_query`, `general_quirky`, `recommendation_movies`, `calendar_remove`, `qa_factoid`, `iot_hue_lighton`, `iot_hue_lightup`, `audio_volume_up`, `social_query`, `general_explain`, `general_confirm`, `news_query`, `qa_definition`, `iot_coffee`, `play_audiobook`, `qa_maths`, `lists_createoradd`, `play_podcasts`, `music_query`, `recommendation_locations`, `play_music`, `calendar_set`, `email_querycontact`, `general_affirm`, `recommendation_events`, `play_radio`, `audio_volume_down`, `social_post`, `general_commandstop`, `iot_hue_lightdim`, `transport_ticket`, `cooking_recipe`, `iot_hue_lightoff`, `audio_volume_mute`, `lists_remove`, `music_settings`, `iot_cleaning`, `takeaway_order`, `music_likeness`, `qa_stock`, `datetime_query`, `play_game`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_textclassification_en_5.1.4_3.4_1698798145186.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_textclassification_en_5.1.4_3.4_1698798145186.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_textclassification","en") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_textclassification","en")
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
|Model Name:|bert_sequence_classifier_textclassification|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|406.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/palakagl/bert_TextClassification