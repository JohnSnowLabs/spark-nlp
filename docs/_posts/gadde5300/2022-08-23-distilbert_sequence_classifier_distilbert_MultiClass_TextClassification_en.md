---
layout: model
title: English DistilBertForSequenceClassification Cased model (from palakagl)
author: John Snow Labs
name: distilbert_sequence_classifier_distilbert_MultiClass_TextClassification
date: 2022-08-23
tags: [distilbert, sequence_classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert_MultiClass_TextClassification` is a English model originally trained by `palakagl`.

## Predicted Entities

`weather_query`, `iot_hue_lightdim`, `audio_volume_up`, `general_praise`, `iot_cleaning`, `alarm_set`, `music_query`, `email_querycontact`, `play_podcasts`, `play_radio`, `transport_query`, `lists_query`, `music_settings`, `play_game`, `general_repeat`, `qa_maths`, `iot_hue_lightoff`, `iot_hue_lightchange`, `play_music`, `play_audiobook`, `alarm_query`, `music_likeness`, `lists_remove`, `qa_definition`, `general_commandstop`, `recommendation_events`, `general_confirm`, `recommendation_locations`, `social_query`, `general_dontcare`, `email_addcontact`, `general_negate`, `general_joke`, `general_quirky`, `cooking_recipe`, `datetime_query`, `news_query`, `qa_factoid`, `general_affirm`, `audio_volume_down`, `lists_createoradd`, `calendar_set`, `audio_volume_mute`, `general_explain`, `datetime_convert`, `iot_wemo_off`, `transport_traffic`, `calendar_query`, `alarm_remove`, `calendar_remove`, `qa_currency`, `iot_hue_lighton`, `iot_wemo_on`, `email_sendemail`, `transport_taxi`, `iot_hue_lightup`, `recommendation_movies`, `social_post`, `qa_stock`, `takeaway_order`, `email_query`, `transport_ticket`, `takeaway_query`, `iot_coffee`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_MultiClass_TextClassification_en_4.1.0_3.0_1661277858811.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_MultiClass_TextClassification_en_4.1.0_3.0_1661277858811.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_MultiClass_TextClassification","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_MultiClass_TextClassification","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_distilbert_MultiClass_TextClassification|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|246.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/palakagl/distilbert_MultiClass_TextClassification