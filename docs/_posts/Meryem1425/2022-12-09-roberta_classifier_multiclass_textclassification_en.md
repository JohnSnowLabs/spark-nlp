---
layout: model
title: English RobertaForSequenceClassification Cased model (from palakagl)
author: John Snow Labs
name: roberta_classifier_multiclass_textclassification
date: 2022-12-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Roberta_Multiclass_TextClassification` is a English model originally trained by `palakagl`.

## Predicted Entities

`qa_maths`, `calendar_remove`, `takeaway_order`, `transport_query`, `recommendation_locations`, `qa_stock`, `play_podcasts`, `music_query`, `audio_volume_down`, `alarm_query`, `recommendation_movies`, `recommendation_events`, `general_repeat`, `datetime_query`, `transport_ticket`, `social_query`, `news_query`, `alarm_set`, `audio_volume_up`, `iot_wemo_on`, `iot_hue_lightup`, `iot_hue_lighton`, `general_praise`, `play_game`, `iot_hue_lightdim`, `iot_cleaning`, `email_sendemail`, `lists_createoradd`, `iot_coffee`, `alarm_remove`, `transport_taxi`, `social_post`, `general_negate`, `qa_factoid`, `general_affirm`, `general_dontcare`, `cooking_recipe`, `music_likeness`, `calendar_query`, `email_addcontact`, `play_audiobook`, `lists_query`, `datetime_convert`, `weather_query`, `calendar_set`, `email_query`, `iot_hue_lightchange`, `qa_definition`, `general_quirky`, `takeaway_query`, `play_music`, `general_confirm`, `email_querycontact`, `play_radio`, `iot_wemo_off`, `lists_remove`, `audio_volume_mute`, `qa_currency`, `general_commandstop`, `music_settings`, `general_explain`, `general_joke`, `transport_traffic`, `iot_hue_lightoff`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_multiclass_textclassification_en_4.2.4_3.0_1670621585623.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_multiclass_textclassification","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, roberta_classifier])

data = spark.createDataFrame([["I love you!"], ["I feel lucky to be here."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols("text")
    .setOutputCols("document")
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_multiclass_textclassification","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, roberta_classifier))

val data = Seq("I love you!").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_multiclass_textclassification|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|422.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/palakagl/Roberta_Multiclass_TextClassification