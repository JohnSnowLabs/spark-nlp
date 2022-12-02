---
layout: model
title: English RobertaForSequenceClassification Cased model (from palakagl)
author: John Snow Labs
name: roberta_classifier_multiclass_textclassification
date: 2022-09-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Roberta_Multiclass_TextClassification` is a English model originally trained by `palakagl`.

## Predicted Entities

`transport_ticket`, `general_commandstop`, `iot_cleaning`, `general_praise`, `music_settings`, `general_quirky`, `recommendation_locations`, `iot_hue_lightoff`, `audio_volume_mute`, `calendar_set`, `iot_coffee`, `datetime_convert`, `general_explain`, `cooking_recipe`, `qa_definition`, `news_query`, `music_likeness`, `recommendation_movies`, `general_dontcare`, `general_affirm`, `recommendation_events`, `alarm_set`, `qa_maths`, `qa_factoid`, `play_podcasts`, `takeaway_query`, `email_sendemail`, `email_addcontact`, `transport_traffic`, `iot_wemo_off`, `general_negate`, `iot_hue_lightdim`, `audio_volume_up`, `general_repeat`, `iot_wemo_on`, `alarm_query`, `lists_createoradd`, `music_query`, `weather_query`, `transport_query`, `alarm_remove`, `takeaway_order`, `social_post`, `general_confirm`, `calendar_query`, `iot_hue_lightup`, `general_joke`, `calendar_remove`, `email_querycontact`, `iot_hue_lightchange`, `iot_hue_lighton`, `play_radio`, `social_query`, `lists_query`, `transport_taxi`, `lists_remove`, `email_query`, `datetime_query`, `play_music`, `qa_stock`, `audio_volume_down`, `qa_currency`, `play_game`, `play_audiobook`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_multiclass_textclassification_en_4.1.0_3.0_1662761168230.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_multiclass_textclassification","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_multiclass_textclassification","en") 
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
|Model Name:|roberta_classifier_multiclass_textclassification|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|422.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/palakagl/Roberta_Multiclass_TextClassification