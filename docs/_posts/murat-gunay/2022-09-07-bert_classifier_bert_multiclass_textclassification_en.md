---
layout: model
title: English BertForSequenceClassification Cased model (from palakagl)
author: John Snow Labs
name: bert_classifier_bert_multiclass_textclassification
date: 2022-09-07
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert_MultiClass_TextClassification` is a English model originally trained by `palakagl`.

## Predicted Entities

`iot_hue_lightchange`, `calendar_set`, `recommendation_movies`, `iot_coffee`, `general_commandstop`, `iot_wemo_on`, `general_negate`, `transport_traffic`, `general_repeat`, `play_music`, `audio_volume_mute`, `transport_query`, `qa_definition`, `calendar_query`, `social_post`, `general_affirm`, `lists_query`, `general_dontcare`, `qa_stock`, `general_confirm`, `datetime_query`, `play_audiobook`, `audio_volume_up`, `iot_hue_lighton`, `weather_query`, `iot_cleaning`, `audio_volume_down`, `play_radio`, `iot_hue_lightoff`, `takeaway_order`, `alarm_query`, `social_query`, `general_joke`, `alarm_set`, `play_podcasts`, `cooking_recipe`, `recommendation_locations`, `calendar_remove`, `news_query`, `takeaway_query`, `email_query`, `transport_ticket`, `lists_createoradd`, `alarm_remove`, `music_settings`, `qa_factoid`, `email_querycontact`, `music_likeness`, `iot_hue_lightup`, `email_sendemail`, `general_quirky`, `play_game`, `qa_maths`, `datetime_convert`, `general_explain`, `iot_wemo_off`, `recommendation_events`, `email_addcontact`, `iot_hue_lightdim`, `music_query`, `transport_taxi`, `general_praise`, `qa_currency`, `lists_remove`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_multiclass_textclassification_en_4.1.0_3.0_1662511115836.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_multiclass_textclassification","en") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_multiclass_textclassification","en") 
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
|Model Name:|bert_classifier_bert_multiclass_textclassification|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/palakagl/bert_MultiClass_TextClassification