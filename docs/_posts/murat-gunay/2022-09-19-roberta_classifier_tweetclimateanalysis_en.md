---
layout: model
title: English RoBertaForSequenceClassification Cased model (from KeithHorgan)
author: John Snow Labs
name: roberta_classifier_tweetclimateanalysis
date: 2022-09-19
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

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `TweetClimateAnalysis` is a English model originally trained by `KeithHorgan`.

## Predicted Entities

`Global warming is not happening - Sea level rise is exaggerated/not accelerating`, `Climate impacts/global warming is beneficial/not bad -Species/plants/reefs aren’t showing climate impacts/are benefiting from climate change`, `Global warming is not happening - Extreme weather isn’t increasing/has happened before/isn’t linked to climate change`, `Climate solutions won’t work - Climate policies areineffective/flawed`, `Climate solutions won’t work - People need energy (e.g. from fossil fuels/nuclear)`, `No claim`, `Global warming is not happening - Climate hasn’t warmed/changed over the last (few) decade(s)`, `Climate movement/science is unreliable - Climate movement is unreliable/alarmist/corrupt`, `Climate impacts/global warming is beneficial/not bad -Climate sensitivity is low/negative feedbacks reduce warming`, `Climate movement/science is unreliable - Climate-related science is unreliable/uncertain/unsound (data, methods & models)`, `Global warming is not happening - Ice/permafrost/snow cover isn’t melting`, `Global warming is not happening - Weather is cold/snowing`, `Climate solutions won’t work - Climate policies (mitigation or adaptation) are harmful`, `Human greenhouse gases are not causing climate change - It’s natural cycles/variation`, `Climate impacts/global warming is beneficial/not bad -CO2 is beneficial/not a pollutant`, `Climate solutions won’t work - Clean energy technology/biofuels won’t work`, `Global warming is not happening - We’re heading into an ice age/global cooling`, `Human greenhouse gases are not causing climate change - There’s no evidence for greenhouse effect/carbon dioxide driving climate change`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_tweetclimateanalysis_en_4.1.0_3.0_1663605095991.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_tweetclimateanalysis_en_4.1.0_3.0_1663605095991.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_tweetclimateanalysis","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_tweetclimateanalysis","en") 
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
|Model Name:|roberta_classifier_tweetclimateanalysis|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/KeithHorgan/TweetClimateAnalysis