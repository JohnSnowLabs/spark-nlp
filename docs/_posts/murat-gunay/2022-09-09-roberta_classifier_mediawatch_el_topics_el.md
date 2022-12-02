---
layout: model
title: Greek RobertaForSequenceClassification Cased model (from cvcio)
author: John Snow Labs
name: roberta_classifier_mediawatch_el_topics
date: 2022-09-09
tags: [el, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: el
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `mediawatch-el-topics` is a Greek model originally trained by `cvcio`.

## Predicted Entities

`ENVIRONMENT`, `MILITARY`, `HEALTH`, `SOCIETY`, `ECONOMY`, `BUSINESS`, `LAW_AND_ORDER`, `POLITICS`, `REFUGEE`, `INTERNATIONAL`, `ELECTIONS`, `FOOD`, `TRANSPORT`, `AGRICULTURE`, `BREAKING_NEWS`, `OPINION`, `AFFAIRS`, `SCIENCE`, `TRAVEL`, `JUSTICE`, `SPORTS`, `ENTERTAINMENT`, `REGIONAL`, `TOURISM`, `RELIGION`, `CRIME`, `TECH`, `COVID`, `SOCIAL_MEDIA`, `WEATHER`, `ARTS_AND_CULTURE`, `EDUCATION`, `NON_PAPER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_mediawatch_el_topics_el_4.1.0_3.0_1662764469221.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_mediawatch_el_topics","el") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_mediawatch_el_topics","el") 
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
|Model Name:|roberta_classifier_mediawatch_el_topics|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|el|
|Size:|469.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/cvcio/mediawatch-el-topics
- https://github.com/andefined
- https://cvcio.org/