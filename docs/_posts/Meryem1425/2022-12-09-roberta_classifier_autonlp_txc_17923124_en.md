---
layout: model
title: English RobertaForSequenceClassification Cased model (from emekaboris)
author: John Snow Labs
name: roberta_classifier_autonlp_txc_17923124
date: 2022-12-09
tags: [en, open_source, roberta, sequence_classification, classification, tensorflow]
task: Text Classification
language: en
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-txc-17923124` is a English model originally trained by `emekaboris`.

## Predicted Entities

`15.0`, `24.0`, `10.0`, `8.0`, `4.0`, `17.0`, `3.0`, `23.0`, `5.0`, `6.0`, `1.0`, `21.0`, `18.0`, `19.0`, `14.0`, `16.0`, `20.0`, `7.0`, `13.0`, `11.0`, `12.0`, `9.0`, `22.0`, `2.0`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_autonlp_txc_17923124_en_4.2.4_3.0_1670623102986.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autonlp_txc_17923124","en") \
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
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autonlp_txc_17923124","en") 
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
|Model Name:|roberta_classifier_autonlp_txc_17923124|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|427.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/emekaboris/autonlp-txc-17923124