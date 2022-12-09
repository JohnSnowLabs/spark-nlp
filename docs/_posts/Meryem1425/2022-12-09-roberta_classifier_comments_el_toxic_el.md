---
layout: model
title: Greek RobertaForSequenceClassification Cased model (from cvcio)
author: John Snow Labs
name: roberta_classifier_comments_el_toxic
date: 2022-12-09
tags: [el, open_source, roberta, sequence_classification, classification, tensorflow]
task: Text Classification
language: el
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `comments-el-toxic` is a Greek model originally trained by `cvcio`.

## Predicted Entities

`obscene`, `toxicity`, `threat`, `severe_toxicity`, `identity_attack`, `insult`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_comments_el_toxic_el_4.2.4_3.0_1670624626019.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_comments_el_toxic","el") \
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
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_comments_el_toxic","el") 
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
|Model Name:|roberta_classifier_comments_el_toxic|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|el|
|Size:|468.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/cvcio/comments-el-toxic