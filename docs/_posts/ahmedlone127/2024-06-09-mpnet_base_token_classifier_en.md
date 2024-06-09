---
layout: model
title: English mpnet_base_token_classifier MPNetForTokenClassification
author: John Snow Labs
name: mpnet_base_token_classifier
date: 2024-06-09
tags: [en, open_source, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.0
spark_version: 3.4
supported: true
engine: onnx
annotator: MPNetForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetForTokenClassification model, trained in house and curated to provide scalability and production-readiness using Spark NLP.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpnet_base_token_classifier_en_5.4.0_3.4_1717953216313.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpnet_base_token_classifier_en_5.4.0_3.4_1717953216313.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = MPNetForTokenClassification.pretrained("camembert_classifier_berties","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler,tokenizer,sequenceClassifier_loaded])

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

val sequenceClassifier_loaded = MPNetForTokenClassification.pretrained("camembert_classifier_berties","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler,tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mpnet_base_token_classifier|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|395.9 MB|
|Case sensitive:|true|