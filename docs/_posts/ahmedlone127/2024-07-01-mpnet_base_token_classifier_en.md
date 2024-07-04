---
layout: model
title: MPnetForTokenClassification Base Model English
author: John Snow Labs
name: mpnet_base_token_classifier
date: 2024-07-01
tags: [token_classification, mpnet, ner, en, open_source, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetForTokenClassification, fine tuned in huggingface in house and then imported to Spark-NLP o provide scalability and production-readiness.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpnet_base_token_classifier_en_5.4.0_3.0_1719843589238.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpnet_base_token_classifier_en_5.4.0_3.0_1719843589238.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier  = MPNetForTokenClassification.pretrained("mpnet_base_token_classifier","en") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = MPNetForTokenClassification.pretrained("mpnet_base_token_classifier", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

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