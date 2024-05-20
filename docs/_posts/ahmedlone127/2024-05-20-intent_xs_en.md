---
layout: model
title: English intent_xs AlbertForSequenceClassification from dejanseo
author: John Snow Labs
name: intent_xs
date: 2024-05-20
tags: [en, open_source, sequence_classification, albert, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`intent_xs` is a English model originally trained by dejanseo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/intent_xs_en_5.2.4_3.0_1716211014633.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/intent_xs_en_5.2.4_3.0_1716211014633.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
        
               
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer()     .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier  = AlbertForSequenceClassification.pretrained("intent_xs","en") \
     .setInputCols(["token","document"]) \
     .setOutputCol("class")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, embeddings])
data = spark.createDataFrame([["Saya suka Spark NLP"]]).toDF("text")
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

val sequenceClassifier = AlbertForSequenceClassification.pretrained("intent_xs", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))
val data = Seq("Où est-ce que je vis?").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|intent_xs|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|44.2 MB|

## References

https://huggingface.co/dejanseo/Intent-XS