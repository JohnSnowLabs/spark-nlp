---
layout: model
title: English sbert_yelp_imdb MPNetForSequenceClassification from Siki-77
author: John Snow Labs
name: sbert_yelp_imdb
date: 2024-08-31
tags: [en, open_source, onnx, sequence_classification, mpnet]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sbert_yelp_imdb` is a English model originally trained by Siki-77.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_yelp_imdb_en_5.4.2_3.0_1725145061319.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sbert_yelp_imdb_en_5.4.2_3.0_1725145061319.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = MPNetForSequenceClassification.pretrained("sbert_yelp_imdb","en") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("class")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, sequenceClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MPNetForSequenceClassification.pretrained("sbert_yelp_imdb", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbert_yelp_imdb|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/Siki-77/sbert_yelp_imdb