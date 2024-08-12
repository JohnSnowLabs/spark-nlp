---
layout: model
title: mxbai large Model
author: John Snow Labs
name: mxbai_large_v1
date: 2024-07-16
tags: [embeddings, mxbai, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: MxbaiEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MxbaiEmbeddings, adataped from huggingface imported to Spark-NLP to provide scalability and production-readiness.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mxbai_large_v1_en_5.4.2_3.0_1721143405168.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mxbai_large_v1_en_5.4.2_3.0_1721143405168.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

mxbai = MxbaiEmbeddings.pretrained("mxbai_large_v1","en") \
    .setInputCols("document") \
    .setOutputCol("embeddings") \

pipeline = Pipeline().setStages([documentAssembler, mxbai])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")

val mxbai = MxbaiEmbeddings.pretrained("mxbai_large_v1", "en")
    .setInputCols("documents")
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, mxbai))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mxbai_large_v1|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[Mxbai]|
|Language:|en|
|Size:|793.8 MB|