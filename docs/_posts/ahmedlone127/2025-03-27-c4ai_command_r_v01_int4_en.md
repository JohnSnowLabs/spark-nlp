---
layout: model
title: c4ai_command_r_v01_int4 model
author: John Snow Labs
name: c4ai_command_r_v01_int4
date: 2025-03-27
tags: [en, open_source, openvino]
task: Text Generation
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: CoHereTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CoHereTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`c4ai_command_r_v01_int4` is a english model originally trained by Qwen.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/c4ai_command_r_v01_int4_en_5.5.1_3.0_1743065483260.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/c4ai_command_r_v01_int4_en_5.5.1_3.0_1743065483260.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = CoHereTransformer.pretrained("c4ai_command_r_v01_int4","en") \
      .setInputCols(["document"]) \
      .setOutputCol("generation")

pipeline = Pipeline().setStages([documentAssembler, seq2seq])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val seq2seq = CoHereTransformer.pretrained("c4ai_command_r_v01_int4","en")
    .setInputCols(Array("document"))
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, seq2seq))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|c4ai_command_r_v01_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|16.6 GB|