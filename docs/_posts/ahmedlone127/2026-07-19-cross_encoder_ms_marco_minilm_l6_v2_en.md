---
layout: model
title: Cross encoder
author: John Snow Labs
name: cross_encoder_ms_marco_minilm_l6_v2
date: 2026-07-19
tags: [en, open_source, onnx]
task: Reranking
language: en
edition: Spark NLP 6.4.2
spark_version: 3.4
supported: true
engine: onnx
annotator: CrossEncoder
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Ths model helps cross encode sentences

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cross_encoder_ms_marco_minilm_l6_v2_en_6.4.2_3.4_1784490146617.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cross_encoder_ms_marco_minilm_l6_v2_en_6.4.2_3.4_1784490146617.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document = MultiDocumentAssembler() \
    .setInputCols(["query", "passage"]) \
    .setOutputCols(["document1", "document2"])

crossEncoder = CrossEncoder.pretrained() \
    .setInputCols(["document1", "document2"]) \
    .setOutputCol("score")

pipeline = Pipeline().setStages([document, crossEncoder])

data = spark.createDataFrame([
    ["How many people live in Berlin?", "Berlin is well known for its museums."]
]).toDF("query", "passage")

result = pipeline.fit(data).transform(data)
result.select("score.result").show(truncate=False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val document = new MultiDocumentAssembler()
  .setInputCols("query", "passage")
  .setOutputCols("document1", "document2")

val crossEncoder = CrossEncoder.pretrained()
  .setInputCols("document1", "document2")
  .setOutputCol("score")

val pipeline = new Pipeline().setStages(Array(document, crossEncoder))

val data = Seq(
  ("How many people live in Berlin?", "Berlin is well known for its museums."))
  .toDF("query", "passage")
val result = pipeline.fit(data).transform(data)

result.select("score.result").show(false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cross_encoder_ms_marco_minilm_l6_v2|
|Compatibility:|Spark NLP 6.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document1, document2]|
|Output Labels:|[score]|
|Language:|en|
|Size:|84.2 MB|