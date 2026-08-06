---
layout: model
title: BGE m3
author: John Snow Labs
name: tmp_bge_m3_model
date: 2026-08-06
tags: [xx, open_source, onnx]
task: Embeddings
language: xx
edition: Spark NLP 7.0.0
spark_version: 3.4
supported: true
engine: onnx
annotator: BGEM3Embeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEM3Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. bge_m3 is a multilingual model originally trained by BAAI.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tmp_bge_m3_model_xx_7.0.0_3.4_1786052423106.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tmp_bge_m3_model_xx_7.0.0_3.4_1786052423106.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = BGEM3Embeddings.pretrained("bge_m3", "xx") \
    .setInputCols(["document"]) \
    .setOutputCol("embeddings") \
    .setReturnSparseEmbeddings(True)

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    embeddings
])

data = spark.createDataFrame([
    ["What is BGE M3?"],
    ["BGE M3 ist ein multilinguales Embedding-Modell."]
]).toDF("text")

result = nlp_pipeline.fit(data).transform(data)

result.selectExpr(
    "text",
    "embeddings.embeddings as dense",
    "embeddings.metadata as sparse"
).show(truncate=60)
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = BGEM3Embeddings.pretrained("bge_m3", "xx")
  .setInputCols(Array("document"))
  .setOutputCol("embeddings")
  .setReturnSparseEmbeddings(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings
))

val data = Seq(
  "What is BGE M3?",
  "BGE M3 ist ein multilinguales Embedding-Modell."
).toDF("text")

val result = pipeline.fit(data).transform(data)

result.selectExpr(
  "text",
  "embeddings.embeddings as dense",
  "embeddings.metadata as sparse"
).show(truncate = 60)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tmp_bge_m3_model|
|Compatibility:|Spark NLP 7.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|1.3 GB|