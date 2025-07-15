---
layout: model
title: all-mpnet-base-v2 from sentence-transformers OpenVINO
author: John Snow Labs
name: all_mpnet_base_v2_openvino
date: 2025-07-15
tags: [openvino, english, embedding, open_source, mpnet, en]
task: Embeddings
language: en
edition: Spark NLP 6.0.0
spark_version: 3.0
supported: true
engine: openvino
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a sentence-transformers model: It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for tasks like clustering or semantic search.

This model is intended to be used as a sentence and short paragraph encoder. Given an input text, it outputs a vector that captures the semantic information. The sentence vector may be used for information retrieval, clustering, or sentence similarity tasks.

By default, input text longer than 384 word pieces is truncated.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/all_mpnet_base_v2_openvino_en_6.0.0_3.0_1752610809513.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/all_mpnet_base_v2_openvino_en_6.0.0_3.0_1752610809513.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import MPNetEmbeddings
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

mpnet_loaded = MPNetEmbeddings.load("all_mpnet_base_v2_openvino")\
    .setInputCols(["document"])\
    .setOutputCol("mpnet_embeddings")\

pipeline = Pipeline(
    stages = [
        document_assembler,
        mpnet_loaded
  ])

data = spark.createDataFrame([
    ['William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist.']
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.selectExpr("explode(mpnet_embeddings.embeddings) as embeddings").show()

```
```scala
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.explode
import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val mpnetEmbeddings = MPNetEmbeddings.load("all_mpnet_base_v2_openvino")
  .setInputCols("document")
  .setOutputCol("mpnet_embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  mpnetEmbeddings
))

val data = Seq(
  "William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist."
).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select(explode($"mpnet_embeddings.embeddings").alias("embeddings")).show(false)

```
</div>

## Results

```bash

+--------------------+
|          embeddings|
+--------------------+
|[-0.020282388, 0....|
+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|all_mpnet_base_v2_openvino|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[mpnet_embeddings]|
|Language:|en|
|Size:|406.5 MB|