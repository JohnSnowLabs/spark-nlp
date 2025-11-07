---
layout: model
title: BGE Base Embeddings v1.5 (ONNX)
author: John Snow Labs
name: bge_base_en_v1_5_onnx
date: 2025-11-07
tags: [embedding, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 6.0.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BGEEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

FlagEmbedding focuses on retrieval-augmented LLMs

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_base_en_v1_5_onnx_en_6.0.0_3.0_1762522014019.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_base_en_v1_5_onnx_en_6.0.0_3.0_1762522014019.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BGEEmbeddings.pretrained("bge_base_en_v1_5_onnx", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    embeddings
])

data = spark.createDataFrame([["This is a test sentence for BGE embeddings."]]).toDF("text")

result = nlp_pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show()
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = BGEEmbeddings.pretrained("bge_base_en_v1_5_onnx", "en")
  .setInputCols(Array("document"))
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings
))

val data = spark.createDataFrame(Seq(
  ("This is a test sentence for BGE embeddings.")
)).toDF("text")

val result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show()
```
</div>

## Results

```bash

+--------------------+
|          embeddings|
+--------------------+
|[[0.023069248, -0...|
+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_base_en_v1_5_onnx|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|256.0 MB|
|Case sensitive:|false|
|Max sentence length:|512|