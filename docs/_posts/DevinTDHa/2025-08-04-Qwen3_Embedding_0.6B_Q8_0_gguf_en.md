---
layout: model
title: Qwen3-Embedding-0.6B-GGUF
author: John Snow Labs
name: Qwen3_Embedding_0.6B_Q8_0_gguf
date: 2025-08-04
tags: [gguf, qwen3, qwen, embedding, q8, llamacpp, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 6.1.1
spark_version: 3.0
supported: true
engine: llamacpp
annotator: AutoGGUFEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

Imported from https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Qwen3_Embedding_0.6B_Q8_0_gguf_en_6.1.1_3.0_1754314464060.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Qwen3_Embedding_0.6B_Q8_0_gguf_en_6.1.1_3.0_1754314464060.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
autoGGUFEmbeddings = AutoGGUFEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("embeddings") \
    .setBatchSize(4) \
    .setNGpuLayers(99) \
    .setPoolingType("MEAN")
pipeline = Pipeline().setStages([document, autoGGUFEmbeddings])

data = spark.createDataFrame([["The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show(truncate = False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val autoGGUFModel = AutoGGUFEmbeddings
  .pretrained()
  .setInputCols("document")
  .setOutputCol("embeddings")
  .setBatchSize(4)
  .setPoolingType("MEAN")

val pipeline = new Pipeline().setStages(Array(document, autoGGUFModel))

val data = Seq(
  "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones.")
  .toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show(truncate = false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Qwen3_Embedding_0.6B_Q8_0_gguf|
|Compatibility:|Spark NLP 6.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|609.4 MB|