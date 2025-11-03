---
layout: model
title: "CODER: Knowledge-Infused Biomedical Embedding Model (ONNX)"
author: John Snow Labs
name: umlsbert_eng_onnx
date: 2025-11-03
tags: [embeddings, en, biomedical, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 6.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is the English version of [CODER](https://github.com/GanjinZero/CODER), a knowledge-infused biomedical embedding model designed for medical concept normalization and cross-lingual representation learning.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/umlsbert_eng_onnx_en_6.1.0_3.0_1762173382824.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/umlsbert_eng_onnx_en_6.1.0_3.0_1762173382824.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("umlsbert_eng_onnx", "en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \

pipeline = Pipeline(stages=[
    documentAssembler,
    tokenizer,
    embeddings
])

data = spark.createDataFrame([
    ["Artificial intelligence is transforming the world."],
    ["Machine learning enables powerful data-driven systems."]
]).toDF("text")

result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show()

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = BertEmbeddings
  .pretrained("umlsbert_eng_onnx", "en")
  .setInputCols(Array("document", "token"))
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings
))

val data = Seq(
  "Artificial intelligence is transforming the world.",
  "Machine learning enables powerful data-driven systems."
).toDF("text")

val result = pipeline.fit(data).transform(data)

result.select("embeddings.embeddings").show(truncate = false)

```
</div>

## Results

```bash

+--------------------+
|          embeddings|
+--------------------+
|[[-0.6771237, 0.5...|
|[[-1.016453, 0.21...|
+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umlsbert_eng_onnx|
|Compatibility:|Healthcare NLP 6.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|408.0 MB|
|Case sensitive:|false|
|Max sentence length:|512|
