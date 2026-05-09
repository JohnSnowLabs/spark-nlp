---
layout: model
title: ModernBERT Base ONNX
author: John Snow Labs
name: modernbert_base_onnx
date: 2026-05-06
tags: [en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 6.2.0
spark_version: 3.4
supported: true
engine: onnx
annotator: ModernBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

ModernBERT is a modernized bidirectional encoder-only Transformer model (BERT-style) pre-trained on 2 trillion tokens of English and code data with a native context length of up to 8,192 tokens. ModernBERT leverages recent architectural improvements such as:

* Rotary Positional Embeddings (RoPE) for long-context support.
* Local-Global Alternating Attention for efficiency on long inputs.
* Unpadding and Flash Attention for efficient inference.

ModernBERT's native long context length makes it ideal for tasks that require processing long documents, such as retrieval, classification, and semantic search within large corpora. The model was trained on a large corpus of text and code, making it suitable for a wide range of downstream tasks, including code retrieval and hybrid (text + code) semantic search.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/modernbert_base_onnx_en_6.2.0_3.4_1778063886598.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/modernbert_base_onnx_en_6.2.0_3.4_1778063886598.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

loaded_model = ModernBertEmbeddings \
    .pretrained("modernbert_base_onnx") \
    .setInputCols(["token", "document"]) \
    .setOutputCol("embeddings")

pipeline = Pipeline(stages=[
    documentAssembler,
    tokenizer,
    loaded_model
])

data = spark.createDataFrame([["Covid cases are increasing fast!"]]).toDF("text")

result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show()
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val loadedModel = ModernBertEmbeddings
  .pretrained("modernbert_base_onnx")
  .setInputCols(Array("token", "document"))
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  loadedModel
))

val data = spark.createDataFrame(Seq("Covid cases are increasing fast!")).toDF("text")

val result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show()
```
</div>

## Results

```bash

+--------------------+
|          embeddings|
+--------------------+
|[[0.721862, 0.107...|
+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|modernbert_base_onnx|
|Compatibility:|Spark NLP 6.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|559.3 MB|