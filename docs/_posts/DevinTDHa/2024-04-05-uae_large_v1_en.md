---
layout: model
title: UAE-Large-V1 for Sentence Embeddings
author: John Snow Labs
name: uae_large_v1
date: 2024-04-05
tags: [uae, en, sentence, embeddings, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.3.3
spark_version: 3.0
supported: true
engine: onnx
annotator: UAEEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

UAE is a novel angle-optimized text embedding model, designed to improve semantic textual
similarity tasks, which are crucial for Large Language Model (LLM) applications. By
introducing angle optimization in a complex space, AnglE effectively mitigates saturation of
the cosine similarity function.

This model is based on UAE-Large-V1 and was orignally exported from https://huggingface.co/WhereIsAI/UAE-Large-V1. Several embedding pooling strategies can be set. Please refer to the class for more information.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/uae_large_v1_en_5.3.3_3.0_1712335736995.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/uae_large_v1_en_5.3.3_3.0_1712335736995.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
embeddings = UAEEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols("embeddings") \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    embeddingsFinisher
])
data = spark.createDataFrame([["hello world", "hello moon"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.UAEEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val embeddings = UAEEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("UAE_embeddings")
val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("UAE_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  embeddingsFinisher
))
val data = Seq("hello world", "hello moon").toDF("text")
val result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
```
</div>

## Results

```bash
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.50387806, 0.5861606, 0.35129607, -0.76046336, -0.32446072, -0.117674336, 0...|
|[0.6660665, 0.961762, 0.24854276, -0.1018044, -0.6569202, 0.027635604, 0.1915...|
+--------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|uae_large_v1|
|Compatibility:|Spark NLP 5.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|1.2 GB|