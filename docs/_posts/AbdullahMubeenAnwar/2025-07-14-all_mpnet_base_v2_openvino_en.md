---
layout: model
title: all-mpnet-base-v2 from sentence-transformers
author: John Snow Labs
name: all_mpnet_base_v2_openvino
date: 2025-07-14
tags: [openvino, english, mpnet, feature_extraction, fill_mask, embeddings, en, open_source]
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

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/all_mpnet_base_v2_openvino_en_6.0.0_3.0_1752496789418.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/all_mpnet_base_v2_openvino_en_6.0.0_3.0_1752496789418.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import MPNetEmbeddings
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

mpnet_loaded = MPNetEmbeddings.load("all_mpnet_base_v2_openvino")\
    .setInputCols(["documents"])\
    .setOutputCol("mpnet")\

pipeline = Pipeline(
    stages = [
        document_assembler,
        mpnet_loaded
  ])

data = spark.createDataFrame([
    ['William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor,and philanthropist.']
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.selectExpr("explode(mpnet.embeddings) as embeddings").show()
```
```scala
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.embeddings.MPNetEmbeddings
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val mpnetLoaded = MPNetEmbeddings
  .load("all_mpnet_base_v2_openvino")
  .setInputCols("documents")
  .setOutputCol("mpnet")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  mpnetLoaded
))

val data = Seq(
  "William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor,and philanthropist."
).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.selectExpr("explode(mpnet.embeddings) as embeddings").show(truncate = false)
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
|Output Labels:|[MPNet]|
|Language:|en|
|Size:|406.7 MB|

## References

https://huggingface.co/datasets/mandarjoshi/trivia_qa
https://huggingface.co/datasets/google-research-datasets/natural_questions
https://huggingface.co/datasets/microsoft/ms_marco