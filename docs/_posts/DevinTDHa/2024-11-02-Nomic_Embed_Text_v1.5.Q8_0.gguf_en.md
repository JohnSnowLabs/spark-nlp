---
layout: model
title: nomic-embed-text-v1.5.Q8_0.gguf
author: John Snow Labs
name: Nomic_Embed_Text_v1.5.Q8_0.gguf
date: 2024-11-02
tags: [gguf, nomic, embeddings, open_source, en, llamacpp]
task: Embeddings
language: en
edition: Spark NLP 5.5.2
spark_version: 3.4
supported: true
engine: llamacpp
annotator: AutoGGUFEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

nomic-embed-text is a 8192 context length text encoder that surpasses OpenAI text-embedding-ada-002 and text-embedding-3-small performance on short and long context tasks.

This model is the updated 1.5 version.

Original model from https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Nomic_Embed_Text_v1.5.Q8_0.gguf_en_5.5.2_3.4_1730556912139.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Nomic_Embed_Text_v1.5.Q8_0.gguf_en_5.5.2_3.4_1730556912139.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
autoGGUFModel = AutoGGUFModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("completions") \
    .setBatchSize(4) \
    .setNPredict(20) \
    .setNGpuLayers(99) \
    .setTemperature(0.4) \
    .setTopK(40) \
    .setTopP(0.9) \
    .setPenalizeNl(True)
pipeline = Pipeline().setStages([document, autoGGUFModel])
data = spark.createDataFrame([[The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("completions").show(truncate = False)
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

## Results

```bash
+--------------------------------------------------------------------------------+
|                                                                      embeddings|
+--------------------------------------------------------------------------------+
|[[-0.034486726, 0.07770534, -0.15982522, -0.017873349, 0.013914132, 0.0365736...|
+--------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Nomic_Embed_Text_v1.5.Q8_0.gguf|
|Compatibility:|Spark NLP 5.5.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|140.7 MB|