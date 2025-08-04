---
layout: model
title: Phi-4-mini-instruct Q4 GGUF
author: John Snow Labs
name: Phi_4_mini_instruct_Q4_K_M_gguf
date: 2025-08-04
tags: [llamacpp, gguf, phi, phi4, en, open_source]
task: Text Generation
language: en
edition: Spark NLP 6.1.1
spark_version: 3.0
supported: true
engine: llamacpp
annotator: AutoGGUFModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Phi-4-mini-instruct is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data. The model belongs to the Phi-4 model family and supports 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning and direct preference optimization to support precise instruction adherence and robust safety measures.

Imported from https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Phi_4_mini_instruct_Q4_K_M_gguf_en_6.1.1_3.0_1754317230916.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Phi_4_mini_instruct_Q4_K_M_gguf_en_6.1.1_3.0_1754317230916.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
autoGGUFModel = AutoGGUFModel.pretrained("Phi_4_mini_instruct_Q4_K_M_gguf") \
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
data = spark.createDataFrame([["Hello, I am a"]]).toDF("text")

result = pipeline.fit(data).transform(data)
result.select("completions").show(truncate = False)
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val autoGGUFModel = AutoGGUFModel
  .pretrained("Phi_4_mini_instruct_Q4_K_M_gguf")
  .setInputCols("document")
  .setOutputCol("completions")
  .setBatchSize(4)
  .setNPredict(20)
  .setNGpuLayers(99)
  .setTemperature(0.4f)
  .setTopK(40)
  .setTopP(0.9f)
  .setPenalizeNl(true)

val pipeline = new Pipeline().setStages(Array(document, autoGGUFModel))

val data = Seq("Hello, I am a").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("completions").show(truncate = false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Phi_4_mini_instruct_Q4_K_M_gguf|
|Compatibility:|Spark NLP 6.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[completions]|
|Language:|en|
|Size:|2.5 GB|