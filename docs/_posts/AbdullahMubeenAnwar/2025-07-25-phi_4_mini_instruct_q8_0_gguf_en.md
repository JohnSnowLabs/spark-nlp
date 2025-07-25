---
layout: model
title: Phi-4-mini-Instruct GGUF (Q8_0 Quantized) by Microsoft
author: John Snow Labs
name: phi_4_mini_instruct_q8_0_gguf
date: 2025-07-25
tags: [gguf, phi4, mini, q8, quantized, instruct, conversational, 128k, en, open_source, llamacpp]
task: Text Generation
language: en
edition: Spark NLP 6.0.0
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

Original model from https://huggingface.co/microsoft/Phi-4-mini-instruct

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/phi_4_mini_instruct_q8_0_gguf_en_6.0.0_3.0_1753405032900.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/phi_4_mini_instruct_q8_0_gguf_en_6.0.0_3.0_1753405032900.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import AutoGGUFModel
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

auto_gguf_model = AutoGGUFModel.pretrained("phi_4_mini_instruct_q8_0_gguf", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("completions") \
    .setBatchSize(4) \
    .setNPredict(20) \
    .setNGpuLayers(99) \
    .setTemperature(0.4) \
    .setTopK(40) \
    .setTopP(0.9) \
    .setPenalizeNl(True)

pipeline = Pipeline().setStages([
    document_assembler,
    auto_gguf_model
])

data = spark.createDataFrame([
    ["The moon is "]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("completions").show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.auto.gguf.AutoGGUFModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val autoGGUFModel = AutoGGUFModel.pretrained("phi_4_mini_instruct_q8_0_gguf", "en")
  .setInputCols("document")
  .setOutputCol("completions")
  .setBatchSize(4)
  .setNPredict(20)
  .setNGpuLayers(99)
  .setTemperature(0.4f)
  .setTopK(40)
  .setTopP(0.9f)
  .setPenalizeNl(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  autoGGUFModel
))

val data = Seq("The moon is ").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("completions").show(false)

```
</div>

## Results

```bash
The moon is Earth's only natural satellite and the fifth largest moon in the Solar System. It orbits,
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|phi_4_mini_instruct_q8_0_gguf|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[completions]|
|Language:|en|
|Size:|3.9 GB|