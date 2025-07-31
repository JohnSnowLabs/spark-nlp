---
layout: model
title: Qwen3-4B GGUF (Q4_K_M Quantized) by Qwen
author: John Snow Labs
name: qwen3_4b_q4_k_m_gguf
date: 2025-07-31
tags: [qwen3, q4, quantized, conversational, en, open_source, llamacpp, gguf]
task: Text Generation
language: en
edition: Spark NLP 6.0.3
spark_version: 3.0
supported: true
engine: llamacpp
annotator: AutoGGUFModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support

Original model from https://huggingface.co/Qwen/Qwen3-4B

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qwen3_4b_q4_k_m_gguf_en_6.0.3_3.0_1753972317683.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qwen3_4b_q4_k_m_gguf_en_6.0.3_3.0_1753972317683.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

auto_gguf_model = AutoGGUFModel.pretrained("qwen3_4b_q4_k_m_gguf", "en") \
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
    ["A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?"]
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

val autoGGUFModel = AutoGGUFModel.pretrained("qwen3_4b_q4_k_m_gguf", "en")
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

val data = Seq("A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("completions").show(false)

```
</div>

## Results

```bash
Explanation:
The phrase "all but 9 run away" means that 9 sheep did not run away, while the remaining (17 - 9 = 8) did. Therefore, the farmer still has the 9 sheep that stayed behind.
Answer: 9.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|qwen3_4b_q4_k_m_gguf|
|Compatibility:|Spark NLP 6.0.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[completions]|
|Language:|en|
|Size:|2.5 GB|