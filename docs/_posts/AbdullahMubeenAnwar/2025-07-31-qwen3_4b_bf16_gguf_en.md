---
layout: model
title: Qwen3-4B GGUF (F16 Quantized) by Qwen
author: John Snow Labs
name: qwen3_4b_bf16_gguf
date: 2025-07-31
tags: [qwen3, float16, quantized, conversational, en, open_source, llamacpp, gguf]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qwen3_4b_bf16_gguf_en_6.0.3_3.0_1753977562790.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qwen3_4b_bf16_gguf_en_6.0.3_3.0_1753977562790.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

auto_gguf_model = AutoGGUFModel.pretrained("qwen3_4b_bf16_gguf", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("completions") \
    .setBatchSize(4) \
    .setNPredict(-1) \
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
    ["Give me a short introduction to large language model."]
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

val autoGGUFModel = AutoGGUFModel.pretrained("qwen3_4b_bf16_gguf", "en")
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

val data = Seq("Give me a short introduction to large language model.").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("completions").show(false)

```
</div>

## Results

```bash
Large language models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. Trained on vast amounts of data, they can answer questions, write essays, code, create stories, and engage in conversations. These models use deep learning algorithms to recognize patterns in language, enabling them to produce coherent and contextually relevant responses. LLMs have revolutionized fields like customer service, content creation, and research, offering powerful tools for tasks ranging from translation to creative writing. While they are highly capable, their outputs depend on the quality of their training data and the specific instructions given.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|qwen3_4b_bf16_gguf|
|Compatibility:|Spark NLP 6.0.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[completions]|
|Language:|en|
|Size:|6.4 GB|