---
layout: model
title: Phi-4-mini-Instruct OpenVINO (Q4 Quantized) by Microsoft
author: John Snow Labs
name: Phi_4_mini_instruct_int4_openvino
date: 2025-07-25
tags: [openvino, phi4, mini, q4, quantized, instruct, conversational, 128k, en, open_source]
task: Text Generation
language: en
edition: Spark NLP 6.0.0
spark_version: 3.0
supported: true
engine: openvino
annotator: Phi4Transformer
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Phi_4_mini_instruct_int4_openvino_en_6.0.0_3.0_1753454352119.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Phi_4_mini_instruct_int4_openvino_en_6.0.0_3.0_1753454352119.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Phi4Transformer
from pyspark.ml import Pipeline

test_data = spark.createDataFrame(
    [
        [
            1,
            """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that provides concise and accurate answers to user queries.
<|start_header_id|>user<|end_header_id|>
Explain the concept of the Internet to a medieval knight.
<|start_header_id|>assistant<|end_header_id|>"""
            .strip().replace("\n", " "),
        ]
    ]
).toDF("id", "text")

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

Phi4 = (
    Phi4Transformer.load(sparknlp_int4_model_name)
    .setMaxOutputLength(120)
    .setInputCols(["documents"])
    .setOutputCol("generation")
)

pipeline = Pipeline().setStages(
    [document_assembler, Phi4]
)

model = pipeline.fit(test_data)
results = model.transform(test_data)

results.select("generation.result").show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.transformer.Phi4Transformer
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val text =
  """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that provides concise and accurate answers to user queries.
<|start_header_id|>user<|end_header_id|>
Explain the concept of the Internet to a medieval knight.
<|start_header_id|>assistant<|end_header_id|>"""
    .stripMargin
    .replaceAll("\n", " ")

val testData = Seq(
  (1, text)
).toDF("id", "text")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val phi4 = Phi4Transformer.load(sparknlpInt4ModelName)
  .setMaxOutputLength(120)
  .setInputCols("documents")
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  phi4
))

val model = pipeline.fit(testData)
val results = model.transform(testData)

results.selectExpr("explode(generation.result) as result").show(false)

```
</div>

## Results

```bash

<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that provides concise and accurate answers to user queries.

<|start_header_id|>user<|end_header_id|>
Explain the concept of the Internet to a medieval knight.

<|start_header_id|>assistant<|end_header_id|>
The internet, my good sir Knight Sir Gareth,
is akin unto an invisible network much like our cobblestone roads but spanning across kingdoms far greater than any map you possess.
Imagine if every squire in your land could send messages instantly over vast distances without crossing paths or carrying physical missives on horseback! 
This is what we call 'the web', where information travels at lightning speed through unseen pathways known as cables buried deep below earth's surface connecting castles worldwide.
Furthermore imagine gathering all knowledge from libraries scattered around realms within moments using magical mirrors reflecting light instead...
Wait no longer!
The magic behind it involves bits

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Phi_4_mini_instruct_int4_openvino|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|2.4 GB|