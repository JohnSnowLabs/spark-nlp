---
layout: model
title: Phi-4-mini-Instruct OpenVINO (Q8 Quantized) by Microsoft
author: John Snow Labs
name: phi_4_mini_instruct_int8_openvino
date: 2025-07-25
tags: [openvino, phi4, mini, q8, quantized, instruct, conversational, 128k, en, open_source]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/phi_4_mini_instruct_int8_openvino_en_6.0.0_3.0_1753475514831.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/phi_4_mini_instruct_int8_openvino_en_6.0.0_3.0_1753475514831.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    Phi4Transformer.pretrained("phi_4_mini_instruct_int8_openvino")
    .setMaxOutputLength(100)
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

val phi4 = Phi4Transformer.pretrained("phi_4_mini_instruct_int8_openvino")
  .setMaxOutputLength(100)
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
The Internet, my noble knight, is akin to a vast and intricate tapestry woven from countless threads. Imagine a world where messages can be sent across great distances in mere moments, where libraries are accessible from every corner of the land, and where knowledge and stories flow freely as the wind. This is the realm of the modern world, connected by invisible threads of communication, allowing people to share information, ideas, and companionship without the need for physical travel.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|phi_4_mini_instruct_int8_openvino|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|3.5 GB|