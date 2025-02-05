---
layout: model
title: Czech barticzech_1_0_pipeline pipeline BartTransformer from UWB-AIR
author: John Snow Labs
name: barticzech_1_0_pipeline
date: 2025-02-05
tags: [cs, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: cs
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`barticzech_1_0_pipeline` is a Czech model originally trained by UWB-AIR.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/barticzech_1_0_pipeline_cs_5.5.1_3.0_1738761077941.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/barticzech_1_0_pipeline_cs_5.5.1_3.0_1738761077941.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("barticzech_1_0_pipeline", lang = "cs")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("barticzech_1_0_pipeline", lang = "cs")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|barticzech_1_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|1.9 GB|

## References

https://huggingface.co/UWB-AIR/barticzech-1.0

## Included Models

- DocumentAssembler
- BartTransformer