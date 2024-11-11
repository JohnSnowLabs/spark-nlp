---
layout: model
title: None malasar_small_dtf_pipeline pipeline WhisperForCTC from vrclc
author: John Snow Labs
name: malasar_small_dtf_pipeline
date: 2024-11-11
tags: [nan, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: nan
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malasar_small_dtf_pipeline` is a None model originally trained by vrclc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malasar_small_dtf_pipeline_nan_5.5.1_3.0_1731342109583.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malasar_small_dtf_pipeline_nan_5.5.1_3.0_1731342109583.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malasar_small_dtf_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malasar_small_dtf_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malasar_small_dtf_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|1.7 GB|

## References

https://huggingface.co/vrclc/Malasar_small_DTF

## Included Models

- AudioAssembler
- WhisperForCTC