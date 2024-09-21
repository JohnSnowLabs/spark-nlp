---
layout: model
title: Twi raydox11_whisper_small_pipeline pipeline WhisperForCTC from Raydox10
author: John Snow Labs
name: raydox11_whisper_small_pipeline
date: 2024-09-17
tags: [tw, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: tw
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`raydox11_whisper_small_pipeline` is a Twi model originally trained by Raydox10.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/raydox11_whisper_small_pipeline_tw_5.5.0_3.0_1726549726228.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/raydox11_whisper_small_pipeline_tw_5.5.0_3.0_1726549726228.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("raydox11_whisper_small_pipeline", lang = "tw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("raydox11_whisper_small_pipeline", lang = "tw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|raydox11_whisper_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tw|
|Size:|1.7 GB|

## References

https://huggingface.co/Raydox10/Raydox11-whisper-small

## Included Models

- AudioAssembler
- WhisperForCTC