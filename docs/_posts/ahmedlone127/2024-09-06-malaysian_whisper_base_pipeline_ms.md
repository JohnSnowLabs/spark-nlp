---
layout: model
title: Malay (macrolanguage) malaysian_whisper_base_pipeline pipeline WhisperForCTC from mesolitica
author: John Snow Labs
name: malaysian_whisper_base_pipeline
date: 2024-09-06
tags: [ms, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ms
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malaysian_whisper_base_pipeline` is a Malay (macrolanguage) model originally trained by mesolitica.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malaysian_whisper_base_pipeline_ms_5.5.0_3.0_1725605966395.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malaysian_whisper_base_pipeline_ms_5.5.0_3.0_1725605966395.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malaysian_whisper_base_pipeline", lang = "ms")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malaysian_whisper_base_pipeline", lang = "ms")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malaysian_whisper_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ms|
|Size:|314.1 MB|

## References

https://huggingface.co/mesolitica/malaysian-whisper-base

## Included Models

- AudioAssembler
- WhisperForCTC