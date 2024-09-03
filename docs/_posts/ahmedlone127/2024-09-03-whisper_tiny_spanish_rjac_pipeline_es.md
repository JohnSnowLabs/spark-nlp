---
layout: model
title: Castilian, Spanish whisper_tiny_spanish_rjac_pipeline pipeline WhisperForCTC from rjac
author: John Snow Labs
name: whisper_tiny_spanish_rjac_pipeline
date: 2024-09-03
tags: [es, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_spanish_rjac_pipeline` is a Castilian, Spanish model originally trained by rjac.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_spanish_rjac_pipeline_es_5.5.0_3.0_1725362103843.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_spanish_rjac_pipeline_es_5.5.0_3.0_1725362103843.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_spanish_rjac_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_spanish_rjac_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_spanish_rjac_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|384.0 MB|

## References

https://huggingface.co/rjac/whisper-tiny-spanish

## Included Models

- AudioAssembler
- WhisperForCTC