---
layout: model
title: Corsican whisper_small_corsican_pipeline pipeline WhisperForCTC from ntviet
author: John Snow Labs
name: whisper_small_corsican_pipeline
date: 2024-09-13
tags: [co, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: co
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_corsican_pipeline` is a Corsican model originally trained by ntviet.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_corsican_pipeline_co_5.5.0_3.0_1726221845878.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_corsican_pipeline_co_5.5.0_3.0_1726221845878.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_corsican_pipeline", lang = "co")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_corsican_pipeline", lang = "co")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_corsican_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|co|
|Size:|1.7 GB|

## References

https://huggingface.co/ntviet/whisper-small-co

## Included Models

- AudioAssembler
- WhisperForCTC