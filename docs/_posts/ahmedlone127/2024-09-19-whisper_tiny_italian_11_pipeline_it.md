---
layout: model
title: Italian whisper_tiny_italian_11_pipeline pipeline WhisperForCTC from FCameCode
author: John Snow Labs
name: whisper_tiny_italian_11_pipeline
date: 2024-09-19
tags: [it, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: it
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_italian_11_pipeline` is a Italian model originally trained by FCameCode.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_italian_11_pipeline_it_5.5.0_3.0_1726787434948.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_italian_11_pipeline_it_5.5.0_3.0_1726787434948.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_italian_11_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_italian_11_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_italian_11_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|390.9 MB|

## References

https://huggingface.co/FCameCode/whisper-tiny-it-11

## Included Models

- AudioAssembler
- WhisperForCTC