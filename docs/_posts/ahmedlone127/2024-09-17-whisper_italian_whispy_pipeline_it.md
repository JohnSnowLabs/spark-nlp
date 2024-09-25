---
layout: model
title: Italian whisper_italian_whispy_pipeline pipeline WhisperForCTC from whispy
author: John Snow Labs
name: whisper_italian_whispy_pipeline
date: 2024-09-17
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_italian_whispy_pipeline` is a Italian model originally trained by whispy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_italian_whispy_pipeline_it_5.5.0_3.0_1726543359182.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_italian_whispy_pipeline_it_5.5.0_3.0_1726543359182.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_italian_whispy_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_italian_whispy_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_italian_whispy_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|1.7 GB|

## References

https://huggingface.co/whispy/whisper_italian

## Included Models

- AudioAssembler
- WhisperForCTC