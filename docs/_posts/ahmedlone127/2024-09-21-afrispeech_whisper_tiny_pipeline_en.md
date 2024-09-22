---
layout: model
title: English afrispeech_whisper_tiny_pipeline pipeline WhisperForCTC from kanyekuthi
author: John Snow Labs
name: afrispeech_whisper_tiny_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`afrispeech_whisper_tiny_pipeline` is a English model originally trained by kanyekuthi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/afrispeech_whisper_tiny_pipeline_en_5.5.0_3.0_1726904737347.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/afrispeech_whisper_tiny_pipeline_en_5.5.0_3.0_1726904737347.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("afrispeech_whisper_tiny_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("afrispeech_whisper_tiny_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|afrispeech_whisper_tiny_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|391.3 MB|

## References

https://huggingface.co/kanyekuthi/AfriSpeech-whisper-tiny

## Included Models

- AudioAssembler
- WhisperForCTC